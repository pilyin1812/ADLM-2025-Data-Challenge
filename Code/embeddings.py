#!/usr/bin/env python3
"""
Build embeddings only (no interactive loop).
Outputs:
  - index.faiss
  - embeddings.npy (float32)
  - chunks.jsonl (id, path, chunk_idx, text)
  - metadata.pkl ([(path, chunk_idx), ...])

Env:
  OPENAI_API_KEY must be set.
"""

import os
import re
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import fitz                 # PyMuPDF
import numpy as np
import faiss
import tiktoken
from openai import OpenAI

# -------------------- Config --------------------
DOCS_DIR              = "DOCS"
OUTPUT_DIR            = "embeddings_out"
EMBEDDING_MODEL       = "text-embedding-3-small"

# Token-aware chunking
MAX_TOKENS_PER_CHUNK  = 1500   # safe cap for embeddings input
OVERLAP_TOKENS        = 150

# Batch size for embedding API calls
BATCH_SIZE_EMBED      = 64
# ------------------------------------------------

# OpenAI client
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment.")
client = OpenAI(api_key=API_KEY)

# Tokenizer
ENC = tiktoken.get_encoding("cl100k_base")

def clean_text(s: str) -> str:
    """Normalize whitespace and tame pathological long tokens."""
    if not s:
        return ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t\f\r]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    s = re.sub(r"[ \t]+(\n)", r"\1", s)
    # Break absurd long runs (e.g., base64 or concatenated columns)
    s = re.sub(r"(\S{300,})",
               lambda m: " ".join(m.group(1)[i:i+100] for i in range(0, len(m.group(1)), 100)),
               s)
    return s.strip()

def extract_pdf_text(path: Path) -> str:
    out = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            t = page.get_text("text")
            if t:
                out.append(t)
    return clean_text("\n".join(out))

def chunk_by_tokens(text: str,
                    max_tokens: int = MAX_TOKENS_PER_CHUNK,
                    overlap_tokens: int = OVERLAP_TOKENS) -> List[str]:
    toks = ENC.encode(text)
    n = len(toks)
    if n == 0:
        return []
    step = max_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("OVERLAP_TOKENS must be smaller than MAX_TOKENS_PER_CHUNK")

    chunks = []
    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        chunk_tokens = toks[start:end]
        chunks.append(ENC.decode(chunk_tokens))
        if end == n:
            break
        start += step
    return chunks

def embed_batch(texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [d.embedding for d in resp.data]

def build_faiss(emb: np.ndarray) -> faiss.IndexFlatL2:
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb.astype("float32"))
    return index

def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: List[str] = []
    all_meta:   List[Tuple[str, int]] = []  # (filepath, chunk_idx)
    chunk_ids:  List[str] = []              # stable ids for jsonl

    print(f"🔍 Scanning PDFs under {DOCS_DIR!r} ...")
    pdf_count = 0
    for path in Path(DOCS_DIR).rglob("*.pdf"):
        pdf_count += 1
        try:
            raw = extract_pdf_text(path)
        except Exception as e:
            print(f"[!] Skipping {path}: {e}")
            continue

        if not raw.strip():
            print(f"[!] Skipping {path} (no extractable text)")
            continue

        chunks = chunk_by_tokens(raw, MAX_TOKENS_PER_CHUNK, OVERLAP_TOKENS)
        print(f"  • {path.name}: {len(chunks)} chunks")
        for i, ch in enumerate(chunks):
            all_chunks.append(ch)
            all_meta.append((str(path), i))
            chunk_ids.append(f"{path.name}::chunk-{i}")

    total = len(all_chunks)
    print(f"\n✅ PDFs: {pdf_count} → Total chunks: {total}")
    if total == 0:
        print("❌ No chunks generated. Aborting.")
        return

    # Embed in batches
    embeddings: List[List[float]] = []
    print("\n🧠 Embedding (batched)...")
    for i in range(0, total, BATCH_SIZE_EMBED):
        batch = all_chunks[i:i+BATCH_SIZE_EMBED]
        safe_batch = []
        for text in batch:
            toks = ENC.encode(text)
            if len(toks) > MAX_TOKENS_PER_CHUNK:
                text = ENC.decode(toks[:MAX_TOKENS_PER_CHUNK])
            safe_batch.append(text)

        try:
            batch_emb = embed_batch(safe_batch)
        except Exception as e:
            print(f"[!] Embedding failed for batch {i}-{i+len(batch)-1}: {e}")
            print("    Skipping this batch.")
            continue

        embeddings.extend(batch_emb)
        print(f"  🔹 {min(i+BATCH_SIZE_EMBED, total)}/{total}")

    if len(embeddings) == 0:
        print("❌ No embeddings created. Aborting.")
        return

    emb_arr = np.array(embeddings, dtype="float32")
    print(f"\nℹ️ Embedding matrix shape: {emb_arr.shape}")

    # Save embeddings matrix
    np.save(out_dir / "embeddings.npy", emb_arr)

    # Save metadata (list of (path, chunk_idx))
    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(all_meta, f)

    # Save chunks (jsonl with ids/paths for easy rehydrate/citations)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for cid, (p, i), text in zip(chunk_ids, all_meta, all_chunks):
            f.write(json.dumps({"id": cid, "path": p, "chunk_idx": i, "text": text}, ensure_ascii=False) + "\n")

    # Build & save FAISS
    index = build_faiss(emb_arr)
    faiss.write_index(index, str(out_dir / "index.faiss"))

    print("\n💾 Saved to", str(out_dir))
    print("   • embeddings.npy")
    print("   • index.faiss")
    print("   • chunks.jsonl")
    print("   • metadata.pkl")
    print("\n✅ Done. No interactive loop was run.")

if __name__ == "__main__":
    main()
