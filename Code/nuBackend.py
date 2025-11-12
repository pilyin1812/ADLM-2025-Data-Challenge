# backend.py
"""
FastAPI RAG MVP
- Loads FAISS + chunks from embeddings_out/
- /api/rag/query  : non-streaming
- /api/rag/stream : SSE streaming
- Returns answer + sources [{title,url?,snippet}]
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
import tiktoken
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse
from openai import OpenAI

# =====================
# Config
# =====================
EMB_DIR = Path("embeddings_out")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-nano"

TOP_K_DEFAULT = 5
MAX_CONTEXT_TOKENS = 4500         # budget for retrieved chunks (safe for mini models)
SYSTEM_PROMPT = (
    "You are a careful clinical laboratory assistant. "
    "Answer using ONLY the provided SOP context when possible. "
    "If the answer depends on clinical judgment beyond SOP scope, say so clearly. "
    "Use concise, stepwise guidance and include bracketed citations like [1], [2] that map to the provided sources."
)

# =====================
# OpenAI client
# =====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment.")
client = OpenAI(api_key=OPENAI_API_KEY)

# Tokenizer for budgeting (cl100k_base covers GPT-4o/4.1 families)
ENC = tiktoken.get_encoding("cl100k_base")


# =====================
# Data loading
# =====================
def load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_index_and_corpus() -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]:
    index = faiss.read_index(str(EMB_DIR / "index.faiss"))
    chunks = load_chunks_jsonl(EMB_DIR / "chunks.jsonl")
    if len(chunks) != index.ntotal:
        # This can happen if a batch failed during embedding; warn loudly.
        print(f"[!] Warning: chunks count ({len(chunks)}) != index.ntotal ({index.ntotal}). "
              "Make sure your build script completed consistently.")
    return index, chunks


INDEX, CORPUS = load_index_and_corpus()


# =====================
# RAG helpers
# =====================
def embed_text(text: str) -> List[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def search_index(query: str, top_k: int) -> List[Tuple[int, float]]:
    q_emb = np.array([embed_text(query)], dtype="float32")
    D, I = INDEX.search(q_emb, top_k)
    return list(zip(I[0].tolist(), D[0].tolist()))

def token_len(s: str) -> int:
    return len(ENC.encode(s))

def trim_context(chunks: List[Dict[str, Any]], budget_tokens: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Take a list of chunk dicts (with 'text'), pack into a single context string
    while respecting token budget. Returns (context_str, used_chunks).
    """
    used = []
    ctx_parts = []
    total = 0
    for i, ch in enumerate(chunks, start=1):
        # Prepend a small header so the model understands chunk boundaries (helps with citations)
        piece = f"\n[CHUNK {i}]\n{ch['text'].strip()}\n"
        tlen = token_len(piece)
        if total + tlen > budget_tokens:
            break
        total += tlen
        used.append(ch)
        ctx_parts.append(piece)
    return "".join(ctx_parts).strip(), used

def build_sources(used_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build UI sources array. Map each used chunk to a source with title/path/snippet.
    """
    sources = []
    for i, ch in enumerate(used_chunks, start=1):
        title = Path(ch["path"]).name
        snippet = ch["text"][:240].replace("\n", " ").strip()
        sources.append({
            "title": f"[{i}] {title}  (chunk {ch['chunk_idx']})",
            # if you have canonical URLs, put them here; otherwise omit or use file path
            "url": None,
            "snippet": snippet
        })
    return sources

def build_messages(system_prompt: str, context: str, question: str, used_chunks_len: int) -> List[Dict[str, str]]:
    citation_hint = (
        f"The following SOP context contains {used_chunks_len} chunks labelled as [CHUNK i]. "
        f"When you use information from a chunk, cite it with the bracket number [i] that matches the source list."
    )
    user_prompt = (
        f"{citation_hint}\n\n"
        f"--- SOP CONTEXT START ---\n{context}\n--- SOP CONTEXT END ---\n\n"
        f"Question: {question}\n"
        f"Answer with clear steps and include citations like [1], [2] that map to the source list."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# =====================
# FastAPI app
# =====================
app = FastAPI(title="Clinical Lab SOP RAG (MVP)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K_DEFAULT
    cite_sources: bool = True

@app.get("/health")
def health():
    return {"ok": True, "chunks": len(CORPUS), "index_ntotal": INDEX.ntotal}

@app.post("/api/rag/query")
def rag_query(req: QueryRequest):
    """
    Non-streaming: returns full answer + sources.
    """
    try:
        results = search_index(req.query, max(1, min(req.top_k, len(CORPUS))))
        retrieved = []
        for idx, dist in results:
            if idx < 0 or idx >= len(CORPUS):
                continue
            retrieved.append(CORPUS[idx])

        context, used_chunks = trim_context(retrieved, MAX_CONTEXT_TOKENS)
        if not context:
            # fall back to answering without context
            messages = [{"role": "user", "content": req.query}]
        else:
            messages = build_messages(SYSTEM_PROMPT, context, req.query, len(used_chunks))

        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
        )
        answer = completion.choices[0].message.content

        sources = build_sources(used_chunks) if req.cite_sources else []
        return {"answer": answer, "sources": sources}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}

@app.post("/api/rag/stream")
def rag_stream(req: QueryRequest):
    """
    Streaming with Server-Sent Events (SSE).
    Emits token deltas as 'data: {"delta": "..."}', then one 'sources' event at the end.
    """
    try:
        results = search_index(req.query, max(1, min(req.top_k, len(CORPUS))))
        retrieved = []
        for idx, dist in results:
            if idx < 0 or idx >= len(CORPUS):
                continue
            retrieved.append(CORPUS[idx])

        context, used_chunks = trim_context(retrieved, MAX_CONTEXT_TOKENS)
        if not context:
            messages = [{"role": "user", "content": req.query}]
        else:
            messages = build_messages(SYSTEM_PROMPT, context, req.query, len(used_chunks))

        def gen():
            stream = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield f"data: {json.dumps({'delta': delta})}\n\n"

            if req.cite_sources:
                sources = build_sources(used_chunks)
            else:
                sources = []
            yield "event: sources\n"
            yield f"data: {json.dumps({'sources': sources})}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    except Exception as e:
        def err_gen():
            yield f"data: {json.dumps({'delta': 'Error: ' + str(e)})}\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")
