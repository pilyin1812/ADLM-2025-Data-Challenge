import os
from typing import Dict, Iterable, Iterator, List, Tuple, Optional

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

# ---- Hardcoded parameters ----
CHROMA_PATH = "TEAM_NAME/chroma"
COLLECTION_NAME = "ADLM_Embeddings_default"
LLM_MODEL = "llama3.1"
TOP_K = 5
MAX_CONTEXT_CHARS = 100_000

PROMPT_TEMPLATE = """
Answer the question using the context below if relevant. 
If the context doesn’t contain the answer, respond 'Context not found.'

{context}

---

Answer the question based on the above context: {question}
""".strip()

# ---- Stub: will find or link local files later ----
def find_relevant_files(citation: Dict) -> Dict:
    """
    Placeholder for your later logic to attach URLs/paths to citations.
    Example (when implemented):
        if "title" in citation:
            citation["path"] = f"docs/{citation['title']}.pdf"
    """
    return citation  # currently no-op


# ---- Internals ----
_db = None
_prompt_tmpl = None
_model = None

def _ensure_db() -> Chroma:
    global _db
    if _db is not None:
        return _db
    if not os.path.isdir(CHROMA_PATH):
        raise RuntimeError(f"Chroma path not found: {CHROMA_PATH}")
    emb_model = os.path.basename(os.path.normpath(CHROMA_PATH))
    _db = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(emb_model),
    )
    return _db

def _ensure_chain() -> Tuple[ChatPromptTemplate, OllamaLLM]:
    global _prompt_tmpl, _model
    if _prompt_tmpl is None:
        _prompt_tmpl = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    if _model is None:
        _model = OllamaLLM(model=LLM_MODEL)
    return _prompt_tmpl, _model


# ---- Public API ----
def query_stream(prompt: str) -> Tuple[Iterable[str], List[Dict]]:
    db = _ensure_db()

    try:
        results = db.similarity_search_with_score(prompt, k=TOP_K)
    except Exception as e:
        def _err() -> Iterator[str]:
            yield f"Retrieval error: {e}"
        return _err(), []

    if not results:
        def _noctx() -> Iterator[str]:
            yield "No relevant context available."
        return _noctx(), []

    # Build context and citations
    context_parts, citations = [], []
    used = 0
    for rank, (doc, score) in enumerate(results, start=1):
        txt = doc.page_content or ""
        if used >= MAX_CONTEXT_CHARS: break
        take = min(len(txt), MAX_CONTEXT_CHARS - used)
        context_parts.append(txt[:take])
        used += take

        meta = dict(doc.metadata or {})
        cite = {
            "title": meta.get("title") or f"doc_{rank}",
            "url": meta.get("url"),
            "path": meta.get("path"),
            "kind": meta.get("kind"),
            "section": meta.get("section"),
            "score": float(score) if score else None,
        }
        cite.update({k: v for k, v in meta.items() if k not in cite})
        cite = find_relevant_files(cite)  # << stub for future linking >>
        citations.append(cite)

    context_text = "\n\n---\n\n".join(context_parts)
    prompt_tmpl, model = _ensure_chain()
    chain = prompt_tmpl | model

    def _gen() -> Iterator[str]:
        for piece in chain.stream({"context": context_text, "question": prompt}):
            yield str(piece)
    return _gen(), citations
