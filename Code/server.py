# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# --------- 1) Pluggable responder (edit this later) ----------
class Responder:
    """
    Super simple responder you can replace with real RAG later.
    Call .respond(prompt: str) -> str
    """
    def __init__(self):
        # put any state/init here (load index, model, etc.)
        pass

    def respond(self, prompt: str) -> str:
        # TODO: replace with your logic
        return "draft"

responder = Responder()

# --------- 2) Fake file catalog (for UI right-rail & citations) ----------
FILES = [
    {
        "id": "sop-001",
        "title": "CSF Gram Stain SOP v3",
        "path": "/files/micro/CSF_Gram_Stain_v3.pdf",
        "kind": "pdf",
        "section": "Microbiology",
        "size": 123456
    },
    {
        "id": "sop-014",
        "title": "Hemolysis Rejection Thresholds",
        "path": "/files/chem/Hemolysis_Thresholds.pdf",
        "kind": "pdf",
        "section": "Chemistry",
        "size": 98765
    },
]

# --------- 3) FastAPI app ----------
app = FastAPI(title="Clinical RAG Sketch API")

# If you’re NOT using the Vite dev proxy, enable CORS for your Vite origin:
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ----- /api/files -----
@app.get("/api/files")
def list_files(q: Optional[str] = None) -> List[Dict[str, Any]]:
    if not q:
        return FILES
    ql = q.lower()
    return [f for f in FILES if ql in f["title"].lower()
            or ql in f.get("section", "").lower()
            or ql in f.get("path", "").lower()]

# ----- /api/chat -----
class ChatBody(BaseModel):
    query: str
    topK: int = 5
    temperature: float = 0.1
    stream: bool = False

@app.post("/api/chat")
def chat(b: ChatBody):
    """
    Minimal shape the UI expects:
      { "answer": str, "sources": [ ...file objects... ] }
    """
    answer = responder.respond(b.query)
    # for now, “cite” the first file or none
    sources = FILES[: min(b.topK, len(FILES))]
    return {"answer": answer, "sources": sources}
