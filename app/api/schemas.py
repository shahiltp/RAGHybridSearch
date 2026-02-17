from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class AskRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    debug: bool = False


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    page: Optional[int] = None


class RetrievedDebugItem(BaseModel):
    chunk_id: str
    doc_id: str
    page: Optional[int] = None
    score: float
    rank: Optional[int] = None
    preview: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    debug: Optional[Dict[str, Any]] = None
