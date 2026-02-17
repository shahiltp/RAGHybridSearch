from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedItem:
    chunk: Chunk
    source: str                 # "semantic" | "bm25" | "fused"
    rank: int                   # 1-based rank in that list
    score: Optional[float] = None


@dataclass(frozen=True)
class FusedItem:
    chunk: Chunk
    fused_score: float
    semantic_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


@dataclass(frozen=True)
class Citation:
    doc_id: str
    chunk_id: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
