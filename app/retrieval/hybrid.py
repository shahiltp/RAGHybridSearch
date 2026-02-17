from __future__ import annotations
from typing import List, Optional, Protocol

from app.core.types import RetrievedItem, FusedItem
from app.retrieval.fusion import weighted_rrf_fuse, RRFWeights


class SemanticRetriever(Protocol):
    def retrieve(self, query: str, top_k: int, doc_id_filter: Optional[str] = None) -> List[RetrievedItem]: ...


class BM25Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> List[RetrievedItem]: ...


class HybridRetriever:
    def __init__(
        self,
        semantic: SemanticRetriever,
        bm25: BM25Retriever,
        weights: RRFWeights,
        fused_top_n: int = 20,
    ):
        self.semantic = semantic
        self.bm25 = bm25
        self.weights = weights
        self.fused_top_n = fused_top_n

    def retrieve(
        self,
        query: str,
        semantic_top_k: int,
        bm25_top_k: int,
        doc_id_filter: Optional[str] = None,
    ) -> List[FusedItem]:
        sem = self.semantic.retrieve(query, semantic_top_k, doc_id_filter=doc_id_filter)
        kw = self.bm25.retrieve(query, bm25_top_k)
        return weighted_rrf_fuse(sem, kw, self.weights, top_n=self.fused_top_n)
