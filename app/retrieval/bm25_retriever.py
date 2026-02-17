from __future__ import annotations

from typing import List

from app.core.types import RetrievedItem
from app.indexing.bm25_index import BM25Index


class BM25Retriever:
    def __init__(self, index: BM25Index):
        self.index = index

    def retrieve(self, query: str, top_k: int) -> List[RetrievedItem]:
        hits = self.index.search(query, top_k=top_k)
        items: List[RetrievedItem] = []
        for rank, (chunk, score) in enumerate(hits, start=1):
            items.append(RetrievedItem(chunk=chunk, source="bm25", rank=rank, score=score))
        return items
