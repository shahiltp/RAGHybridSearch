from __future__ import annotations

from typing import List, Optional

from app.core.types import RetrievedItem
from app.indexing.pgvector_store import PGVectorStore


class PGVectorSemanticRetriever:
    def __init__(self, store: PGVectorStore, embed_fn):
        """
        embed_fn(query: str) -> List[float]
        Keep as dependency injection so we can swap OpenAI embeddings later.
        """
        self.store = store
        self.embed_fn = embed_fn

    def retrieve(self, query: str, top_k: int, doc_id_filter: Optional[str] = None) -> List[RetrievedItem]:
        q_emb = self.embed_fn(query)
        results = self.store.semantic_search(q_emb, top_k=top_k, doc_id_filter=doc_id_filter)

        # distance: smaller is better. rank is 1-based.
        items: List[RetrievedItem] = []
        for i, (chunk, dist) in enumerate(results, start=1):
            items.append(RetrievedItem(chunk=chunk, source="semantic", rank=i, score=-dist))
        return items
