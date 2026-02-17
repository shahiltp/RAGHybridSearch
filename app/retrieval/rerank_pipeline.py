from __future__ import annotations
from typing import List

from app.core.types import Chunk
from app.core.types import FusedItem


def rerank_fused(
    query: str,
    fused: List[FusedItem],
    reranker,
    rerank_top_n: int = 20,
    final_top_k: int = 5,
) -> List[Chunk]:
    # Take top N fused items for cross-encoder rerank
    candidates = fused[: min(rerank_top_n, len(fused))]
    chunks = [c.chunk for c in candidates]

    reranked = reranker.rerank(query, chunks, top_k=final_top_k)
    return [r.chunk for r in reranked]
