from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.core.types import RetrievedItem, FusedItem


@dataclass(frozen=True)
class RRFWeights:
    k: int = 60
    w_semantic: float = 1.0
    w_bm25: float = 1.0


def _rrf_contribution(rank: int, k: int) -> float:
    # rank is 1-based. Higher rank number => smaller contribution
    return 1.0 / (k + rank)


def weighted_rrf_fuse(
    semantic: List[RetrievedItem],
    bm25: List[RetrievedItem],
    weights: RRFWeights,
    top_n: int = 20,
) -> List[FusedItem]:
    """
    Merge semantic + BM25 result lists using Weighted Reciprocal Rank Fusion (WRRF).

    WRRF(d) = ws/(k+rank_semantic(d)) + wb/(k+rank_bm25(d))

    Scale-free, robust, and ideal before cross-encoder reranking.
    """
    # Map chunk_id -> (chunk, semantic_rank, bm25_rank)
    seen: Dict[str, Tuple] = {}

    for item in semantic:
        cid = item.chunk.chunk_id
        if cid not in seen:
            seen[cid] = (item.chunk, item.rank, None)
        else:
            chunk, s_rank, b_rank = seen[cid]
            seen[cid] = (chunk, s_rank or item.rank, b_rank)

    for item in bm25:
        cid = item.chunk.chunk_id
        if cid not in seen:
            seen[cid] = (item.chunk, None, item.rank)
        else:
            chunk, s_rank, b_rank = seen[cid]
            seen[cid] = (chunk, s_rank, b_rank or item.rank)

    fused: List[FusedItem] = []
    for chunk_id, (chunk, s_rank, b_rank) in seen.items():
        score = 0.0
        if s_rank is not None:
            score += weights.w_semantic * _rrf_contribution(s_rank, weights.k)
        if b_rank is not None:
            score += weights.w_bm25 * _rrf_contribution(b_rank, weights.k)

        fused.append(
            FusedItem(
                chunk=chunk,
                fused_score=score,
                semantic_rank=s_rank,
                bm25_rank=b_rank,
            )
        )

    fused.sort(key=lambda x: x.fused_score, reverse=True)
    return fused[:top_n]
