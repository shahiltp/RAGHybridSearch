from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cohere

from app.core.types import Chunk


@dataclass
class RerankResult:
    chunk: Chunk
    score: float
    original_rank: int


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self.client = cohere.Client(api_key)
        self.model = model

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[RerankResult]:
        docs = [c.text for c in chunks]
        resp = self.client.rerank(
            model=self.model,
            query=query,
            documents=docs,
            top_n=min(top_k, len(docs)),
        )

        out: List[RerankResult] = []
        for r in resp.results:
            idx = r.index
            out.append(
                RerankResult(
                    chunk=chunks[idx],
                    score=float(r.relevance_score),
                    original_rank=idx + 1,
                )
            )
        return out
