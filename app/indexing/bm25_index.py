from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from rank_bm25 import BM25Okapi
from sqlalchemy import text

from app.core.types import Chunk
from app.indexing.pgvector_store import PGVectorStore


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def simple_tokenize(s: str) -> List[str]:
    # Simple tokenizer (good enough for baseline). Swap later if needed.
    return [t.lower() for t in _WORD_RE.findall(s)]


@dataclass
class BM25Index:
    chunks: List[Chunk]
    tokenized: List[List[str]]
    bm25: Optional[BM25Okapi]

    @classmethod
    def build_from_pg(cls, store: PGVectorStore, doc_id_filter: Optional[str] = None) -> "BM25Index":
        where_clause = ""
        params: Dict[str, Any] = {}
        if doc_id_filter:
            where_clause = "WHERE doc_id = :doc_id"
            params["doc_id"] = doc_id_filter

        sql = text(f"""
        SELECT chunk_id, doc_id, text, metadata
        FROM chunks
        {where_clause}
        ORDER BY doc_id, chunk_index;
        """)

        chunks: List[Chunk] = []
        with store.engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
            for r in rows:
                chunks.append(
                    Chunk(
                        chunk_id=r["chunk_id"],
                        doc_id=r["doc_id"],
                        text=r["text"],
                        metadata=r["metadata"] or {},
                    )
                )

        tokenized = [simple_tokenize(c.text) for c in chunks]

        # Safety: if no chunks, return empty BM25 (prevents crash from empty corpus)
        if len(chunks) == 0:
            return cls(chunks=[], tokenized=[], bm25=None)  # type: ignore

        bm25 = BM25Okapi(tokenized)
        return cls(chunks=chunks, tokenized=tokenized, bm25=bm25)

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        # Safety: if no chunks or BM25 not initialized, return empty results
        if self.bm25 is None or not self.chunks:
            return []

        q_tokens = simple_tokenize(query)
        scores = self.bm25.get_scores(q_tokens)

        # Get top_k indices by score desc
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked]
