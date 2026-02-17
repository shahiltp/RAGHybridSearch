from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.core.types import Chunk


class PGVectorStore:
    def __init__(self, dsn: str):
        self.engine: Engine = create_engine(dsn, pool_pre_ping=True, future=True)

    def upsert_document(self, doc_id: str, title: Optional[str] = None, source: Optional[str] = None) -> None:
        q = text("""
        INSERT INTO documents (doc_id, title, source)
        VALUES (:doc_id, :title, :source)
        ON CONFLICT (doc_id) DO UPDATE SET
          title = COALESCE(EXCLUDED.title, documents.title),
          source = COALESCE(EXCLUDED.source, documents.source);
        """)
        with self.engine.begin() as conn:
            conn.execute(q, {"doc_id": doc_id, "title": title, "source": source})

    def upsert_chunks_with_embeddings(
        self,
        chunks: Sequence[Chunk],
        chunk_indices: Sequence[int],
        embeddings: Sequence[List[float]],
    ) -> None:
        """
        Inserts/updates chunks and embeddings.
        - chunks[i].chunk_id is the PK
        - embeddings are stored in pgvector column
        """
        if not (len(chunks) == len(chunk_indices) == len(embeddings)):
            raise ValueError("chunks, chunk_indices, embeddings must have same length")

        with self.engine.begin() as conn:
            for chunk, idx, emb in zip(chunks, chunk_indices, embeddings):
                # 1) Insert chunk first
                conn.execute(
                    text("""
                    INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, metadata)
                    VALUES (:chunk_id, :doc_id, :chunk_index, :text, CAST(:metadata AS jsonb))
                    ON CONFLICT (chunk_id) DO UPDATE SET
                      text = EXCLUDED.text,
                      metadata = EXCLUDED.metadata,
                      chunk_index = EXCLUDED.chunk_index,
                      doc_id = EXCLUDED.doc_id;
                    """),
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": chunk.doc_id,
                        "chunk_index": idx,
                        "text": chunk.text,
                        "metadata": _to_json(chunk.metadata),
                    },
                )

                # 2) Assert chunk exists
                exists = conn.execute(
                    text("SELECT 1 FROM chunks WHERE chunk_id = :chunk_id"),
                    {"chunk_id": chunk.chunk_id},
                ).scalar()

                if exists is None:
                    raise RuntimeError(f"Chunk insert failed for chunk_id={chunk.chunk_id}")

                # 3) Insert embedding after chunk exists
                conn.execute(
                    text("""
                    INSERT INTO embeddings (chunk_id, embedding)
                    VALUES (:chunk_id, CAST(:embedding AS vector))
                    ON CONFLICT (chunk_id) DO UPDATE SET
                      embedding = EXCLUDED.embedding;
                    """),
                    {
                        "chunk_id": chunk.chunk_id,
                        "embedding": _to_pgvector_literal(emb),
                    },
                )


    def semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        doc_id_filter: Optional[str] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Returns (Chunk, distance) sorted by cosine distance ascending.
        """
        where_clause = ""
        params: Dict[str, Any] = {"k": top_k}

        if doc_id_filter:
            where_clause = "WHERE c.doc_id = :doc_id"
            params["doc_id"] = doc_id_filter

        # Use a casted parameter for the query vector to avoid inlining large literals.
        params["q"] = _to_pgvector_literal(query_embedding)

        sql = text(f"""
        SELECT c.chunk_id, c.doc_id, c.text, c.metadata,
               (e.embedding <=> CAST(:q AS vector)) AS distance
        FROM embeddings e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        {where_clause}
        ORDER BY e.embedding <=> CAST(:q AS vector)
        LIMIT :k;
        """)

        out: List[Tuple[Chunk, float]] = []
        with self.engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()
            for r in rows:
                chunk = Chunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    text=r["text"],
                    metadata=r["metadata"] or {},
                )
                out.append((chunk, float(r["distance"])))
        return out


def _to_pgvector_literal(vec: List[float]) -> str:
    # pgvector accepts array-like string: '[1,2,3]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def _to_json(d: Dict[str, Any]) -> str:
    # avoid bringing json libs into core; keep simple
    import json
    return json.dumps(d, ensure_ascii=False)
