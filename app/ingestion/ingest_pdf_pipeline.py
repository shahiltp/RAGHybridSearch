from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from app.core.types import Chunk
from app.indexing.pgvector_store import PGVectorStore
from app.ingestion.chunker import TokenChunker
from app.ingestion.pdf_loader import load_pdf
from sqlalchemy import text


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def doc_id_from_path(path: str) -> str:
    p = Path(path).resolve()
    return f"doc_{_sha1(str(p))[:12]}"


def chunk_id(doc_id: str, page: int, chunk_index: int, text: str) -> str:
    # stable across reruns: doc + page + index + content
    return f"ch_{_sha1(doc_id + f':p{page}:' + str(chunk_index) + text)[:12]}"


class PDFIngestor:
    def __init__(
        self,
        store: PGVectorStore,
        openai_client: OpenAI,
        embedding_model: str,
        chunker: TokenChunker,
    ):
        self.store = store
        self.client = openai_client
        self.embedding_model = embedding_model
        self.chunker = chunker

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [d.embedding for d in resp.data]

    def ingest_pdf(self, path: str) -> str:
        
        did = doc_id_from_path(path)
        with self.store.engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM documents WHERE doc_id = :doc_id"),
                {"doc_id": did},
            ).scalar()
        if exists:
            return did
        loaded = load_pdf(path)
        self.store.upsert_document(doc_id=did, title=loaded.title, source=loaded.source)

        all_chunks: List[Chunk] = []
        all_chunk_indices: List[int] = []
        all_texts: List[str] = []

        for page_obj in loaded.pages:
            page_text = page_obj["text"]
            base_meta: Dict = {
                "source": loaded.source,
                "title": loaded.title,
                **page_obj["meta"],  # includes page
            }

            chunked = self.chunker.chunk(page_text, base_meta)
            for ct in chunked:
                idx = int(ct.meta["chunk_index"])
                page = int(ct.meta.get("page", 0))
                cid = chunk_id(did, page, idx, ct.text)

                all_chunks.append(Chunk(chunk_id=cid, doc_id=did, text=ct.text, metadata=ct.meta))
                all_chunk_indices.append(idx)
                all_texts.append(ct.text)

        # embeddings in batches
        batch_size = 64
        all_embeddings: List[List[float]] = []
        for i in range(0, len(all_texts), batch_size):
            all_embeddings.extend(self.embed_batch(all_texts[i:i + batch_size]))

        self.store.upsert_chunks_with_embeddings(all_chunks, all_chunk_indices, all_embeddings)
        return did
