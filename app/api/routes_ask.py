from __future__ import annotations

from typing import Dict, Any, Optional, List

from fastapi import APIRouter
from dotenv import load_dotenv
from openai import OpenAI

from app.api.schemas import AskRequest, AskResponse, Citation
from app.core.config import settings

from app.indexing.pgvector_store import PGVectorStore
from app.retrieval.semantic_pgvector import PGVectorSemanticRetriever
from app.indexing.bm25_index import BM25Index
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.fusion import RRFWeights

from app.rerank.cohere_reranker import CohereReranker
from app.retrieval.rerank_pipeline import rerank_fused

from app.generation.openai_client import OpenAILLM
from app.generation.answerer import Answerer
from app.generation.citation_guard import citations_with_pages

load_dotenv()
router = APIRouter()

# Shared clients
_store = PGVectorStore(settings.pg_dsn)
_oai = OpenAI(api_key=settings.openai_api_key)

def _embed(text: str) -> list[float]:
    r = _oai.embeddings.create(model=settings.embedding_model, input=text)
    return r.data[0].embedding

_semantic = PGVectorSemanticRetriever(_store, embed_fn=_embed)

# Reranker (Cohere)
_reranker = CohereReranker(
    api_key=settings.cohere_api_key or "",
    model=settings.cohere_rerank_model or "rerank-english-v3.0",
)

_llm = OpenAILLM(api_key=settings.openai_api_key, model=settings.llm_model)
_answerer = Answerer(_llm)


def _debug_items_from_fused(fused, limit=10) -> List[Dict[str, Any]]:
    out = []
    for f in fused[:limit]:
        page = None
        if isinstance(f.chunk.metadata, dict):
            p = f.chunk.metadata.get("page")
            if p is not None:
                try:
                    page = int(p)
                except Exception:
                    page = None
        out.append({
            "chunk_id": f.chunk.chunk_id,
            "doc_id": f.chunk.doc_id,
            "page": page,
            "fused_score": f.fused_score,
            "semantic_rank": f.semantic_rank,
            "bm25_rank": f.bm25_rank,
            "preview": (f.chunk.text or "")[:160],
        })
    return out


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    doc_id_filter = req.doc_id

    # BM25 index per request if doc_id provided (simple + correct)
    # (Later we can cache BM25Index per doc_id)
    bm25_index = BM25Index.build_from_pg(_store, doc_id_filter=doc_id_filter)
    bm25 = BM25Retriever(bm25_index)

    hybrid = HybridRetriever(
        semantic=_semantic,
        bm25=bm25,
        weights=RRFWeights(
            k=settings.rrf_k,
            w_semantic=settings.w_semantic,
            w_bm25=settings.w_bm25,
        ),
        fused_top_n=settings.fused_top_n,
    )

    fused = hybrid.retrieve(
        req.query,
        settings.semantic_top_k,
        settings.bm25_top_k,
        doc_id_filter=doc_id_filter,
    )

    context_chunks = rerank_fused(
        query=req.query,
        fused=fused,
        reranker=_reranker,
        rerank_top_n=20,
        final_top_k=settings.final_top_k,
    )

    answer = _answerer.answer(req.query, context_chunks)
    clean_answer = answer
    clean_answer = clean_answer.replace("ANSWER:", "").strip()
    if "CITATIONS:" in clean_answer:
        clean_answer = clean_answer.split("CITATIONS:")[0].strip()
    
    cits = citations_with_pages(answer, context_chunks)

    debug: Optional[Dict[str, Any]] = None
    if req.debug:
        debug = debug or {}
        debug["contexts"] = [c.text for c in context_chunks]
        debug = {
            "doc_id_filter": doc_id_filter,
            "fused_top": _debug_items_from_fused(fused, limit=10),
            "contexts": [c.text for c in context_chunks],
            "context_pages": [
                {"chunk_id": c.chunk_id, "page": c.metadata.get("page") if isinstance(c.metadata, dict) else None}
                for c in context_chunks
            ],
        }

    return AskResponse(
        answer=clean_answer,
        citations=[Citation(**c) for c in cits],
        debug=debug,
    )
