import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from app.core.config import settings
from app.indexing.pgvector_store import PGVectorStore
from app.retrieval.semantic_pgvector import PGVectorSemanticRetriever
from app.indexing.bm25_index import BM25Index
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.fusion import RRFWeights
from app.rerank.cohere_reranker import CohereReranker
from app.retrieval.rerank_pipeline import rerank_fused

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
store = PGVectorStore(os.environ["PG_DSN"])

def embed(text: str) -> list[float]:
    r = client.embeddings.create(model=os.environ.get("EMBEDDING_MODEL","text-embedding-3-small"), input=text)
    return r.data[0].embedding

semantic = PGVectorSemanticRetriever(store, embed_fn=embed)
bm25 = BM25Retriever(BM25Index.build_from_pg(store, doc_id_filter="doc_d95f9a5dd762"))

hybrid = HybridRetriever(
    semantic=semantic,
    bm25=bm25,
    weights=RRFWeights(k=settings.rrf_k, w_semantic=settings.w_semantic, w_bm25=settings.w_bm25),
    fused_top_n=settings.fused_top_n,
)

query = "What law governs the agreement?"
fused = hybrid.retrieve(query, settings.semantic_top_k, settings.bm25_top_k)

reranker = CohereReranker(
    api_key=os.environ["COHERE_API_KEY"],
    model=os.environ.get("COHERE_RERANK_MODEL","rerank-english-v3.0")
)

final_chunks = rerank_fused(
    query=query,
    fused=fused,
    reranker=reranker,
    rerank_top_n=20,
    final_top_k=settings.final_top_k,
)

print("FINAL CONTEXT (after rerank):")
for i, ch in enumerate(final_chunks, start=1):
    print(i, ch.chunk_id, "|", ch.text[:90])
