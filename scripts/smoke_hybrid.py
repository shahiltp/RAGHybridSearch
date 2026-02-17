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

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
store = PGVectorStore(os.environ["PG_DSN"])

def embed(text: str) -> list[float]:
    r = client.embeddings.create(model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"), input=text)
    return r.data[0].embedding

# Build retrievers
semantic = PGVectorSemanticRetriever(store, embed_fn=embed)
bm25_index = BM25Index.build_from_pg(store)
bm25 = BM25Retriever(BM25Index.build_from_pg(store, doc_id_filter="doc_d95f9a5dd762"))

hybrid = HybridRetriever(
    semantic=semantic,
    bm25=bm25,
    weights=RRFWeights(k=settings.rrf_k, w_semantic=settings.w_semantic, w_bm25=settings.w_bm25),
    fused_top_n=settings.fused_top_n,
)

query = "What law governs the agreement?"
fused = hybrid.retrieve(query, settings.semantic_top_k, settings.bm25_top_k)

print("FUSED TOP:")
for i, f in enumerate(fused[:10], start=1):
    print(i, f.chunk.chunk_id, "score=", round(f.fused_score, 6), "|", f.chunk.text[:80], "| ranks:", f.semantic_rank, f.bm25_rank)
