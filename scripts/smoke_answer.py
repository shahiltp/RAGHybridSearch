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
from app.generation.openai_client import OpenAILLM
from app.generation.answerer import Answerer

load_dotenv()

# Embeddings (OpenAI)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
store = PGVectorStore(os.environ["PG_DSN"])

def embed(text: str) -> list[float]:
    r = client.embeddings.create(model=os.environ.get("EMBEDDING_MODEL","text-embedding-3-small"), input=text)
    return r.data[0].embedding

# Retrieve
semantic = PGVectorSemanticRetriever(store, embed_fn=embed)
DOC_ID = "doc_d95f9a5dd762"
bm25 = BM25Retriever(BM25Index.build_from_pg(store, doc_id_filter=DOC_ID))
hybrid = HybridRetriever(
    semantic=semantic,
    bm25=bm25,
    weights=RRFWeights(k=settings.rrf_k, w_semantic=settings.w_semantic, w_bm25=settings.w_bm25),
    fused_top_n=settings.fused_top_n,
)

query = "What is English law part 1 about?"
#fused = hybrid.retrieve(query, settings.semantic_top_k, settings.bm25_top_k)
fused = hybrid.retrieve(query, settings.semantic_top_k, settings.bm25_top_k, doc_id_filter=DOC_ID)

# Rerank
reranker = CohereReranker(
    api_key=os.environ["COHERE_API_KEY"],
    model=os.environ.get("COHERE_RERANK_MODEL","rerank-english-v3.0")
)
context_chunks = rerank_fused(query, fused, reranker, rerank_top_n=20, final_top_k=settings.final_top_k)

# Generate
llm = OpenAILLM(api_key=os.environ["OPENAI_API_KEY"], model=os.environ.get("LLM_MODEL","gpt-4o-mini"))
answerer = Answerer(llm)

ans = answerer.answer(query, context_chunks)
print("\nANSWER:\n", ans)
