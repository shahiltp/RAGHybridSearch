import os
import sys
from pathlib import Path

# Add the workspace root to Python path so app module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from app.indexing.pgvector_store import PGVectorStore
from app.core.types import Chunk

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
store = PGVectorStore(os.environ["PG_DSN"])

def embed(text: str) -> list[float]:
    r = client.embeddings.create(model=os.environ.get("EMBEDDING_MODEL","text-embedding-3-small"), input=text)
    return r.data[0].embedding

# 1) insert a sample doc+chunks
doc_id = "doc_test_1"
store.upsert_document(doc_id, title="Test Doc", source="smoke")

chunks = [
    Chunk(chunk_id="c1", doc_id=doc_id, text="The contract shall be governed by the laws of England.", metadata={"page": 1}),
    Chunk(chunk_id="c2", doc_id=doc_id, text="Termination requires thirty days written notice.", metadata={"page": 2}),
]
embs = [embed(c.text) for c in chunks]
store.upsert_chunks_with_embeddings(chunks, chunk_indices=[0,1], embeddings=embs)

# 2) search
q = "What law governs the agreement?"
q_emb = embed(q)
print(f"Query embedding length: {len(q_emb)}")
print(f"Query embedding (first 5 values): {q_emb[:5]}")

# build literal like pgvector expects
def _to_pgvector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

q_literal = _to_pgvector_literal(q_emb)
print("Query literal length:", len(q_literal))
print("Query literal preview:\n", q_literal[:1000])

hits = store.semantic_search(q_emb, top_k=3)
print(f"Number of hits: {len(hits)}")

print("Top hits:")
for ch, dist in hits:
    print(ch.chunk_id, "dist=", round(dist, 4), "|", ch.text[:80])
