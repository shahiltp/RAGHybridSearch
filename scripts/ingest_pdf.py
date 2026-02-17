import os
import sys
from pathlib import Path
# Ensure project root is on sys.path so `import app` works when running this file directly.
# File is at <project>/scripts/ingest_pdf.py, so go up two levels to reach the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from openai import OpenAI

from app.indexing.pgvector_store import PGVectorStore
from app.ingestion.chunker import TokenChunker
from app.ingestion.ingest_pdf_pipeline import PDFIngestor

load_dotenv()

if len(sys.argv) < 2:
    print("Usage: python scripts/ingest_pdf.py <path_to_pdf>")
    raise SystemExit(1)

path = sys.argv[1]

store = PGVectorStore(os.environ["PG_DSN"])
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

chunker = TokenChunker(
    model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
    chunk_tokens=int(os.environ.get("CHUNK_TOKENS", "350")),
    overlap_tokens=int(os.environ.get("CHUNK_OVERLAP_TOKENS", "40")),
)

ingestor = PDFIngestor(
    store=store,
    openai_client=client,
    embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
    chunker=chunker,
)

doc_id = ingestor.ingest_pdf(path)
print("✅ Ingested:", path)
print("✅ doc_id:", doc_id)
