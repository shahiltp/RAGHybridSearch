# Hybrid Search Legal Mind 🏛️

A production-ready **hybrid RAG (Retrieval-Augmented Generation) system** for legal documents, combining semantic search, BM25 full-text search, RRF fusion, and intelligent reranking to deliver precise legal information retrieval.

---

## 🎯 Features

✅ **Hybrid Retrieval**: Semantic search (pgvector embeddings) + BM25 full-text with RRF fusion  
✅ **Intelligent Reranking**: Cohere reranker to refine top results  
✅ **LLM-Powered Generation**: OpenAI GPT-4o for legal question answering  
✅ **Production API**: FastAPI endpoints for ingestion, retrieval, and QA  
✅ **Multi-format Support**: PDF and DOCX document ingestion  
✅ **Evaluation Suite**: RAGAS metrics for RAG quality assessment  
✅ **Docker Ready**: PostgreSQL + pgvector in Docker for easy setup  

---

## 📋 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │   Hybrid Retrieval      │
        │  ┌─────────────────────┐│
        │  │ Semantic Search     ││ (pgvector + OpenAI embeddings)
        │  └─────────────────────┘│
        │  ┌─────────────────────┐│
        │  │ BM25 Full-Text      ││ (rank-bm25)
        │  └─────────────────────┘│
        └────────────┬─────────────┘
                     │
        ┌────────────▼───────────┐
        │  RRF Fusion (rank_bm25)│ Reciprocal Rank Fusion
        └────────────┬───────────┘
                     │
        ┌────────────▼──────────────┐
        │  Reranking (Cohere)       │ Optional intelligent refinement
        └────────────┬──────────────┘
                     │
        ┌────────────▼──────────────┐
        │  Top-K Results            │
        └────────────┬──────────────┘
                     │
        ┌────────────▼──────────────┐
        │  LLM Generation (GPT-4o)  │ Answer synthesis
        └────────────┬──────────────┘
                     │
        ┌────────────▼──────────────┐
        │  Final Answer             │
        └───────────────────────────┘

Data Flow:
Document → Chunking → Embedding → pgvector + BM25 Index → Database
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Docker & Docker Compose** (for PostgreSQL + pgvector)
- **OpenAI API Key** (for embeddings and LLM)
- **Cohere API Key** (optional, for reranking)

### 1. Clone & Setup

```bash
# Clone the repository
git clone <repo-url>
cd hybridsearchLegalMind

# Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

**`.env` template:**
```env
# OpenAI API
OPENAI_API_KEY=sk-proj-your-key-here
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# PostgreSQL + pgvector
PG_DSN=postgresql+psycopg://postgres:postgres@localhost:5433/legalmind

# Retrieval parameters
SEMANTIC_TOP_K=30
BM25_TOP_K=30
FUSED_TOP_N=20

# RRF fusion weights
RRF_K=60
W_SEMANTIC=1.0
W_BM25=1.2

# Final context size
FINAL_TOP_K=5

# Reranking (optional)
RERANK_PROVIDER=cohere
COHERE_API_KEY=your-cohere-key
COHERE_RERANK_MODEL=rerank-english-v3.0

# Ingestion chunking
CHUNK_TOKENS=350
CHUNK_OVERLAP_TOKENS=40
```

### 3. Start PostgreSQL + pgvector

```bash
# Start Docker services
docker-compose up -d

# Initialize database schema
docker exec -it legalmind_pg psql -U postgres -d legalmind -f scripts/init_db.sql
```

### 4. Verify Setup

```bash
# Run semantic search smoke test
python scripts/smoke_semantic.py

# Run reranking smoke test
python scripts/smoke_rerank.py

# Run full answer pipeline smoke test
python scripts/smoke_answer.py
```

---

## 🔧 Core Modules

### Indexing

#### `app/indexing/pgvector_store.py`
Manages semantic search using PostgreSQL pgvector extension.

**Key methods:**
- `upsert_document(doc_id, title, source)` - Register document
- `upsert_chunks_with_embeddings(chunks, indices, embeddings)` - Insert chunks + vectors
- `semantic_search(query_embedding, top_k, doc_id_filter)` - Vector similarity search

**Usage:**
```python
from app.indexing.pgvector_store import PGVectorStore
from openai import OpenAI

store = PGVectorStore(os.environ["PG_DSN"])
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Generate embedding
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input="What are contract terms?"
).data[0].embedding

# Search
results = store.semantic_search(query_embedding, top_k=5)
for chunk, distance in results:
    print(f"{chunk.chunk_id}: {chunk.text} (distance: {distance:.4f})")
```

#### `app/indexing/bm25_index.py`
Full-text search using BM25 ranking algorithm.

**Key methods:**
- `BM25Index.build_from_pg(store, doc_id_filter)` - Build index from database
- `search(query, top_k)` - BM25 search

**Usage:**
```python
from app.indexing.bm25_index import BM25Index

bm25 = BM25Index.build_from_pg(store, doc_id_filter="doc_123")
results = bm25.search("termination clause", top_k=5)
for chunk, score in results:
    print(f"{chunk.chunk_id}: {score:.2f}")
```

#### `app/indexing/fusion.py`
Combines semantic + BM25 results using Reciprocal Rank Fusion (RRF).

**Usage:**
```python
from app.indexing.fusion import fuse_results

semantic_results = store.semantic_search(...)
bm25_results = bm25.search(...)

fused = fuse_results(
    semantic_results=semantic_results,
    bm25_results=bm25_results,
    w_semantic=1.0,
    w_bm25=1.2,
    rrf_k=60,
    top_n=20
)
```

### Ingestion

#### `scripts/ingest_pdf.py`
Ingest PDF documents into the system.

```bash
python scripts/ingest_pdf.py "/path/to/contract.pdf"
# Output: ✅ Ingested: /path/to/contract.pdf
#         ✅ doc_id: doc_abc123def
```

#### `scripts/ingest_docx.py`
Ingest DOCX documents into the system.

```bash
python scripts/ingest_docx.py "/path/to/document.docx"
```

**Process:**
1. Load document (PDF/DOCX)
2. Chunk into sentences (token-based, configurable overlap)
3. Generate embeddings via OpenAI
4. Store chunks + embeddings + metadata in PostgreSQL

### Retrieval

#### `app/retrieval/hybrid.py`
Orchestrates the full hybrid retrieval pipeline.

**Usage:**
```python
from app.retrieval.hybrid import HybridRetriever

retriever = HybridRetriever(store=store, settings=settings)
results = retriever.retrieve(
    query="What are termination clauses?",
    doc_id="doc_123",
    top_k=5
)
```

### Reranking

#### `app/rerank/cohere.py`
Intelligent reranking using Cohere API (optional).

Improves result relevance by rescoring top-K results from hybrid search.

### Generation

#### `app/generation/llm.py`
LLM-powered answer synthesis from retrieved context.

Formats context and query into prompt, calls GPT-4o, returns structured answer.

---

## 🛠️ Configuration

All settings are managed via environment variables (see `.env`).

### Retrieval Tuning

- **`SEMANTIC_TOP_K`** (default: 30) - Results from semantic search
- **`BM25_TOP_K`** (default: 30) - Results from BM25 search
- **`FUSED_TOP_N`** (default: 20) - Results after RRF fusion
- **`FINAL_TOP_K`** (default: 5) - Final context chunks sent to LLM

### RRF Weights

- **`RRF_K`** (default: 60) - RRF normalization parameter
- **`W_SEMANTIC`** (default: 1.0) - Semantic search weight
- **`W_BM25`** (default: 1.2) - BM25 search weight (higher = prioritize keyword matches)

### Ingestion Chunking

- **`CHUNK_TOKENS`** (default: 350) - Tokens per chunk
- **`CHUNK_OVERLAP_TOKENS`** (default: 40) - Overlap between consecutive chunks

---

## 🌐 API Endpoints

### Start API Server

```bash
python -m uvicorn app.api.routes:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

#### `POST /ask`
Ask a legal question and get an AI-generated answer.

**Request:**
```json
{
  "query": "What are the primary sources of English law?",
  "doc_id": "doc_abc123",
  "debug": false
}
```

**Response:**
```json
{
  "answer": "According to the documents, the primary sources of English law are...",
  "retrieved_chunks": [
    {
      "chunk_id": "c1",
      "text": "The contract shall be governed by the laws of England.",
      "distance": 0.12,
      "source": "doc_abc123"
    }
  ],
  "debug_info": {
    "semantic_results": [...],
    "bm25_results": [...],
    "fused_results": [...],
    "reranked_results": [...]
  }
}
```

#### `POST /ingest`
Ingest a document (PDF/DOCX).

**Request:**
```json
{
  "file_path": "/path/to/document.pdf"
}
```

**Response:**
```json
{
  "doc_id": "doc_abc123",
  "chunks_count": 42,
  "embeddings_stored": 42
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "embeddings_api": "connected"
}
```

---

## 📊 Evaluation & Quality

Run the RAGAS evaluation suite to assess RAG quality:

```bash
python -m app.eval.run_eval
```

**Metrics:**
- **Faithfulness** - Answer adherence to retrieved context
- **Answer Relevance** - Answer relevance to query
- **Context Precision** - Precision of retrieved context
- **Context Recall** - Recall of relevant context

---

## 🧪 Testing

### Smoke Tests

```bash
# Test semantic search
python scripts/smoke_semantic.py

# Test reranking pipeline
python scripts/smoke_rerank.py

# Test full QA pipeline
python scripts/smoke_answer.py
```

### Unit Tests

```bash
pytest tests/ -v
```

---

## 🐳 Docker

### Start PostgreSQL + pgvector

```bash
docker-compose up -d
```

### Initialize Database

```bash
docker exec -it legalmind_pg psql -U postgres -d legalmind -f scripts/init_db.sql
```

### View Database

```bash
docker exec -it legalmind_pg psql -U postgres -d legalmind

# Inside psql:
SELECT doc_id, title, source FROM documents LIMIT 10;
SELECT COUNT(*) FROM chunks WHERE doc_id = 'doc_abc123';
SELECT COUNT(*) FROM embeddings;
```

### Stop Services

```bash
docker-compose down
```

---

## ⚠️ Known Limitations & Future Work

- **Scanned PDFs**: Documents without extractable text (images) require OCR support (e.g., Tesseract, AWS Textract)
- **BM25 Empty Handling**: Returns 0 results for docs with 0 chunks (safe fallback)
- **Embedding Costs**: High-dimensional embeddings (1536/3072 dims) increase pgvector index size
- **Reranking Optional**: Cohere reranking is optional; hybrid search works without it

### Future Enhancements

- [ ] OCR support for scanned PDFs
- [ ] Multi-language support
- [ ] Streaming LLM responses
- [ ] Custom reranking models
- [ ] Batch document ingestion API
- [ ] Caching layer for frequent queries
- [ ] Advanced metrics dashboard

---

## 🤝 Contributing

1. Create a feature branch
2. Make changes (ensure smoke tests pass)
3. Update README with new features
4. Submit PR

---

## 📝 License

MIT License - See LICENSE file for details

---

## 💡 Example Workflow

```bash
# 1. Start services
docker-compose up -d

# 2. Initialize DB
docker exec -it legalmind_pg psql -U postgres -d legalmind -f scripts/init_db.sql

# 3. Ingest document
python scripts/ingest_pdf.py "./contracts/sample_contract.pdf"
# Output: doc_id: doc_sample123

# 4. Test retrieval
python scripts/smoke_semantic.py

# 5. Ask question via API
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are payment terms?","doc_id":"doc_sample123"}'

# 6. Evaluate quality
python -m app.eval.run_eval
```

---

## 📞 Support

For issues or questions:
1. Check `.env` configuration
2. Review logs from `docker-compose logs legalmind_pg`
3. Ensure PostgreSQL is healthy: `docker ps`
4. Test with smoke scripts first

---

**Built with ❤️ for legal document intelligence**
