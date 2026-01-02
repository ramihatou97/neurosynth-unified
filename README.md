# NeuroSynth Unified

**Neurosurgical Knowledge Retrieval and Question Answering System**

A production-ready RAG (Retrieval-Augmented Generation) system for neurosurgical knowledge, combining semantic search with Claude AI for accurate, citation-backed answers.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Features

- **PDF Ingestion Pipeline** - Extract text, images, and medical entities from neurosurgical documents
- **Semantic Search** - Hybrid FAISS + pgvector search with UMLS concept boosting
- **RAG Question Answering** - Claude-powered answers with inline citations
- **Multi-modal** - Text-to-image and image-to-image search via BiomedCLIP
- **REST API** - 20+ endpoints for search, RAG, and document management
- **Production Ready** - Docker deployment, health checks, comprehensive tests

---

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Docker & Docker Compose (for containerized deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/neurosynth-unified.git
cd neurosynth-unified

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

```bash
# Required
DATABASE_URL=postgresql+asyncpg://neurosynth:password@localhost:5432/neurosynth
VOYAGE_API_KEY=your-voyage-ai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Optional
FAISS_INDEX_DIR=./indexes
CLAUDE_MODEL=claude-sonnet-4-20250514
DEBUG=false
```

### Database Setup

```bash
# Start PostgreSQL with pgvector
docker-compose up -d postgres

# Run migrations
python scripts/init_database.py
```

### Run the API

```bash
# Development
uvicorn src.api.main:app --reload --port 8000

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Visit http://localhost:8000/docs for interactive API documentation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI REST API                         │
│  /search  │  /rag/ask  │  /documents  │  /health                │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        Service Layer                             │
│  SearchService  │  RAGEngine  │  ContextAssembler               │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       Retrieval Layer                            │
│  FAISS Manager  │  pgvector Search  │  Re-ranker                │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       Storage Layer                              │
│  PostgreSQL + pgvector  │  FAISS Indexes                        │
└─────────────────────────────────────────────────────────────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

---

## API Endpoints

### Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/search` | Semantic search with filters |
| GET | `/api/v1/search/quick` | Quick autocomplete search |
| GET | `/api/v1/search/similar/{id}` | Find similar items |

### RAG

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/rag/ask` | Ask question with citations |
| POST | `/api/v1/rag/ask/stream` | Streaming response (SSE) |
| POST | `/api/v1/rag/conversation` | Multi-turn conversation |
| GET | `/api/v1/rag/summarize/{doc_id}` | Summarize document |

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/documents` | List documents |
| GET | `/api/v1/documents/{id}` | Get document details |
| DELETE | `/api/v1/documents/{id}` | Delete document |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe |
| GET | `/stats` | System statistics |

---

## Usage Examples

### Search

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "retrosigmoid approach for acoustic neuroma",
    "mode": "hybrid",
    "top_k": 10,
    "filters": {
      "chunk_types": ["PROCEDURE", "ANATOMY"]
    }
  }'
```

### RAG Question Answering

```bash
curl -X POST http://localhost:8000/api/v1/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the retrosigmoid approach?",
    "include_citations": true
  }'
```

Response:
```json
{
  "answer": "The retrosigmoid approach provides excellent exposure of the cerebellopontine angle [1]. Patient positioning is lateral with the head turned [2]...",
  "citations": [
    {"index": 1, "chunk_id": "...", "snippet": "...", "page_number": 45}
  ],
  "generation_time_ms": 2500
}
```

### Python SDK

```python
import httpx

async with httpx.AsyncClient() as client:
    # Search
    response = await client.post(
        "http://localhost:8000/api/v1/search",
        json={"query": "facial nerve preservation"}
    )
    results = response.json()
    
    # RAG
    response = await client.post(
        "http://localhost:8000/api/v1/rag/ask",
        json={"question": "How to preserve the facial nerve?"}
    )
    answer = response.json()
    print(answer["answer"])
```

---

## Ingesting Documents

### Using the Pipeline

```python
from src.ingest import UnifiedPipeline, UnifiedPipelineConfig

# Configure for database output
config = UnifiedPipelineConfig.for_database(
    connection_string="postgresql+asyncpg://..."
)

# Process document
async with UnifiedPipeline(config) as pipeline:
    result = await pipeline.process_document("neurosurgery_atlas.pdf")
    print(f"Ingested: {result.chunk_count} chunks, {result.image_count} images")
```

### Building FAISS Indexes

```bash
python scripts/build_indexes.py \
  --database postgresql://... \
  --output ./indexes
```

---

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t neurosynth:latest .

# Run with Docker Compose
docker-compose up -d
```

### Docker Compose Services

```yaml
services:
  api:        # FastAPI application
  postgres:   # PostgreSQL + pgvector
  redis:      # Optional caching
```

See [docker-compose.yml](docker-compose.yml) for full configuration.

---

## Development

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Type checking
mypy src/

# Linting
ruff check src/

# Formatting
black src/ tests/
```

---

## Project Structure

```
neurosynth-unified/
├── src/
│   ├── api/           # FastAPI application
│   ├── database/      # PostgreSQL + repositories
│   ├── ingest/        # PDF ingestion pipeline
│   ├── rag/           # RAG engine + prompts
│   ├── retrieval/     # FAISS + search service
│   ├── shared/        # Shared models
│   └── ...
├── tests/
│   ├── unit/          # Unit tests
│   └── integration/   # API tests
├── scripts/           # Utility scripts
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Configuration

### RAG Configuration

```python
from src.rag import RAGConfig

config = RAGConfig(
    model="claude-sonnet-4-20250514",
    max_tokens=2048,
    temperature=0.3,
    max_context_tokens=8000,
    max_context_chunks=10
)
```

### Search Configuration

```python
from src.retrieval import SearchService

service = SearchService(
    database=db,
    faiss_manager=faiss,
    embedder=embedder,
    config={
        "faiss_k_multiplier": 10,
        "text_weight": 0.7,
        "cui_boost": 1.2
    }
)
```

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| pgvector HNSW search | <10ms | Default backend |
| FAISS search | <10ms | At scale (>500K) |
| Hybrid search | ~50ms | With filtering |
| RAG response | ~2-3s | Claude generation |
| PDF ingestion | ~30s/page | With VLM captions |

---

## Scaling: pgvector vs FAISS

The system supports **both** pgvector and FAISS for vector search. Choose based on library size:

| Vector Count | Recommended Backend | Command |
|--------------|---------------------|---------|
| < 50K | pgvector HNSW (default) | No change needed |
| 50K - 500K | Either | Benchmark both |
| > 500K | FAISS IVFFlat | See below |

### Switching to FAISS at Scale

```bash
# Enable FAISS when approaching 500K+ vectors
export USE_FAISS=true
export USE_PGVECTOR=false

# Build FAISS indexes
python scripts/build_indexes.py --faiss

# Restart API
```

See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md#vector-search-scaling-strategy) for detailed scaling guidance.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Anthropic Claude](https://anthropic.com) - Language model
- [Voyage AI](https://voyage.ai) - Text embeddings
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP) - Medical image embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [pgvector](https://github.com/pgvector/pgvector) - PostgreSQL vectors
