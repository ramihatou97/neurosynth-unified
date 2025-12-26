# NeuroSynth Architecture

## Overview

NeuroSynth is a Retrieval-Augmented Generation (RAG) system designed for neurosurgical knowledge. It combines semantic search over medical documents with Claude AI to provide accurate, citation-backed answers to clinical questions.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interface                                  │
│                    (REST API / Web Client / CLI)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Application                                │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Search    │  │     RAG     │  │  Documents   │  │     Health      │   │
│  │   Routes    │  │   Routes    │  │   Routes     │  │    Routes       │   │
│  └─────────────┘  └─────────────┘  └──────────────┘  └─────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Dependency Injection                             │   │
│  │    ServiceContainer  │  Settings  │  Repositories                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Service Layer                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐      │
│  │  SearchService   │  │    RAGEngine     │  │  ContextAssembler    │      │
│  │                  │  │                  │  │                      │      │
│  │  - search()      │  │  - ask()         │  │  - assemble()        │      │
│  │  - find_similar()│  │  - summarize()   │  │  - build_citations() │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Retrieval Layer                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐      │
│  │  FAISSManager    │  │  pgvector Search │  │     Re-ranker        │      │
│  │                  │  │                  │  │                      │      │
│  │  Text: 1024d     │  │  Filter by:      │  │  CrossEncoder        │      │
│  │  Image: 512d     │  │  - document_id   │  │  LLMReranker         │      │
│  │  Caption: 1024d  │  │  - chunk_type    │  │  MedicalReranker     │      │
│  └──────────────────┘  │  - specialty     │  └──────────────────────┘      │
│                        │  - CUIs          │                                 │
│                        └──────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Storage Layer                                     │
│  ┌─────────────────────────────────────┐  ┌───────────────────────────┐    │
│  │         PostgreSQL + pgvector        │  │      FAISS Indexes        │    │
│  │                                      │  │                           │    │
│  │  Documents  │  Chunks  │  Images     │  │  text.faiss (1024d)       │    │
│  │  Links      │  Embeddings            │  │  image.faiss (512d)       │    │
│  │                                      │  │  caption.faiss (1024d)    │    │
│  └─────────────────────────────────────┘  └───────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          External Services                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐      │
│  │   Voyage AI      │  │  Anthropic Claude │  │    BiomedCLIP       │      │
│  │   (Embeddings)   │  │  (Generation)     │  │  (Image Embeddings) │      │
│  │   voyage-3       │  │  claude-sonnet-4  │  │  512d vectors       │      │
│  │   1024d vectors  │  │                   │  │                     │      │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### 1. Ingestion Flow

```
PDF Document
     │
     ▼
┌────────────────────┐
│  PyMuPDF Extract   │ → Raw text + images
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  Visual Triage     │ → Filter decorative images (60-70% reduction)
│  (Gemini Flash)    │
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  Entity Extraction │ → Anatomical structures, procedures, CUIs
│  (14 categories)   │
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  Semantic Chunking │ → Context-aware chunk boundaries
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  VLM Captioning    │ → Medical image descriptions
│  (Gemini Pro)      │
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  Embedding         │ → Voyage-3 (text), BiomedCLIP (images)
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  PostgreSQL +      │ → Persistent storage with vectors
│  FAISS Indexes     │
└────────────────────┘
```

### 2. Search Flow

```
User Query: "retrosigmoid approach for acoustic neuroma"
     │
     ▼
┌────────────────────┐
│  Query Embedding   │ → Voyage-3: 1024d vector
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  FAISS Search      │ → Fast ANN: top 100-200 candidates
│  (< 10ms)          │
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  pgvector Filter   │ → Apply document, type, specialty filters
│  + CUI Boost       │ → Boost results with matching CUIs
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  Re-ranking        │ → Cross-encoder or LLM re-ranking
│  (optional)        │
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  Link Images       │ → Attach relevant images to chunks
└────────────────────┘
     │
     ▼
Top K Results with Scores
```

### 3. RAG Flow

```
Question: "What is the retrosigmoid approach?"
     │
     ▼
┌────────────────────┐
│  Search Service    │ → Get top 20 relevant chunks
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  Context Assembler │ → Token budget: 8000
│                    │ → Deduplicate content
│                    │ → Build [1][2] citations
│                    │ → Attach linked images
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  Prompt Builder    │ → System: Medical domain prompt
│                    │ → Context: Numbered sources
│                    │ → Question: User query
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  Claude API        │ → Generate answer with inline [N]
│  (claude-sonnet-4) │
└────────────────────┘
     │
     ▼
┌────────────────────┐
│  Citation Extract  │ → Parse [1][2] from answer
│                    │ → Link to source chunks
└────────────────────┘
     │
     ▼
RAGResponse {
  answer: "The retrosigmoid approach provides [1]...",
  citations: [...],
  images: [...]
}
```

---

## Component Details

### Database Schema

```sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    source_path TEXT NOT NULL,
    title TEXT,
    total_pages INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Chunks table with vector embeddings
CREATE TABLE chunks (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    content TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER,
    chunk_type TEXT,          -- PROCEDURE, ANATOMY, CLINICAL, etc.
    specialty TEXT,           -- skull_base, spine, vascular, etc.
    cuis TEXT[],              -- UMLS concept IDs
    embedding VECTOR(1024),   -- Voyage-3 embedding
    created_at TIMESTAMP DEFAULT NOW()
);

-- Images table with dual embeddings
CREATE TABLE images (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    file_path TEXT,
    page_number INTEGER,
    image_type TEXT,
    is_decorative BOOLEAN DEFAULT FALSE,
    vlm_caption TEXT,
    cuis TEXT[],
    embedding VECTOR(512),           -- BiomedCLIP visual
    caption_embedding VECTOR(1024),  -- Voyage-3 text
    created_at TIMESTAMP DEFAULT NOW()
);

-- Links between chunks and images
CREATE TABLE links (
    id UUID PRIMARY KEY,
    chunk_id UUID REFERENCES chunks(id),
    image_id UUID REFERENCES images(id),
    link_type TEXT,           -- proximity, semantic, cui_match
    score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vector indexes
CREATE INDEX chunks_embedding_idx ON chunks 
    USING ivfflat (embedding vector_cosine_ops);
    
CREATE INDEX images_embedding_idx ON images 
    USING ivfflat (embedding vector_cosine_ops);
```

### FAISS Index Configuration

| Index | Dimension | Type | Vectors | Use Case |
|-------|-----------|------|---------|----------|
| text | 1024 | IVFFlat | ~1M | Text chunk search |
| image | 512 | IVFFlat | ~100K | Image similarity |
| caption | 1024 | IVFFlat | ~100K | Text-to-image search |

Configuration:
```python
TEXT_CONFIG = FAISSIndexConfig(
    name="text",
    dimension=1024,
    index_type="IVFFlat",
    nlist=100,      # Number of clusters
    nprobe=10,      # Clusters to search
    metric="IP"     # Inner product (cosine after normalization)
)
```

### Hybrid Search Strategy

```
Query: "facial nerve preservation"
           │
           ▼
┌─────────────────────────────────────┐
│         FAISS (Speed)               │
│  • Approximate nearest neighbors    │
│  • O(log n) search complexity       │
│  • Returns: 100-200 candidate IDs   │
│  • Latency: < 10ms                  │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│       pgvector (Filtering)          │
│  • Exact filtering on metadata      │
│  • CUI-based boosting               │
│  • JOIN with images via links       │
│  • Returns: Enriched results        │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│       Re-ranking (Quality)          │
│  • Cross-encoder scoring            │
│  • Query-document relevance         │
│  • Returns: Top K reordered         │
└─────────────────────────────────────┘
```

### Citation Tracking

The system maintains full provenance from source to answer:

```
Source Document (PDF)
    │
    ├── Page 45, Paragraph 3
    │       │
    │       ▼
    │   Chunk (id: abc-123)
    │       │
    │       ▼
    │   Citation [1] in context
    │       │
    │       ▼
    │   Referenced as [1] in answer
    │       │
    │       ▼
    │   CitationItem {
    │       index: 1,
    │       chunk_id: "abc-123",
    │       snippet: "...",
    │       page_number: 45
    │   }
    │
    └── User can verify source
```

---

## Scalability

### Horizontal Scaling

```
                    Load Balancer
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     ┌─────────┐   ┌─────────┐   ┌─────────┐
     │  API 1  │   │  API 2  │   │  API 3  │
     └─────────┘   └─────────┘   └─────────┘
          │              │              │
          └──────────────┼──────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        ┌──────────┐          ┌──────────┐
        │ Primary  │◀────────▶│ Replica  │
        │ Postgres │          │ Postgres │
        └──────────┘          └──────────┘
```

### Performance Characteristics

| Component | Scaling Factor | Notes |
|-----------|---------------|-------|
| API Servers | Horizontal | Stateless, add instances |
| FAISS | Memory | Keep in RAM, ~4GB per 1M vectors |
| PostgreSQL | Vertical + Read replicas | pgvector indexes |
| Claude API | Rate limited | Queue for high load |

---

## Security

### Authentication (Recommended)

```python
# Add to production deployment
from fastapi import Security
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/api/v1/search")
async def search(
    request: SearchRequest,
    token: str = Security(security)
):
    validate_token(token)
    ...
```

### Data Privacy

- No PII stored in embeddings
- Document content stored encrypted at rest
- API keys rotated regularly
- Audit logging for all requests

---

## Monitoring

### Health Checks

```python
# Kubernetes probes
GET /health/live   → Container alive
GET /health/ready  → Services initialized

# Detailed health
GET /health → {
    "status": "healthy",
    "components": {
        "database": {"status": "healthy", "latency_ms": 5},
        "faiss": {"status": "healthy", "details": {"text_size": 100000}},
        "search": {"status": "healthy"},
        "rag": {"status": "healthy"}
    }
}
```

### Metrics (Recommended)

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram

search_requests = Counter('search_requests_total', 'Total search requests')
search_latency = Histogram('search_latency_seconds', 'Search latency')
rag_requests = Counter('rag_requests_total', 'Total RAG requests')
rag_latency = Histogram('rag_latency_seconds', 'RAG latency')
```

---

## Future Enhancements

1. **Knowledge Graph** - Entity relationships for graph-based retrieval
2. **Active Learning** - User feedback to improve rankings
3. **Multi-language** - Support for non-English documents
4. **Real-time Ingestion** - Stream processing for new documents
5. **Caching Layer** - Redis for frequent queries
6. **Fine-tuned Models** - Domain-specific embeddings

---

## References

- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [pgvector](https://github.com/pgvector/pgvector)
- [Anthropic API](https://docs.anthropic.com/)
- [Voyage AI](https://docs.voyageai.com/)
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
