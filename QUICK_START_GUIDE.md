# NeuroSynth Unified - Quick Start Guide
**Date:** January 4, 2026  
**For:** Developers onboarding to the codebase

---

## 🚀 Getting Started (5 Minutes)

### 1. Prerequisites
```bash
# Required
- Python 3.11+
- PostgreSQL 15+ with pgvector extension
- Node.js 18+ (for frontend)

# API Keys (get from .env)
- VOYAGE_API_KEY (text embeddings)
- ANTHROPIC_API_KEY (RAG + VLM)
- GOOGLE_API_KEY (optional - deep research)
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
vim .env

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### 3. Database Setup
```bash
# Create database
createdb neurosynth

# Enable pgvector extension
psql neurosynth -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run migrations
psql neurosynth < src/database/schema.sql
psql neurosynth < src/database/migrations/004_authority_registry.sql
psql neurosynth < src/database/migrations/005_enhanced_chunk_metadata.sql
psql neurosynth < src/database/migrations/006_add_pages_table.sql
```

### 4. Start Services
```bash
# Terminal 1: Backend API
uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev

# Access
# - API: http://localhost:8000
# - Frontend: http://localhost:5173
# - API Docs: http://localhost:8000/docs
```

---

## 🔧 Common Tasks

### Ingest a Document
```bash
# Via API
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@path/to/document.pdf" \
  -F "metadata={\"title\":\"My Document\"}"

# Via Python script
python scripts/ingest_document.py path/to/document.pdf
```

### Search for Content
```bash
# Text search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "pterional craniotomy", "limit": 10}'

# Image search
curl -X POST http://localhost:8000/api/search/images \
  -H "Content-Type: application/json" \
  -d '{"query": "brain MRI", "limit": 5}'
```

### RAG Query (V3)
```bash
# Standard mode (fast)
curl -X POST http://localhost:8000/api/rag/v3 \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the steps for pterional approach?", "mode": "standard"}'

# Deep research mode (comprehensive)
curl -X POST http://localhost:8000/api/rag/v3 \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare pterional vs orbitozygomatic approach", "mode": "deep_research"}'

# External mode (web-enriched)
curl -X POST http://localhost:8000/api/rag/v3 \
  -H "Content-Type: application/json" \
  -d '{"query": "Latest research on glioblastoma treatment", "mode": "external"}'
```

### Generate Synthesis
```bash
# V3 synthesis with web enrichment
curl -X POST http://localhost:8000/api/synthesis/v3 \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Pterional Craniotomy",
    "include_images": true,
    "enable_web_enrichment": true,
    "format": "pdf"
  }'
```

---

## 🐛 Troubleshooting

### Issue: Missing Embeddings
**Symptom:** Search returns no results or very few results

**Diagnosis:**
```bash
psql neurosynth -c "
SELECT 
    COUNT(*) as total_chunks,
    COUNT(embedding) as with_embeddings,
    COUNT(*) - COUNT(embedding) as missing_embeddings
FROM chunks;
"
```

**Fix:**
```bash
python scripts/backfill_missing_data.py --chunk-embeddings
```

### Issue: Missing VLM Captions
**Symptom:** Images have no captions in search results

**Diagnosis:**
```bash
psql neurosynth -c "
SELECT 
    COUNT(*) as total_images,
    COUNT(vlm_caption) as with_captions
FROM images;
"
```

**Fix:**
```bash
python scripts/backfill_missing_data.py --vlm-captions
```

### Issue: Slow Search
**Symptom:** Search takes >100ms

**Diagnosis:**
```bash
# Check if HNSW indexes exist
psql neurosynth -c "\d chunks"
psql neurosynth -c "\d images"
```

**Fix:**
```bash
# Create HNSW indexes
psql neurosynth -c "
CREATE INDEX CONCURRENTLY idx_chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX CONCURRENTLY idx_images_clip_hnsw
ON images USING hnsw (clip_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"
```

---

## 📚 Key Files to Know

### Core Pipeline
- `src/ingest/pipeline.py` - Main ingestion orchestrator (1,982 LOC)
- `src/core/neuro_chunker.py` - Type-aware semantic chunking (1,143 LOC)
- `src/ingest/database_writer.py` - Transactional DB writes (1,041 LOC)

### RAG & Search
- `src/rag/unified_engine.py` - V3 tri-modal RAG (1,379 LOC)
- `src/retrieval/search_service.py` - Hybrid search (1,395 LOC)
- `src/retrieval/reranker.py` - Cross-encoder reranking (1,011 LOC)

### API
- `src/api/main.py` - FastAPI app (270 LOC)
- `src/api/dependencies.py` - Service container (650 LOC)
- `src/api/routes/rag_v3.py` - V3 RAG endpoints (493 LOC)

### Database
- `src/database/repositories/chunk.py` - Chunk repository (514 LOC)
- `src/database/repositories/image.py` - Image repository (557 LOC)
- `src/database/connection.py` - Connection pooling

---

## 🎯 Next Steps

1. **Read the full analysis:** `CODEBASE_ANALYSIS_2026-01-04.md`
2. **Fix data integrity:** Run backfill scripts
3. **Run tests:** `python -m pytest tests/`
4. **Explore API:** http://localhost:8000/docs

