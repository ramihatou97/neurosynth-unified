# NeuroSynth Unified - Action Plan
**Date:** January 4, 2026  
**Priority:** Immediate Actions Required

---

## 🚨 Critical Issues (Fix This Week)

### 1. Data Integrity - Missing Embeddings (HIGH PRIORITY)

**Problem:**
- 445/1791 chunks (24.8%) missing embeddings → **invisible to search**
- 181/320 images (56.6%) missing caption embeddings → **not searchable**
- 233/320 images (72.8%) missing caption summaries → **verbose UI**

**Impact:** Core search functionality degraded

**Action:**
```bash
# Backfill all missing data (estimated time: 30-60 minutes)
python scripts/backfill_missing_data.py --all

# Verify completion
python scripts/verify_data_integrity.py
```

**Verification:**
```sql
-- Should show 0 missing embeddings
SELECT 
    COUNT(*) as total_chunks,
    COUNT(embedding) as with_embeddings,
    COUNT(*) - COUNT(embedding) as missing_embeddings
FROM chunks;

-- Should show all images with captions have embeddings
SELECT 
    COUNT(*) as total_images,
    COUNT(vlm_caption) as with_captions,
    COUNT(caption_embedding) as with_caption_embeddings,
    COUNT(caption_summary) as with_summaries
FROM images
WHERE is_decorative = FALSE;
```

**Owner:** Data Team  
**Deadline:** January 6, 2026

---

### 2. Version Consolidation (MEDIUM PRIORITY)

**Problem:**
- V1, V2.0, V2.2, and V3 coexist
- Unclear which version to use
- Duplicate code paths

**Action:**

**Step 1: Document Version Strategy**
```markdown
# Create VERSION_STRATEGY.md

## API Endpoints
- `/api/rag` (V1) → **DEPRECATED** - Use `/api/rag/v3`
- `/api/rag/v3` (V3) → **RECOMMENDED** - Tri-modal RAG
- `/api/synthesis` (V1) → **DEPRECATED** - Use `/api/synthesis/v3`
- `/api/synthesis/v3` (V3) → **RECOMMENDED** - Web-enriched

## Chunking
- `NeuroSemanticChunker` (V2.0) → **CURRENT** - Production use
- `EnhancedChunker` (V2.2) → **EXPERIMENTAL** - Not integrated

## Decision: Use V2.2 chunking for new ingestions
```

**Step 2: Add Deprecation Warnings**
```python
# In src/api/routes/rag.py
@router.post("/api/rag")
async def rag_v1(request: RAGRequest):
    logger.warning("V1 RAG endpoint is deprecated. Use /api/rag/v3 instead.")
    # ... existing code
```

**Step 3: Update Documentation**
- Update README.md with version guidance
- Update API docs with migration path
- Add version badges to endpoints

**Owner:** Engineering Team  
**Deadline:** January 10, 2026

---

### 3. Testing Coverage (MEDIUM PRIORITY)

**Problem:**
- V3 features not tested end-to-end
- No integration tests for tri-modal RAG
- No tests for backfill scripts

**Action:**

**Step 1: Run Existing Tests**
```bash
# Run all existing tests
python -m pytest tests/ -v

# Run specific test suites
python tests/test_v22_chunk_optimization.py
python scripts/test_api_comprehensive.py
python scripts/test_chat_hardening.py
python scripts/test_synthesis_fixes.py
python scripts/test_full_pipeline.py
```

**Step 2: Add V3 Integration Tests**
```python
# Create tests/test_v3_rag_modes.py

async def test_standard_rag_mode():
    """Test standard RAG mode (fast, local-only)."""
    engine = UnifiedRAGEngine(...)
    result = await engine.query("pterional craniotomy", mode="standard")
    assert result.answer is not None
    assert result.sources is not None
    assert result.latency_ms < 5000  # Should be fast

async def test_deep_research_mode():
    """Test deep research mode (comprehensive)."""
    engine = UnifiedRAGEngine(...)
    result = await engine.query("compare approaches", mode="deep_research")
    assert result.answer is not None
    assert len(result.sources) > 5  # Should have many sources
    assert result.expanded_queries is not None

async def test_external_mode():
    """Test external mode (web-enriched)."""
    engine = UnifiedRAGEngine(...)
    result = await engine.query("latest research", mode="external")
    assert result.answer is not None
    assert result.external_sources is not None
```

**Step 3: Add Data Integrity Tests**
```python
# Create tests/test_data_integrity.py

async def test_all_chunks_have_embeddings():
    """Verify all chunks have embeddings."""
    conn = await get_connection()
    result = await conn.fetchrow("""
        SELECT COUNT(*) as total, COUNT(embedding) as with_embeddings
        FROM chunks
    """)
    assert result['total'] == result['with_embeddings']

async def test_embedding_dimensions():
    """Verify embedding dimensions are correct."""
    conn = await get_connection()
    # Check chunk embeddings are 1024d
    result = await conn.fetchrow("""
        SELECT array_length(embedding, 1) as dim
        FROM chunks
        WHERE embedding IS NOT NULL
        LIMIT 1
    """)
    assert result['dim'] == 1024
    
    # Check image embeddings are 512d
    result = await conn.fetchrow("""
        SELECT array_length(clip_embedding, 1) as dim
        FROM images
        WHERE clip_embedding IS NOT NULL
        LIMIT 1
    """)
    assert result['dim'] == 512
```

**Owner:** QA Team  
**Deadline:** January 13, 2026

---

## 📈 Short-Term Improvements (Next 2 Weeks)

### 4. Performance Optimization

**Action Items:**
- [ ] Implement query result caching (Redis)
- [ ] Batch embeddings in ingestion pipeline (128 per batch)
- [ ] Profile pipeline to identify bottlenecks
- [ ] Add connection pooling metrics

**Expected Impact:**
- 50% reduction in query latency (caching)
- 30% faster ingestion (batching)

---

### 5. Reliability Enhancements

**Action Items:**
- [ ] Add environment variable validation at startup
- [ ] Add retry logic for API calls (exponential backoff)
- [ ] Add health checks for external services
- [ ] Add graceful degradation (fallback to V1 if V3 fails)

**Expected Impact:**
- Fail-fast on misconfiguration
- 90% reduction in transient failures

---

### 6. Observability

**Action Items:**
- [ ] Add structured logging for all pipeline stages
- [ ] Add metrics collection (Prometheus/StatsD)
- [ ] Add error tracking (Sentry)
- [ ] Add performance monitoring (APM)

**Expected Impact:**
- Faster debugging
- Proactive issue detection

---

## 🎯 Success Metrics

### Week 1 (January 6)
- ✅ 0 chunks missing embeddings
- ✅ 0 images missing caption embeddings
- ✅ Version strategy documented

### Week 2 (January 13)
- ✅ All existing tests passing
- ✅ V3 integration tests added
- ✅ Data integrity tests added

### Week 3 (January 20)
- ✅ Query caching implemented
- ✅ Batch embeddings implemented
- ✅ Environment validation added

### Week 4 (January 27)
- ✅ Structured logging added
- ✅ Metrics collection added
- ✅ Health checks added

---

## 📞 Contacts

**Data Integrity Issues:** Data Team  
**API/Backend Issues:** Engineering Team  
**Testing Issues:** QA Team  
**Performance Issues:** DevOps Team

---

## 📝 Notes

- All scripts are in `scripts/` directory
- All tests are in `tests/` directory
- Database migrations are in `src/database/migrations/`
- API documentation: http://localhost:8000/docs

