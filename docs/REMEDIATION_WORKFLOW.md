# NeuroSynth Image Linking Remediation Workflow
## Document: dcc1124f-5ab0-477e-baa1-6cf06c22c571

**Assessment Score**: 7/10 → Target: 9/10
**Date**: Generated based on implementation assessment report

---

## Executive Summary

| Issue | Severity | Impact | Fix Complexity |
|-------|----------|--------|----------------|
| Page number compression | **HIGH** | 60% images unreachable | Medium |
| Missing VLM captions | Medium | 7 images unlinkable | Low |
| Link coverage gap | Medium | 31% chunks unlinked | Resolved by #1 |

---

## Root Cause Analysis

### 1. Page Number Mismatch (Critical)

**Location**: `backfill_page_numbers.py` + `src/ingest/fusion.py:TriPassLinker._get_candidate_chunks()`

**Mechanism**:
```
backfill_page_numbers.py:
  - Found total_pages = NULL
  - Used fallback: 59 chunks ÷ 4 chunks/page = 15 pages
  - Result: Chunks assigned to pages 1-15

PDF reality:
  - Actual pages: 36
  - Images extracted with correct page_number: 1-36

TriPassLinker._get_candidate_chunks():
  - page_buffer = 1
  - Filter: abs(chunk.page_start - image.page_number) <= 1
  - Images on pages 18-36: abs(15 - 18) = 3 > 1 → EXCLUDED
```

**Evidence**:
```sql
-- Chunks page range
SELECT MIN(page_number), MAX(page_number) FROM chunks 
WHERE document_id = 'dcc1124f-...'; 
-- Result: 1, 15

-- Images page range  
SELECT MIN(page_number), MAX(page_number) FROM images
WHERE document_id = 'dcc1124f-...';
-- Result: 1, 36

-- Unreachable images
SELECT COUNT(*) FROM images 
WHERE document_id = 'dcc1124f-...' 
  AND page_number > 18
  AND caption_embedding IS NOT NULL;
-- Result: 10 (images with embeddings that can never match)
```

---

## Remediation Steps

### Step 1: Fix Document Page Count (5 minutes)

```sql
-- Verify current state
SELECT id, title, total_pages FROM documents 
WHERE id = 'dcc1124f-5ab0-477e-baa1-6cf06c22c571';

-- Apply fix
UPDATE documents 
SET total_pages = 36, 
    updated_at = NOW()
WHERE id = 'dcc1124f-5ab0-477e-baa1-6cf06c22c571';
```

### Step 2: Re-run Page Number Backfill (2 minutes)

```bash
# Using improved script with PDF introspection
python scripts/backfill_page_numbers_improved.py \
  --document-id dcc1124f-5ab0-477e-baa1-6cf06c22c571

# Verify
psql $DATABASE_URL -c "
  SELECT MIN(page_number), MAX(page_number), COUNT(*) 
  FROM chunks 
  WHERE document_id = 'dcc1124f-5ab0-477e-baa1-6cf06c22c571'
"
# Expected: 1, 36, 59
```

### Step 3: Re-run Image Linking (3 minutes)

```bash
python scripts/backfill_links.py \
  --document-id dcc1124f-5ab0-477e-baa1-6cf06c22c571

# Or via API with relaxed mode if page numbers still uncertain:
curl -X POST "http://localhost:8000/api/documents/dcc1124f-.../relink?page_confidence_mode=relaxed"
```

### Step 4: Re-caption Failed Images (5-10 minutes)

```bash
# Identify uncaptioned images
python scripts/recaption_failed_images.py \
  --document-id dcc1124f-5ab0-477e-baa1-6cf06c22c571 \
  --dry-run

# Execute with embedding generation
python scripts/recaption_failed_images.py \
  --document-id dcc1124f-5ab0-477e-baa1-6cf06c22c571 \
  --with-embeddings
```

### Step 5: Final Relink After Captions (2 minutes)

```bash
# Now that images have captions, re-run linking to capture semantic matches
python scripts/backfill_links.py \
  --document-id dcc1124f-5ab0-477e-baa1-6cf06c22c571
```

---

## Verification Queries

```sql
-- 1. Check page distribution is now correct
SELECT 
  'chunks' as type,
  MIN(page_number) as min_page,
  MAX(page_number) as max_page,
  COUNT(*) as count
FROM chunks WHERE document_id = 'dcc1124f-...'
UNION ALL
SELECT 
  'images',
  MIN(page_number),
  MAX(page_number),
  COUNT(*)
FROM images WHERE document_id = 'dcc1124f-...';

-- 2. Check image linkage coverage
SELECT 
  COUNT(*) as total_images,
  COUNT(*) FILTER (WHERE is_decorative) as decorative,
  COUNT(*) FILTER (WHERE caption_embedding IS NOT NULL) as with_embedding,
  COUNT(DISTINCT l.image_id) as linked
FROM images i
LEFT JOIN links l ON i.id = l.image_id
WHERE i.document_id = 'dcc1124f-...';
-- Target: linked >= with_embedding * 0.8 (80%+)

-- 3. Check link quality distribution
SELECT 
  link_type,
  COUNT(*) as count,
  ROUND(AVG(score)::numeric, 3) as avg_score,
  ROUND(MIN(score)::numeric, 3) as min_score,
  ROUND(MAX(score)::numeric, 3) as max_score
FROM links l
JOIN chunks c ON l.chunk_id = c.id
WHERE c.document_id = 'dcc1124f-...'
GROUP BY link_type;

-- 4. Verify no orphaned images on high pages
SELECT i.id, i.page_number, i.vlm_caption IS NOT NULL as has_caption
FROM images i
LEFT JOIN links l ON i.id = l.image_id
WHERE i.document_id = 'dcc1124f-...'
  AND i.page_number > 15
  AND NOT i.is_decorative
  AND l.id IS NULL;
-- Target: 0 rows (all high-page images should now be linked)
```

---

## Expected Outcomes

| Metric | Before | After |
|--------|--------|-------|
| Images with embeddings linked | 6/15 (40%) | 15/15 (100%) |
| Chunks with image links | 41/59 (69%) | 55+/59 (93%+) |
| Link score range | 0.559-0.805 | 0.55-0.85 |
| Images with captions | 8/15 | 15/15 |
| Quality score | 7/10 | 9/10 |

---

## Preventive Measures

### 1. Pipeline Enhancement

Add page count verification to ingestion pipeline:

```python
# In src/ingest/pipeline.py, after PDF extraction
async def _validate_page_consistency(self, doc, chunks, images):
    """Verify page numbers are consistent across artifacts."""
    actual_pages = len(doc)
    max_chunk_page = max(c.page_number for c in chunks)
    max_image_page = max(i.page_number for i in images)
    
    if max_chunk_page < max_image_page - 3:
        logger.warning(
            f"Page mismatch detected: chunks max={max_chunk_page}, "
            f"images max={max_image_page}, actual={actual_pages}"
        )
        # Trigger automatic re-estimation
        await self._reestimate_chunk_pages(chunks, actual_pages)
```

### 2. Database Constraint

```sql
-- Add check constraint to catch NULL total_pages early
ALTER TABLE documents 
ADD CONSTRAINT chk_total_pages_not_null 
CHECK (total_pages IS NOT NULL AND total_pages > 0);
```

### 3. Monitoring Query

Add to periodic health checks:

```sql
-- Identify documents with page mismatches
SELECT 
  d.id,
  d.title,
  d.total_pages,
  MAX(c.page_number) as max_chunk_page,
  MAX(i.page_number) as max_image_page,
  CASE 
    WHEN MAX(c.page_number) < MAX(i.page_number) - 3 THEN 'MISMATCH'
    ELSE 'OK'
  END as status
FROM documents d
LEFT JOIN chunks c ON d.id = c.document_id
LEFT JOIN images i ON d.id = i.document_id
GROUP BY d.id, d.title, d.total_pages
HAVING MAX(c.page_number) < MAX(i.page_number) - 3;
```

---

## Files Created

| File | Purpose |
|------|---------|
| `patches/tripasslinker_page_confidence.patch` | Add page_confidence_mode to TriPassLinker |
| `scripts/backfill_page_numbers_improved.py` | PDF introspection + conservative estimation |
| `scripts/recaption_failed_images.py` | Re-caption images that failed VLM |
| `patches/relink_api_page_confidence.patch` | API support for page_confidence_mode |
