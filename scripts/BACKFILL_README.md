# NeuroSynth Backfill Scripts

Scripts to repair chunk-image linking and related data issues caused by the pipeline ordering bug.

## Problem Statement

**Critical Bug in `src/ingest/pipeline.py`:**

```
Stage 4: LINKING happens FIRST
    chunks, images, links = self.linker.link(chunks, images)  # ← caption_embedding is NULL!
    
Stage 5: EMBEDDINGS generated AFTER
    await self._generate_embeddings(chunks, images, tracker, document)  # ← Too late!
```

**Impact:** TriPassLinker's Pass 3 (semantic similarity) cannot work because `caption_embedding` doesn't exist yet during linking.

## Scripts Overview

| Script | Purpose | Prerequisites |
|--------|---------|---------------|
| `backfill_links.py` | Re-run TriPassLinker on documents with embeddings | Embeddings must exist |
| `backfill_page_numbers.py` | Estimate page numbers for chunks | None |
| `backfill_cuis.py` | Extract UMLS CUIs using SciSpacy | SciSpacy + en_core_sci_lg |

## Execution Order

```
┌─────────────────────────────────────────────────────────────────┐
│  IMMEDIATE FIX                                                   │
├─────────────────────────────────────────────────────────────────┤
│  1. Run generate_image_captions.py   (if images lack captions)  │
│     → Generates VLM captions + caption embeddings               │
│                                                                 │
│  2. Run backfill_links.py                                       │
│     → Creates chunk-image links using TriPassLinker             │
│                                                                 │
│  3. Run backfill_page_numbers.py                                │
│     → Chunks get page numbers for citations                     │
│                                                                 │
│  4. Rebuild FAISS indexes                                       │
│     → POST /api/v1/indexes/rebuild                              │
├─────────────────────────────────────────────────────────────────┤
│  OPTIONAL                                                        │
├─────────────────────────────────────────────────────────────────┤
│  5. Run backfill_cuis.py            (requires SciSpacy)         │
│     → Enables entity-based filtering                            │
└─────────────────────────────────────────────────────────────────┘
```

## Usage Examples

### 1. Backfill Links

```bash
# Process all documents
DATABASE_URL=$DATABASE_URL python scripts/backfill_links.py

# Process specific document
python scripts/backfill_links.py --document-id dcc1124f-5ab0-477e-baa1-6cf06c22c571

# Preview without changes
python scripts/backfill_links.py --dry-run

# Custom score threshold
python scripts/backfill_links.py --min-score 0.6
```

### 2. Backfill Page Numbers

```bash
# Process all documents
DATABASE_URL=$DATABASE_URL python scripts/backfill_page_numbers.py

# Process specific document
python scripts/backfill_page_numbers.py --document-id dcc1124f-5ab0-477e-baa1-6cf06c22c571

# Preview without changes
python scripts/backfill_page_numbers.py --dry-run
```

### 3. Backfill CUIs (Optional)

```bash
# Install prerequisites
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz

# Process all documents
DATABASE_URL=$DATABASE_URL python scripts/backfill_cuis.py

# Process chunks only (faster)
python scripts/backfill_cuis.py --chunks-only

# Limit items processed
python scripts/backfill_cuis.py --limit 100

# Preview without changes
python scripts/backfill_cuis.py --dry-run
```

## Verification Queries

### After Link Backfill

```sql
-- Count total links
SELECT COUNT(*) FROM links;

-- Links per document
SELECT 
    d.title,
    COUNT(l.id) as link_count
FROM documents d
JOIN chunks c ON c.document_id = d.id
LEFT JOIN links l ON l.chunk_id = c.id
GROUP BY d.id, d.title
ORDER BY link_count DESC;

-- Link type distribution
SELECT link_type, COUNT(*) as count
FROM links
GROUP BY link_type
ORDER BY count DESC;
```

### After Page Number Backfill

```sql
-- Chunks with page numbers
SELECT COUNT(*) FROM chunks WHERE page_number IS NOT NULL;

-- Page distribution per document
SELECT 
    d.title,
    MIN(c.page_number) as min_page,
    MAX(c.page_number) as max_page,
    COUNT(*) as chunk_count
FROM documents d
JOIN chunks c ON c.document_id = d.id
WHERE c.page_number IS NOT NULL
GROUP BY d.id, d.title;
```

### After CUI Backfill

```sql
-- Chunks with CUIs
SELECT COUNT(*) FROM chunks WHERE array_length(cuis, 1) > 0;

-- Images with CUIs
SELECT COUNT(*) FROM images WHERE array_length(cuis, 1) > 0;

-- Top CUIs
SELECT unnest(cuis) as cui, COUNT(*) as count
FROM chunks
WHERE array_length(cuis, 1) > 0
GROUP BY cui
ORDER BY count DESC
LIMIT 20;
```

## TriPassLinker Algorithm

The linking algorithm uses three passes:

### Pass 1: Deterministic (Early Exit)
- Matches "Figure 6.3" references in chunk text to `image.figure_id`
- Returns `strength=1.0` if match found
- Regex: `(?i)(?:figure|fig\.?)\s*(\d+(?:\.\d+)?[a-z]?)`

### Pass 2: CUI Overlap
- Jaccard similarity between chunk.cuis and image.cuis
- Threshold: 0.25
- Requires SciSpacy CUI extraction

### Pass 3: Semantic Similarity
- Cosine similarity between `chunk.text_embedding` and `image.caption_embedding`
- Threshold: 0.55
- **This is what was broken** - caption_embedding didn't exist during ingestion

### Fusion Scoring
```
fusion = (semantic × 0.55) + (cui × 0.45)
Link accepted if:
  - fusion ≥ 0.55 (full acceptance)
  - OR cui_only ≥ 0.25 (CUI fallback at 80% confidence)
```

## Permanent Fix

Apply the patch to `src/ingest/pipeline.py`:

```bash
# Review the patch
cat patches/pipeline_fix.patch

# Apply the patch (from project root)
patch -p1 < patches/pipeline_fix.patch
```

Or manually reorder the pipeline stages:

```python
# BEFORE (broken):
Stage 4: Link images to chunks
Stage 5: Generate embeddings

# AFTER (fixed):
Stage 4: Generate embeddings  ← NEW POSITION
Stage 5: Link images to chunks ← Moved after embeddings
Stage 6: Fuse embeddings      ← Only fusion now
```

## Troubleshooting

### "No chunks with embeddings"
```bash
# Generate embeddings first
python scripts/generate_embeddings.py
```

### "No images with caption_embedding"
```bash
# Generate captions and caption embeddings first
python scripts/generate_image_captions.py
```

### "SciSpacy not found"
```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
```

### "Dimension mismatch in cosine similarity"
This happens when comparing 1024d text embeddings with 512d image embeddings. The `backfill_links.py` script uses `caption_embedding` (1024d) which matches `text_embedding` (1024d).

## Expected Outcomes

### Before Backfill
- Links: 0
- Images in search: 0
- Captioned images: 15/22
- Page numbers: 0/59

### After Backfill
- Links: ~30-50 per document
- Images in search: 3-5 per result
- Captioned images: 22/22
- Page numbers: 59/59

## Files Reference

```
scripts/
├── backfill_links.py          # ← NEW: Re-run TriPassLinker
├── backfill_page_numbers.py   # ← NEW: Estimate page numbers
├── backfill_cuis.py           # ← NEW: Extract UMLS CUIs
├── generate_image_captions.py # Existing: VLM captions
├── generate_embeddings.py     # Existing: Text embeddings
└── build_indexes.py           # Existing: FAISS indexes

patches/
└── pipeline_fix.patch         # ← NEW: Pipeline ordering fix
```
