# Surgical Integration Patches

Patches for integrating synthesis quality improvements into existing codebase.

## Patch Application Order

Apply patches in numerical order to ensure clean integration:

```bash
cd /Users/ramihatoum/neurosynth-unified

# 1. ChunkerConfig - Add adaptive overlap by chunk type
patch -p1 < patches/001_chunker_adaptive_overlap.patch

# 2. CaptionQualityScorer - Add quality gating for VLM captions
patch -p1 < patches/002_caption_quality_scorer.patch

# 3. SearchService - Add synthesis-optimized search method
patch -p1 < patches/003_search_synthesis_method.patch

# 4. SynthesisEngine - Add quality scoring method
patch -p1 < patches/004_synthesis_score_method.patch

# 5. TriPassLinker - Add page confidence modes
patch -p1 < patches/005_tripasslinker_page_confidence.patch
```

## Patch Descriptions

### 001_chunker_adaptive_overlap.patch
**File**: `src/core/neuro_chunker.py`
**Action**: PATCH existing ChunkerConfig class

Adds adaptive overlap sentences by chunk type:
- `procedure_overlap_sentences: int = 3` (more overlap for step continuity)
- `anatomy_overlap_sentences: int = 1` (less overlap for definitions)
- `pathology_overlap_sentences: int = 2` (standard overlap)
- `clinical_overlap_sentences: int = 2` (standard overlap)
- `get_overlap_for_type(chunk_type)` method

### 002_caption_quality_scorer.patch
**File**: `src/retrieval/vlm_captioner.py`
**Action**: ADD new class (no collision)

Adds `CaptionQualityScorer` class for VLM caption quality gating:
- Medical terminology density scoring
- Length adequacy scoring
- Generic phrase penalization
- Quality threshold checking

### 003_search_synthesis_method.patch
**File**: `src/retrieval/search_service.py`
**Action**: EXTEND existing class

Adds `hybrid_search_for_synthesis()` method:
- Synthesis-optimized search with MMR diversity
- Authority tier filtering
- Multi-index fusion (text, summary, caption embeddings)

### 004_synthesis_score_method.patch
**File**: `src/synthesis/engine.py`
**Action**: EXTEND existing class

Adds `compute_synthesis_score()` method:
- Content coverage scoring (40%)
- Source authority scoring (25%)
- Visual integration scoring (15%)
- Coherence scoring (20%)
- Letter grade assignment

### 005_tripasslinker_page_confidence.patch
**File**: `src/ingest/fusion.py`
**Action**: PATCH existing class

Adds page confidence modes to TriPassLinker:
- `page_confidence_mode: str = "strict"` parameter
- Modes: "strict", "relaxed", "semantic_only"
- Automatic unreliable page data detection

## Verification

After applying patches, verify no syntax errors:

```bash
# Check Python syntax
python -m py_compile src/core/neuro_chunker.py
python -m py_compile src/retrieval/vlm_captioner.py
python -m py_compile src/retrieval/search_service.py
python -m py_compile src/synthesis/engine.py
python -m py_compile src/ingest/fusion.py

# Run tests if available
pytest tests/ -v --tb=short
```

## Rollback

To rollback a patch:

```bash
patch -R -p1 < patches/00X_patch_name.patch
```

## New Scripts

In addition to patches, the following remediation scripts were added:

- `scripts/recaption_failed_images.py` - Re-caption images with NULL vlm_caption
- `scripts/backfill_page_numbers_improved.py` - Multi-strategy page number estimation
- `docs/REMEDIATION_WORKFLOW.md` - Complete workflow for Chapter 6 remediation
