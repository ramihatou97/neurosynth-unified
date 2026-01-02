#!/usr/bin/env python3
"""
NeuroSynth Synthesis Engine Alignment Audit
============================================

Audits the synthesis layer alignment with upgraded:
- Extraction/Indexing pipeline (quality scores, type-specific chunking)
- Search/Retrieval layer (authority boost, type boost, CUI boost)

Checks:
1. SearchResult → ContextAdapter field compatibility
2. Quality score handling (3 scores vs 1)
3. Chunk type utilization for section mapping
4. Authority score preservation and weighting
5. Image catalog with caption embeddings
6. CUI preservation for validation

Usage:
    DATABASE_URL=... VOYAGE_API_KEY=... python scripts/audit_synthesis_alignment.py
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# AUDIT RESULTS MODEL
# =============================================================================

class AuditStatus(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    INFO = "INFO"


@dataclass
class AuditCheck:
    name: str
    status: AuditStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    fix_required: bool = False
    fix_description: str = ""


@dataclass
class AuditReport:
    checks: List[AuditCheck] = field(default_factory=list)

    def add(self, check: AuditCheck):
        self.checks.append(check)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.status == AuditStatus.PASS)

    @property
    def warnings(self) -> int:
        return sum(1 for c in self.checks if c.status == AuditStatus.WARN)

    @property
    def failures(self) -> int:
        return sum(1 for c in self.checks if c.status == AuditStatus.FAIL)

    @property
    def fixes_needed(self) -> List[AuditCheck]:
        return [c for c in self.checks if c.fix_required]

    def print_report(self):
        print("\n" + "="*70)
        print("SYNTHESIS ENGINE ALIGNMENT AUDIT REPORT")
        print("="*70)

        status_icons = {
            AuditStatus.PASS: "[PASS]",
            AuditStatus.WARN: "[WARN]",
            AuditStatus.FAIL: "[FAIL]",
            AuditStatus.INFO: "[INFO]",
        }

        for check in self.checks:
            print(f"\n{status_icons[check.status]} {check.name}")
            print(f"   {check.message}")
            if check.details:
                for k, v in check.details.items():
                    print(f"   - {k}: {v}")
            if check.fix_required:
                print(f"   FIX: {check.fix_description}")

        print("\n" + "-"*70)
        print(f"SUMMARY: {self.passed} passed, {self.warnings} warnings, {self.failures} failures")
        print(f"FIXES NEEDED: {len(self.fixes_needed)}")
        print("-"*70)


# =============================================================================
# AUDIT FUNCTIONS
# =============================================================================

async def audit_searchresult_fields(db, embedder, report: AuditReport):
    """Check that SearchResult has all fields needed by ContextAdapter."""

    from src.retrieval.search_service import SearchService

    service = SearchService(database=db, embedder=embedder)

    query = "retrosigmoid approach acoustic neuroma"
    response = await service.search(query, top_k=5, include_images=True)

    if not response.results:
        report.add(AuditCheck(
            name="SearchResult Fields",
            status=AuditStatus.FAIL,
            message="No search results returned - cannot audit fields",
            fix_required=True,
            fix_description="Ensure database has indexed content"
        ))
        return None

    r = response.results[0]

    # Required fields for ContextAdapter
    required_fields = {
        'chunk_id': 'Unique chunk identifier',
        'content': 'Chunk text content',
        'document_id': 'Parent document ID',
        'document_title': 'Document title for citations',
        'chunk_type': 'Type classification (PROCEDURE, ANATOMY, etc.)',
        'page_start': 'Page number for citations',
        'authority_score': 'Source authority weight',
        'semantic_score': 'Vector similarity score',
        'final_score': 'Combined ranking score',
        'entity_names': 'Extracted medical entities',
        'cuis': 'UMLS concept identifiers',
        'images': 'Linked images list',
    }

    # New quality score fields from upgraded pipeline
    quality_fields = {
        'readability_score': 'Text readability metric',
        'coherence_score': 'Logical coherence metric',
        'completeness_score': 'Content completeness metric',
    }

    present = {}
    missing = {}

    for fld, desc in {**required_fields, **quality_fields}.items():
        has_field = hasattr(r, fld)
        value = getattr(r, fld, None) if has_field else None
        present[fld] = {
            'exists': has_field,
            'value': _truncate_value(value),
            'description': desc
        }
        if not has_field:
            missing[fld] = desc

    if missing:
        # Check if missing are quality scores (warn) or required (fail)
        missing_required = {k: v for k, v in missing.items() if k in required_fields}
        missing_quality = {k: v for k, v in missing.items() if k in quality_fields}

        if missing_required:
            report.add(AuditCheck(
                name="Required SearchResult Fields",
                status=AuditStatus.FAIL,
                message=f"Missing {len(missing_required)} required fields",
                details=missing_required,
                fix_required=True,
                fix_description="Add missing fields to SearchResult model and search enrichment query"
            ))

        if missing_quality:
            report.add(AuditCheck(
                name="Quality Score Fields",
                status=AuditStatus.WARN,
                message=f"Missing {len(missing_quality)} quality score fields",
                details=missing_quality,
                fix_required=True,
                fix_description="Add quality scores to search enrichment query (readability, coherence, completeness)"
            ))
    else:
        report.add(AuditCheck(
            name="SearchResult Fields",
            status=AuditStatus.PASS,
            message="All required and quality fields present",
            details={'total_fields': len(present)}
        ))

    return response


async def audit_quality_score_handling(response, report: AuditReport):
    """Check how quality scores flow to ContextAdapter."""

    from src.synthesis.engine import ContextAdapter, TemplateType

    if not response or not response.results:
        report.add(AuditCheck(
            name="Quality Score Handling",
            status=AuditStatus.FAIL,
            message="No results to audit quality scores"
        ))
        return

    r = response.results[0]

    # Check what quality score fields exist
    has_composite = hasattr(r, 'quality_score')
    has_readability = hasattr(r, 'readability_score')
    has_coherence = hasattr(r, 'coherence_score')
    has_completeness = hasattr(r, 'completeness_score')

    # Get values
    quality_scores = {
        'quality_score (composite)': getattr(r, 'quality_score', None),
        'readability_score': getattr(r, 'readability_score', None),
        'coherence_score': getattr(r, 'coherence_score', None),
        'completeness_score': getattr(r, 'completeness_score', None),
    }

    # Check ContextAdapter usage
    adapter = ContextAdapter()

    # The adapter currently uses: quality_score = getattr(result, 'quality_score', 0.7)
    # This means it expects a single composite score but pipeline provides 3 separate scores

    if has_readability and has_coherence and has_completeness and not has_composite:
        report.add(AuditCheck(
            name="Quality Score Handling",
            status=AuditStatus.WARN,
            message="Pipeline provides 3 separate scores but ContextAdapter expects single composite",
            details=quality_scores,
            fix_required=True,
            fix_description="Update ContextAdapter to compute composite from (readability + coherence + completeness) / 3"
        ))
    elif has_composite:
        report.add(AuditCheck(
            name="Quality Score Handling",
            status=AuditStatus.PASS,
            message="Composite quality_score available",
            details=quality_scores
        ))
    else:
        report.add(AuditCheck(
            name="Quality Score Handling",
            status=AuditStatus.WARN,
            message="No quality scores available - using default 0.7",
            details=quality_scores,
            fix_required=True,
            fix_description="Add quality score computation to extraction pipeline"
        ))


async def audit_chunk_type_utilization(response, report: AuditReport):
    """Check if chunk_type is used for section mapping."""

    from src.synthesis.engine import ContextAdapter, TemplateType, TEMPLATE_SECTIONS

    if not response or not response.results:
        return

    adapter = ContextAdapter()

    # Test adaptation
    context = adapter.adapt(
        topic="test topic",
        search_results=response.results,
        template_type=TemplateType.PROCEDURAL
    )

    # Analyze section distribution by chunk_type
    type_to_section = {}
    for section_name, chunks in context['sections'].items():
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', 'UNKNOWN')
            if chunk_type not in type_to_section:
                type_to_section[chunk_type] = {}
            if section_name not in type_to_section[chunk_type]:
                type_to_section[chunk_type][section_name] = 0
            type_to_section[chunk_type][section_name] += 1

    # Check if chunk types are scattered across sections (suboptimal)
    # or concentrated (good alignment)
    scattered_types = []
    for chunk_type, sections in type_to_section.items():
        if len(sections) > 3:  # Spread across many sections
            scattered_types.append(chunk_type)

    if scattered_types:
        report.add(AuditCheck(
            name="Chunk Type -> Section Mapping",
            status=AuditStatus.WARN,
            message=f"Chunk types scattered across sections (keyword-based classification)",
            details={
                'scattered_types': scattered_types,
                'distribution': type_to_section
            },
            fix_required=True,
            fix_description="Add type-based section routing: PROCEDURE->'Step-by-Step Technique', ANATOMY->'Surgical Approach'"
        ))
    else:
        report.add(AuditCheck(
            name="Chunk Type -> Section Mapping",
            status=AuditStatus.PASS,
            message="Chunk types reasonably concentrated in relevant sections",
            details={'distribution': type_to_section}
        ))

    # Check if classify_section uses chunk_type
    import inspect
    classify_source = ""
    if hasattr(adapter, '_classify_to_section'):
        classify_source = inspect.getsource(adapter._classify_to_section)
    if hasattr(adapter, 'classify_section'):
        classify_source = inspect.getsource(adapter.classify_section)

    uses_chunk_type = 'chunk_type' in classify_source

    report.add(AuditCheck(
        name="Section Classifier Uses chunk_type",
        status=AuditStatus.PASS if uses_chunk_type else AuditStatus.WARN,
        message="Section classifier " + ("uses" if uses_chunk_type else "does NOT use") + " chunk_type field",
        fix_required=not uses_chunk_type,
        fix_description="Update classify_section to use chunk_type as primary signal before keyword fallback"
    ))


async def audit_authority_score_flow(response, report: AuditReport):
    """Check authority score preservation and weighting."""

    from src.synthesis.engine import ContextAdapter, TemplateType

    if not response or not response.results:
        return

    adapter = ContextAdapter()

    context = adapter.adapt(
        topic="test topic",
        search_results=response.results,
        template_type=TemplateType.PROCEDURAL
    )

    # Check authority scores in adapted chunks
    authority_scores = []
    for section, chunks in context['sections'].items():
        for chunk in chunks:
            auth = chunk.get('authority_score', 0)
            authority_scores.append({
                'section': section,
                'authority': auth,
                'combined_score': chunk.get('combined_score', 0),
                'final_score': chunk.get('final_score', 0)
            })

    if not authority_scores:
        report.add(AuditCheck(
            name="Authority Score Flow",
            status=AuditStatus.FAIL,
            message="No chunks in context to check authority scores"
        ))
        return

    # Verify authority is used in combined_score
    avg_authority = sum(a['authority'] for a in authority_scores) / len(authority_scores)
    has_varied_authority = len(set(a['authority'] for a in authority_scores)) > 1

    # Check if combined_score reflects authority
    combined_uses_authority = all(
        a['combined_score'] != a['final_score'] or a['authority'] == 1.0
        for a in authority_scores if a['combined_score'] > 0
    )

    report.add(AuditCheck(
        name="Authority Score Flow",
        status=AuditStatus.PASS if combined_uses_authority else AuditStatus.WARN,
        message=f"Authority scores preserved (avg: {avg_authority:.2f})",
        details={
            'average_authority': round(avg_authority, 3),
            'authority_variance': has_varied_authority,
            'weighted_in_combined': combined_uses_authority,
            'sample_scores': authority_scores[:3]
        },
        fix_required=not combined_uses_authority,
        fix_description="Ensure combined_score = final_score * authority_score * quality_score"
    ))

    # Check if chunks are sorted by authority within sections
    for section, chunks in context['sections'].items():
        if len(chunks) > 1:
            combined_scores = [c.get('combined_score', 0) for c in chunks]
            is_sorted = combined_scores == sorted(combined_scores, reverse=True)
            if not is_sorted:
                report.add(AuditCheck(
                    name=f"Section '{section}' Sorted by Score",
                    status=AuditStatus.WARN,
                    message="Chunks not sorted by combined_score",
                    fix_required=True,
                    fix_description="Sort chunks by combined_score descending after section assignment"
                ))
                break


async def audit_image_catalog(response, report: AuditReport):
    """Check image catalog includes caption embeddings."""

    from src.synthesis.engine import ContextAdapter, TemplateType

    if not response or not response.results:
        return

    adapter = ContextAdapter()

    context = adapter.adapt(
        topic="test topic",
        search_results=response.results,
        template_type=TemplateType.PROCEDURAL
    )

    image_catalog = context.get('image_catalog', [])

    if not image_catalog:
        # Check if source results have images
        source_images = sum(len(r.images or []) for r in response.results)
        if source_images > 0:
            report.add(AuditCheck(
                name="Image Catalog",
                status=AuditStatus.WARN,
                message=f"Source has {source_images} images but catalog is empty",
                fix_required=True,
                fix_description="Verify ContextAdapter image extraction from SearchResult.images"
            ))
        else:
            report.add(AuditCheck(
                name="Image Catalog",
                status=AuditStatus.INFO,
                message="No images in source results"
            ))
        return

    # Check image fields
    sample_img = image_catalog[0]
    expected_fields = ['id', 'caption', 'path', 'page', 'image_type']
    optional_fields = ['caption_embedding', 'vlm_caption', 'document_title']

    present_fields = list(sample_img.keys())
    missing_required = [f for f in expected_fields if f not in present_fields]
    missing_optional = [f for f in optional_fields if f not in present_fields]

    if missing_required:
        report.add(AuditCheck(
            name="Image Catalog Fields",
            status=AuditStatus.FAIL,
            message=f"Missing required fields: {missing_required}",
            details={'present': present_fields, 'missing': missing_required},
            fix_required=True,
            fix_description="Add missing fields to image catalog construction"
        ))
    else:
        has_caption_embedding = 'caption_embedding' in present_fields
        report.add(AuditCheck(
            name="Image Catalog Fields",
            status=AuditStatus.PASS if has_caption_embedding else AuditStatus.WARN,
            message=f"{len(image_catalog)} images cataloged" +
                    (" with caption embeddings" if has_caption_embedding else " WITHOUT caption embeddings"),
            details={
                'image_count': len(image_catalog),
                'fields': present_fields,
                'has_caption_embedding': has_caption_embedding
            },
            fix_required=not has_caption_embedding,
            fix_description="Pass caption_embedding to image catalog for semantic figure resolution"
        ))


async def audit_cui_preservation(response, report: AuditReport):
    """Check CUI preservation for content validation."""

    from src.synthesis.engine import ContextAdapter, TemplateType

    if not response or not response.results:
        return

    adapter = ContextAdapter()

    context = adapter.adapt(
        topic="test topic",
        search_results=response.results,
        template_type=TemplateType.PROCEDURAL
    )

    # Count CUIs in source vs adapted
    source_cuis = set()
    for r in response.results:
        cuis = getattr(r, 'cuis', []) or []
        source_cuis.update(cuis)

    adapted_cuis = set()
    for section, chunks in context['sections'].items():
        for chunk in chunks:
            cuis = chunk.get('cuis', []) or []
            adapted_cuis.update(cuis)

    # Check if CUIs are preserved
    preserved_ratio = len(adapted_cuis) / len(source_cuis) if source_cuis else 0

    if not source_cuis:
        report.add(AuditCheck(
            name="CUI Preservation",
            status=AuditStatus.WARN,
            message="No CUIs in source results (CUI boost ineffective)",
            fix_required=True,
            fix_description="Ensure UMLS extraction is run on chunks (backfill_cuis.py)"
        ))
    elif preserved_ratio < 0.5:
        report.add(AuditCheck(
            name="CUI Preservation",
            status=AuditStatus.WARN,
            message=f"Only {preserved_ratio*100:.1f}% CUIs preserved through adaptation",
            details={
                'source_cuis': len(source_cuis),
                'adapted_cuis': len(adapted_cuis)
            },
            fix_required=True,
            fix_description="Ensure ContextAdapter passes CUIs to chunk_data dict"
        ))
    else:
        report.add(AuditCheck(
            name="CUI Preservation",
            status=AuditStatus.PASS,
            message=f"CUIs preserved ({len(source_cuis)} unique)",
            details={
                'source_cuis': len(source_cuis),
                'adapted_cuis': len(adapted_cuis),
                'sample': list(source_cuis)[:5]
            }
        ))


async def audit_figure_resolver(report: AuditReport):
    """Check if FigureResolver uses semantic matching."""

    from src.synthesis.engine import FigureResolver
    import inspect

    resolver = FigureResolver()

    # Check resolve method source
    resolve_source = inspect.getsource(resolver.resolve)

    uses_embedding = 'embedding' in resolve_source.lower()
    uses_semantic = 'semantic' in resolve_source.lower() or 'cosine' in resolve_source.lower()
    uses_keyword = 'keyword' in resolve_source.lower() or 'match' in resolve_source.lower()

    if uses_embedding or uses_semantic:
        report.add(AuditCheck(
            name="Figure Resolver",
            status=AuditStatus.PASS,
            message="FigureResolver uses semantic/embedding matching"
        ))
    else:
        report.add(AuditCheck(
            name="Figure Resolver",
            status=AuditStatus.WARN,
            message="FigureResolver uses keyword matching only (not semantic)",
            details={
                'uses_embedding': uses_embedding,
                'uses_keyword': uses_keyword
            },
            fix_required=True,
            fix_description="Enhance FigureResolver to use caption_embedding for semantic matching"
        ))


def _truncate_value(value) -> str:
    """Truncate value for display."""
    if value is None:
        return "None"
    if isinstance(value, list):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return f"{{{len(value)} keys}}"
    if isinstance(value, str) and len(value) > 50:
        return f"{value[:50]}..."
    return str(value)


# =============================================================================
# MAIN AUDIT
# =============================================================================

async def main():
    """Run complete synthesis alignment audit."""

    print("="*70)
    print("NEUROSYNTH SYNTHESIS ENGINE ALIGNMENT AUDIT")
    print("="*70)

    # Check environment
    db_url = os.getenv("DATABASE_URL")
    voyage_key = os.getenv("VOYAGE_API_KEY")

    if not db_url:
        print("[FAIL] DATABASE_URL not set")
        sys.exit(1)

    if not voyage_key:
        print("[FAIL] VOYAGE_API_KEY not set")
        sys.exit(1)

    print("\n[1] Initializing services...")

    from src.database.connection import DatabaseConnection
    from src.ingest.embeddings import VoyageTextEmbedder

    db = await DatabaseConnection.initialize(db_url)
    embedder = VoyageTextEmbedder(api_key=voyage_key)

    print("[OK] Services initialized")

    report = AuditReport()

    # Run audits
    print("\n[2] Auditing SearchResult fields...")
    response = await audit_searchresult_fields(db, embedder, report)

    print("[3] Auditing quality score handling...")
    await audit_quality_score_handling(response, report)

    print("[4] Auditing chunk type utilization...")
    await audit_chunk_type_utilization(response, report)

    print("[5] Auditing authority score flow...")
    await audit_authority_score_flow(response, report)

    print("[6] Auditing image catalog...")
    await audit_image_catalog(response, report)

    print("[7] Auditing CUI preservation...")
    await audit_cui_preservation(response, report)

    print("[8] Auditing figure resolver...")
    await audit_figure_resolver(report)

    # Print report
    report.print_report()

    # Close database
    await db.close()

    # Summary
    if report.failures > 0:
        print("\n[FAIL] AUDIT FAILED - Critical issues found")
        sys.exit(1)
    elif report.fixes_needed:
        print(f"\n[WARN] AUDIT PASSED WITH WARNINGS - {len(report.fixes_needed)} fixes recommended")
        print("\nRecommended fixes:")
        for i, fix in enumerate(report.fixes_needed, 1):
            print(f"  {i}. {fix.name}: {fix.fix_description}")
        sys.exit(0)
    else:
        print("\n[PASS] AUDIT PASSED - Synthesis engine aligned with pipeline")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
