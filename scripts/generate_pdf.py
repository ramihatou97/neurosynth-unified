#!/usr/bin/env python
"""
Generate Synthesis with Properly Embedded Images

This script:
1. Runs synthesis with include_images=True
2. Resolves figure placeholders to actual images from the database
3. Exports to publication-ready PDF with embedded images

Usage:
    python generate_synthesis_pdf.py "translabyrinthine approach" --output synthesis.pdf
    python generate_synthesis_pdf.py "pterional craniotomy" --template PROCEDURAL --output output.pdf
"""

import asyncio
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


async def generate_synthesis_with_images(
    topic: str,
    template_type: str = "PROCEDURAL",
    max_chunks: int = 50,
    include_verification: bool = False,
    output_json: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Generate synthesis with images properly attached.
    
    Returns dict with:
        - title, abstract, sections
        - resolved_figures (with actual image paths)
        - references
    """
    # Import here to avoid issues if not in backend env
    try:
        from src.api.dependencies import ServiceContainer
        from src.synthesis.engine import SynthesisEngine, TemplateType
        from anthropic import AsyncAnthropic
    except ImportError:
        print("⚠️  Backend modules not found. Using mock data.")
        return _generate_mock_data(topic)
    
    print(f"🔍 Searching for: {topic}")
    
    # Initialize services
    container = ServiceContainer()
    await container.initialize()
    
    # Search WITH images attached
    search_response = await container.search.search(
        query=topic,
        mode="hybrid",
        top_k=max_chunks,
        include_images=True,  # ✅ CRITICAL: Enable image attachment
        rerank=False
    )
    
    print(f"   Found {len(search_response.results)} chunks")
    total_images = sum(len(r.images) for r in search_response.results)
    print(f"   Attached {total_images} images")
    
    if total_images == 0:
        print("   ⚠️  No images found. Checking database...")
        # Try to get images directly
        images = await _fetch_images_for_topic(container, topic)
        print(f"   Found {len(images)} images in database")
    
    # Initialize Claude client
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Get template type enum
    template = getattr(TemplateType, template_type.upper(), TemplateType.PROCEDURAL)
    
    # Create synthesis engine
    engine = SynthesisEngine(client)
    
    print(f"📝 Generating {template.value} synthesis...")
    
    # Generate synthesis WITH figures
    result = await engine.synthesize(
        topic=topic,
        template_type=template,
        search_results=search_response.results,
        include_verification=include_verification,
        include_figures=True  # ✅ CRITICAL: Enable figure resolution
    )
    
    print(f"   Title: {result.title}")
    print(f"   Sections: {len(result.sections)}")
    print(f"   Figure requests: {len(result.figure_requests)}")
    print(f"   Resolved figures: {len(result.resolved_figures)}")
    
    # Convert to dict
    data = {
        'title': result.title,
        'abstract': result.abstract,
        'sections': [s.to_dict() for s in result.sections],
        'figure_requests': [f.to_dict() for f in result.figure_requests],
        'resolved_figures': result.resolved_figures,
        'references': result.references if hasattr(result, 'references') else [],
        'metadata': {
            'topic': topic,
            'template': template_type,
            'generated_at': datetime.now().isoformat(),
            'total_words': result.total_words,
            'total_figures': result.total_figures
        }
    }
    
    # If no resolved figures, try to resolve manually
    if len(result.resolved_figures) == 0 and len(result.figure_requests) > 0:
        print("   🔧 Manually resolving figures...")
        data['resolved_figures'] = await _resolve_figures_manually(
            container, 
            result.figure_requests
        )
        print(f"   Resolved: {len(data['resolved_figures'])} figures")
    
    # Save JSON if requested
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"   Saved: {output_json}")
    
    await container.cleanup()
    
    return data


async def _fetch_images_for_topic(container, topic: str) -> List[Dict]:
    """Fetch images related to topic from database."""
    try:
        # Search for relevant chunks first
        response = await container.search.search(
            query=topic,
            mode="text",
            top_k=20
        )
        
        # Get chunk IDs
        chunk_ids = [r.chunk_id for r in response.results]
        
        # Query images linked to these chunks
        async with container.db.session() as session:
            from sqlalchemy import text
            
            result = await session.execute(text("""
                SELECT DISTINCT i.id, i.storage_path, i.vlm_caption, i.image_type
                FROM images i
                JOIN chunk_image_links cil ON i.id = cil.image_id
                WHERE cil.chunk_id = ANY(:chunk_ids)
                LIMIT 20
            """), {"chunk_ids": chunk_ids})
            
            images = []
            for row in result:
                images.append({
                    'id': row.id,
                    'path': row.storage_path,
                    'caption': row.vlm_caption or '',
                    'type': row.image_type
                })
            
            return images
            
    except Exception as e:
        print(f"   Error fetching images: {e}")
        return []


async def _resolve_figures_manually(container, figure_requests) -> List[Dict]:
    """Manually resolve figure requests to images."""
    resolved = []
    
    try:
        async with container.db.session() as session:
            from sqlalchemy import text
            
            for fig in figure_requests:
                topic = fig.topic if hasattr(fig, 'topic') else fig.get('topic', '')
                fig_type = fig.figure_type if hasattr(fig, 'figure_type') else fig.get('type', '')
                
                # Search for matching images
                result = await session.execute(text("""
                    SELECT id, storage_path, vlm_caption, image_type,
                           similarity(vlm_caption, :topic) as sim
                    FROM images
                    WHERE vlm_caption IS NOT NULL
                    ORDER BY sim DESC
                    LIMIT 1
                """), {"topic": topic})
                
                row = result.first()
                if row and row.sim > 0.1:
                    resolved.append({
                        'placeholder_id': fig.placeholder_id if hasattr(fig, 'placeholder_id') else f'fig_{len(resolved)+1}',
                        'topic': topic,
                        'type': fig_type,
                        'path': row.storage_path,
                        'caption': row.vlm_caption,
                        'confidence': float(row.sim)
                    })
                    
    except Exception as e:
        print(f"   Error resolving figures: {e}")
    
    return resolved


def _generate_mock_data(topic: str) -> Dict[str, Any]:
    """Generate mock synthesis data for testing without backend."""
    return {
        'title': f"Surgical Approach: {topic.title()}",
        'abstract': f"This comprehensive review covers the key aspects of {topic}, including anatomical considerations, surgical technique, and outcomes.",
        'sections': [
            {
                'title': 'Introduction',
                'content': f'''The {topic} represents a cornerstone technique in modern neurosurgery.

[REQUEST_FIGURE: type="anatomical_diagram" topic="surgical anatomy overview"]

Understanding the relevant anatomy is crucial for successful outcomes.

**PEARL:** Always identify key landmarks before proceeding with deeper dissection.''',
                'word_count': 45
            },
            {
                'title': 'Surgical Technique',
                'content': f'''The patient is positioned appropriately for optimal access.

[REQUEST_FIGURE: type="surgical_photograph" topic="patient positioning"]

The incision is marked along anatomical landmarks.

[REQUEST_FIGURE: type="intraoperative_view" topic="surgical exposure"]

**HAZARD:** Avoid injury to adjacent neurovascular structures.

Careful hemostasis is maintained throughout the procedure.''',
                'word_count': 55
            },
            {
                'title': 'Outcomes and Complications',
                'content': '''Outcomes are generally favorable with appropriate patient selection.

Complications may include infection, hemorrhage, and neurological deficits.

**PEARL:** Early mobilization reduces thromboembolic complications.''',
                'word_count': 30
            }
        ],
        'figure_requests': [
            {'placeholder_id': 'fig_1', 'type': 'anatomical_diagram', 'topic': 'surgical anatomy overview'},
            {'placeholder_id': 'fig_2', 'type': 'surgical_photograph', 'topic': 'patient positioning'},
            {'placeholder_id': 'fig_3', 'type': 'intraoperative_view', 'topic': 'surgical exposure'}
        ],
        'resolved_figures': [],  # Will need actual images
        'references': [
            {'source': 'Rhoton AL. Cranial Anatomy and Surgical Approaches. 2003.', 'chunks_used': 5},
            {'source': 'Youmans and Winn Neurological Surgery. 8th ed. 2022.', 'chunks_used': 3}
        ],
        'metadata': {
            'topic': topic,
            'template': 'PROCEDURAL',
            'generated_at': datetime.now().isoformat(),
            'total_words': 130,
            'total_figures': 3
        }
    }


def export_to_pdf(
    synthesis_data: Dict[str, Any],
    output_path: Path,
    image_base_path: Path = Path("data/images"),
    author: str = "NeuroSynth",
    quality: str = "high"
):
    """Export synthesis data to publication-ready PDF."""
    from synthesis_export import SynthesisExporter, ExportConfig
    
    config = ExportConfig(
        title=synthesis_data.get('title', 'Synthesis'),
        author=author,
        image_quality=quality,
        include_toc=True,
        include_abstract=True,
        include_references=True
    )
    
    exporter = SynthesisExporter(
        image_base_path=image_base_path,
        config=config
    )
    
    print(f"📄 Generating PDF...")
    exporter.to_pdf(synthesis_data, output_path)
    print(f"✅ PDF saved: {output_path}")
    
    # Also save HTML for preview
    html_path = output_path.with_suffix('.html')
    html = exporter.to_html(synthesis_data, embed_images=True)
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"✅ HTML saved: {html_path}")


def export_to_markdown(
    synthesis_data: Dict[str, Any],
    output_path: Path,
    image_base_path: Path = Path("data/images")
):
    """Export synthesis data to Markdown with image links."""
    from synthesis_export import SynthesisExporter, ExportConfig
    
    config = ExportConfig(
        title=synthesis_data.get('title', 'Synthesis')
    )
    
    exporter = SynthesisExporter(
        image_base_path=image_base_path,
        config=config
    )
    
    md = exporter.to_markdown(synthesis_data)
    
    with open(output_path, 'w') as f:
        f.write(md)
    
    print(f"✅ Markdown saved: {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready synthesis with embedded images"
    )
    parser.add_argument(
        "topic",
        type=str,
        help="Topic to synthesize (e.g., 'translabyrinthine approach')"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("synthesis_output.pdf"),
        help="Output file path (.pdf, .html, .md)"
    )
    parser.add_argument(
        "--template", "-t",
        type=str,
        default="PROCEDURAL",
        choices=["PROCEDURAL", "DISORDER", "ANATOMY", "ENCYCLOPEDIA"],
        help="Synthesis template type"
    )
    parser.add_argument(
        "--chunks", "-c",
        type=int,
        default=50,
        help="Maximum chunks to use"
    )
    parser.add_argument(
        "--images", "-i",
        type=Path,
        default=Path("data/images"),
        help="Base path for images"
    )
    parser.add_argument(
        "--author",
        type=str,
        default="NeuroSynth",
        help="Author name for document"
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=["high", "medium", "low"],
        default="high",
        help="Image quality for PDF"
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        help="Also save raw synthesis as JSON"
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        help="Load existing synthesis from JSON instead of generating"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Include Gemini verification"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NEUROSYNTH PUBLICATION GENERATOR")
    print("=" * 60)
    print()
    
    # Get synthesis data
    if args.from_json:
        print(f"📂 Loading from: {args.from_json}")
        with open(args.from_json, 'r') as f:
            synthesis_data = json.load(f)
    else:
        synthesis_data = await generate_synthesis_with_images(
            topic=args.topic,
            template_type=args.template,
            max_chunks=args.chunks,
            include_verification=args.verify,
            output_json=args.save_json
        )
    
    print()
    
    # Export based on output format
    suffix = args.output.suffix.lower()
    
    if suffix == '.pdf':
        export_to_pdf(
            synthesis_data,
            args.output,
            image_base_path=args.images,
            author=args.author,
            quality=args.quality
        )
    elif suffix == '.html':
        from synthesis_export import SynthesisExporter, ExportConfig
        config = ExportConfig(title=synthesis_data['title'], author=args.author)
        exporter = SynthesisExporter(args.images, config)
        html = exporter.to_html(synthesis_data, embed_images=True)
        with open(args.output, 'w') as f:
            f.write(html)
        print(f"✅ HTML saved: {args.output}")
    elif suffix in ('.md', '.markdown'):
        export_to_markdown(synthesis_data, args.output, args.images)
    else:
        print(f"❌ Unknown format: {suffix}")
        print("   Supported: .pdf, .html, .md")
        return 1
    
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())
