#!/usr/bin/env python
"""
Simple PDF export using the MedicalPDFGenerator directly.
Bypasses ServiceContainer compatibility issues.
"""
import sys
import json
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.synthesis.pdf_generator import MedicalPDFGenerator

def main():
    # Load synthesis JSON
    input_file = Path("synthesis_for_pdf.json")

    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return 1

    print(f"Loading synthesis from: {input_file}")
    with open(input_file, 'r') as f:
        synthesis_data = json.load(f)

    print(f"✅ Loaded:")
    print(f"   Title: {synthesis_data['title']}")
    print(f"   Sections: {len(synthesis_data['sections'])}")
    print(f"   Figure requests: {len(synthesis_data.get('figure_requests', []))}")
    print(f"   Resolved figures: {len(synthesis_data.get('resolved_figures', []))}")

    # Create PDF
    output_file = Path("translabyrinthine_synthesis.pdf")
    print(f"\nGenerating PDF: {output_file}")

    generator = MedicalPDFGenerator(
        title=synthesis_data['title'],
        author="NeuroSynth AI",
        subject="Neurosurgical Synthesis"
    )

    try:
        # Build story (ReportLab flowables)
        story = []

        # Add abstract
        if synthesis_data.get('abstract'):
            story.extend(generator._create_abstract(synthesis_data['abstract']))

        # Add sections
        for section in synthesis_data['sections']:
            story.extend(generator._create_section(
                title=section['title'],
                content=section['content'],
                level=section.get('level', 2)
            ))

        # Generate PDF
        generator.build(story, str(output_file))

        print(f"\n✅ PDF created successfully!")
        print(f"   File: {output_file}")
        print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")

        return 0

    except Exception as e:
        print(f"\n❌ PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
