#!/usr/bin/env python
"""Generate sample synthesis and save to readable file."""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

from src.api.dependencies import ServiceContainer
from src.synthesis.engine import SynthesisEngine, TemplateType
from anthropic import AsyncAnthropic

async def generate():
    print("Generating synthesis sample...")

    # Initialize
    container = ServiceContainer()
    await container.initialize()

    # Search
    response = await container.search.search(
        query="translabyrinthine approach vestibular schwannoma",
        mode="hybrid",
        top_k=30,
        include_images=True,  # Enable image attachment for figure resolution
        rerank=False
    )

    print(f"Found {len(response.results)} chunks")

    # Synthesize
    anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    engine = SynthesisEngine(anthropic_client=anthropic_client)

    print("Generating synthesis (this takes ~2 minutes)...")
    result = await engine.synthesize(
        topic="translabyrinthine approach for vestibular schwannoma",
        template_type=TemplateType.PROCEDURAL,
        search_results=response.results,
        include_verification=False,
        include_figures=True
    )

    # Save to file
    output_file = Path("synthesis_output.md")
    with open(output_file, 'w') as f:
        f.write(f"# {result.title}\n\n")
        f.write(f"## Abstract\n\n{result.abstract}\n\n")
        f.write("---\n\n")

        for section in result.sections:
            f.write(f"## {section.title}\n\n")
            f.write(f"{section.content}\n\n")
            f.write(f"*({section.word_count} words)*\n\n")
            f.write("---\n\n")

        f.write("## References\n\n")
        for ref in result.references:
            if isinstance(ref, dict):
                f.write(f"- {ref.get('source', 'Unknown')}\n")
            else:
                f.write(f"- {str(ref)}\n")

        f.write(f"\n---\n\n")
        f.write(f"**Statistics:**\n")
        f.write(f"- Total words: {result.total_words}\n")
        f.write(f"- Sections: {len(result.sections)}\n")
        f.write(f"- Generation time: {result.synthesis_time_ms/1000:.1f}s\n")

    print(f"\n✅ Synthesis saved to: {output_file}")
    print(f"   Words: {result.total_words}")
    print(f"   Sections: {len(result.sections)}")

    return str(output_file)

if __name__ == "__main__":
    output = asyncio.run(generate())
    print(f"\nView output: cat {output}")
