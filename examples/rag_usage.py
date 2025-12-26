#!/usr/bin/env python3
"""
NeuroSynth RAG - Usage Examples
================================

Demonstrates the RAG layer for neurosurgical question answering.
"""

import asyncio
import os
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def example_basic_rag():
    """Basic RAG question answering."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic RAG Question Answering")
    print("=" * 60)
    
    # Note: Requires initialized search_service
    # This is a demonstration of the API
    
    code = '''
from src.rag import RAGEngine, RAGConfig
from src.retrieval import SearchService, FAISSManager, VoyageEmbedder
from src.database import init_database, get_repositories

# Initialize components
db = await init_database("postgresql://...")
repos = get_repositories(db)

faiss = FAISSManager("./indexes")
faiss.load()

embedder = VoyageEmbedder(api_key=os.getenv("VOYAGE_API_KEY"))
search = SearchService(db, faiss, embedder)

# Create RAG engine
engine = RAGEngine(
    search_service=search,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Ask a question
response = await engine.ask(
    question="What is the retrosigmoid approach for acoustic neuroma?",
    include_citations=True,
    include_images=True
)

# Print answer
print(response.answer)

# Print citations used
print("\\nCitations used:")
for c in response.used_citations:
    print(f"  [{c.index}] {c.snippet}")
    print(f"      Page {c.page_number} | {c.chunk_type}")

# Print linked images
if response.images:
    print("\\nRelated images:")
    for img in response.images:
        print(f"  📷 {img.caption}")

# Timing info
print(f"\\nSearch: {response.search_time_ms}ms")
print(f"Context: {response.context_time_ms}ms")
print(f"Generation: {response.generation_time_ms}ms")
'''
    print(code)


async def example_filtered_search():
    """RAG with filtered search."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: RAG with Filtered Search")
    print("=" * 60)
    
    code = '''
from src.rag import RAGEngine
from src.retrieval import SearchFilters

# Filter to specific document or chunk types
filters = SearchFilters(
    document_ids=["doc-uuid-here"],  # Specific document
    chunk_types=["PROCEDURE", "ANATOMY"],  # Content types
    specialties=["skull_base"],  # Subspecialty
    cuis=["C0001418"]  # Boost results with this CUI
)

response = await engine.ask(
    question="Describe the surgical technique",
    filters=filters,
    include_citations=True
)
'''
    print(code)


async def example_conversation():
    """Multi-turn RAG conversation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Multi-turn Conversation")
    print("=" * 60)
    
    code = '''
from src.rag import RAGEngine, RAGConversation

# Create conversation manager
conversation = RAGConversation(engine, max_history=10)

# First question
r1 = await conversation.ask(
    "What approaches are available for cerebellopontine angle tumors?"
)
print("Q1:", r1.answer[:200], "...")

# Follow-up question (uses conversation context)
r2 = await conversation.ask(
    "Tell me more about the retrosigmoid approach"
)
print("Q2:", r2.answer[:200], "...")

# Another follow-up
r3 = await conversation.ask(
    "What are the main complications to watch for?"
)
print("Q3:", r3.answer[:200], "...")

# Get all citations from the conversation
all_citations = conversation.get_all_citations()
print(f"\\nTotal unique sources: {len(all_citations)}")

# Clear and start new conversation
conversation.clear()
'''
    print(code)


async def example_streaming():
    """Streaming RAG responses."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Streaming Response")
    print("=" * 60)
    
    code = '''
from src.rag import RAGEngine

# Stream the response
async for chunk in await engine.ask(
    question="Explain the retrosigmoid approach",
    stream=True
):
    # Print each token as it arrives
    print(chunk, end="", flush=True)
    
    # Last chunk contains metadata
    if "RAG_METADATA" in chunk:
        # Parse metadata for citations
        import json
        meta_start = chunk.find("{")
        meta_end = chunk.rfind("}") + 1
        metadata = json.loads(chunk[meta_start:meta_end])
        print(f"\\n\\nUsed {len(metadata['used_citations'])} citations")
'''
    print(code)


async def example_specialized_prompts():
    """Using specialized medical prompts."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Specialized Medical Prompts")
    print("=" * 60)
    
    code = '''
from src.rag import RAGEngine, PromptLibrary, QuestionType

library = PromptLibrary()

# Use procedural system prompt
procedural_prompt = library.get_system_prompt(QuestionType.PROCEDURAL)

engine = RAGEngine(
    search_service=search,
    api_key=api_key,
    system_prompt=procedural_prompt  # Override default
)

# Or use question templates
question = library.format_question(
    "procedure_steps",
    procedure="far lateral approach"
)

response = await engine.ask(question)
'''
    print(code)


async def example_document_summary():
    """Summarize an entire document."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Document Summarization")
    print("=" * 60)
    
    code = '''
# Summarize a document by ID
response = await engine.summarize_document(
    document_id="doc-uuid-here",
    max_chunks=20
)

print(response.answer)
print(f"\\nBased on {response.context_chunks_used} chunks")
'''
    print(code)


async def example_compare_approaches():
    """Compare surgical approaches."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Compare Approaches")
    print("=" * 60)
    
    code = '''
# Compare two surgical approaches
response = await engine.compare_approaches(
    approach1="retrosigmoid",
    approach2="translabyrinthine"
)

print(response.answer)
'''
    print(code)


async def example_custom_config():
    """RAG with custom configuration."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Custom Configuration")
    print("=" * 60)
    
    code = '''
from src.rag import RAGEngine, RAGConfig, ContextFormat

config = RAGConfig(
    # Model settings
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    temperature=0.2,
    
    # Context settings
    max_context_tokens=12000,
    max_context_chunks=15,
    max_images=8,
    context_format=ContextFormat.XML,
    
    # Search settings
    search_top_k=30,
    search_mode="hybrid",
    enable_rerank=True,
    
    # Output settings
    include_citations=True,
    stream=False
)

engine = RAGEngine(
    search_service=search,
    api_key=api_key,
    config=config
)
'''
    print(code)


async def example_error_handling():
    """Proper error handling."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Error Handling")
    print("=" * 60)
    
    code = '''
from src.rag import RAGEngine

try:
    response = await engine.ask("My question")
    
    if response.context_chunks_used == 0:
        print("Warning: No relevant context found")
        print("The answer may not be based on available sources")
    
    print(response.answer)
    
except anthropic.APIError as e:
    print(f"API error: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
'''
    print(code)


async def main():
    """Run all examples."""
    print("NeuroSynth RAG Layer - Usage Examples")
    print("=" * 60)
    print()
    print("These examples demonstrate the RAG API.")
    print("To run them, you need:")
    print("  1. PostgreSQL database with indexed content")
    print("  2. Built FAISS indexes")
    print("  3. VOYAGE_API_KEY environment variable")
    print("  4. ANTHROPIC_API_KEY environment variable")
    
    await example_basic_rag()
    await example_filtered_search()
    await example_conversation()
    await example_streaming()
    await example_specialized_prompts()
    await example_document_summary()
    await example_compare_approaches()
    await example_custom_config()
    await example_error_handling()
    
    print("\n" + "=" * 60)
    print("END OF EXAMPLES")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
