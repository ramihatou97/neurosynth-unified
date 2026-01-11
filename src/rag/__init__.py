"""
NeuroSynth Unified - RAG Layer
===============================

Retrieval-Augmented Generation for neurosurgical knowledge.

Components:
- context.py: Context assembly and citation tracking
- engine.py: RAG engine with Claude integration
- prompts.py: Medical domain prompt templates

Architecture:
    Question → Search → Context Assembly → Claude → Answer + Citations

Quick Start:
    from src.rag import RAGEngine, RAGConfig
    
    # Initialize engine
    engine = RAGEngine(
        search_service=search_service,
        api_key="your-anthropic-key"
    )
    
    # Ask a question
    response = await engine.ask(
        question="What is the retrosigmoid approach?",
        include_citations=True
    )
    
    # Access results
    print(response.answer)
    
    for citation in response.used_citations:
        print(f"[{citation.index}] {citation.snippet}")
        print(f"    Page {citation.page_number}, {citation.chunk_type}")
    
    for image in response.images:
        print(f"📷 {image.caption}")

Multi-turn Conversations:
    from src.rag import RAGConversation
    
    conversation = RAGConversation(engine)
    
    r1 = await conversation.ask("What approaches exist for CPA tumors?")
    r2 = await conversation.ask("Tell me more about the retrosigmoid approach")
    r3 = await conversation.ask("What are the complications?")
    
    # Get all citations from conversation
    all_citations = conversation.get_all_citations()

Medical Prompts:
    from src.rag import PromptLibrary, QuestionType
    
    library = PromptLibrary()
    
    # Get specialized system prompt
    system_prompt = library.get_system_prompt(QuestionType.PROCEDURAL)
    
    # Format templated questions
    question = library.format_question(
        "procedure_steps",
        procedure="retrosigmoid craniotomy"
    )
"""

# Context Assembly
from src.rag.context import (
    ContextAssembler,
    AssembledContext,
    Citation,
    ContextImage,
    CitationExtractor,
    ContextFormat,
    estimate_tokens,
    truncate_to_tokens
)

# RAG Engine
from src.rag.engine import (
    RAGEngine,
    RAGResponse,
    RAGConfig,
    RAGConversation,
    SYSTEM_PROMPT_MEDICAL,
    SYSTEM_PROMPT_GENERAL
)

# Prompts
from src.rag.prompts import (
    PromptLibrary,
    QuestionType,
    FormattedPrompt,
    get_medical_system_prompt,
    format_question,
    SYSTEM_PROMPTS,
    QUESTION_TEMPLATES
)

# V2 Unified RAG Engine (consolidated from V1+V2+V3)
from src.rag.unified_rag_engine import (
    UnifiedRAGEngine,
    SearchMode,
    QueryComplexity,
    QueryAnalysis,
    QueryRouter,
)

__all__ = [
    # Context
    'ContextAssembler',
    'AssembledContext',
    'Citation',
    'ContextImage',
    'CitationExtractor',
    'ContextFormat',
    'estimate_tokens',
    'truncate_to_tokens',
    
    # Engine
    'RAGEngine',
    'RAGResponse',
    'RAGConfig',
    'RAGConversation',
    'SYSTEM_PROMPT_MEDICAL',
    'SYSTEM_PROMPT_GENERAL',
    
    # Prompts
    'PromptLibrary',
    'QuestionType',
    'FormattedPrompt',
    'get_medical_system_prompt',
    'format_question',
    'SYSTEM_PROMPTS',
    'QUESTION_TEMPLATES',

    # V2 Unified Engine (consolidated)
    'UnifiedRAGEngine',
    'SearchMode',
    'QueryComplexity',
    'QueryAnalysis',
    'QueryRouter',
]
