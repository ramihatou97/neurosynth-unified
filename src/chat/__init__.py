"""
NeuroSynth Chat Module
======================

Production-ready conversational RAG with:
- Synthesis context linking
- Redis-backed persistence
- Token budget management
- Enhanced citations with medical concepts
"""

from .store import (
    Turn,
    Conversation,
    ConversationStore,
    SynthesisContextStore,
    get_conversation_store,
    get_synthesis_store,
    estimate_tokens,
    MAX_CONTEXT_TOKENS,
    MAX_HISTORY_TURNS,
)

from .engine import (
    EnhancedRAGEngine,
    EnhancedCitation,
    EnhancedRAGResponse,
    SynthesisContext,
    ConversationTurn,
    EnhancedConversationManager,
)

__all__ = [
    # Store
    "Turn",
    "Conversation",
    "ConversationStore",
    "SynthesisContextStore",
    "get_conversation_store",
    "get_synthesis_store",
    "estimate_tokens",
    "MAX_CONTEXT_TOKENS",
    "MAX_HISTORY_TURNS",
    # Engine
    "EnhancedRAGEngine",
    "EnhancedCitation",
    "EnhancedRAGResponse",
    "SynthesisContext",
    "ConversationTurn",
    "EnhancedConversationManager",
]
