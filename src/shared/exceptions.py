"""Custom exceptions for NeuroSynth unified system."""


class NeuroSynthException(Exception):
    """Base exception for all NeuroSynth errors."""
    pass


# Phase 1: Extraction Pipeline Exceptions

class ExtractionException(NeuroSynthException):
    """Base exception for extraction errors."""
    pass


class PDFProcessingError(ExtractionException):
    """Error during PDF processing."""
    pass


class OCRError(ExtractionException):
    """Error during OCR fallback."""
    pass


class ChunkingError(ExtractionException):
    """Error during semantic chunking."""
    pass


class EntityExtractionError(ExtractionException):
    """Error during entity extraction."""
    pass


class EmbeddingError(ExtractionException):
    """Error generating embeddings."""
    pass


class VLMCaptioningError(ExtractionException):
    """Error during VLM image captioning."""
    pass


class TriageError(ExtractionException):
    """Error during visual triage."""
    pass


class UMLSLinkingError(ExtractionException):
    """Error during UMLS CUI linking."""
    pass


# Phase 2: Retrieval and RAG Exceptions

class RetrievalException(NeuroSynthException):
    """Base exception for retrieval errors."""
    pass


class SearchError(RetrievalException):
    """Error during search operation."""
    pass


class FAISSError(RetrievalException):
    """Error in FAISS index operations."""
    pass


class DatabaseError(RetrievalException):
    """Error in database operations."""
    pass


class ConnectionError(DatabaseError):
    """Database connection error."""
    pass


class QueryError(DatabaseError):
    """Database query error."""
    pass


class RAGError(RetrievalException):
    """Error in RAG generation."""
    pass


class ContextAssemblyError(RAGError):
    """Error assembling context for RAG."""
    pass


class PromptConstructionError(RAGError):
    """Error constructing prompt."""
    pass


class LLMError(RAGError):
    """Error calling LLM (Claude)."""
    pass


# Configuration and Validation Exceptions

class ConfigurationError(NeuroSynthException):
    """Error in configuration."""
    pass


class ValidationError(NeuroSynthException):
    """Data validation error."""
    pass


class ModelError(ValidationError):
    """Error with data model."""
    pass


# API and External Service Exceptions

class APIError(NeuroSynthException):
    """Error in API operation."""
    pass


class ExternalServiceError(NeuroSynthException):
    """Error calling external service (Anthropic, Voyage, etc.)."""
    pass


class APIKeyError(ExternalServiceError):
    """API key missing or invalid."""
    pass


class RateLimitError(ExternalServiceError):
    """Rate limit exceeded."""
    pass
