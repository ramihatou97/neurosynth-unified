"""
NeuroSynth Unified - Re-ranking Module
=======================================

Re-ranking models for improving search result quality.

Supports:
- Cross-encoder re-ranking (sentence-transformers)
- LLM-based re-ranking (Claude)
- Custom scoring functions

Usage:
    from src.retrieval.reranker import CrossEncoderReranker
    
    reranker = CrossEncoderReranker()
    scores = await reranker.score(query, documents)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import asyncio

logger = logging.getLogger(__name__)


# =============================================================================
# Base Reranker
# =============================================================================

class BaseReranker(ABC):
    """Abstract base class for re-ranking models."""
    
    @abstractmethod
    async def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Score documents against query.
        
        Args:
            query: Search query
            documents: List of document texts
        
        Returns:
            List of relevance scores (higher = more relevant)
        """
        pass
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        Re-rank documents and return sorted indices with scores.
        
        Returns:
            List of (original_index, score) tuples, sorted by score desc
        """
        scores = await self.score(query, documents)
        
        # Pair with indices
        indexed_scores = list(enumerate(scores))
        
        # Sort by score descending
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            indexed_scores = indexed_scores[:top_k]
        
        return indexed_scores


# =============================================================================
# Cross-Encoder Reranker
# =============================================================================

class CrossEncoderReranker(BaseReranker):
    """
    Re-ranking using cross-encoder models.
    
    Cross-encoders jointly encode query and document,
    providing more accurate relevance scores than bi-encoders.
    
    Models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, general)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (balanced)
    - BAAI/bge-reranker-base (high quality)
    - BAAI/bge-reranker-large (highest quality)
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                    max_length=self.max_length
                )
                logger.info(f"Loaded cross-encoder: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. "
                    "Install with: pip install sentence-transformers"
                )
    
    async def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """Score documents using cross-encoder."""
        if not documents:
            return []
        
        self._load_model()
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, batch_size=self.batch_size)
        )
        
        return scores.tolist()


# =============================================================================
# LLM Reranker (Claude)
# =============================================================================

class LLMReranker(BaseReranker):
    """
    Re-ranking using LLM (Claude).
    
    Uses Claude to score document relevance to query.
    More expensive but can handle nuanced relevance judgments.
    
    Best for:
    - Complex queries requiring reasoning
    - Domain-specific relevance
    - Small result sets (top 10-20)
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-sonnet-4-20250514",
        batch_size: int = 5,
        temperature: float = 0.0
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self._client = None
    
    def _get_client(self):
        """Get Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client
    
    async def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """Score documents using Claude."""
        if not documents:
            return []
        
        scores = []
        
        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_scores = await self._score_batch(query, batch)
            scores.extend(batch_scores)
        
        return scores
    
    async def _score_batch(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """Score a batch of documents."""
        # Build prompt
        docs_text = "\n\n".join([
            f"[Document {i+1}]\n{doc[:500]}..."
            if len(doc) > 500 else f"[Document {i+1}]\n{doc}"
            for i, doc in enumerate(documents)
        ])
        
        prompt = f"""Rate the relevance of each document to the query on a scale of 0-10.

Query: {query}

{docs_text}

For each document, provide ONLY a number from 0-10 indicating relevance.
Format: One number per line, in order.
10 = Highly relevant, directly answers the query
5 = Somewhat relevant, contains related information
0 = Not relevant

Scores:"""

        # Call Claude
        client = self._get_client()
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        
        # Parse scores
        text = response.content[0].text.strip()
        lines = text.split('\n')
        
        scores = []
        for line in lines:
            try:
                score = float(line.strip()) / 10.0  # Normalize to 0-1
                scores.append(min(max(score, 0), 1))  # Clamp
            except ValueError:
                scores.append(0.5)  # Default
        
        # Pad if needed
        while len(scores) < len(documents):
            scores.append(0.5)
        
        return scores[:len(documents)]


# =============================================================================
# Ensemble Reranker
# =============================================================================

class EnsembleReranker(BaseReranker):
    """
    Combines multiple re-rankers with weighted scoring.
    
    Usage:
        ensemble = EnsembleReranker([
            (CrossEncoderReranker(), 0.6),
            (LLMReranker(), 0.4)
        ])
    """
    
    def __init__(
        self,
        rerankers: List[Tuple[BaseReranker, float]]
    ):
        """
        Args:
            rerankers: List of (reranker, weight) tuples
        """
        self.rerankers = rerankers
        
        # Normalize weights
        total_weight = sum(w for _, w in rerankers)
        self.weights = [w / total_weight for _, w in rerankers]
    
    async def score(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """Score using ensemble of re-rankers."""
        if not documents:
            return []
        
        # Get scores from all rerankers
        all_scores = await asyncio.gather(*[
            reranker.score(query, documents)
            for reranker, _ in self.rerankers
        ])
        
        # Weighted combination
        combined = [0.0] * len(documents)
        for scores, weight in zip(all_scores, self.weights):
            for i, score in enumerate(scores):
                combined[i] += score * weight
        
        return combined


# =============================================================================
# Medical Domain Reranker
# =============================================================================

class MedicalReranker(BaseReranker):
    """
    Medical domain-specific re-ranking.
    
    Combines:
    - Semantic similarity
    - Medical entity overlap (UMLS CUIs)
    - Section type relevance
    """
    
    def __init__(
        self,
        base_reranker: BaseReranker = None,
        cui_weight: float = 0.2,
        section_weight: float = 0.1
    ):
        self.base_reranker = base_reranker
        self.cui_weight = cui_weight
        self.section_weight = section_weight
        
        # Section type relevance for common queries
        self.section_relevance = {
            'PROCEDURE': {'surgical', 'approach', 'technique', 'step'},
            'ANATOMY': {'anatomy', 'structure', 'location', 'nerve'},
            'PATHOLOGY': {'tumor', 'lesion', 'disease', 'condition'},
            'CLINICAL': {'symptom', 'diagnosis', 'treatment', 'outcome'}
        }
    
    async def score(
        self,
        query: str,
        documents: List[str],
        metadata: List[Dict] = None
    ) -> List[float]:
        """Score with medical domain features."""
        if not documents:
            return []
        
        # Base scores
        if self.base_reranker:
            base_scores = await self.base_reranker.score(query, documents)
        else:
            base_scores = [0.5] * len(documents)
        
        # Adjust with metadata if available
        if metadata:
            query_lower = query.lower()
            
            for i, meta in enumerate(metadata):
                # CUI overlap bonus
                query_cuis = meta.get('query_cuis', [])
                doc_cuis = meta.get('cuis', [])
                if query_cuis and doc_cuis:
                    overlap = len(set(query_cuis) & set(doc_cuis))
                    base_scores[i] += overlap * self.cui_weight * 0.1
                
                # Section type relevance
                section_type = meta.get('chunk_type', '')
                if section_type in self.section_relevance:
                    keywords = self.section_relevance[section_type]
                    if any(kw in query_lower for kw in keywords):
                        base_scores[i] += self.section_weight
        
        # Normalize
        max_score = max(base_scores) if base_scores else 1
        if max_score > 0:
            base_scores = [s / max_score for s in base_scores]
        
        return base_scores


# =============================================================================
# Factory
# =============================================================================

def create_reranker(
    reranker_type: str = "cross-encoder",
    **kwargs
) -> BaseReranker:
    """
    Factory function to create re-rankers.
    
    Args:
        reranker_type: Type of reranker
            - "cross-encoder": Fast cross-encoder model
            - "llm": Claude-based reranker
            - "medical": Medical domain reranker
            - "ensemble": Combination of multiple
        **kwargs: Reranker-specific arguments
    
    Returns:
        BaseReranker instance
    """
    if reranker_type == "cross-encoder":
        return CrossEncoderReranker(**kwargs)
    
    elif reranker_type == "llm":
        return LLMReranker(**kwargs)
    
    elif reranker_type == "medical":
        base = CrossEncoderReranker(**kwargs.pop('cross_encoder_kwargs', {}))
        return MedicalReranker(base_reranker=base, **kwargs)
    
    elif reranker_type == "ensemble":
        rerankers = kwargs.get('rerankers', [
            (CrossEncoderReranker(), 0.7),
            (LLMReranker(), 0.3)
        ])
        return EnsembleReranker(rerankers)
    
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing Rerankers...")
        
        # Test cross-encoder (requires sentence-transformers)
        try:
            reranker = CrossEncoderReranker()
            
            query = "surgical approach for acoustic neuroma"
            documents = [
                "The retrosigmoid approach provides excellent exposure of the cerebellopontine angle.",
                "Patient demographics and baseline characteristics were collected.",
                "Facial nerve preservation is a key goal in acoustic neuroma surgery."
            ]
            
            scores = await reranker.score(query, documents)
            print(f"Cross-encoder scores: {scores}")
            
            ranked = await reranker.rerank(query, documents)
            print(f"Ranked order: {ranked}")
            
        except ImportError:
            print("sentence-transformers not installed, skipping cross-encoder test")
        
        print("✓ Test complete")
    
    asyncio.run(test())
