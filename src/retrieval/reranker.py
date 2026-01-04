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
import numpy as np

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
    
    elif reranker_type == "diversity":
        return DiversityReranker(**kwargs)

    elif reranker_type == "quality":
        return QualityReranker(**kwargs)

    elif reranker_type == "pipeline":
        return create_pipeline_reranker(**kwargs)

    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")


# =============================================================================
# Diversity Reranker (MMR)
# =============================================================================

class DiversityReranker:
    """
    Maximal Marginal Relevance reranker for result diversity.

    Balances relevance with diversity to avoid redundant results.

    Algorithm:
        MMR = λ * Sim(d, q) - (1-λ) * max(Sim(d, d_selected))

    Where:
        - λ: Balance parameter (1.0 = pure relevance, 0.0 = pure diversity)
        - Sim(d, q): Query-document similarity
        - Sim(d, d_selected): Document-document similarity with selected docs

    Expected improvement: +10-15% synthesis coverage
    """

    def __init__(
        self,
        lambda_mult: float = 0.7,
        fallback_score: float = 0.5
    ):
        """
        Initialize diversity reranker.

        Args:
            lambda_mult: Balance between relevance (1.0) and diversity (0.0)
                        Recommended: 0.6-0.8 for medical retrieval
            fallback_score: Score to use when embeddings unavailable
        """
        if not 0 <= lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")

        self.lambda_mult = lambda_mult
        self.fallback_score = fallback_score

    async def score(
        self,
        query: str,
        documents: List[str],
        embeddings: List[np.ndarray] = None,
        query_embedding: np.ndarray = None,
        initial_scores: List[float] = None
    ) -> List[float]:
        """
        Score documents using MMR algorithm.

        Args:
            query: Search query (unused if embeddings provided)
            documents: List of document texts
            embeddings: List of document embedding vectors
            query_embedding: Query embedding vector
            initial_scores: Optional initial relevance scores

        Returns:
            List of MMR-adjusted scores
        """
        n = len(documents)
        if n == 0:
            return []

        # Fallback if no embeddings
        if embeddings is None or query_embedding is None:
            logger.warning("No embeddings for MMR, using fallback scores")
            return [self.fallback_score] * n

        # Validate embeddings
        if len(embeddings) != n:
            logger.warning(f"Embedding count mismatch: {len(embeddings)} vs {n} docs")
            embeddings = embeddings[:n] if len(embeddings) > n else embeddings + [embeddings[0]] * (n - len(embeddings))

        return self._calculate_mmr_scores(
            query_embedding,
            embeddings,
            initial_scores
        )

    def _calculate_mmr_scores(
        self,
        query_emb: np.ndarray,
        doc_embs: List[np.ndarray],
        initial_scores: List[float] = None
    ) -> List[float]:
        """
        Calculate MMR scores using iterative selection.

        Returns:
            List of scores (higher = selected earlier)
        """
        n = len(doc_embs)
        if n == 0:
            return []

        # Convert to numpy array
        query_emb = np.asarray(query_emb, dtype=np.float32)
        doc_matrix = np.vstack([np.asarray(e, dtype=np.float32) for e in doc_embs])

        # Query-document similarities
        query_sim = self._cosine_similarity_batch(query_emb, doc_matrix)

        # Use initial scores if provided, otherwise use query similarity
        if initial_scores is not None:
            relevance = np.array(initial_scores, dtype=np.float32)
        else:
            relevance = query_sim

        # Document-document similarity matrix
        doc_sim = self._pairwise_cosine(doc_matrix)

        # MMR selection
        selected_indices = []
        remaining = list(range(n))
        mmr_scores = np.zeros(n, dtype=np.float32)

        for rank in range(n):
            if not remaining:
                break

            if not selected_indices:
                # First document: highest relevance
                best_idx = max(remaining, key=lambda i: relevance[i])
            else:
                # Calculate MMR for each remaining document
                best_mmr = float('-inf')
                best_idx = remaining[0]

                for idx in remaining:
                    # Relevance component
                    rel = relevance[idx]

                    # Diversity component: max similarity to already selected
                    max_sim_to_selected = max(doc_sim[idx][s] for s in selected_indices)

                    # MMR formula
                    mmr = self.lambda_mult * rel - (1 - self.lambda_mult) * max_sim_to_selected

                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = idx

            # Assign rank-based score (first selected = highest score)
            mmr_scores[best_idx] = 1.0 - (rank / n)
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return mmr_scores.tolist()

    async def rerank(
        self,
        query: str,
        documents: List[str],
        embeddings: List[np.ndarray] = None,
        query_embedding: np.ndarray = None,
        top_k: int = None,
        initial_scores: List[float] = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents and return sorted indices with scores.

        Returns:
            List of (original_index, score) tuples, sorted by score desc
        """
        scores = await self.score(query, documents, embeddings, query_embedding, initial_scores)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    @staticmethod
    def _cosine_similarity_batch(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all documents."""
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        docs_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-8)
        return np.dot(docs_norm, query_norm)

    @staticmethod
    def _pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarities."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normalized = matrix / (norms + 1e-8)
        return np.dot(normalized, normalized.T)


# =============================================================================
# Quality Reranker
# =============================================================================

class QualityReranker:
    """
    Quality-aware reranker using chunk metadata scores.

    Combines multiple quality signals:
    - Authority score (Rhoton > random paper)
    - Readability score (text clarity)
    - Coherence score (internal consistency)
    - Completeness score (information density)
    - Type appropriateness (procedure chunks for "how to" queries)

    Expected improvement: +10-12% synthesis quality
    """

    def __init__(
        self,
        authority_weight: float = 0.30,
        readability_weight: float = 0.20,
        coherence_weight: float = 0.20,
        completeness_weight: float = 0.20,
        type_weight: float = 0.10,
        default_score: float = 0.5
    ):
        """
        Initialize quality reranker.

        Args:
            authority_weight: Weight for source authority (0.3 recommended)
            readability_weight: Weight for text readability
            coherence_weight: Weight for chunk coherence
            completeness_weight: Weight for information completeness
            type_weight: Weight for type-query matching
            default_score: Default when metadata missing
        """
        self.weights = {
            'authority': authority_weight,
            'readability': readability_weight,
            'coherence': coherence_weight,
            'completeness': completeness_weight,
            'type': type_weight
        }

        self.default_score = default_score

        # Query type detection patterns
        self.type_patterns = {
            'PROCEDURE': {
                'keywords': {'how', 'step', 'technique', 'approach', 'surgical', 'operative',
                            'method', 'perform', 'conduct', 'execute', 'do'},
                'phrases': ['how to', 'steps for', 'technique for', 'approach to']
            },
            'ANATOMY': {
                'keywords': {'where', 'structure', 'location', 'anatomy', 'landmark',
                            'relationship', 'course', 'origin', 'insertion', 'boundary'},
                'phrases': ['where is', 'located at', 'anatomy of', 'structures of']
            },
            'PATHOLOGY': {
                'keywords': {'what', 'cause', 'symptom', 'disease', 'pathology',
                            'etiology', 'mechanism', 'presentation', 'diagnosis'},
                'phrases': ['what is', 'causes of', 'symptoms of', 'pathology of']
            },
            'CLINICAL': {
                'keywords': {'treatment', 'outcome', 'prognosis', 'management',
                            'complication', 'indication', 'contraindication', 'result'},
                'phrases': ['treatment for', 'outcome of', 'management of', 'indications for']
            }
        }

        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Quality weights sum to {total_weight}, normalizing...")
            self.weights = {k: v/total_weight for k, v in self.weights.items()}

    async def score(
        self,
        query: str,
        documents: List[str],
        metadata: List[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Score documents based on quality metrics.

        Args:
            query: Search query
            documents: List of document texts
            metadata: List of metadata dicts with quality scores

        Returns:
            List of quality-adjusted scores
        """
        n = len(documents)
        if n == 0:
            return []

        if not metadata:
            logger.warning("No metadata for quality reranking, using default scores")
            return [self.default_score] * n

        # Detect query type
        query_type = self._detect_query_type(query)
        logger.debug(f"Detected query type: {query_type}")

        scores = []
        for i, meta in enumerate(metadata):
            if meta is None:
                scores.append(self.default_score)
            else:
                score = self._calculate_quality_score(meta, query_type)
                scores.append(score)

        return scores

    async def rerank(
        self,
        query: str,
        documents: List[str],
        metadata: List[Dict[str, Any]] = None,
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents by quality and return sorted indices.

        Returns:
            List of (original_index, score) tuples, sorted by score desc
        """
        scores = await self.score(query, documents, metadata)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    def _detect_query_type(self, query: str) -> str:
        """
        Detect query type from keywords and phrases.

        Returns:
            Detected type ('PROCEDURE', 'ANATOMY', 'PATHOLOGY', 'CLINICAL', or 'GENERAL')
        """
        query_lower = query.lower()

        type_scores = {}
        for qtype, patterns in self.type_patterns.items():
            score = 0

            # Check keywords
            for keyword in patterns['keywords']:
                if keyword in query_lower:
                    score += 1

            # Check phrases (weighted higher)
            for phrase in patterns['phrases']:
                if phrase in query_lower:
                    score += 2

            type_scores[qtype] = score

        if not any(type_scores.values()):
            return 'GENERAL'

        return max(type_scores, key=type_scores.get)

    def _calculate_quality_score(
        self,
        meta: Dict[str, Any],
        query_type: str
    ) -> float:
        """
        Calculate weighted quality score from metadata.

        Args:
            meta: Metadata dict with quality scores
            query_type: Detected query type

        Returns:
            Quality score between 0 and 1
        """
        # Extract scores with defaults
        authority = float(meta.get('authority_score', self.default_score))
        readability = float(meta.get('readability_score', self.default_score))
        coherence = float(meta.get('coherence_score', self.default_score))
        completeness = float(meta.get('completeness_score', self.default_score))

        # Calculate type match score
        chunk_type = meta.get('chunk_type', 'GENERAL')
        if isinstance(chunk_type, str):
            chunk_type = chunk_type.upper()

        if query_type == 'GENERAL' or chunk_type == query_type:
            type_match = 1.0
        elif chunk_type == 'GENERAL':
            type_match = 0.7  # General chunks are acceptable
        else:
            type_match = 0.4  # Mismatched type

        # Weighted combination
        score = (
            self.weights['authority'] * authority +
            self.weights['readability'] * readability +
            self.weights['coherence'] * coherence +
            self.weights['completeness'] * completeness +
            self.weights['type'] * type_match
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))

    def adjust_weights(self, **kwargs) -> None:
        """Adjust scoring weights dynamically."""
        for key, value in kwargs.items():
            if key in self.weights:
                self.weights[key] = value

        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}


# =============================================================================
# Combined Pipeline Reranker
# =============================================================================

class PipelineReranker:
    """
    Combined reranker that applies multiple stages.

    Typical pipeline:
    1. Quality reranking (filter low-quality)
    2. Cross-encoder (relevance boost)
    3. Diversity reranking (MMR)
    """

    def __init__(
        self,
        quality_reranker: QualityReranker = None,
        diversity_reranker: DiversityReranker = None,
        quality_weight: float = 0.3,
        diversity_weight: float = 0.3,
        relevance_weight: float = 0.4
    ):
        self.quality_reranker = quality_reranker
        self.diversity_reranker = diversity_reranker
        self.quality_weight = quality_weight
        self.diversity_weight = diversity_weight
        self.relevance_weight = relevance_weight

    async def rerank(
        self,
        query: str,
        documents: List[str],
        embeddings: List[np.ndarray] = None,
        query_embedding: np.ndarray = None,
        metadata: List[Dict[str, Any]] = None,
        initial_scores: List[float] = None,
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        Apply full reranking pipeline.

        Returns:
            List of (original_index, combined_score) tuples
        """
        n = len(documents)
        if n == 0:
            return []

        # Initialize with relevance scores
        if initial_scores:
            combined_scores = np.array(initial_scores, dtype=np.float32)
        else:
            combined_scores = np.ones(n, dtype=np.float32) * 0.5

        # Stage 1: Quality scores
        if self.quality_reranker and metadata:
            quality_scores = await self.quality_reranker.score(query, documents, metadata)
            combined_scores = (
                self.relevance_weight * combined_scores +
                self.quality_weight * np.array(quality_scores)
            )

        # Stage 2: Diversity scores (MMR)
        if self.diversity_reranker and embeddings is not None:
            diversity_scores = await self.diversity_reranker.score(
                query, documents, embeddings, query_embedding,
                initial_scores=combined_scores.tolist()
            )
            combined_scores = (
                (1 - self.diversity_weight) * combined_scores +
                self.diversity_weight * np.array(diversity_scores)
            )

        # Normalize to [0, 1]
        if combined_scores.max() > 0:
            combined_scores = combined_scores / combined_scores.max()

        # Sort and return
        indexed_scores = list(enumerate(combined_scores.tolist()))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores


def create_pipeline_reranker(
    enable_quality: bool = True,
    enable_diversity: bool = True,
    **kwargs
) -> PipelineReranker:
    """
    Factory function for creating pipeline reranker.

    Args:
        enable_quality: Whether to include quality reranking
        enable_diversity: Whether to include diversity reranking
        **kwargs: Additional config options

    Returns:
        Configured PipelineReranker
    """
    quality = QualityReranker(**kwargs.get('quality_config', {})) if enable_quality else None
    diversity = DiversityReranker(**kwargs.get('diversity_config', {})) if enable_diversity else None

    return PipelineReranker(
        quality_reranker=quality,
        diversity_reranker=diversity,
        quality_weight=kwargs.get('quality_weight', 0.3),
        diversity_weight=kwargs.get('diversity_weight', 0.3),
        relevance_weight=kwargs.get('relevance_weight', 0.4)
    )


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
