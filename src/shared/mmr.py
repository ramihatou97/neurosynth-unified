"""
NeuroSynth - Maximal Marginal Relevance (MMR) Deduplication
===========================================================

Implements MMR for diverse chunk selection in retrieval.
Prevents near-duplicate chunks from dominating synthesis context.

Reference: Carbonell & Goldstein, 1998 - "The use of MMR, diversity-based
reranking for reordering documents and producing summaries"
"""

import numpy as np
from typing import List, Optional, Tuple, Any, Dict
import logging

logger = logging.getLogger(__name__)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity in range [-1, 1]
    """
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm_v1 * norm_v2))


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: Matrix of shape (n_samples, embedding_dim)

    Returns:
        Similarity matrix of shape (n_samples, n_samples)
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Compute similarity matrix
    return np.dot(normalized, normalized.T)


def mmr_sort(
    query_embedding: List[float],
    candidate_embeddings: List[List[float]],
    candidate_indices: List[int],
    top_k: int,
    lambda_mult: float = 0.7
) -> List[int]:
    """
    Select top_k indices using Maximal Marginal Relevance.

    MMR balances relevance to query with diversity among selected items.

    MMR = λ * Sim(doc, query) - (1-λ) * max(Sim(doc, selected_docs))

    Args:
        query_embedding: Vector of the user's search query
        candidate_embeddings: Vectors for all retrieved chunks
        candidate_indices: Original indices to map back to objects
        top_k: Number of results to return
        lambda_mult: Diversity slider (0.0 to 1.0)
            - 1.0 = Pure Relevance (standard search, no diversity)
            - 0.7 = Medical default (mostly relevant, slight diversity)
            - 0.5 = Balanced
            - 0.0 = Pure Diversity (ignore relevance)

    Returns:
        List of selected original indices from candidate_indices

    Example:
        >>> query_emb = [0.1, 0.2, 0.3]
        >>> candidates = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.35], [0.5, 0.6, 0.7]]
        >>> indices = [0, 1, 2]
        >>> mmr_sort(query_emb, candidates, indices, top_k=2, lambda_mult=0.7)
        [0, 2]  # First is most relevant, second is diverse
    """
    if len(candidate_embeddings) == 0:
        return []

    if len(candidate_embeddings) <= top_k:
        return candidate_indices

    # Convert to numpy arrays
    query_v = np.array(query_embedding)
    cand_vs = np.array(candidate_embeddings)

    # Pre-calculate similarity to query for all candidates
    sims_to_query = np.array([
        cosine_similarity(c, query_v) for c in cand_vs
    ])

    # Initialize tracking
    selected_indices = []  # Indices into candidate arrays
    remaining_indices = list(range(len(candidate_embeddings)))

    # Iteratively select best candidate
    while len(selected_indices) < top_k and len(remaining_indices) > 0:
        best_mmr_score = -float('inf')
        best_idx_in_remaining = -1

        for i, idx in enumerate(remaining_indices):
            # Relevance component
            relevance = sims_to_query[idx]

            # Diversity component (penalty for similarity to already selected)
            if not selected_indices:
                diversity_penalty = 0.0
            else:
                # Find max similarity to any already selected item
                similarities_to_selected = [
                    cosine_similarity(cand_vs[idx], cand_vs[sel_idx])
                    for sel_idx in selected_indices
                ]
                diversity_penalty = max(similarities_to_selected)

            # MMR Formula
            mmr_score = (lambda_mult * relevance) - ((1 - lambda_mult) * diversity_penalty)

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx_in_remaining = i

        # Move winner from remaining to selected
        original_idx = remaining_indices[best_idx_in_remaining]
        selected_indices.append(original_idx)
        remaining_indices.pop(best_idx_in_remaining)

    # Map back to original indices
    result = [candidate_indices[i] for i in selected_indices]

    logger.debug(f"MMR selected {len(result)} items with λ={lambda_mult}")
    return result


def mmr_rerank(
    query_embedding: List[float],
    documents: List[Any],
    embedding_key: str = "embedding",
    top_k: int = 10,
    lambda_mult: float = 0.7
) -> List[Any]:
    """
    Convenience function to rerank a list of document dicts/objects using MMR.

    Args:
        query_embedding: Query vector
        documents: List of document dicts or objects, each containing an embedding
        embedding_key: Key/attribute to access embedding in each document
        top_k: Number of documents to return
        lambda_mult: Diversity parameter

    Returns:
        Reranked list of documents
    """
    if not documents:
        return []

    # Extract embeddings
    embeddings = []
    valid_docs = []
    valid_indices = []

    for i, doc in enumerate(documents):
        # Handle both dict and object interfaces
        if isinstance(doc, dict):
            emb = doc.get(embedding_key)
        else:
            emb = getattr(doc, embedding_key, None)

        if emb is not None:
            embeddings.append(emb)
            valid_docs.append(doc)
            valid_indices.append(i)

    if not embeddings:
        logger.warning("No documents with embeddings found for MMR")
        return documents[:top_k]

    # Run MMR
    selected_indices = mmr_sort(
        query_embedding=query_embedding,
        candidate_embeddings=embeddings,
        candidate_indices=list(range(len(valid_docs))),
        top_k=top_k,
        lambda_mult=lambda_mult
    )

    # Return reranked documents
    return [valid_docs[i] for i in selected_indices]


class MMRSelector:
    """
    Class-based MMR selector with caching and configuration.

    Usage:
        selector = MMRSelector(lambda_mult=0.7)
        results = selector.select(query_emb, candidate_embs, top_k=50)
    """

    def __init__(
        self,
        lambda_mult: float = 0.7,
        similarity_threshold: float = 0.95
    ):
        """
        Initialize MMR selector.

        Args:
            lambda_mult: Default diversity parameter
            similarity_threshold: Skip candidates too similar to selected
        """
        self.lambda_mult = lambda_mult
        self.similarity_threshold = similarity_threshold
        self._cache: Dict[str, Any] = {}

    def select(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int,
        lambda_mult: Optional[float] = None
    ) -> List[int]:
        """
        Select diverse candidates using MMR.

        Args:
            query_embedding: Query vector
            candidate_embeddings: Candidate vectors
            top_k: Number to select
            lambda_mult: Override default diversity parameter

        Returns:
            List of selected indices
        """
        lm = lambda_mult if lambda_mult is not None else self.lambda_mult

        return mmr_sort(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_indices=list(range(len(candidate_embeddings))),
            top_k=top_k,
            lambda_mult=lm
        )

    def select_with_scores(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int,
        lambda_mult: Optional[float] = None
    ) -> List[Tuple[int, float, float]]:
        """
        Select with relevance and diversity scores.

        Returns:
            List of (index, relevance_score, mmr_score) tuples
        """
        lm = lambda_mult if lambda_mult is not None else self.lambda_mult

        query_v = np.array(query_embedding)
        cand_vs = np.array(candidate_embeddings)

        # Calculate relevance scores
        relevance_scores = [cosine_similarity(c, query_v) for c in cand_vs]

        # Run MMR selection
        selected = mmr_sort(
            query_embedding=query_embedding,
            candidate_embeddings=candidate_embeddings,
            candidate_indices=list(range(len(candidate_embeddings))),
            top_k=top_k,
            lambda_mult=lm
        )

        # Calculate MMR scores for selected
        results = []
        selected_set = set()

        for idx in selected:
            relevance = relevance_scores[idx]

            if not selected_set:
                diversity_penalty = 0.0
            else:
                similarities = [
                    cosine_similarity(cand_vs[idx], cand_vs[s])
                    for s in selected_set
                ]
                diversity_penalty = max(similarities)

            mmr_score = (lm * relevance) - ((1 - lm) * diversity_penalty)
            results.append((idx, relevance, mmr_score))
            selected_set.add(idx)

        return results
