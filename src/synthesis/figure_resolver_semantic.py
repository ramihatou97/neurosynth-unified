"""
NeuroSynth - Semantic Figure Resolver
=====================================

Vector-based figure matching replacing naive word overlap.
Uses caption embeddings + keyword matching + authority tier boost.

Fixes Issue #2: Figure resolution uses word overlap, ignores caption_embedding.
"""

import numpy as np
import re
import logging
from typing import List, Dict, Optional, Protocol, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class EmbeddingService(Protocol):
    """Protocol for embedding service interface."""
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single string."""
        ...


@dataclass
class FigureMatch:
    """Result of matching a figure to a section."""
    image: Any  # ExtractedImage or similar
    score: float
    match_reason: str  # "semantic", "keyword", or "hybrid"
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    tier_boost: float = 0.0


class SemanticFigureResolver:
    """
    Resolves figure requests to actual images using semantic similarity.

    Replaces naive word overlap with:
    1. Vector similarity between section text and caption embedding
    2. Keyword overlap as secondary signal
    3. Authority tier boost for Tier 1 sources (Rhoton/Lawton)

    Usage:
        resolver = SemanticFigureResolver(embedding_service)
        matches = await resolver.resolve_figures(section_text, available_images)
        for match in matches:
            print(f"Image {match.image.id}: {match.score:.2f} ({match.match_reason})")
    """

    # Weight distribution for scoring components
    WEIGHT_SEMANTIC = 0.60   # Primary signal from vector similarity
    WEIGHT_KEYWORD = 0.25    # Secondary signal from keyword matching
    WEIGHT_TIER = 0.15       # Authority boost for Tier 1 sources

    # Tier score mapping
    TIER_SCORES = {
        1: 1.0,   # Rhoton, Lawton, Spetzler
        2: 0.7,   # Major references
        3: 0.4,   # Standard references
        4: 0.1,   # General sources
    }

    # Minimum score to consider a match valid
    MIN_SCORE_THRESHOLD = 0.45

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the semantic figure resolver.

        Args:
            embedding_service: Service for generating embeddings
            weights: Custom weight distribution (semantic, keyword, tier)
        """
        self.embedder = embedding_service

        if weights:
            self.WEIGHT_SEMANTIC = weights.get("semantic", self.WEIGHT_SEMANTIC)
            self.WEIGHT_KEYWORD = weights.get("keyword", self.WEIGHT_KEYWORD)
            self.WEIGHT_TIER = weights.get("tier", self.WEIGHT_TIER)

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec_a or not vec_b:
            return 0.0

        a = np.array(vec_a)
        b = np.array(vec_b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def _calculate_keyword_overlap(self, text_a: str, text_b: str) -> float:
        """
        Calculate Jaccard similarity on word sets.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if not text_a or not text_b:
            return 0.0

        # Extract words, filter stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were'}

        set_a = set(w.lower() for w in re.findall(r'\w+', text_a) if w.lower() not in stopwords and len(w) > 2)
        set_b = set(w.lower() for w in re.findall(r'\w+', text_b) if w.lower() not in stopwords and len(w) > 2)

        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))

        return intersection / union if union > 0 else 0.0

    def _get_tier_score(self, image: Any) -> float:
        """Get authority tier score for an image."""
        # Handle both dict and object interfaces
        if isinstance(image, dict):
            tier = image.get('source_tier', 4)
        else:
            tier = getattr(image, 'source_tier', 4)

        return self.TIER_SCORES.get(tier, 0.1)

    def _get_caption(self, image: Any) -> str:
        """Extract caption from image object."""
        if isinstance(image, dict):
            return image.get('caption', '') or image.get('vlm_caption', '')
        else:
            return getattr(image, 'caption', '') or getattr(image, 'vlm_caption', '')

    def _get_caption_embedding(self, image: Any) -> Optional[List[float]]:
        """Extract caption embedding from image object."""
        if isinstance(image, dict):
            return image.get('caption_embedding')
        else:
            return getattr(image, 'caption_embedding', None)

    async def _calculate_match_score(
        self,
        section_text: str,
        section_embedding: Optional[List[float]],
        image: Any
    ) -> FigureMatch:
        """
        Calculate comprehensive match score for a single image.

        Args:
            section_text: Text content of the section
            section_embedding: Pre-computed embedding of section text
            image: Image object/dict

        Returns:
            FigureMatch with detailed scoring breakdown
        """
        caption = self._get_caption(image)
        caption_embedding = self._get_caption_embedding(image)

        # 1. Semantic Score (Primary Signal)
        semantic_score = 0.0
        if section_embedding and caption_embedding:
            semantic_score = self._cosine_similarity(section_embedding, caption_embedding)
        elif self.embedder and section_embedding and caption:
            # Fallback: embed caption on the fly if needed
            try:
                caption_emb = await self.embedder.embed_text(caption[:500])
                semantic_score = self._cosine_similarity(section_embedding, caption_emb)
            except Exception as e:
                logger.warning(f"Failed to embed caption: {e}")

        # 2. Keyword Score (Secondary Signal)
        keyword_score = self._calculate_keyword_overlap(section_text, caption)

        # 3. Tier Boost (Authority Signal)
        tier_score = self._get_tier_score(image)

        # 4. Weighted Combination
        final_score = (
            (semantic_score * self.WEIGHT_SEMANTIC) +
            (keyword_score * self.WEIGHT_KEYWORD) +
            (tier_score * self.WEIGHT_TIER)
        )

        # Determine primary match reason
        if semantic_score > keyword_score and semantic_score > 0.3:
            match_reason = "semantic"
        elif keyword_score > semantic_score and keyword_score > 0.2:
            match_reason = "keyword"
        else:
            match_reason = "hybrid"

        return FigureMatch(
            image=image,
            score=final_score,
            match_reason=match_reason,
            semantic_score=semantic_score,
            keyword_score=keyword_score,
            tier_boost=tier_score * self.WEIGHT_TIER
        )

    async def resolve_figures(
        self,
        section_text: str,
        available_images: List[Any],
        top_k: int = 2,
        min_score: Optional[float] = None
    ) -> List[FigureMatch]:
        """
        Resolve figures for a section.

        Args:
            section_text: Text content of the section
            available_images: List of available images (ExtractedImage or dicts)
            top_k: Maximum number of figures to return
            min_score: Minimum score threshold (defaults to MIN_SCORE_THRESHOLD)

        Returns:
            List of FigureMatch objects, sorted by score descending
        """
        if not available_images:
            return []

        threshold = min_score if min_score is not None else self.MIN_SCORE_THRESHOLD

        # Pre-compute section embedding
        section_embedding = None
        if self.embedder:
            try:
                # Truncate for efficiency
                section_embedding = await self.embedder.embed_text(section_text[:1000])
            except Exception as e:
                logger.warning(f"Failed to embed section text: {e}")

        # Score all images
        matches = []
        for image in available_images:
            match = await self._calculate_match_score(
                section_text=section_text,
                section_embedding=section_embedding,
                image=image
            )

            if match.score >= threshold:
                matches.append(match)

        # Sort by score descending
        matches.sort(key=lambda x: x.score, reverse=True)

        # Log results
        if matches:
            logger.info(f"Found {len(matches)} figure matches above threshold {threshold}")
            for m in matches[:top_k]:
                logger.debug(
                    f"  Match: score={m.score:.3f} "
                    f"(sem={m.semantic_score:.2f}, kw={m.keyword_score:.2f}, tier={m.tier_boost:.2f})"
                )

        return matches[:top_k]

    async def resolve_figures_batch(
        self,
        sections: List[Dict[str, Any]],
        available_images: List[Any],
        top_k_per_section: int = 2
    ) -> Dict[str, List[FigureMatch]]:
        """
        Resolve figures for multiple sections.

        Args:
            sections: List of section dicts with 'name' and 'content' keys
            available_images: List of all available images
            top_k_per_section: Max figures per section

        Returns:
            Dict mapping section names to their matched figures
        """
        results = {}
        used_images = set()

        for section in sections:
            name = section.get('name', section.get('title', 'Unknown'))
            content = section.get('content', '')

            # Filter out already-used images
            remaining_images = [
                img for img in available_images
                if self._get_image_id(img) not in used_images
            ]

            matches = await self.resolve_figures(
                section_text=content,
                available_images=remaining_images,
                top_k=top_k_per_section
            )

            results[name] = matches

            # Track used images to avoid duplicates
            for match in matches:
                used_images.add(self._get_image_id(match.image))

        return results

    def _get_image_id(self, image: Any) -> str:
        """Get unique identifier for an image."""
        if isinstance(image, dict):
            return image.get('id', image.get('image_id', str(id(image))))
        else:
            return getattr(image, 'id', getattr(image, 'image_id', str(id(image))))


class LegacyFigureResolver:
    """
    Fallback figure resolver using only keyword matching.

    Use when embeddings are unavailable.
    """

    def resolve_figures(
        self,
        section_text: str,
        available_images: List[Any],
        top_k: int = 2
    ) -> List[FigureMatch]:
        """Resolve using keyword overlap only."""
        matches = []

        for image in available_images:
            if isinstance(image, dict):
                caption = image.get('caption', '')
            else:
                caption = getattr(image, 'caption', '')

            # Simple word overlap
            section_words = set(re.findall(r'\w+', section_text.lower()))
            caption_words = set(re.findall(r'\w+', caption.lower()))

            if not section_words or not caption_words:
                continue

            overlap = len(section_words & caption_words) / len(section_words | caption_words)

            if overlap > 0.1:
                matches.append(FigureMatch(
                    image=image,
                    score=overlap,
                    match_reason="keyword",
                    keyword_score=overlap
                ))

        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:top_k]
