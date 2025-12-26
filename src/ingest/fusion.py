"""
NeuroSynth v2.0 - Image-Chunk Fusion
=====================================

Link images to chunks based on references and context.

Linking strategies:
1. Explicit reference matching ("Figure 3" in text) - Deterministic
2. UMLS CUI Jaccard overlap - Semantic ontology matching
3. Embedding cosine similarity - Neural semantic matching

Upgrade in Phase 1 Perfected:
- TriPassLinker with fusion scoring: (semantic × 0.55) + (cui × 0.45) ≥ 0.55
- Returns LinkResult objects for Phase 2 export
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set

import numpy as np

from src.shared.models import SemanticChunk, ExtractedImage, LinkResult, LinkMatchType

logger = logging.getLogger(__name__)


class ImageChunkLinker:
    """
    Link images to chunks bidirectionally.
    
    Key insight: Images and text must be connected for:
    - Multimodal retrieval
    - Proper citation
    - Context preservation
    """
    
    def __init__(self, min_overlap_score: int = 2):
        """
        Initialize the linker.
        
        Args:
            min_overlap_score: Minimum entity overlap score for proximity linking
        """
        self.min_overlap_score = min_overlap_score
    
    def link(
        self,
        chunks: List[SemanticChunk],
        images: List[ExtractedImage]
    ) -> Tuple[List[SemanticChunk], List[ExtractedImage]]:
        """
        Link images to chunks bidirectionally.
        
        Modifies objects in place and returns them.
        
        Args:
            chunks: List of semantic chunks
            images: List of extracted images
            
        Returns:
            Tuple of (chunks, images) with links populated
        """
        # Build figure reference map
        figure_map = self._build_figure_map(images)
        
        # Pass 1: Link by explicit reference
        self._link_by_reference(chunks, images, figure_map)
        
        # Pass 2: Link unlinked images by proximity + entity overlap
        self._link_by_proximity(chunks, images)
        
        return chunks, images
    
    def _build_figure_map(
        self,
        images: List[ExtractedImage]
    ) -> Dict[str, ExtractedImage]:
        """
        Build map of normalized figure IDs to images.
        """
        figure_map = {}
        
        for image in images:
            if image.figure_id:
                normalized = self._normalize_figure_ref(image.figure_id)
                figure_map[normalized] = image
                
                # Also map original
                figure_map[image.figure_id.lower()] = image
        
        return figure_map
    
    def _normalize_figure_ref(self, ref: str) -> str:
        """
        Normalize figure reference for matching.
        
        Examples:
        - "Fig. 3A" -> "figure 3a"
        - "Figure 3.2" -> "figure 3.2"
        - "FIGURE 3" -> "figure 3"
        """
        ref = ref.lower().strip()
        ref = re.sub(r"^fig\.?\s*", "figure ", ref)
        ref = re.sub(r"\s+", " ", ref)
        return ref
    
    def _link_by_reference(
        self,
        chunks: List[SemanticChunk],
        images: List[ExtractedImage],
        figure_map: Dict[str, ExtractedImage]
    ):
        """
        Link chunks to images by explicit figure references.
        """
        for chunk in chunks:
            for ref in chunk.figure_refs:
                normalized = self._normalize_figure_ref(ref)
                
                image = figure_map.get(normalized)
                if image:
                    # Add bidirectional links
                    if image.id not in chunk.image_ids:
                        chunk.image_ids.append(image.id)
                    if chunk.id not in image.chunk_ids:
                        image.chunk_ids.append(chunk.id)
    
    def _link_by_proximity(
        self,
        chunks: List[SemanticChunk],
        images: List[ExtractedImage]
    ):
        """
        Link unlinked images to nearby chunks with entity overlap.
        """
        # Find unlinked images
        unlinked = [img for img in images if not img.chunk_ids]
        
        for image in unlinked:
            # Find chunks on same page
            page_chunks = [
                c for c in chunks
                if c.page_start <= image.page_number <= c.page_end
            ]
            
            if not page_chunks:
                continue
            
            # Score and find best match
            best_chunk = self._find_best_match(image, page_chunks)
            
            if best_chunk:
                # Add bidirectional links
                if image.id not in best_chunk.image_ids:
                    best_chunk.image_ids.append(image.id)
                if best_chunk.id not in image.chunk_ids:
                    image.chunk_ids.append(best_chunk.id)
    
    def _find_best_match(
        self,
        image: ExtractedImage,
        chunks: List[SemanticChunk]
    ) -> Optional[SemanticChunk]:
        """
        Find best matching chunk for an image based on entity overlap.
        """
        if not chunks:
            return None
        
        # Build image context for matching
        image_context = f"{image.caption or ''} {image.surrounding_text}".lower()
        
        best_score = 0
        best_chunk = None
        
        for chunk in chunks:
            score = 0
            
            # Entity name overlap (highest weight)
            for entity in chunk.entity_names:
                if entity.lower() in image_context:
                    score += 3
            
            # Keyword overlap
            for keyword in chunk.keywords:
                if keyword.lower() in image_context:
                    score += 1
            
            # Type matching bonus
            image_type_str = str(image.image_type.value).lower()
            chunk_type_str = chunk.chunk_type.value.lower()
            
            if "surgical" in image_type_str and chunk_type_str == "procedure":
                score += 2
            if "anatomy" in image_type_str and chunk_type_str == "anatomy":
                score += 2
            if "imaging" in image_type_str and chunk_type_str in ["clinical", "pathology"]:
                score += 1
            
            if score > best_score:
                best_score = score
                best_chunk = chunk
        
        # Return best if above threshold, otherwise first chunk
        if best_score >= self.min_overlap_score:
            return best_chunk
        elif chunks:
            return chunks[0]  # Default to first chunk on same page
        
        return None
    
    def get_linked_images(
        self,
        chunk: SemanticChunk,
        images: List[ExtractedImage]
    ) -> List[ExtractedImage]:
        """
        Get all images linked to a chunk.
        
        Convenience method for retrieval.
        """
        image_map = {img.id: img for img in images}
        return [image_map[img_id] for img_id in chunk.image_ids if img_id in image_map]
    
    def get_linked_chunks(
        self,
        image: ExtractedImage,
        chunks: List[SemanticChunk]
    ) -> List[SemanticChunk]:
        """
        Get all chunks linked to an image.
        
        Convenience method for retrieval.
        """
        chunk_map = {c.id: c for c in chunks}
        return [chunk_map[c_id] for c_id in image.chunk_ids if c_id in chunk_map]


class EmbeddingFuser:
    """
    Create fused text+image embeddings for multimodal retrieval.
    
    Strategy:
    1. For chunks with linked images, combine text and image embeddings
    2. Weight combination based on image relevance
    3. Normalize final embedding
    """
    
    def __init__(self, image_weight: float = 0.3):
        """
        Initialize the fuser.
        
        Args:
            image_weight: Weight for image embedding in fusion (0-1)
        """
        self.image_weight = image_weight
    
    def fuse_embeddings(
        self,
        chunks: List[SemanticChunk],
        images: List[ExtractedImage]
    ) -> List[SemanticChunk]:
        """
        Create fused embeddings for all chunks.
        
        Modifies chunks in place.
        """
        import numpy as np
        
        image_map = {img.id: img for img in images}
        
        for chunk in chunks:
            # Default: fused = text embedding
            if chunk.text_embedding is None:
                continue
            
            if not chunk.image_ids:
                chunk.fused_embedding = chunk.text_embedding.copy()
                continue
            
            # Get linked image embeddings
            image_embeddings = []
            for img_id in chunk.image_ids:
                img = image_map.get(img_id)
                if img and img.embedding is not None:
                    image_embeddings.append(img.embedding)
            
            if not image_embeddings:
                chunk.fused_embedding = chunk.text_embedding.copy()
                continue
            
            # Average image embeddings
            avg_image = np.mean(image_embeddings, axis=0)
            
            # Handle dimension mismatch
            text_dim = len(chunk.text_embedding)
            image_dim = len(avg_image)
            
            if image_dim != text_dim:
                # Project to same dimension (simple padding/truncation)
                if image_dim < text_dim:
                    avg_image = np.pad(avg_image, (0, text_dim - image_dim))
                else:
                    avg_image = avg_image[:text_dim]
            
            # Normalize both
            text_norm = chunk.text_embedding / (np.linalg.norm(chunk.text_embedding) + 1e-8)
            image_norm = avg_image / (np.linalg.norm(avg_image) + 1e-8)
            
            # Weighted combination
            fused = (1 - self.image_weight) * text_norm + self.image_weight * image_norm
            
            # Normalize result
            chunk.fused_embedding = fused / (np.linalg.norm(fused) + 1e-8)

        return chunks


class TriPassLinker:
    """
    Three-pass image-chunk linker with fusion scoring.

    Pass 1: Deterministic regex matching (early exit)
        - Matches "Figure X" references in chunk text to image figure_id
        - Returns with strength=1.0 if match found

    Pass 2: UMLS CUI Jaccard overlap
        - Compares CUI sets between chunk and image caption
        - Provides ontology-based semantic matching

    Pass 3: Embedding cosine similarity
        - Compares chunk.text_embedding with image.caption_embedding
        - Provides neural semantic matching

    Fusion scoring:
        fusion = (semantic × 0.55) + (cui × 0.45)
        Link accepted if fusion ≥ 0.55 OR cui_only ≥ 0.25

    Returns LinkResult objects for Phase 2 export.
    """

    # Regex pattern for figure references (supports decimal notation like "Fig. 3.1")
    FIGURE_PATTERN = re.compile(
        r"(?i)(?:figure|fig\.?)\s*(\d+(?:\.\d+)?[a-z]?(?:\s*[-–,]\s*\d+(?:\.\d+)?[a-z]?)*)",
        re.IGNORECASE
    )

    def __init__(
        self,
        semantic_threshold: float = 0.55,
        cui_threshold: float = 0.25,
        semantic_weight: float = 0.55,
        cui_weight: float = 0.45,
        page_buffer: int = 1,
    ):
        """
        Initialize the tri-pass linker.

        Args:
            semantic_threshold: Minimum fusion score for acceptance
            cui_threshold: Minimum CUI-only score for fallback acceptance
            semantic_weight: Weight for semantic similarity in fusion
            cui_weight: Weight for CUI overlap in fusion
            page_buffer: Page proximity range (+/- pages)
        """
        self.semantic_threshold = semantic_threshold
        self.cui_threshold = cui_threshold
        self.semantic_weight = semantic_weight
        self.cui_weight = cui_weight
        self.page_buffer = page_buffer

        # Validation
        if abs((semantic_weight + cui_weight) - 1.0) > 0.001:
            logger.warning(
                f"Weights don't sum to 1.0: {semantic_weight} + {cui_weight} = "
                f"{semantic_weight + cui_weight}"
            )

    def link(
        self,
        chunks: List[SemanticChunk],
        images: List[ExtractedImage]
    ) -> Tuple[List[SemanticChunk], List[ExtractedImage], List[LinkResult]]:
        """
        Link images to chunks using tri-pass algorithm.

        Modifies chunks and images in place, returns LinkResult list.

        Args:
            chunks: List of semantic chunks
            images: List of extracted images

        Returns:
            Tuple of (chunks, images, links) with bidirectional references
        """
        links: List[LinkResult] = []

        # Build figure reference map for deterministic matching
        figure_map = self._build_figure_map(images)

        for image in images:
            # Skip decorative images or those without caption embedding
            if image.is_decorative:
                continue

            # Get candidate chunks within page buffer
            candidates = self._get_candidate_chunks(chunks, image)

            if not candidates:
                continue

            for chunk in candidates:
                result = self._calculate_link(chunk, image, figure_map)

                if result:
                    links.append(result)

                    # Add bidirectional references
                    if image.id not in chunk.image_ids:
                        chunk.image_ids.append(image.id)
                    if chunk.id not in image.chunk_ids:
                        image.chunk_ids.append(chunk.id)

                    # Store link strength on image for retrieval boosting
                    if hasattr(image, 'link_strengths'):
                        image.link_strengths[chunk.id] = result.strength

        logger.info(
            f"TriPassLinker: Created {len(links)} links "
            f"({sum(1 for l in links if l.match_type == LinkMatchType.DETERMINISTIC.value)} deterministic, "
            f"{sum(1 for l in links if 'fusion' in l.match_type)} fusion, "
            f"{sum(1 for l in links if l.match_type == LinkMatchType.CUI_ONLY.value)} CUI-only)"
        )

        return chunks, images, links

    def _build_figure_map(
        self,
        images: List[ExtractedImage]
    ) -> Dict[str, ExtractedImage]:
        """Build map of normalized figure IDs to images."""
        figure_map = {}

        for image in images:
            if image.figure_id:
                # Normalize figure ID
                normalized = self._normalize_figure_ref(image.figure_id)
                figure_map[normalized] = image

                # Also map original lowercase
                figure_map[image.figure_id.lower().strip()] = image

        return figure_map

    def _normalize_figure_ref(self, ref: str) -> str:
        """Normalize figure reference for matching."""
        ref = ref.lower().strip()
        ref = re.sub(r"^fig\.?\s*", "figure ", ref)
        # Handle underscore format: "fig_1" → "figure 1"
        ref = ref.replace("figure _", "figure ")
        ref = re.sub(r"\s+", " ", ref)
        return ref

    def _normalize_chapter_figure(self, ref: str) -> str:
        """
        Convert chapter.figure notation to sequential figure number.

        Examples:
            '3.1' → '1'
            '3.2a' → '2a'
            '5' → '5' (no change)
        """
        if '.' in ref:
            parts = ref.split('.')
            if len(parts) == 2:
                # Return just the figure number part (e.g., "1" from "3.1" or "2a" from "3.2a")
                return parts[1]
        return ref

    def _get_candidate_chunks(
        self,
        chunks: List[SemanticChunk],
        image: ExtractedImage
    ) -> List[SemanticChunk]:
        """Get chunks within page buffer of image."""
        return [
            c for c in chunks
            if abs(c.page_start - image.page_number) <= self.page_buffer
            or abs(c.page_end - image.page_number) <= self.page_buffer
        ]

    def _calculate_link(
        self,
        chunk: SemanticChunk,
        image: ExtractedImage,
        figure_map: Dict[str, ExtractedImage]
    ) -> Optional[LinkResult]:
        """
        Calculate link between chunk and image using tri-pass algorithm.

        Returns LinkResult if link should be created, None otherwise.
        """
        # PASS 1: Deterministic regex match
        if self._check_deterministic_match(chunk, image, figure_map):
            return LinkResult(
                chunk_id=chunk.id,
                image_id=image.id,
                strength=1.0,
                match_type=LinkMatchType.DETERMINISTIC.value,
                details={"pass": 1, "method": "figure_reference"}
            )

        # PASS 2: UMLS CUI overlap (requires CUIs on both)
        cui_score = 0.0
        if chunk.cuis and image.cuis:
            cui_score = self._calculate_cui_overlap(
                set(chunk.cuis),
                set(image.cuis)
            )

        # PASS 3: Semantic similarity (requires embeddings)
        sem_score = 0.0
        if (
            chunk.text_embedding is not None
            and image.caption_embedding is not None
        ):
            sem_score = self._calculate_cosine_similarity(
                chunk.text_embedding,
                image.caption_embedding
            )

        # Fusion scoring
        if sem_score > 0 or cui_score > 0:
            fusion = (sem_score * self.semantic_weight) + (cui_score * self.cui_weight)

            if fusion >= self.semantic_threshold:
                # Determine primary match type
                match_type = (
                    LinkMatchType.FUSION_SEMANTIC.value
                    if sem_score > cui_score
                    else LinkMatchType.FUSION_CUI.value
                )

                return LinkResult(
                    chunk_id=chunk.id,
                    image_id=image.id,
                    strength=fusion,
                    match_type=match_type,
                    details={
                        "pass": 3,
                        "semantic_score": round(sem_score, 4),
                        "cui_score": round(cui_score, 4),
                        "fusion": round(fusion, 4),
                    }
                )

            # Fallback: CUI-only threshold
            if cui_score >= self.cui_threshold:
                return LinkResult(
                    chunk_id=chunk.id,
                    image_id=image.id,
                    strength=cui_score * 0.8,  # Reduced confidence
                    match_type=LinkMatchType.CUI_ONLY.value,
                    details={
                        "pass": 2,
                        "cui_score": round(cui_score, 4),
                        "reason": "cui_threshold_fallback"
                    }
                )

        return None

    def _check_deterministic_match(
        self,
        chunk: SemanticChunk,
        image: ExtractedImage,
        figure_map: Dict[str, ExtractedImage]
    ) -> bool:
        """Check if chunk explicitly references this image."""
        # Check figure_refs on chunk
        if hasattr(chunk, 'figure_refs') and chunk.figure_refs:
            for ref in chunk.figure_refs:
                normalized = self._normalize_figure_ref(ref)
                matched_image = figure_map.get(normalized)
                if matched_image and matched_image.id == image.id:
                    return True

        # Also check chunk content for figure references
        if image.figure_id:
            normalized_fig_id = self._normalize_figure_ref(image.figure_id)
            chunk_text = chunk.content.lower()

            # Check for "Figure X" or "Fig. X" in text
            matches = self.FIGURE_PATTERN.findall(chunk_text)
            for match in matches:
                # Try direct match first
                ref = f"figure {match}"
                if ref == normalized_fig_id:
                    return True

                # Try matching raw figure number
                if match == image.figure_id.replace("Figure ", "").replace("Fig. ", ""):
                    return True

                # Try chapter-relative normalization: "3.1" → "1" to match "fig_1" → "figure 1"
                normalized_match = self._normalize_chapter_figure(match)
                ref_normalized = f"figure {normalized_match}"
                if ref_normalized == normalized_fig_id:
                    return True

        return False

    def _calculate_cui_overlap(
        self,
        cuis1: Set[str],
        cuis2: Set[str]
    ) -> float:
        """Calculate Jaccard similarity between CUI sets."""
        if not cuis1 or not cuis2:
            return 0.0

        intersection = len(cuis1 & cuis2)
        union = len(cuis1 | cuis2)

        return intersection / union if union > 0 else 0.0

    def _calculate_cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        # Handle dimension mismatch
        if len(embedding1) != len(embedding2):
            min_dim = min(len(embedding1), len(embedding2))
            embedding1 = embedding1[:min_dim]
            embedding2 = embedding2[:min_dim]

        # Normalize
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Clamp to [0, 1]
        return float(max(0.0, min(1.0, similarity)))

    def get_links_for_chunk(
        self,
        chunk_id: str,
        links: List[LinkResult]
    ) -> List[LinkResult]:
        """Get all links for a specific chunk."""
        return [l for l in links if l.chunk_id == chunk_id]

    def get_links_for_image(
        self,
        image_id: str,
        links: List[LinkResult]
    ) -> List[LinkResult]:
        """Get all links for a specific image."""
        return [l for l in links if l.image_id == image_id]

    def get_stats(self, links: List[LinkResult]) -> Dict[str, int]:
        """Get linking statistics."""
        stats = {
            "total_links": len(links),
            "deterministic": 0,
            "fusion_semantic": 0,
            "fusion_cui": 0,
            "cui_only": 0,
            "avg_strength": 0.0,
        }

        if not links:
            return stats

        for link in links:
            if link.match_type == LinkMatchType.DETERMINISTIC.value:
                stats["deterministic"] += 1
            elif link.match_type == LinkMatchType.FUSION_SEMANTIC.value:
                stats["fusion_semantic"] += 1
            elif link.match_type == LinkMatchType.FUSION_CUI.value:
                stats["fusion_cui"] += 1
            elif link.match_type == LinkMatchType.CUI_ONLY.value:
                stats["cui_only"] += 1

        stats["avg_strength"] = sum(l.strength for l in links) / len(links)

        return stats
