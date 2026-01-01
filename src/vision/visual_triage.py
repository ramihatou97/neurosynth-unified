"""
NeuroSynth Phase 1 - Visual Triage

Fast image classification to skip VLM captioning for low-value images.

Saves 60-70% VLM costs by filtering:
- Decorative images (logos, icons, borders)
- Duplicate images
- Low-information images (solid colors, gradients)

Usage:
    triage = VisualTriage()
    result = triage.evaluate(image_path)
    if result.should_process:
        caption = await vlm.caption(image_path)
"""

import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional, Set, Dict, Any, Callable
from pathlib import Path
from enum import Enum
import numpy as np

logger = logging.getLogger("neurosynth.vision.triage")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

class SkipReason(Enum):
    """Reasons for skipping VLM processing."""
    TOO_SMALL = "too_small"
    EXTREME_ASPECT = "extreme_aspect_ratio"
    LOW_ENTROPY = "low_entropy_solid"
    HIGH_ENTROPY = "high_entropy_noise"
    LOW_EDGES = "low_edge_density"
    SINGLE_COLOR = "single_dominant_color"
    LOW_VARIANCE = "low_color_variance"
    DUPLICATE = "duplicate"
    TRIAGE_ERROR = "triage_error"
    PROCESSED = "processed"
    MEDICAL_CONTENT = "medical_content"


@dataclass
class TriageConfig:
    """Visual triage configuration thresholds."""
    
    # Size thresholds (pixels)
    min_width: int = 100
    min_height: int = 100
    max_aspect_ratio: float = 10.0
    
    # Entropy thresholds (bits, 0-8 range for 8-bit images)
    min_entropy: float = 4.0      # Below = solid color / gradient
    max_entropy: float = 7.5      # Above = noise / compression artifacts
    
    # Edge density thresholds (ratio of edge pixels)
    min_edge_ratio: float = 0.015  # Below = too simple (icons, logos)
    max_edge_ratio: float = 0.5    # Above = likely noise
    
    # Color thresholds
    max_dominant_ratio: float = 0.85  # Above = single dominant color
    min_color_variance: float = 500   # Below = low contrast image
    
    # Duplicate detection (perceptual hash)
    hash_size: int = 8            # Hash grid size (8x8 = 64 bits)
    hash_threshold: int = 8       # Hamming distance for duplicates
    
    # Processing
    thumbnail_size: Tuple[int, int] = (100, 100)
    edge_threshold: int = 30      # Gradient magnitude for edge detection


@dataclass
class TriageResult:
    """Result of visual triage evaluation."""
    should_process: bool
    reason: str
    confidence: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "should_process": self.should_process,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "metrics": self.metrics
        }


@dataclass
class TriageStats:
    """Aggregated triage statistics."""
    total: int = 0
    processed: int = 0
    skipped: int = 0
    skip_reasons: Dict[str, int] = field(default_factory=dict)
    
    def record(self, result: TriageResult):
        """Record a triage result."""
        self.total += 1
        if result.should_process:
            self.processed += 1
        else:
            self.skipped += 1
            reason = result.reason
            self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1
    
    def get_savings(self) -> Dict:
        """Calculate savings from triage."""
        if self.total == 0:
            return {
                "total_images": 0,
                "vlm_calls_made": 0,
                "vlm_calls_avoided": 0,
                "savings_pct": 0.0,
                "skip_breakdown": {}
            }
        
        return {
            "total_images": self.total,
            "vlm_calls_made": self.processed,
            "vlm_calls_avoided": self.skipped,
            "savings_pct": round((self.skipped / self.total) * 100, 1),
            "skip_breakdown": dict(self.skip_reasons)
        }
    
    def reset(self):
        """Reset statistics."""
        self.total = 0
        self.processed = 0
        self.skipped = 0
        self.skip_reasons.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# PERCEPTUAL HASH (SIMPLE IMPLEMENTATION)
# ═══════════════════════════════════════════════════════════════════════════════

class SimplePerceptualHash:
    """
    Simple perceptual hash implementation.
    
    Uses average hash (aHash) algorithm:
    1. Resize to small thumbnail
    2. Convert to grayscale
    3. Compute average pixel value
    4. Create binary hash: pixel > average = 1, else 0
    """
    
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
    
    def compute(self, img) -> str:
        """Compute perceptual hash for an image."""
        from PIL import Image
        
        # Resize to hash_size x hash_size
        thumb = img.resize((self.hash_size, self.hash_size), Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        gray = thumb.convert('L')
        
        # Get pixels as numpy array
        pixels = np.array(gray, dtype=np.float32).flatten()
        
        # Compute average
        avg = pixels.mean()
        
        # Create binary hash
        bits = pixels > avg
        
        # Convert to hex string
        hash_int = 0
        for bit in bits:
            hash_int = (hash_int << 1) | int(bit)
        
        return format(hash_int, f'0{self.hash_size * self.hash_size // 4}x')
    
    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two hashes."""
        if len(hash1) != len(hash2):
            return max(len(hash1), len(hash2)) * 4  # Max distance
        
        # Convert hex to binary and count differences
        int1 = int(hash1, 16)
        int2 = int(hash2, 16)
        xor = int1 ^ int2
        
        return bin(xor).count('1')


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL TRIAGE
# ═══════════════════════════════════════════════════════════════════════════════

class VisualTriage:
    """
    Fast image classification for VLM cost optimization.
    
    Filters out decorative/low-value images before expensive VLM captioning.
    
    5-Stage Pipeline:
    1. Size Filter - Skip tiny images and extreme aspect ratios
    2. Entropy Filter - Skip solid colors and noise
    3. Edge Density Filter - Skip overly simple images
    4. Color Analysis - Skip single-color images
    5. Duplicate Detection - Skip repeated images
    """
    
    def __init__(self, config: TriageConfig = None):
        self.config = config or TriageConfig()
        self._seen_hashes: Set[str] = set()
        self._hasher = SimplePerceptualHash(hash_size=self.config.hash_size)
        self.stats = TriageStats()
        
        # Try to import imagehash for better hashing
        self._use_imagehash = False
        try:
            import imagehash
            self._imagehash = imagehash
            self._use_imagehash = True
            logger.debug("Using imagehash library for perceptual hashing")
        except ImportError:
            logger.debug("imagehash not available, using simple implementation")
        
        logger.info(f"Visual triage initialized (thresholds: entropy={self.config.min_entropy}-{self.config.max_entropy}, edges>{self.config.min_edge_ratio})")
    
    def evaluate(self, image_input) -> TriageResult:
        """
        Evaluate whether an image should be sent to VLM.
        
        Args:
            image_input: Path to image file, PIL Image, or bytes
        
        Returns:
            TriageResult with decision, reason, confidence, and metrics
        """
        try:
            from PIL import Image
            
            # Load image
            if isinstance(image_input, (str, Path)):
                img = Image.open(image_input)
            elif isinstance(image_input, bytes):
                import io
                img = Image.open(io.BytesIO(image_input))
            elif hasattr(image_input, 'mode'):  # PIL Image
                img = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Convert to RGB if needed
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            metrics = {}
            
            # ─────────────────────────────────────────────────────────────────
            # Stage 1: SIZE FILTER
            # ─────────────────────────────────────────────────────────────────
            
            width, height = img.size
            metrics['width'] = width
            metrics['height'] = height
            aspect_ratio = max(width, height) / max(min(width, height), 1)
            metrics['aspect_ratio'] = round(aspect_ratio, 2)
            
            if width < self.config.min_width or height < self.config.min_height:
                result = TriageResult(
                    should_process=False,
                    reason=SkipReason.TOO_SMALL.value,
                    confidence=0.95,
                    metrics=metrics
                )
                self.stats.record(result)
                return result
            
            if aspect_ratio > self.config.max_aspect_ratio:
                result = TriageResult(
                    should_process=False,
                    reason=SkipReason.EXTREME_ASPECT.value,
                    confidence=0.90,
                    metrics=metrics
                )
                self.stats.record(result)
                return result
            
            # ─────────────────────────────────────────────────────────────────
            # Stage 2: ENTROPY FILTER
            # ─────────────────────────────────────────────────────────────────
            
            entropy = self._compute_entropy(img)
            metrics['entropy'] = round(entropy, 2)
            
            if entropy < self.config.min_entropy:
                result = TriageResult(
                    should_process=False,
                    reason=SkipReason.LOW_ENTROPY.value,
                    confidence=0.90,
                    metrics=metrics
                )
                self.stats.record(result)
                return result
            
            if entropy > self.config.max_entropy:
                result = TriageResult(
                    should_process=False,
                    reason=SkipReason.HIGH_ENTROPY.value,
                    confidence=0.70,
                    metrics=metrics
                )
                self.stats.record(result)
                return result
            
            # ─────────────────────────────────────────────────────────────────
            # Stage 3: EDGE DENSITY FILTER
            # ─────────────────────────────────────────────────────────────────
            
            edge_ratio = self._compute_edge_density(img)
            metrics['edge_ratio'] = round(edge_ratio, 4)
            
            if edge_ratio < self.config.min_edge_ratio:
                result = TriageResult(
                    should_process=False,
                    reason=SkipReason.LOW_EDGES.value,
                    confidence=0.85,
                    metrics=metrics
                )
                self.stats.record(result)
                return result
            
            # ─────────────────────────────────────────────────────────────────
            # Stage 4: COLOR ANALYSIS
            # ─────────────────────────────────────────────────────────────────
            
            dominant_ratio, color_variance = self._analyze_colors(img)
            metrics['dominant_color_ratio'] = round(dominant_ratio, 3)
            metrics['color_variance'] = round(color_variance, 1)
            
            if dominant_ratio > self.config.max_dominant_ratio:
                result = TriageResult(
                    should_process=False,
                    reason=SkipReason.SINGLE_COLOR.value,
                    confidence=0.85,
                    metrics=metrics
                )
                self.stats.record(result)
                return result
            
            if color_variance < self.config.min_color_variance:
                result = TriageResult(
                    should_process=False,
                    reason=SkipReason.LOW_VARIANCE.value,
                    confidence=0.80,
                    metrics=metrics
                )
                self.stats.record(result)
                return result
            
            # ─────────────────────────────────────────────────────────────────
            # Stage 5: DUPLICATE DETECTION
            # ─────────────────────────────────────────────────────────────────
            
            img_hash = self._compute_hash(img)
            metrics['phash'] = img_hash
            
            is_duplicate, duplicate_distance = self._check_duplicate(img_hash)
            
            if is_duplicate:
                metrics['duplicate_distance'] = duplicate_distance
                result = TriageResult(
                    should_process=False,
                    reason=SkipReason.DUPLICATE.value,
                    confidence=0.95,
                    metrics=metrics
                )
                self.stats.record(result)
                return result
            
            # Register hash for future duplicate detection
            self._seen_hashes.add(img_hash)
            
            # ─────────────────────────────────────────────────────────────────
            # PASS: Send to VLM
            # ─────────────────────────────────────────────────────────────────
            
            confidence = self._compute_medical_confidence(metrics)
            metrics['medical_confidence'] = round(confidence, 3)
            
            result = TriageResult(
                should_process=True,
                reason=SkipReason.MEDICAL_CONTENT.value,
                confidence=confidence,
                metrics=metrics
            )
            self.stats.record(result)
            return result
            
        except Exception as e:
            logger.warning(f"Triage evaluation failed: {e}")
            # Default to processing on error (conservative approach)
            result = TriageResult(
                should_process=True,
                reason=SkipReason.TRIAGE_ERROR.value,
                confidence=0.5,
                metrics={"error": str(e)}
            )
            self.stats.record(result)
            return result
    
    def _compute_entropy(self, img) -> float:
        """
        Compute image entropy (information content).
        
        High entropy = complex image (photos, detailed diagrams)
        Low entropy = simple image (solid colors, gradients, icons)
        """
        from PIL import Image
        
        # Resize for speed
        thumb = img.resize(self.config.thumbnail_size, Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        gray = thumb.convert('L')
        
        # Get histogram
        hist = np.array(gray.histogram(), dtype=np.float64)
        
        # Normalize to probabilities
        hist = hist / hist.sum()
        
        # Remove zeros (log(0) is undefined)
        hist = hist[hist > 0]
        
        # Shannon entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return float(entropy)
    
    def _compute_edge_density(self, img) -> float:
        """
        Compute edge density using gradient magnitude.
        
        High edge density = detailed content
        Low edge density = simple shapes, solid areas
        """
        from PIL import Image
        
        # Resize and convert to grayscale
        thumb = img.resize(self.config.thumbnail_size, Image.Resampling.LANCZOS)
        gray = thumb.convert('L')
        
        # Get as numpy array
        arr = np.array(gray, dtype=np.float32)
        
        # Compute gradients (Sobel-like)
        gx = np.abs(np.diff(arr, axis=1))  # Horizontal gradient
        gy = np.abs(np.diff(arr, axis=0))  # Vertical gradient
        
        # Count edge pixels (above threshold)
        threshold = self.config.edge_threshold
        edge_pixels = np.sum(gx > threshold) + np.sum(gy > threshold)
        
        # Total possible edge pixels
        total_pixels = gx.size + gy.size
        
        return float(edge_pixels / total_pixels) if total_pixels > 0 else 0.0
    
    def _analyze_colors(self, img) -> Tuple[float, float]:
        """
        Analyze color distribution.
        
        Returns:
            dominant_ratio: Fraction of pixels that are the dominant color
            color_variance: Overall color variance (higher = more colorful)
        """
        from PIL import Image
        
        # Resize for speed
        thumb = img.resize((50, 50), Image.Resampling.LANCZOS)
        
        # Ensure RGB
        if thumb.mode != 'RGB':
            thumb = thumb.convert('RGB')
        
        # Get pixels as numpy array
        pixels = np.array(thumb).reshape(-1, 3)
        
        # Quantize colors to reduce uniqueness (16 levels per channel)
        quantized = (pixels // 16) * 16
        
        # Find unique colors and their counts
        unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
        
        # Dominant color ratio
        dominant_ratio = float(counts.max() / counts.sum())
        
        # Color variance
        color_variance = float(np.var(pixels))
        
        return dominant_ratio, color_variance
    
    def _compute_hash(self, img) -> str:
        """Compute perceptual hash for duplicate detection."""
        if self._use_imagehash:
            return str(self._imagehash.phash(img))
        else:
            return self._hasher.compute(img)
    
    def _check_duplicate(self, img_hash: str) -> Tuple[bool, int]:
        """
        Check if image is a duplicate of previously seen image.
        
        Returns:
            is_duplicate: True if duplicate found
            distance: Hamming distance to closest match
        """
        min_distance = float('inf')
        
        for seen_hash in self._seen_hashes:
            if self._use_imagehash:
                # imagehash returns ImageHash objects
                distance = self._imagehash.hex_to_hash(img_hash) - self._imagehash.hex_to_hash(seen_hash)
            else:
                distance = SimplePerceptualHash.hamming_distance(img_hash, seen_hash)
            
            if distance < min_distance:
                min_distance = distance
            
            if distance <= self.config.hash_threshold:
                return True, distance
        
        return False, int(min_distance) if min_distance != float('inf') else -1
    
    def _compute_medical_confidence(self, metrics: dict) -> float:
        """
        Compute confidence that image contains medical content.
        
        Medical images typically have:
        - Moderate to high entropy (5-7)
        - Moderate edge density (0.05-0.3)
        - Color variance
        """
        # Entropy score (optimal around 5.5-6.5)
        entropy = metrics.get('entropy', 5.0)
        if entropy < 5.0:
            entropy_score = entropy / 5.0
        elif entropy > 7.0:
            entropy_score = 1.0 - (entropy - 7.0) / 1.5
        else:
            entropy_score = 1.0
        entropy_score = max(0, min(1, entropy_score))
        
        # Edge score (optimal around 0.05-0.2)
        edge_ratio = metrics.get('edge_ratio', 0.1)
        if edge_ratio < 0.05:
            edge_score = edge_ratio / 0.05
        elif edge_ratio > 0.3:
            edge_score = 1.0 - (edge_ratio - 0.3) / 0.2
        else:
            edge_score = 1.0
        edge_score = max(0, min(1, edge_score))
        
        # Combined score (weight entropy more)
        confidence = (entropy_score * 0.6) + (edge_score * 0.4)
        
        return confidence
    
    def reset(self):
        """Reset for new document (clears duplicate detection cache)."""
        self._seen_hashes.clear()
        self.stats.reset()
        logger.debug("Visual triage reset")
    
    def get_stats(self) -> Dict:
        """Get triage statistics."""
        return {
            "unique_hashes_seen": len(self._seen_hashes),
            **self.stats.get_savings()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAGE-AWARE VLM WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class TriageAwareVLMCaptioner:
    """
    VLM captioner with visual triage pre-filtering.
    
    Wraps any VLM captioner and skips low-value images automatically.
    
    Usage:
        vlm = create_vlm_captioner()
        triage_vlm = TriageAwareVLMCaptioner(vlm)
        
        caption, info = await triage_vlm.caption(image_path)
        if info['skipped']:
            print(f"Skipped: {info['reason']}")
    """
    
    def __init__(
        self,
        vlm_captioner,
        triage: VisualTriage = None,
        config: TriageConfig = None
    ):
        """
        Initialize triage-aware captioner.
        
        Args:
            vlm_captioner: Underlying VLM captioner (must have async caption() method)
            triage: VisualTriage instance (creates one if None)
            config: TriageConfig for new VisualTriage
        """
        self.vlm = vlm_captioner
        self.triage = triage or VisualTriage(config)
        
        logger.info("Triage-aware VLM captioner initialized")
    
    async def caption(
        self,
        image_path: Path = None,
        image_bytes: bytes = None,
        context: str = "",
        force: bool = False,
        **kwargs
    ) -> Tuple[str, Dict]:
        """
        Caption image with triage pre-check.
        
        Args:
            image_path: Path to image file
            image_bytes: Raw image bytes
            context: Surrounding text context
            force: Skip triage and always process
            **kwargs: Additional args passed to VLM
        
        Returns:
            Tuple of (caption, triage_info)
        """
        # Determine image input
        image_input = image_path if image_path else image_bytes
        
        if image_input is None:
            raise ValueError("Either image_path or image_bytes required")
        
        # Triage check (unless forced)
        if not force:
            result = self.triage.evaluate(image_input)
            
            if not result.should_process:
                logger.debug(
                    f"Skipped VLM for {image_path.name if image_path else 'bytes'}: "
                    f"{result.reason} (confidence={result.confidence:.2f})"
                )
                
                return "", result.to_dict()
        
        # Process with VLM
        try:
            if image_path:
                caption = await self.vlm.caption(
                    image_path=image_path,
                    context=context,
                    **kwargs
                )
            else:
                caption = await self.vlm.caption(
                    image_bytes=image_bytes,
                    context=context,
                    **kwargs
                )
            
            return caption, {
                "skipped": False,
                "reason": SkipReason.PROCESSED.value,
                "confidence": 1.0
            }
            
        except Exception as e:
            logger.error(f"VLM captioning failed: {e}")
            return "", {
                "skipped": False,
                "reason": "vlm_error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_savings(self) -> Dict:
        """Get cost/time savings from triage."""
        return self.triage.get_stats()

    def get_stats(self) -> Dict:
        """Get VLM captioning statistics from underlying captioner."""
        return self.vlm.get_stats()

    def reset(self):
        """Reset triage for new document."""
        self.triage.reset()

    async def caption_batch(
        self,
        images: list,
        on_progress: Optional[Callable] = None,
        skip_decorative: bool = True
    ) -> list:
        """
        Caption multiple images with triage pre-filtering.

        Args:
            images: List of ImageInput objects to caption
            on_progress: Progress callback (processed, total)
            skip_decorative: Skip decorative images (passed to underlying VLM)

        Returns:
            List of (CaptionResult, triage_info) tuples
        """
        # Apply triage to filter images
        to_process = []
        skip_indices = []

        for i, image_input in enumerate(images):
            # Evaluate with triage
            triage_result = self.triage.evaluate(image_input.file_path)

            if not triage_result.should_process:
                # Record skip
                skip_indices.append((i, triage_result.to_dict()))
                logger.debug(
                    f"Triage skipped {image_input.id}: {triage_result.reason} "
                    f"(confidence={triage_result.confidence:.2f})"
                )
            else:
                # Queue for processing
                to_process.append((i, image_input))

        # Process non-skipped images in batch
        vlm_results = []
        indices = []
        if to_process:
            indices, inputs_to_process = zip(*to_process)
            vlm_results = await self.vlm.caption_batch(
                list(inputs_to_process),
                on_progress=on_progress,
                skip_decorative=skip_decorative
            )

        # Reconstruct results in original order
        results = [None] * len(images)

        # Fill in VLM results
        for idx, vlm_result in zip(indices, vlm_results):
            triage_info = {
                "skipped": False,
                "reason": SkipReason.PROCESSED.value,
                "confidence": 1.0
            }
            results[idx] = (vlm_result, triage_info)

        # Fill in skipped results with empty CaptionResult
        from src.retrieval.vlm_captioner import CaptionResult, ImageType

        for idx, triage_info in skip_indices:
            skipped_result = CaptionResult(
                image_id=images[idx].id,
                caption="",  # Empty caption for skipped images
                image_type=ImageType.UNKNOWN,
                confidence=0.0,
                model="triage-skipped",
                tokens_used=0,
                processing_time_ms=0.0
            )
            results[idx] = (skipped_result, triage_info)

        return results


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def batch_evaluate(
    image_paths: list,
    config: TriageConfig = None,
    reset_per_document: bool = False
) -> Tuple[list, list, Dict]:
    """
    Evaluate multiple images and split into process/skip lists.
    
    Args:
        image_paths: List of paths to evaluate
        config: Triage configuration
        reset_per_document: Whether to reset duplicate detection between documents
    
    Returns:
        Tuple of (to_process, to_skip, stats)
    """
    triage = VisualTriage(config)
    
    to_process = []
    to_skip = []
    
    for path in image_paths:
        result = triage.evaluate(path)
        
        if result.should_process:
            to_process.append((path, result))
        else:
            to_skip.append((path, result))
    
    return to_process, to_skip, triage.get_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python visual_triage.py <image_path_or_directory>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    triage = VisualTriage()
    
    if path.is_file():
        # Single file
        result = triage.evaluate(path)
        print(f"\n{path.name}:")
        print(json.dumps(result.to_dict(), indent=2))
    else:
        # Directory
        images = list(path.glob("*.jpg")) + list(path.glob("*.jpeg")) + list(path.glob("*.png"))
        
        print(f"\nEvaluating {len(images)} images in {path}...\n")
        
        for img_path in sorted(images):
            result = triage.evaluate(img_path)
            status = "✅ PROCESS" if result.should_process else "❌ SKIP"
            print(f"{status} {img_path.name}: {result.reason} (conf={result.confidence:.2f})")
        
        print(f"\n{'-'*60}")
        stats = triage.get_stats()
        print(f"Summary:")
        print(f"  Total: {stats['total_images']}")
        print(f"  Process: {stats['vlm_calls_made']}")
        print(f"  Skip: {stats['vlm_calls_avoided']}")
        print(f"  Savings: {stats['savings_pct']}%")
        print(f"\nSkip breakdown:")
        for reason, count in stats['skip_breakdown'].items():
            print(f"  {reason}: {count}")
