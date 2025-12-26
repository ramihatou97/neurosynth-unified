"""
NeuroSynth v2.0 - Image Extractor
==================================

Extract images with full medical context.

Key features:
1. Surrounding text extraction
2. Caption detection with figure ID parsing
3. Image type classification
4. Quality filtering (entropy, size)
5. Deduplication via content hash
"""

import hashlib
import io
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from src.shared.models import ExtractedImage, ImageType

logger = logging.getLogger(__name__)


@dataclass
class ImageExtractionConfig:
    """Configuration for image extraction."""
    min_size: int = 200            # Minimum width/height in pixels (raised for medical quality)
    max_aspect_ratio: float = 5.0  # Maximum width/height ratio
    min_entropy: float = 4.0       # Minimum entropy for content filtering (raised to catch gradients)
    context_margin: float = 60.0   # Pixels around image for context extraction
    caption_search_height: float = 120.0  # Pixels below image to search for caption


class NeuroImageExtractor:
    """
    Extract images with full medical context.
    
    Key insight: Medical images are meaningless without:
    - Caption (what the image shows)
    - Surrounding text (why it matters)
    - Figure ID (for cross-referencing)
    """
    
    # Caption detection patterns
    CAPTION_PATTERNS = [
        r"^(Figure|Fig\.?|FIGURE)\s*(\d+[\.\-]?\d*[A-Za-z]?)",
        r"^(Plate|Image|Photo)\s*(\d+[\.\-]?\d*)",
        r"^(Panel)\s*([A-Za-z])",
    ]
    
    # Image type classification keywords
    TYPE_KEYWORDS = {
        ImageType.SURGICAL_PHOTO: [
            "intraoperative", "surgical view", "operative field", "exposure",
            "retractor", "dissection", "clip application", "surgical photo"
        ],
        ImageType.IMAGING_SCAN: [
            "mri", "mr imaging", "ct scan", "x-ray", "xray",
            "t1-weighted", "t2-weighted", "t1w", "t2w", "flair",
            "angiogram", "angiography", "dsa", "cta", "mra",
            "axial", "sagittal", "coronal", "diffusion"
        ],
        ImageType.ANATOMY_DIAGRAM: [
            "anatomy", "anatomical", "schematic", "diagram",
            "illustration", "drawing", "cross-section", "labeled"
        ],
        ImageType.FLOWCHART: [
            "algorithm", "flowchart", "decision tree", "protocol",
            "management", "pathway", "step 1", "step 2"
        ],
    }
    
    def __init__(
        self,
        output_dir: Path,
        config: ImageExtractionConfig = None
    ):
        """
        Initialize the image extractor.
        
        Args:
            output_dir: Directory to save extracted images
            config: Extraction configuration
        """
        if not HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) is required for image extraction")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ImageExtractionConfig()
        
        self._compiled_caption_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.CAPTION_PATTERNS
        ]
    
    def extract_from_page(
        self,
        doc: "fitz.Document",
        page: "fitz.Page",
        page_num: int,
        document_id: str,
        seen_hashes: Set[str]
    ) -> List[ExtractedImage]:
        """
        Extract all meaningful images from a page.
        
        Args:
            doc: PyMuPDF document
            page: Page to extract from
            page_num: Page number (0-indexed)
            document_id: Parent document ID
            seen_hashes: Set of already seen content hashes for deduplication
            
        Returns:
            List of ExtractedImage objects
        """
        images = []
        
        # Get image list from page
        image_list = page.get_images(full=True)
        
        # Get image positions
        try:
            image_info = page.get_image_info()
            pos_map = {info["xref"]: info.get("bbox") for info in image_info if "xref" in info}
        except (AttributeError, RuntimeError, MemoryError, KeyError) as e:
            logger.debug(f"Image info extraction fallback: {e}")
            pos_map = {}
        
        for img_tuple in image_list:
            xref = img_tuple[0]
            
            try:
                image = self._extract_single_image(
                    doc=doc,
                    page=page,
                    page_num=page_num,
                    xref=xref,
                    bbox=pos_map.get(xref),
                    document_id=document_id,
                    seen_hashes=seen_hashes
                )
                
                if image:
                    images.append(image)
                    
            except Exception as e:
                # Log but continue - don't fail entire page
                print(f"Warning: Failed to extract image xref {xref}: {e}")
                continue
        
        return images
    
    def _extract_single_image(
        self,
        doc: "fitz.Document",
        page: "fitz.Page",
        page_num: int,
        xref: int,
        bbox: Optional[Tuple[float, float, float, float]],
        document_id: str,
        seen_hashes: Set[str]
    ) -> Optional[ExtractedImage]:
        """
        Extract a single image with full context.
        """
        # Extract raw image data
        try:
            base_image = doc.extract_image(xref)
        except (ValueError, RuntimeError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Image extraction failed for xref {xref}: {e}")
            return None
        
        if not base_image:
            return None
        
        image_bytes = base_image["image"]
        width = base_image["width"]
        height = base_image["height"]
        ext = base_image.get("ext", "png")
        
        # Compute content hash for deduplication
        content_hash = hashlib.md5(image_bytes).hexdigest()
        if content_hash in seen_hashes:
            return None
        seen_hashes.add(content_hash)

        # Apply filters

        # Filter: Severely degraded images (high pixel count but tiny file size)
        pixels = width * height
        file_size_kb = len(image_bytes) / 1024
        if file_size_kb > 0:
            compression_ratio = pixels / file_size_kb
            # If > 20,000 pixels per KB, image is severely degraded
            if compression_ratio > 20000 and file_size_kb < 10:
                return None

        # Filter: Minimum size
        if width < self.config.min_size or height < self.config.min_size:
            return None
        
        # Filter: Aspect ratio (catches banners, lines, separators)
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > self.config.max_aspect_ratio:
            return None
        
        # Filter: Entropy (catches solid colors, simple patterns)
        entropy = self._compute_entropy(image_bytes)
        if entropy < self.config.min_entropy:
            return None

        # Filter: Gradient patterns (catches decorative backgrounds)
        if self._is_gradient_or_solid(image_bytes):
            return None

        # Save to disk
        file_path = self.output_dir / f"{content_hash}.{ext}"
        file_path.write_bytes(image_bytes)
        
        # Create image object
        image = ExtractedImage.create(
            document_id=document_id,
            page_number=page_num,
            file_path=file_path,
            width=width,
            height=height,
            content_hash=content_hash
        )
        image.format = ext
        
        # Extract context if we have position
        if bbox:
            image.surrounding_text = self._extract_surrounding_text(page, bbox)
            caption, confidence, figure_id = self._detect_caption(page, bbox)
            image.caption = caption
            image.caption_confidence = confidence
            image.figure_id = figure_id
        else:
            # Try to get context from page text
            page_text = page.get_text("text")
            image.surrounding_text = page_text[:500] if page_text else ""
        
        # Classify image type
        image.image_type = self._classify_image(
            caption=image.caption,
            surrounding_text=image.surrounding_text
        )
        
        # Compute quality score
        image.quality_score = self._compute_quality(
            image_bytes=image_bytes,
            width=width,
            height=height,
            entropy=entropy
        )
        
        return image
    
    def _extract_surrounding_text(
        self,
        page: "fitz.Page",
        image_bbox: Tuple[float, float, float, float]
    ) -> str:
        """
        Extract text surrounding an image.
        
        Strategy:
        1. Expand bbox by margin
        2. Get all text blocks in expanded area
        3. Exclude text inside image bbox
        4. Sort by position and concatenate
        """
        x0, y0, x1, y1 = image_bbox
        margin = self.config.context_margin
        
        # Create expanded search area
        search_rect = fitz.Rect(
            max(0, x0 - margin),
            max(0, y0 - margin),
            x1 + margin,
            y1 + margin
        )
        
        image_rect = fitz.Rect(image_bbox)
        
        # Get text blocks in search area
        try:
            blocks = page.get_text("dict", clip=search_rect)["blocks"]
        except (TypeError, RuntimeError, IndexError, KeyError) as e:
            logger.debug(f"Text extraction failed in rect {search_rect}: {e}")
            return ""
        
        surrounding = []
        
        for block in blocks:
            if block.get("type") != 0:  # Text blocks only
                continue
            
            block_rect = fitz.Rect(block["bbox"])
            
            # Skip text that's inside the image area
            if image_rect.contains(block_rect):
                continue
            
            # Extract text from block
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "") + " "
            
            text = text.strip()
            if text:
                # Store with y-position for ordering
                y_pos = block["bbox"][1]
                surrounding.append((y_pos, text))
        
        # Sort by vertical position (top to bottom)
        surrounding.sort(key=lambda x: x[0])
        
        # Join and limit length
        result = "\n".join(t[1] for t in surrounding)
        return result[:1000]
    
    def _detect_caption(
        self,
        page: "fitz.Page",
        image_bbox: Tuple[float, float, float, float]
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Detect caption for an image.
        
        Searches below and beside the image for caption text.
        
        Returns:
            Tuple of (caption_text, confidence, figure_id)
        """
        x0, y0, x1, y1 = image_bbox
        page_rect = page.rect
        
        # Search area: below image
        caption_rect = fitz.Rect(
            max(0, x0 - 30),
            y1,
            min(page_rect.width, x1 + 30),
            min(page_rect.height, y1 + self.config.caption_search_height)
        )
        
        try:
            blocks = page.get_text("dict", clip=caption_rect)["blocks"]
        except (TypeError, RuntimeError, IndexError, KeyError) as e:
            logger.debug(f"Caption detection failed in rect {caption_rect}: {e}")
            return None, 0.0, None
        
        for block in blocks:
            if block.get("type") != 0:
                continue
            
            # Extract text from block
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "") + " "
            
            text = text.strip()
            if not text:
                continue
            
            # Check for figure pattern
            for pattern in self._compiled_caption_patterns:
                match = pattern.match(text)
                if match:
                    figure_type = match.group(1)
                    figure_num = match.group(2)
                    figure_id = f"{figure_type} {figure_num}"
                    
                    # Caption is the rest of the text
                    caption = text[match.end():].strip(":.- ")
                    if not caption:
                        caption = text
                    
                    return caption, 0.95, figure_id
            
            # Short text near image might be caption (lower confidence)
            if len(text) < 300:
                return text, 0.5, None
        
        return None, 0.0, None
    
    def _classify_image(
        self,
        caption: Optional[str],
        surrounding_text: str
    ) -> ImageType:
        """
        Classify image type based on context.
        """
        context = f"{caption or ''} {surrounding_text}".lower()
        
        scores: Dict[ImageType, int] = {}
        
        for img_type, keywords in self.TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in context)
            if score > 0:
                scores[img_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        return ImageType.ILLUSTRATION
    
    def _compute_entropy(self, image_bytes: bytes) -> float:
        """
        Compute Shannon entropy of image.
        
        Higher entropy = more information content.
        Low entropy images are likely decorative (solid colors, gradients).
        """
        if not HAS_PIL:
            return 5.0  # Default to passing if PIL unavailable
        
        try:
            img = PILImage.open(io.BytesIO(image_bytes)).convert("L")
            histogram = img.histogram()
            total = sum(histogram)
            
            if total == 0:
                return 0.0
            
            entropy = 0.0
            for count in histogram:
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
            
            return entropy

        except (IOError, AttributeError, ValueError) as e:
            logger.debug(f"Entropy computation failed, using default: {e}")
            return 5.0  # Fallback: assume non-decorative

    def _is_gradient_or_solid(self, image_bytes: bytes) -> bool:
        """
        Detect solid colors and gradient patterns that indicate decorative images.

        Returns True if image appears to be a solid color or simple gradient.
        """
        if not HAS_PIL:
            return False

        try:
            img = PILImage.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Sample pixels from different regions
            width, height = img.size
            pixels = []

            # Sample 9 points in a 3x3 grid
            for x_frac in [0.25, 0.5, 0.75]:
                for y_frac in [0.25, 0.5, 0.75]:
                    x = int(width * x_frac)
                    y = int(height * y_frac)
                    pixels.append(img.getpixel((x, y)))

            # Check color variance
            r_values = [p[0] for p in pixels]
            g_values = [p[1] for p in pixels]
            b_values = [p[2] for p in pixels]

            # Calculate variance for each channel
            def variance(values):
                mean = sum(values) / len(values)
                return sum((v - mean) ** 2 for v in values) / len(values)

            r_var = variance(r_values)
            g_var = variance(g_values)
            b_var = variance(b_values)

            # If all channels have very low variance, it's solid/gradient
            # Threshold: variance < 500 per channel indicates low color diversity
            max_var = max(r_var, g_var, b_var)

            if max_var < 500:
                return True  # Likely solid color or gentle gradient

            # Check for monotonic gradient pattern (values consistently increasing/decreasing)
            # Sample a horizontal line
            h_pixels = [img.getpixel((int(width * f), height // 2)) for f in [0.1, 0.3, 0.5, 0.7, 0.9]]
            h_grays = [sum(p) // 3 for p in h_pixels]  # Convert to grayscale

            # Check if monotonically increasing or decreasing
            diffs = [h_grays[i+1] - h_grays[i] for i in range(len(h_grays)-1)]
            if all(d >= -5 for d in diffs) or all(d <= 5 for d in diffs):
                # Consistent gradient direction
                if max(h_grays) - min(h_grays) > 20:  # But has some variation (not solid)
                    if max_var < 2000:  # Still relatively uniform
                        return True  # Likely a gradient

            return False

        except (IOError, IndexError, AttributeError) as e:
            logger.debug(f"Gradient detection failed: {e}")
            return False

    def _compute_quality(
        self,
        image_bytes: bytes,
        width: int,
        height: int,
        entropy: float
    ) -> float:
        """
        Compute overall quality score (0-1).
        
        Based on:
        - Resolution
        - Entropy (information content)
        """
        # Resolution component (normalized to 1024x768 as "good" for medical imaging)
        pixels = width * height
        resolution_score = min(pixels / (1024 * 768), 1.0)

        # Entropy component (normalized, typical range 0-8)
        entropy_score = min(entropy / 7.0, 1.0)

        # Combined score - weight resolution higher for medical imaging
        return (resolution_score * 0.6 + entropy_score * 0.4)
    
    def extract_from_document(
        self,
        doc: "fitz.Document",
        document_id: str
    ) -> List[ExtractedImage]:
        """
        Extract all images from a document.
        
        Convenience method for full document extraction.
        """
        all_images = []
        seen_hashes: Set[str] = set()
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_images = self.extract_from_page(
                doc=doc,
                page=page,
                page_num=page_num,
                document_id=document_id,
                seen_hashes=seen_hashes
            )
            all_images.extend(page_images)
        
        return all_images
