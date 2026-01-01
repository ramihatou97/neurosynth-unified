# src/ingest/memory_safe_pipeline.py
"""
Memory-Safe Document Processing Pipeline

Implements staged execution with explicit garbage collection between phases.
Prevents OOM errors by:
1. Loading models only when needed
2. Unloading immediately after use
3. Running heavy operations in subprocesses (optional)
"""
import gc
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from src.core.model_manager import model_manager, ModelType

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for memory-safe processing."""
    min_available_memory_mb: int = 1500  # Pause/Warn if RAM drops below this

    # Feature Toggles
    enable_scispacy: bool = True
    enable_biomedclip: bool = True
    enable_vlm_captioning: bool = True

    # CRITICAL: Switch between in-process (fast) and subprocess (safe) embedding
    use_subprocess: bool = True

    # Batch sizes
    chunk_batch_size: int = 50
    image_batch_size: int = 4
    vlm_batch_size: int = 2

    # Chunking parameters
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Garbage Collection trigger
    gc_every_n_items: int = 20


class MemorySafePipeline:
    """
    Document processing pipeline designed to prevent OOM errors
    by strictly enforcing sequential model loading.

    Usage:
        pipeline = MemorySafePipeline()
        result = await pipeline.process(Path("document.pdf"))
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        logger.info(f"MemorySafePipeline initialized (subprocess={self.config.use_subprocess})")

    def _force_cleanup(self):
        """Aggressive memory cleanup."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _check_memory(self) -> Dict[str, float]:
        """Check current memory status."""
        import psutil
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)

        if available_mb < self.config.min_available_memory_mb:
            logger.warning(
                f"Low memory warning: {available_mb:.0f}MB available "
                f"(threshold: {self.config.min_available_memory_mb}MB)"
            )

        return {
            "available_mb": available_mb,
            "percent_used": mem.percent,
            "total_mb": mem.total / (1024 * 1024),
        }

    # -------------------------------------------------------------------------
    # STAGE 1: PDF Parsing (CPU Bound, Low Memory)
    # -------------------------------------------------------------------------
    async def stage_parse_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Parse PDF and extract text + images."""
        logger.info(f"Stage 1: Parsing PDF {pdf_path.name}")
        import fitz  # PyMuPDF

        result = {
            "metadata": {},
            "pages": [],
            "images": []
        }

        try:
            doc = fitz.open(pdf_path)
            result["metadata"] = {
                "title": doc.metadata.get("title", pdf_path.stem),
                "page_count": len(doc),
                "file_name": pdf_path.name,
            }

            for page_index, page in enumerate(doc):
                # Extract Text
                text = page.get_text()
                result["pages"].append({
                    "page_num": page_index + 1,
                    "text": text
                })

                # Extract Images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        if base_image and base_image.get("image"):
                            result["images"].append({
                                "page_num": page_index + 1,
                                "image_index": img_index,
                                "image_bytes": base_image["image"],
                                "ext": base_image.get("ext", "png"),
                                "width": base_image.get("width", 0),
                                "height": base_image.get("height", 0),
                            })
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_index + 1}: {e}")

            doc.close()
            logger.info(f"Parsed {len(result['pages'])} pages and {len(result['images'])} images.")
            return result

        except Exception as e:
            logger.error(f"PDF Parse failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # STAGE 2: Text Chunking (CPU Bound)
    # -------------------------------------------------------------------------
    async def stage_chunk_text(self, pages: List[Dict]) -> List[Dict]:
        """Chunk text into smaller segments for processing."""
        logger.info("Stage 2: Chunking text")
        chunks = []
        chunk_id = 0

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        for page in pages:
            words = page["text"].split()
            for i in range(0, len(words), chunk_size - overlap):
                segment = words[i:i + chunk_size]
                if len(segment) < 20:  # Skip noise
                    continue

                chunks.append({
                    "chunk_id": chunk_id,
                    "page_num": page["page_num"],
                    "content": " ".join(segment),
                    "entities": [],  # Populated in Stage 3
                    "word_count": len(segment),
                })
                chunk_id += 1

        logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
        return chunks

    # -------------------------------------------------------------------------
    # STAGE 3: Entity Extraction (Heavy Model: SciSpacy)
    # -------------------------------------------------------------------------
    async def stage_extract_entities(self, chunks: List[Dict]) -> List[Dict]:
        """Extract medical/scientific entities using SciSpacy."""
        if not self.config.enable_scispacy:
            logger.info("Stage 3: SciSpacy disabled, skipping entity extraction")
            return chunks

        logger.info(f"Stage 3: Extracting entities for {len(chunks)} chunks")
        self._check_memory()

        # The context manager handles loading AND unloading
        with model_manager.load(ModelType.SCISPACY) as nlp:
            for i, chunk in enumerate(chunks):
                try:
                    doc = nlp(chunk["content"])
                    chunk["entities"] = [
                        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
                        for ent in doc.ents
                    ]
                except Exception as e:
                    logger.warning(f"Entity extraction failed for chunk {i}: {e}")
                    chunk["entities"] = []

                # Periodic GC within the loop for large documents
                if i > 0 and i % self.config.gc_every_n_items == 0:
                    gc.collect()
                    logger.debug(f"Processed {i}/{len(chunks)} chunks")

        self._force_cleanup()  # Ensure memory is returned before next stage
        logger.info(f"Entity extraction complete. Memory: {self._check_memory()['available_mb']:.0f}MB available")
        return chunks

    # -------------------------------------------------------------------------
    # STAGE 4: Image Embeddings (Heavy Model: BiomedCLIP)
    # -------------------------------------------------------------------------
    async def stage_image_embeddings(self, images: List[Dict]) -> List[Dict]:
        """Generate embeddings for images using BiomedCLIP."""
        if not self.config.enable_biomedclip or not images:
            logger.info("Stage 4: BiomedCLIP disabled or no images, skipping")
            return images

        logger.info(f"Stage 4: Embedding {len(images)} images")
        self._check_memory()

        # STRATEGY A: Subprocess Isolation (Maximum Safety)
        if self.config.use_subprocess:
            logger.info("Using Subprocess Isolation for embeddings.")
            try:
                from src.ingest.subprocess_embedder import SubprocessEmbedder

                embedder = SubprocessEmbedder()

                # Run in thread pool to avoid blocking event loop during file I/O
                images = await asyncio.to_thread(embedder.embed_images, images)

            except Exception as e:
                logger.error(f"Subprocess embedding failed: {e}")
                # Mark all images as failed but continue pipeline
                for img in images:
                    img["embedding"] = []
                    img["embedding_error"] = str(e)

        # STRATEGY B: In-Process (Maximum Speed)
        else:
            logger.info("Using In-Process ModelManager for embeddings.")
            import torch
            from PIL import Image
            import io

            with model_manager.load(ModelType.BIOMEDCLIP) as clip_bundle:
                model = clip_bundle["model"]
                preprocess = clip_bundle["preprocess"]
                device = clip_bundle["device"]

                batch_size = self.config.image_batch_size
                for i in range(0, len(images), batch_size):
                    batch = images[i:i + batch_size]

                    for img_data in batch:
                        try:
                            pil_img = Image.open(io.BytesIO(img_data["image_bytes"]))
                            if pil_img.mode != "RGB":
                                pil_img = pil_img.convert("RGB")
                            tensor = preprocess(pil_img).unsqueeze(0).to(device)

                            with torch.no_grad():
                                embedding = model.encode_image(tensor)
                                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                                img_data["embedding"] = embedding.cpu().squeeze().tolist()
                        except Exception as e:
                            logger.warning(f"Image embed failed: {e}")
                            img_data["embedding"] = []

                    gc.collect()

        self._force_cleanup()
        logger.info(f"Image embedding complete. Memory: {self._check_memory()['available_mb']:.0f}MB available")
        return images

    # -------------------------------------------------------------------------
    # STAGE 5: VLM Captioning (API Call - Low Memory)
    # -------------------------------------------------------------------------
    async def stage_vlm_captioning(self, images: List[Dict]) -> List[Dict]:
        """Generate captions using Vision-Language Model API."""
        if not self.config.enable_vlm_captioning or not images:
            logger.info("Stage 5: VLM captioning disabled or no images, skipping")
            return images

        logger.info(f"Stage 5: Generating captions for {len(images)} images")

        # This would call an external API (e.g., GPT-4V, Claude)
        # Placeholder for now - implement based on your VLM provider
        for img in images:
            img["caption"] = ""  # Placeholder

        return images

    # -------------------------------------------------------------------------
    # MAIN EXECUTION
    # -------------------------------------------------------------------------
    async def process(self, pdf_path: Path) -> Dict[str, Any]:
        """Run the full sequential pipeline."""
        logger.info(f"Starting Memory-Safe Pipeline for {pdf_path.name}")
        logger.info(f"Initial memory: {self._check_memory()}")
        self._force_cleanup()  # Start clean

        try:
            # 1. Parse PDF
            raw_data = await self.stage_parse_pdf(pdf_path)
            self._force_cleanup()

            # 2. Chunk Text
            chunks = await self.stage_chunk_text(raw_data["pages"])

            # 3. Extract Entities (Loads -> Runs -> Unloads SciSpacy)
            chunks = await self.stage_extract_entities(chunks)

            # 4. Image Embeddings (Loads -> Runs -> Unloads BiomedCLIP)
            images = await self.stage_image_embeddings(raw_data["images"])

            # 5. VLM Captioning (API calls, low memory)
            images = await self.stage_vlm_captioning(images)

            # Remove raw image bytes from result to save memory
            for img in images:
                if "image_bytes" in img:
                    del img["image_bytes"]

            result = {
                "metadata": raw_data["metadata"],
                "chunks": chunks,
                "images": images,
                "stats": {
                    "chunk_count": len(chunks),
                    "image_count": len(images),
                    "total_entities": sum(len(c.get("entities", [])) for c in chunks),
                    "embedded_images": sum(1 for img in images if img.get("embedding")),
                }
            }

            logger.info(f"Pipeline completed successfully: {result['stats']}")
            logger.info(f"Final memory: {model_manager.get_memory_stats()}")
            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            model_manager.unload_all()  # Emergency cleanup
            self._force_cleanup()
            raise
