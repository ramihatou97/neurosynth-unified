# src/ingest/subprocess_embedder.py
"""
Robust subprocess-based image embedding with retry and fallback.

Features:
- Complete memory isolation (subprocess dies → OS reclaims 100%)
- Retry on failure (2 attempts)
- Graceful degradation (pipeline continues if embedding fails)
- Batch processing (limits memory per subprocess)
- Timeout handling
"""

import sys
import json
import subprocess
import tempfile
import logging
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbedderConfig:
    """Configuration for subprocess embedder."""
    max_retries: int = 2
    timeout_seconds: int = 300  # 5 minutes
    batch_size: int = 10  # Images per subprocess call
    retry_delay: float = 2.0


class SubprocessEmbedder:
    """
    Robust BiomedCLIP embedding via subprocess isolation.

    Why subprocess?
    - BiomedCLIP + SciSpacy in same process causes segfaults
    - Subprocess crashes don't affect main server
    - OS guarantees 100% memory reclamation when subprocess exits

    Usage:
        embedder = SubprocessEmbedder()
        images = embedder.embed_images(images_with_bytes)
        # Each image now has 'embedding' field (list of floats or None)
    """

    SCRIPT_NAME = "embed_images_subprocess.py"

    def __init__(self, config: Optional[EmbedderConfig] = None, project_root: Path = None):
        self.config = config or EmbedderConfig()
        self.python_exe = sys.executable

        # Find project root and script path
        if project_root:
            self.project_root = Path(project_root)
        else:
            # Assume we're in src/ingest/, go up two levels
            self.project_root = Path(__file__).parent.parent.parent

        self.script_path = self.project_root / "scripts" / self.SCRIPT_NAME

        if not self.script_path.exists():
            raise FileNotFoundError(
                f"Worker script not found at {self.script_path.absolute()}. "
                f"Expected at: scripts/{self.SCRIPT_NAME}"
            )

        logger.info(f"SubprocessEmbedder initialized (retries={self.config.max_retries})")

    def _run_subprocess(self, image_paths: List[str], temp_dir: Path) -> Dict[str, List[float]]:
        """Run subprocess with given image paths."""
        input_json = temp_dir / "input.json"
        output_json = temp_dir / "output.json"

        with open(input_json, 'w') as f:
            json.dump(image_paths, f)

        cmd = [
            self.python_exe,
            str(self.script_path),
            str(input_json),
            str(output_json)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            cwd=str(self.project_root),
        )

        # Log subprocess output for debugging
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-10:]:  # Last 10 lines
                if line and 'FutureWarning' not in line:
                    logger.debug(f"[Worker] {line}")

        if result.returncode != 0:
            logger.warning(f"Subprocess exit code: {result.returncode}")
            raise RuntimeError(f"Worker exit code: {result.returncode}")

        if not output_json.exists():
            raise FileNotFoundError("No output from worker")

        with open(output_json) as f:
            return json.load(f)

    def _embed_batch_with_retry(self, images: List[Dict], temp_dir: Path) -> int:
        """
        Embed a batch with retry logic.
        Returns number of successful embeddings.
        """
        # Write images to temp files
        image_paths = []
        path_to_idx = {}

        for idx, img_data in enumerate(images):
            if "image_bytes" not in img_data:
                continue

            ext = img_data.get("ext", "png")
            if not ext.startswith("."):
                ext = f".{ext}"

            img_path = temp_dir / f"img_{idx}{ext}"
            try:
                with open(img_path, 'wb') as f:
                    f.write(img_data["image_bytes"])
                image_paths.append(str(img_path))
                path_to_idx[str(img_path)] = idx
            except Exception as e:
                logger.warning(f"Failed to write image {idx}: {e}")

        if not image_paths:
            return 0

        # Retry loop
        for attempt in range(self.config.max_retries):
            try:
                logger.info(
                    f"BiomedCLIP subprocess attempt {attempt + 1}/{self.config.max_retries} "
                    f"({len(image_paths)} images)"
                )

                embeddings = self._run_subprocess(image_paths, temp_dir)

                # Map results back
                success = 0
                for path, vector in embeddings.items():
                    if vector and path in path_to_idx:
                        idx = path_to_idx[path]
                        images[idx]["embedding"] = vector
                        success += 1

                logger.info(f"Embedded {success}/{len(image_paths)} images")
                return success

            except subprocess.TimeoutExpired:
                logger.warning(f"Attempt {attempt + 1}: Timeout after {self.config.timeout_seconds}s")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}: {e}")

            # Delay before retry
            if attempt < self.config.max_retries - 1:
                logger.info(f"Retrying in {self.config.retry_delay}s...")
                time.sleep(self.config.retry_delay)

        logger.error("All embedding attempts failed")
        return 0

    def embed_images(self, images: List[Dict]) -> List[Dict]:
        """
        Main entry point. Embeds images with full retry and fallback.

        Args:
            images: List of dicts with 'image_bytes' key

        Returns:
            Same list with 'embedding' field added:
            - List of 512 floats if successful
            - None if failed (can still use captions for search)
        """
        if not images:
            return images

        logger.info(f"SubprocessEmbedder: Processing {len(images)} images")

        # Process in batches to limit subprocess memory
        batch_size = self.config.batch_size
        total_success = 0

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(images) + batch_size - 1) // batch_size

            logger.info(f"Processing batch {batch_num}/{total_batches}")

            # Each batch gets its own temp directory
            temp_dir = Path(tempfile.mkdtemp(prefix="biomedclip_"))

            try:
                success = self._embed_batch_with_retry(batch, temp_dir)
                total_success += success
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # Mark failed images explicitly
        for img in images:
            if "embedding" not in img:
                img["embedding"] = None

        logger.info(f"SubprocessEmbedder complete: {total_success}/{len(images)} embedded")
        return images

    def test_worker(self) -> bool:
        """
        Test if the worker script can run successfully.
        Useful for health checks.
        """
        try:
            from PIL import Image
            import io

            temp_dir = Path(tempfile.mkdtemp(prefix="biomedclip_test_"))

            # Create 1x1 test image
            img = Image.new('RGB', (1, 1), color='red')
            test_img_path = temp_dir / "test.png"
            img.save(test_img_path)

            input_json = temp_dir / "input.json"
            output_json = temp_dir / "output.json"

            with open(input_json, 'w') as f:
                json.dump([str(test_img_path)], f)

            result = subprocess.run(
                [self.python_exe, str(self.script_path), str(input_json), str(output_json)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.project_root),
            )

            success = result.returncode == 0 and output_json.exists()

            shutil.rmtree(temp_dir, ignore_errors=True)

            if success:
                logger.info("Worker test: PASSED")
            else:
                logger.warning(f"Worker test: FAILED (exit code {result.returncode})")

            return success

        except Exception as e:
            logger.error(f"Worker test failed: {e}")
            return False
