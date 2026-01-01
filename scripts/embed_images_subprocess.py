#!/usr/bin/env python3
# scripts/embed_images_subprocess.py
"""
Isolated BiomedCLIP Worker Process

Crashes here don't affect main server. OS guarantees 100% memory reclamation.

Features:
- Force CPU mode (avoids MPS segfaults on Apple Silicon)
- Aggressive per-image GC
- Timeout handling via SIGALRM
- Partial results on failure
"""

import sys
import json
import gc
import logging
import signal
import traceback

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='%(asctime)s - BiomedCLIP-Worker - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BiomedCLIP-Worker")


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Worker timeout - killing process")


def main(input_path: str, output_path: str) -> int:
    """
    Load BiomedCLIP, process images, exit.

    Returns:
        0 on success, 1 on failure
    """
    results = {}

    # Set hard timeout (Unix only)
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(240)  # 4 minute hard limit
    except (AttributeError, ValueError):
        pass  # Windows doesn't have SIGALRM

    try:
        logger.info("Starting BiomedCLIP worker...")

        # Lazy imports inside subprocess
        import torch
        from PIL import Image
        import open_clip

        # Force CPU to avoid MPS/CUDA issues
        device = "cpu"
        logger.info(f"Using device: {device}")

        # Load model
        logger.info("Loading BiomedCLIP model...")
        model, preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            device=device
        )
        model.eval()
        logger.info("Model loaded successfully")

        # Load input manifest
        with open(input_path) as f:
            image_paths = json.load(f)

        logger.info(f"Processing {len(image_paths)} images...")

        success_count = 0
        fail_count = 0

        for idx, img_path in enumerate(image_paths):
            try:
                # Load and preprocess
                image = Image.open(img_path).convert("RGB")
                tensor = preprocess(image).unsqueeze(0).to(device)

                # Encode
                with torch.no_grad():
                    features = model.encode_image(tensor)
                    # Normalize
                    features = features / features.norm(dim=-1, keepdim=True)
                    # Convert to list
                    results[img_path] = features.cpu().numpy().tolist()[0]

                success_count += 1

                # Aggressive cleanup per image
                del tensor, features, image
                gc.collect()

                # Progress logging
                if (idx + 1) % 5 == 0:
                    logger.info(f"Progress: {idx + 1}/{len(image_paths)}")

            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                results[img_path] = None
                fail_count += 1

        logger.info(f"Completed: {success_count} success, {fail_count} failed")

    except TimeoutError as e:
        logger.error(f"Worker timeout: {e}")
    except ImportError as e:
        logger.critical(f"Missing dependency: {e}")
        return 1
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Always write output (even partial results)
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f)
            logger.info(f"Wrote {len(results)} results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to write output: {e}")

        # Cancel alarm
        try:
            signal.alarm(0)
        except:
            pass

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python embed_images_subprocess.py <input.json> <output.json>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Input JSON: Array of image file paths", file=sys.stderr)
        print("Output JSON: Object mapping paths to embeddings (512-dim) or null", file=sys.stderr)
        sys.exit(1)

    exit_code = main(sys.argv[1], sys.argv[2])
    sys.exit(exit_code)
