#!/usr/bin/env python3
"""
Memory-Safe Pipeline Test Script
================================

Tests both "Fast" (in-process) and "Safe" (subprocess) modes of the
memory-safe ingestion pipeline.

Usage:
    python scripts/test_memory_safe_ingest.py

Requirements:
    - Server running on localhost:8000
    - PIL/Pillow installed
    - reportlab installed (for PDF generation)
"""

import io
import sys
import time
import json
import tempfile
import requests
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
MEMORY_STATS_URL = f"{BASE_URL}/api/v1/ingest/memory-safe/memory-stats"
UPLOAD_URL = f"{BASE_URL}/api/v1/ingest/memory-safe/upload"
JOB_STATUS_URL = f"{BASE_URL}/api/v1/ingest/job"
UNLOAD_URL = f"{BASE_URL}/api/v1/ingest/memory-safe/unload-all"


def create_test_pdf(num_pages: int = 3, include_images: bool = True) -> bytes:
    """
    Generate a test PDF with text and optional images.
    Uses PyMuPDF (fitz) which is already installed.
    """
    import fitz  # PyMuPDF
    from PIL import Image

    doc = fitz.open()

    # Sample neuroscience text
    sample_texts = [
        """The cerebral cortex is the outer layer of neural tissue of the cerebrum.
        It plays a key role in memory, attention, perception, cognition, awareness,
        thought, language, and consciousness. The cerebral cortex is composed of gray
        matter, consisting mainly of cell bodies and capillaries. The hippocampus is
        a critical brain structure involved in learning and memory formation.""",

        """The brainstem connects the cerebrum with the spinal cord. It consists of
        the midbrain, pons, and medulla oblongata. The brainstem controls many vital
        functions including heart rate, breathing, sleeping, and eating. The vertebral
        artery supplies blood to the brainstem and posterior brain regions.""",

        """Neurosurgical approaches to the skull base require detailed knowledge of
        cranial anatomy. The pterional craniotomy provides access to the anterior
        circulation aneurysms and tumors of the sellar region. The retrosigmoid
        approach allows access to the cerebellopontine angle.""",
    ]

    for page_num in range(num_pages):
        page = doc.new_page(width=612, height=792)  # Letter size

        # Add title
        title_rect = fitz.Rect(72, 72, 540, 100)
        page.insert_textbox(
            title_rect,
            f"Test Document - Page {page_num + 1}",
            fontsize=16,
            fontname="helv",
            align=fitz.TEXT_ALIGN_CENTER
        )

        # Add body text
        text_rect = fitz.Rect(72, 120, 540, 400)
        text = sample_texts[page_num % len(sample_texts)]
        page.insert_textbox(
            text_rect,
            text,
            fontsize=11,
            fontname="helv",
            align=fitz.TEXT_ALIGN_LEFT
        )

        # Add a test image (colored rectangle)
        if include_images:
            # Create a simple test image
            img = Image.new('RGB', (200, 150), color=(100, 150, 200))

            # Add some variation
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            draw.rectangle([20, 20, 180, 130], fill=(200, 100, 100))
            draw.ellipse([50, 40, 150, 110], fill=(100, 200, 100))

            # Convert to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()

            # Insert into PDF
            img_rect = fitz.Rect(72, 420, 272, 570)
            page.insert_image(img_rect, stream=img_bytes)

            # Add caption
            caption_rect = fitz.Rect(72, 575, 272, 595)
            page.insert_textbox(
                caption_rect,
                f"Figure {page_num + 1}: Test anatomical diagram",
                fontsize=9,
                fontname="helv",
                align=fitz.TEXT_ALIGN_CENTER
            )

    # Save to bytes
    pdf_bytes = doc.tobytes()
    doc.close()

    return pdf_bytes


def check_server():
    """Verify server is running."""
    try:
        resp = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def get_memory_stats():
    """Get current memory statistics."""
    resp = requests.get(MEMORY_STATS_URL)
    return resp.json()


def upload_pdf(pdf_bytes: bytes, filename: str, force_subprocess: bool = False) -> dict:
    """Upload PDF and return job info."""
    files = {'file': (filename, pdf_bytes, 'application/pdf')}
    data = {'force_subprocess': 'true' if force_subprocess else 'false'}

    resp = requests.post(UPLOAD_URL, files=files, data=data)
    return resp.json()


def wait_for_job(job_id: str, timeout: int = 120) -> dict:
    """Poll job status until complete or timeout."""
    start = time.time()

    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{JOB_STATUS_URL}/{job_id}")
            if resp.status_code == 200:
                job = resp.json()
                status = job.get('status', 'unknown')

                if status in ('completed', 'failed'):
                    return job

            time.sleep(1)
        except Exception as e:
            print(f"  Warning: {e}")
            time.sleep(2)

    return {"status": "timeout", "error": f"Job did not complete within {timeout}s"}


def run_test(mode: str, force_subprocess: bool):
    """Run a single test."""
    print(f"\n{'='*60}")
    print(f"TEST: {mode} Mode")
    print(f"{'='*60}")

    # Get initial memory
    print("\n1. Checking initial memory state...")
    initial_mem = get_memory_stats()
    print(f"   Process RSS: {initial_mem['process']['rss_mb']:.1f} MB")
    print(f"   System Available: {initial_mem['system']['available_gb']:.2f} GB")
    print(f"   Loaded Models: {initial_mem['model_manager']['loaded_models']}")

    # Generate test PDF
    print("\n2. Generating test PDF...")
    pdf_bytes = create_test_pdf(num_pages=3, include_images=True)
    print(f"   PDF size: {len(pdf_bytes) / 1024:.1f} KB")

    # Upload
    print(f"\n3. Uploading with force_subprocess={force_subprocess}...")
    start_time = time.time()

    result = upload_pdf(pdf_bytes, f"test_{mode.lower()}.pdf", force_subprocess)

    if 'job_id' not in result:
        print(f"   ERROR: Upload failed: {result}")
        return False

    job_id = result['job_id']
    config = result.get('config', {})
    print(f"   Job ID: {job_id}")
    print(f"   Mode: {config.get('mode', 'unknown')}")
    print(f"   RAM Available: {config.get('available_ram_gb', '?')} GB")

    # Wait for completion
    print("\n4. Waiting for processing...")
    job = wait_for_job(job_id)

    elapsed = time.time() - start_time
    status = job.get('status', 'unknown')

    if status == 'completed':
        summary = job.get('summary', {})
        print(f"   ✓ COMPLETED in {elapsed:.1f}s")
        print(f"   Chunks: {summary.get('chunks', '?')}")
        print(f"   Images: {summary.get('images', '?')}")
        print(f"   Entities: {summary.get('entities', '?')}")
        print(f"   Embedded Images: {summary.get('embedded_images', '?')}")
    elif status == 'failed':
        print(f"   ✗ FAILED: {job.get('error', 'Unknown error')}")
        return False
    else:
        print(f"   ? Status: {status}")
        return False

    # Check final memory
    print("\n5. Checking final memory state...")
    final_mem = get_memory_stats()
    print(f"   Process RSS: {final_mem['process']['rss_mb']:.1f} MB")
    print(f"   Loaded Models: {final_mem['model_manager']['loaded_models']}")

    # Verify models were unloaded
    if not final_mem['model_manager']['loaded_models']:
        print("   ✓ All models unloaded (GC working)")
    else:
        print("   ⚠ Some models still loaded")

    return True


def main():
    print("=" * 60)
    print("MEMORY-SAFE PIPELINE TEST")
    print("=" * 60)

    # Check server
    print("\nChecking server...")
    if not check_server():
        print("ERROR: Server not running at", BASE_URL)
        print("Start the server with: ./start312.sh")
        sys.exit(1)
    print("✓ Server is running")

    # Unload any existing models
    print("\nClearing any loaded models...")
    requests.post(UNLOAD_URL)
    print("✓ Models cleared")

    results = {}

    # Test 1: Fast Mode (In-Process)
    try:
        results['fast'] = run_test("Fast (In-Process)", force_subprocess=False)
    except Exception as e:
        print(f"\nFast mode test failed with exception: {e}")
        results['fast'] = False

    # Clear between tests
    print("\n" + "-" * 60)
    print("Clearing models between tests...")
    requests.post(UNLOAD_URL)
    time.sleep(2)

    # Test 2: Safe Mode (Subprocess)
    try:
        results['safe'] = run_test("Safe (Subprocess)", force_subprocess=True)
    except Exception as e:
        print(f"\nSafe mode test failed with exception: {e}")
        results['safe'] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for mode, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {mode.upper()} mode: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n✓ All tests passed! Memory-safe pipeline is operational.")
    else:
        print("\n⚠ Some tests failed. Check the logs above for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
