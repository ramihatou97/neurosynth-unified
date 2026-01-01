#!/usr/bin/env python
"""Test Stage 1-2: PDF Parsing and Text Extraction

Usage:
    python tests/manual/stage01_pdf_parsing.py <path-to-pdf>

Or set the TEST_PDF_PATH environment variable.
"""
import sys
import os
from pathlib import Path

# Get PDF path from command line or environment
if len(sys.argv) > 1:
    PDF_PATH = Path(sys.argv[1])
else:
    PDF_PATH = Path(os.getenv("TEST_PDF_PATH", ""))
    if not PDF_PATH.exists():
        print("Usage: python tests/manual/stage01_pdf_parsing.py <path-to-pdf>")
        print("   Or: TEST_PDF_PATH=/path/to/file.pdf python tests/manual/stage01_pdf_parsing.py")
        sys.exit(1)

def test_pdf_parsing():
    print("="*70)
    print("STAGE 1-2: PDF PARSING & TEXT EXTRACTION")
    print("="*70)

    # Test 1: File exists
    assert PDF_PATH.exists(), f"PDF not found: {PDF_PATH}"
    print(f"✅ PDF file exists: {PDF_PATH.name}")
    print(f"   Size: {PDF_PATH.stat().st_size / 1024 / 1024:.2f} MB")

    # Test 2: PyMuPDF can open
    import fitz
    doc = fitz.open(str(PDF_PATH))
    print(f"\n✅ PDF opened with PyMuPDF")
    print(f"   Pages: {len(doc)}")
    print(f"   Title: {doc.metadata.get('title', 'N/A')}")
    print(f"   Author: {doc.metadata.get('author', 'N/A')}")
    print(f"   Subject: {doc.metadata.get('subject', 'N/A')}")

    # Test 3: Extract sample page
    page = doc[0]
    text = page.get_text("text")
    print(f"\n✅ Text extraction working")
    print(f"   Page 1 text length: {len(text)} chars")
    print(f"   First 150 chars: {text[:150]}...")

    # Test 4: Image extraction
    images = page.get_images()
    print(f"\n✅ Image extraction working")
    print(f"   Page 1 images: {len(images)}")

    # Test 5: Check a few more pages for content
    total_text = 0
    total_images = 0
    for i in range(min(10, len(doc))):
        page_text = doc[i].get_text("text")
        page_images = doc[i].get_images()
        total_text += len(page_text)
        total_images += len(page_images)

    print(f"\n✅ Sample of first 10 pages:")
    print(f"   Total text: {total_text} chars")
    print(f"   Total images: {total_images}")
    print(f"   Avg text per page: {total_text/min(10, len(doc)):.0f} chars")

    # Assertions (before closing)
    page_count = len(doc)
    assert page_count > 20, f"PDF too short: {page_count} pages"
    assert total_text > 1000, f"Too little text extracted: {total_text} chars"

    doc.close()

    print("\n" + "="*70)
    print("STAGE 1-2: PASSED ✅")
    print("="*70)
    return True

if __name__ == "__main__":
    try:
        test_pdf_parsing()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ STAGE 1-2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
