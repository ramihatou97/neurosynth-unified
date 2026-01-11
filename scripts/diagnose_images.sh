#!/bin/bash
# =============================================================================
# IMAGE EMBEDDING DIAGNOSTIC SCRIPT
# =============================================================================
# Run this script when images are not appearing in synthesis output.
# It tests each stage of the 7-stage pipeline and identifies the failure point.
#
# Usage: ./scripts/diagnose_images.sh
# =============================================================================

set -e

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
IMAGE_OUTPUT_DIR="${IMAGE_OUTPUT_DIR:-./output/images}"

echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║           IMAGE EMBEDDING DIAGNOSTIC TOOL                             ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Track failures
FAILURES=0

# =============================================================================
# STAGE 1: Check if images exist in storage
# =============================================================================
echo "┌─────────────────────────────────────────────────────────────────────────┐"
echo "│ STAGE 1: Image Storage                                                  │"
echo "└─────────────────────────────────────────────────────────────────────────┘"

if [ -d "$IMAGE_OUTPUT_DIR" ]; then
    IMAGE_COUNT=$(find "$IMAGE_OUTPUT_DIR" -type f -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l | tr -d ' ')
    echo "✓ IMAGE_OUTPUT_DIR exists: $IMAGE_OUTPUT_DIR"
    echo "  Found $IMAGE_COUNT image files"

    if [ "$IMAGE_COUNT" -eq 0 ]; then
        echo "⚠ WARNING: No images found in storage!"
        ((FAILURES++))
    else
        # Get a sample image for testing
        SAMPLE_IMAGE=$(find "$IMAGE_OUTPUT_DIR" -type f \( -name "*.png" -o -name "*.jpg" \) | head -1)
        SAMPLE_PATH="${SAMPLE_IMAGE#$IMAGE_OUTPUT_DIR/}"
        echo "  Sample image: $SAMPLE_PATH"
    fi
else
    echo "✗ FAIL: IMAGE_OUTPUT_DIR does not exist: $IMAGE_OUTPUT_DIR"
    ((FAILURES++))
fi
echo ""

# =============================================================================
# STAGE 2: Check backend is running
# =============================================================================
echo "┌─────────────────────────────────────────────────────────────────────────┐"
echo "│ STAGE 2: Backend Health                                                 │"
echo "└─────────────────────────────────────────────────────────────────────────┘"

HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/api/v1/health" 2>/dev/null || echo "000")
if [ "$HEALTH_STATUS" = "200" ]; then
    echo "✓ Backend is running at $BACKEND_URL"
else
    echo "✗ FAIL: Backend not responding (HTTP $HEALTH_STATUS)"
    echo "  Start backend with: ./start-dev.sh"
    ((FAILURES++))
fi
echo ""

# =============================================================================
# STAGE 3: Check image serving endpoint
# =============================================================================
echo "┌─────────────────────────────────────────────────────────────────────────┐"
echo "│ STAGE 3: Image Serving Endpoint                                         │"
echo "└─────────────────────────────────────────────────────────────────────────┘"

if [ -n "$SAMPLE_PATH" ]; then
    IMG_URL="$BACKEND_URL/api/v1/images/$SAMPLE_PATH"
    IMG_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$IMG_URL" 2>/dev/null || echo "000")

    if [ "$IMG_STATUS" = "200" ]; then
        echo "✓ Image serving works: $IMG_URL"
    elif [ "$IMG_STATUS" = "404" ]; then
        echo "✗ FAIL: Image returns 404"
        echo "  URL: $IMG_URL"
        echo "  Check IMAGE_OUTPUT_DIR environment variable"
        ((FAILURES++))
    elif [ "$IMG_STATUS" = "403" ]; then
        echo "✗ FAIL: Image returns 403 (Forbidden)"
        echo "  Path security check may be failing"
        ((FAILURES++))
    else
        echo "✗ FAIL: Unexpected status $IMG_STATUS for $IMG_URL"
        ((FAILURES++))
    fi
else
    echo "⚠ SKIP: No sample image available for testing"
fi
echo ""

# =============================================================================
# STAGE 4: Check synthesis engine
# =============================================================================
echo "┌─────────────────────────────────────────────────────────────────────────┐"
echo "│ STAGE 4: Synthesis Engine                                               │"
echo "└─────────────────────────────────────────────────────────────────────────┘"

SYNTH_HEALTH=$(curl -s "$BACKEND_URL/api/synthesis/health" 2>/dev/null || echo '{}')
ENGINE_INIT=$(echo "$SYNTH_HEALTH" | grep -o '"engine_initialized":[^,}]*' | cut -d: -f2)

if [ "$ENGINE_INIT" = "true" ]; then
    echo "✓ Synthesis engine is initialized"
else
    echo "⚠ Synthesis engine not initialized (will init on first request)"
fi
echo ""

# =============================================================================
# STAGE 5: Check JSON serialization
# =============================================================================
echo "┌─────────────────────────────────────────────────────────────────────────┐"
echo "│ STAGE 5: JSON Serialization                                             │"
echo "└─────────────────────────────────────────────────────────────────────────┘"

cd /Users/ramihatoum/neurosynth-unified
SERIAL_TEST=$(./venv/bin/python -c "
import warnings
import logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from src.synthesis.engine import FigureRequest
import json
from src.api.routes.synthesis import SynthesisEncoder

fr = FigureRequest('test_id', 'anatomy', 'test topic', 'test context')
try:
    result = json.dumps(fr, cls=SynthesisEncoder)
    print('SUCCESS')
except Exception as e:
    print(f'FAIL: {e}')
" 2>/dev/null)

# Extract just the last line (SUCCESS or FAIL message)
SERIAL_RESULT=$(echo "$SERIAL_TEST" | tail -1)

if [[ "$SERIAL_RESULT" == "SUCCESS" ]]; then
    echo "✓ FigureRequest serialization works"
else
    echo "✗ FAIL: Serialization test failed"
    echo "  Check SynthesisEncoder in src/api/routes/synthesis.py"
    ((FAILURES++))
fi
echo ""

# =============================================================================
# STAGE 6: Check parseMarkdown regex
# =============================================================================
echo "┌─────────────────────────────────────────────────────────────────────────┐"
echo "│ STAGE 6: Frontend Markdown Parsing                                      │"
echo "└─────────────────────────────────────────────────────────────────────────┘"

# Check if helpers.js has image regex before link regex
HELPERS_FILE="/Users/ramihatoum/Desktop/frontend/src/utils/helpers.js"
if [ -f "$HELPERS_FILE" ]; then
    IMG_LINE=$(grep -n '!\[' "$HELPERS_FILE" | head -1 | cut -d: -f1)
    LINK_LINE=$(grep -n '\[(.+?)\]' "$HELPERS_FILE" | grep -v '!\[' | head -1 | cut -d: -f1)

    if [ -n "$IMG_LINE" ] && [ -n "$LINK_LINE" ]; then
        if [ "$IMG_LINE" -lt "$LINK_LINE" ]; then
            echo "✓ Image regex (line $IMG_LINE) comes BEFORE link regex (line $LINK_LINE)"
        else
            echo "✗ FAIL: Link regex (line $LINK_LINE) comes BEFORE image regex (line $IMG_LINE)"
            echo "  This will cause images to render as links!"
            ((FAILURES++))
        fi
    else
        echo "⚠ Could not determine regex ordering"
    fi
else
    echo "⚠ helpers.js not found at expected location"
fi
echo ""

# =============================================================================
# STAGE 7: Check CSS styles
# =============================================================================
echo "┌─────────────────────────────────────────────────────────────────────────┐"
echo "│ STAGE 7: CSS Styles                                                     │"
echo "└─────────────────────────────────────────────────────────────────────────┘"

STYLES_FILE="/Users/ramihatoum/Desktop/frontend/src/styles.css"
if [ -f "$STYLES_FILE" ]; then
    if grep -q "synthesis-figure" "$STYLES_FILE"; then
        echo "✓ .synthesis-figure CSS class is defined"
    else
        echo "⚠ WARNING: .synthesis-figure CSS class not found"
        echo "  Images may render but without styling"
    fi
else
    echo "⚠ styles.css not found at expected location"
fi
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                           DIAGNOSTIC SUMMARY                          ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"

if [ "$FAILURES" -eq 0 ]; then
    echo ""
    echo "  ✓ All checks passed! Image embedding pipeline is healthy."
    echo ""
    echo "  If images still don't appear:"
    echo "    1. Check browser console for errors"
    echo "    2. Check Network tab for failed requests"
    echo "    3. Verify synthesis actually generates figure_requests"
    echo ""
else
    echo ""
    echo "  ✗ Found $FAILURES failure(s) in the pipeline"
    echo ""
    echo "  Fix the issues above and re-run this script."
    echo ""
fi

exit $FAILURES
