# Image Embedding Retrospective: Complete Analysis

## Executive Summary

The "missing embedded images in synthesis" issue is a **multi-stage pipeline failure** that has manifested in different forms across multiple conversations. This document provides a complete forensic analysis of every failure mode, attempted fix, and the architectural understanding required to prevent recurrence.

---

## Part 1: The 7-Stage Image Embedding Pipeline

Images must successfully traverse **7 distinct stages** from database to browser. Failure at ANY stage results in "missing images" with different symptoms.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMAGE EMBEDDING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STAGE 1: FIGURE RESOLUTION (Backend)                                       │
│  ├── Input: LLM outputs [REQUEST_FIGURE: type="..." topic="..."]            │
│  ├── Input: image_catalog from SearchResult.images                          │
│  ├── Process: FigureResolver.resolve() matches requests to images           │
│  └── Output: resolved_figures list with image_path, image_caption           │
│                                                                              │
│  STAGE 2: JSON SERIALIZATION (Backend)                                      │
│  ├── Input: SynthesisResult containing FigureRequest dataclasses            │
│  ├── Process: json.dumps() for SSE streaming                                │
│  ├── Requirement: Custom encoder for non-serializable objects               │
│  └── Output: Valid JSON string                                              │
│                                                                              │
│  STAGE 3: IMAGE FILE SERVING (Backend)                                      │
│  ├── Input: IMAGE_OUTPUT_DIR environment variable                           │
│  ├── Process: FileResponse at /api/v1/images/{path}                         │
│  ├── Requirement: Path must point to actual image storage                   │
│  └── Output: HTTP 200 with image bytes                                      │
│                                                                              │
│  STAGE 4: MARKDOWN GENERATION (Frontend)                                    │
│  ├── Input: resolved_figures from API response                              │
│  ├── Process: buildMarkdownFromResult() in useApi.js                        │
│  ├── Requirement: Construct ![caption](/api/v1/images/path) syntax          │
│  └── Output: Markdown string with image tags                                │
│                                                                              │
│  STAGE 5: HTML CONVERSION (Frontend)                                        │
│  ├── Input: Markdown string                                                 │
│  ├── Process: parseMarkdown() in helpers.js                                 │
│  ├── Requirement: Image regex MUST precede link regex                       │
│  └── Output: HTML with <figure><img></figure> elements                      │
│                                                                              │
│  STAGE 6: CSS STYLING (Frontend)                                            │
│  ├── Input: HTML with class="synthesis-figure"                              │
│  ├── Process: CSS rules applied                                             │
│  ├── Requirement: .synthesis-figure styles defined                          │
│  └── Output: Visible, styled images                                         │
│                                                                              │
│  STAGE 7: NETWORK/PROXY (Browser)                                           │
│  ├── Input: <img src="/api/v1/images/...">                                  │
│  ├── Process: Vite proxy forwards to backend                                │
│  ├── Requirement: Proxy config includes /api route                          │
│  └── Output: Image loaded in browser                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Complete Failure History

### Failure Mode A: TypeError During SSE Streaming

**Symptom**: Synthesis starts, generates content, then crashes with:
```
TypeError: Object of type FigureRequest is not JSON serializable
```

**Root Cause**: Python dataclasses (`FigureRequest`, `SynthesisSection`) don't have built-in JSON serialization.

**Failed Fix Attempts**:

1. **Adding `serialize_figures()` helper function**
   - Location: `src/api/routes/synthesis.py`
   - Problem: Python bytecode caching meant old code was still running
   - Symptom: Fix was in source file but error persisted

2. **Cleaning `__pycache__` directories**
   - Command: `find . -type d -name __pycache__ -exec rm -rf {} +`
   - Result: Partial success - serialization error changed but didn't disappear
   - Reason: The helper function approach was incomplete

**Successful Fix**:

```python
# src/api/routes/synthesis.py:215-233
class SynthesisEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles synthesis-specific objects."""
    def default(self, obj):
        # Handle dataclass instances (like FigureRequest, SynthesisSection)
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dataclass_fields__'):
            return {k: getattr(obj, k) for k in obj.__dataclass_fields__.keys()}
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        # Fallback for other types
        return str(obj)

def _sse_event(data) -> str:
    """Format data as Server-Sent Event."""
    if isinstance(data, BaseModel):
        data = data.model_dump()
    return f"data: {json.dumps(data, cls=SynthesisEncoder)}\n\n"
```

**Why This Fix Is Permanent**:
- Generic: Handles ANY object with `to_dict()`, `__dataclass_fields__`, or `__dict__`
- Fallback: Converts unknown types to strings rather than crashing
- Architectural: Applied at the SSE layer, covering all synthesis output

---

### Failure Mode B: Images Return 404

**Symptom**: Network tab shows 404 for `/api/v1/images/...` requests

**Root Cause**: `IMAGE_OUTPUT_DIR` environment variable pointed to wrong directory.

**The Mismatch**:
```
Configured: IMAGE_OUTPUT_DIR="./data/images"
Actual:     Images stored in "./output/images"
```

**Failed Fix Attempts**:
- None - this was identified and fixed immediately

**Successful Fix**:
```bash
# Start backend with correct path
export IMAGE_OUTPUT_DIR="./output/images"
```

**Verification**:
```bash
curl -I http://localhost:8000/api/v1/images/ed8cd171-5aea-4b80-827e-5190f9a991a9/31a8e583847b08fbd8c8b92c75c82c24.png
# Returns: HTTP/1.1 200 OK
```

**Why This Could Recur**:
- Environment variable must be set correctly at every startup
- The startup script `/tmp/start_backend.sh` now includes correct value

---

### Failure Mode C: Images Render as Links (THE CRITICAL BUG)

**Symptom**:
- Synthesis completes successfully
- Backend logs show figures resolved
- Images load (HTTP 200)
- But images appear as clickable text links, not embedded images

**Root Cause**: `parseMarkdown()` regex ordering error.

**The Bug**:
```javascript
// BEFORE FIX (helpers.js)
.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2">$1</a>')  // Link regex

// This regex matches BOTH:
// - [text](url)     → correct: becomes <a>
// - ![text](url)    → WRONG: "!" becomes literal, rest becomes <a>
```

**What Was Happening**:
```
Input:  "![Caption](/api/v1/images/foo.png)"
Output: "!<a href="/api/v1/images/foo.png">Caption</a>"
        ↑
        The "!" became literal text, image became a link
```

**Failed Fix Attempts**:
- None - this was the deepest bug, not identified until all other issues were fixed

**Successful Fix**:
```javascript
// AFTER FIX (helpers.js:178-181)
// Images (must come before links to avoid conflict)
.replace(/!\[([^\]]*)\]\(([^)]+)\)/g,
    '<figure class="synthesis-figure"><img src="$2" alt="$1" loading="lazy" /><figcaption>$1</figcaption></figure>')
// Links
.replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
```

**Why Order Matters**:
```
Pattern specificity:
  !\[...\](...) is MORE SPECIFIC than \[...\](...)

The image pattern includes "!" which the link pattern doesn't check for.
By running image regex FIRST, we capture all images.
The remaining [...](...)  patterns are then correctly treated as links.
```

**Why This Fix Is Permanent**:
- The regex order is now correct: specific before general
- The image regex captures the `!` prefix explicitly
- Any `![...]()` will be matched as image, never as link

---

### Failure Mode D: No Figures Generated by LLM

**Symptom**: `figure_requests` array is empty

**Root Cause**: LLM didn't output `[REQUEST_FIGURE: ...]` tags, or used wrong format.

**The Prompt Issue**:
```
The prompt asks for: [REQUEST_FIGURE: type="..." topic="..."]
But LLM sometimes outputs: [Figure: description]
```

**Fix Applied** (engine.py:1069-1091):
```python
# Pattern 1: Structured format [REQUEST_FIGURE: type="..." topic="..."]
structured_pattern = r'\[REQUEST_FIGURE:\s*type="([^"]+)"\s*topic="([^"]+)"\]'

# Pattern 2: Simple format [Figure: description] - fallback
simple_pattern = r'\[Figure:\s*([^\]]+)\]'
```

**Why This Works**:
- Accepts both formats the LLM might produce
- Infers figure type from description keywords if using simple format

---

### Failure Mode E: Python Bytecode Caching

**Symptom**: Code changes don't take effect after editing

**Root Cause**: Python `.pyc` files in `__pycache__` contain compiled old code.

**The Problem**:
```
1. Edit synthesis.py with fix
2. Restart server with --reload
3. Server loads OLD .pyc files instead of new .py
4. Fix doesn't work, debugging goes in circles
```

**Fixes Applied**:
```bash
# Clean all bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Prevent future caching
export PYTHONDONTWRITEBYTECODE=1
```

**Permanent Solution** (in start scripts):
```bash
#!/bin/bash
export PYTHONDONTWRITEBYTECODE=1
./venv/bin/python -m uvicorn src.api.main:app --reload
```

---

## Part 3: The Complete Fix Summary

| Stage | Issue | Fix | File | Lines |
|-------|-------|-----|------|-------|
| 2 | FigureRequest not serializable | SynthesisEncoder class | synthesis.py | 215-233 |
| 2 | Section figures not serializable | serialize_figures() helper | synthesis.py | 615-628 |
| 3 | Wrong IMAGE_OUTPUT_DIR | Corrected env variable | start_backend.sh | - |
| 5 | Image regex missing | Added before link regex | helpers.js | 178-179 |
| 6 | No figure styles | Added CSS rules | styles.css | 1630-1652 |
| - | Bytecode caching | PYTHONDONTWRITEBYTECODE=1 | start_backend.sh | - |

---

## Part 4: Diagnostic Decision Tree

```
USER REPORTS: "Images not showing in synthesis"
│
├─► Check backend logs for "Resolving X figures"
│   ├─► 0 figures → STAGE 1 FAILURE (LLM not generating requests)
│   │   └─► Solution: Check prompts, add fallback patterns
│   │
│   └─► N figures (N > 0) → Continue to Stage 2
│
├─► Check for TypeError in logs
│   ├─► "not JSON serializable" → STAGE 2 FAILURE
│   │   └─► Solution: Verify SynthesisEncoder is used in _sse_event()
│   │
│   └─► No TypeError → Continue to Stage 3
│
├─► Test image URL directly: curl -I http://localhost:8000/api/v1/images/{path}
│   ├─► 404 Not Found → STAGE 3 FAILURE
│   │   └─► Solution: Check IMAGE_OUTPUT_DIR matches actual path
│   │
│   ├─► 403 Forbidden → Security path check failing
│   │   └─► Solution: Check is_relative_to() logic in main.py
│   │
│   └─► 200 OK → Continue to Stage 4
│
├─► Check frontend markdown: console.log(content)
│   ├─► No "![" in content → STAGE 4 FAILURE
│   │   └─► Solution: Check buildMarkdownFromResult() in useApi.js
│   │
│   └─► Has "![...](...)" → Continue to Stage 5
│
├─► Check parsed HTML: console.log(parseMarkdown(content))
│   ├─► No "<img" in output → STAGE 5 FAILURE
│   │   └─► Solution: Verify image regex precedes link regex in helpers.js
│   │
│   ├─► Has "<a" where "<img" expected → Regex ordering wrong
│   │   └─► Solution: Move image regex BEFORE link regex
│   │
│   └─► Has "<img" → Continue to Stage 6
│
├─► Inspect element in browser
│   ├─► No .synthesis-figure styles → STAGE 6 FAILURE
│   │   └─► Solution: Add CSS rules for .synthesis-figure
│   │
│   └─► Styles applied → Continue to Stage 7
│
└─► Check Network tab for image requests
    ├─► No requests made → Image src not rendering
    │   └─► Solution: Check dangerouslySetInnerHTML usage
    │
    ├─► Requests to wrong URL → Vite proxy misconfigured
    │   └─► Solution: Check vite.config.js proxy settings
    │
    └─► Requests 200 but no image → Content-Type issue
        └─► Solution: Check FileResponse in main.py
```

---

## Part 5: The Mental Model

### Core Principle: Images Are a Data Transformation Chain

```
Database Image → SearchResult.images → image_catalog → resolved_figures
                                                              ↓
                              ![caption](url) ← buildMarkdownFromResult()
                                    ↓
                     <img src="url"> ← parseMarkdown()
                                    ↓
                          Rendered Image ← Browser
```

### Key Invariants That Must Hold

1. **Every resolved figure must have non-empty `image_path`**
   ```javascript
   if (fig.image_path && fig.image_path !== '.') { ... }
   ```

2. **Image paths must be served at the URL they're referenced**
   ```
   Frontend constructs: /api/v1/images/${fig.image_path}
   Backend serves at:   /api/v1/images/{file_path:path}
   These MUST match.
   ```

3. **Regex ordering: Specific before General**
   ```
   !\[...\](...) must be processed BEFORE \[...\](...)
   ```

4. **JSON serialization must handle all types**
   ```python
   # This cascade handles everything:
   to_dict() → __dataclass_fields__ → __dict__ → str()
   ```

### Testing Checklist (Run Before Any Synthesis Release)

```bash
# 1. Verify image serving
curl -s -o /dev/null -w "%{http_code}" \
  http://localhost:8000/api/v1/images/$(ls output/images | head -1)
# Expected: 200

# 2. Verify serialization
python -c "
from src.synthesis.engine import FigureRequest
import json
from src.api.routes.synthesis import SynthesisEncoder
fr = FigureRequest('id', 'type', 'topic', 'ctx')
print(json.dumps(fr, cls=SynthesisEncoder))
"
# Expected: Valid JSON, no error

# 3. Verify regex ordering
node -e "
const { parseMarkdown } = require('./frontend/src/utils/helpers');
const md = '![Test](http://img.png)';
const html = parseMarkdown(md);
console.log(html.includes('<img') ? 'PASS' : 'FAIL');
"
# Expected: PASS
```

---

## Part 6: Architectural Recommendations

### 1. Add Startup Validation

```python
# In src/api/main.py create_app()
def validate_image_config():
    image_dir = Path(os.getenv("IMAGE_OUTPUT_DIR", "./output/images"))
    if not image_dir.exists():
        logger.error(f"IMAGE_OUTPUT_DIR does not exist: {image_dir}")
        raise RuntimeError(f"Image directory not found: {image_dir}")
    logger.info(f"Image serving configured: {image_dir.resolve()}")
```

### 2. Add Frontend Debug Mode

```javascript
// In buildMarkdownFromResult()
if (process.env.NODE_ENV === 'development') {
    console.log('[Synthesis] resolved_figures:', resolvedFigures.length);
    console.log('[Synthesis] figureMap entries:', figureMap.size);
}
```

### 3. Add End-to-End Test

```python
# tests/e2e/test_image_embedding.py
async def test_images_embedded_in_synthesis():
    """Verify images appear in synthesis output."""
    # 1. Generate synthesis
    response = await client.post("/api/synthesis/generate", json={
        "topic": "pterional craniotomy",
        "template_type": "PROCEDURAL"
    })
    result = response.json()

    # 2. Verify figures resolved
    assert len(result["resolved_figures"]) > 0

    # 3. Verify paths are valid
    for fig in result["resolved_figures"]:
        assert fig["image_path"]
        assert fig["image_path"] != "."

        # 4. Verify image serves
        img_response = await client.get(f"/api/v1/images/{fig['image_path']}")
        assert img_response.status_code == 200
```

---

## Conclusion

The image embedding issue was a **compound failure** requiring fixes at multiple pipeline stages:

| Root Cause | Impact | Fix Type |
|------------|--------|----------|
| No custom JSON encoder | Synthesis crashes | Architectural |
| Wrong IMAGE_OUTPUT_DIR | 404 errors | Configuration |
| Regex ordering | Images become links | Code fix |
| Bytecode caching | Fixes don't apply | Environment |

The fixes applied are **universal and permanent** because they address the issues at the architectural level rather than patching symptoms. The custom `SynthesisEncoder` handles any future dataclass. The regex ordering is now correct and won't regress unless explicitly changed.

This document serves as the definitive reference for understanding and debugging image embedding in NeuroSynth synthesis.
