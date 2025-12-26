"""
Image serving routes with security validation.

Serves extracted images from the IMAGE_OUTPUT_DIR with path traversal protection.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/images", tags=["Images"])

IMAGE_DIR = Path(os.getenv("IMAGE_OUTPUT_DIR", "./data/images"))


@router.get("/{file_path:path}")
async def serve_image(file_path: str):
    """
    Serve image files with path validation.

    Args:
        file_path: Relative path to image file

    Returns:
        FileResponse with image content

    Raises:
        HTTPException 403: Path traversal attempt detected
        HTTPException 400: Invalid path format
        HTTPException 404: Image file not found

    Security:
        - Prevents directory traversal attacks
        - Validates path is within IMAGE_DIR
        - Only serves files (not directories)
    """
    try:
        # Construct full path and resolve to absolute path
        full_path = IMAGE_DIR / file_path
        full_path = full_path.resolve()

        # Security: Prevent directory traversal
        if not full_path.is_relative_to(IMAGE_DIR.resolve()):
            logger.warning(f"Path traversal attempt blocked: {file_path}")
            raise HTTPException(status_code=403, detail="Access denied")

    except (ValueError, OSError) as e:
        logger.error(f"Invalid path format: {file_path} - {e}")
        raise HTTPException(status_code=400, detail="Invalid path")

    # Check file exists and is a regular file
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    if not full_path.is_file():
        logger.warning(f"Attempt to access non-file: {file_path}")
        raise HTTPException(status_code=403, detail="Access denied")

    # Serve the file
    return FileResponse(full_path)
