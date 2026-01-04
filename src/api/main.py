"""
NeuroSynth Unified - FastAPI Application
=========================================

Main FastAPI application with all routes and middleware.

Run with:
    uvicorn src.api.main:app --reload

Or:
    python -m src.api.main
"""

from dotenv import load_dotenv
load_dotenv()  # Load .env file before anything else

# Validate environment at startup (before anything else)
from src.api.dependencies import validate_environment
validate_environment(exit_on_error=True)

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.api.dependencies import (
    ServiceContainer,
    get_settings
)
from src.api.routes import (
    search_router,
    rag_router,
    documents_router,
    health_router,
    synthesis_router,
    ingest_router,
    entities_router,
    indexes_router,
    # V3 Routes
    rag_v3_router,
    synthesis_v3_router,
)
from src.api.routes.images import router as images_router
from src.api.routes.knowledge_graph import router as knowledge_graph_router
from src.api.routes.registry import router as registry_router, load_registry_from_db
from src.chat.routes import router as chat_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    Initializes services on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Starting NeuroSynth API...")
    
    settings = get_settings()
    container = ServiceContainer.get_instance()
    
    try:
        await container.initialize(settings)
        logger.info("✓ Services initialized")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Fail fast - don't allow degraded startup that will cause 500 errors
        raise RuntimeError(f"Cannot start API without database: {e}") from e

    # Initialize chat stores (non-blocking - falls back to in-memory)
    try:
        from src.chat.routes import get_stores
        await get_stores()
        logger.info("✓ Chat stores initialized")
    except Exception as e:
        logger.warning(f"Chat store initialization failed (using in-memory fallback): {e}")

    # Load authority registry from database (if table exists)
    try:
        await load_registry_from_db(container.database)
        logger.info("✓ Authority registry loaded")
    except Exception as e:
        logger.warning(f"Authority registry not loaded: {e}. Using defaults.")

    yield
    
    # Shutdown
    logger.info("Shutting down NeuroSynth API...")
    await container.shutdown()
    logger.info("✓ Shutdown complete")


# =============================================================================
# Create Application
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.api_title,
        description="""
# NeuroSynth API

Neurosurgical knowledge retrieval and question answering API.

## Features

- **Semantic Search**: Search indexed content using natural language
- **RAG Q&A**: Ask questions and get answers with citations
- **Multi-turn Conversations**: Continue conversations with context
- **Document Management**: Browse and manage indexed documents

## Authentication

Currently open access. Production deployments should add authentication.

## Rate Limiting

- 100 requests per minute per client
- RAG endpoints may have lower limits due to LLM costs

        """,
        version=settings.api_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )
    
    # Register routes
    # All main API routes use /api/v1 prefix for frontend compatibility
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(search_router, prefix="/api/v1")
    app.include_router(rag_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(entities_router, prefix="/api/v1")
    app.include_router(indexes_router, prefix="/api/v1")
    app.include_router(ingest_router)  # Has /api/v1/ingest prefix
    app.include_router(synthesis_router)
    app.include_router(images_router)  # Image serving with security
    app.include_router(knowledge_graph_router)  # Knowledge graph endpoints
    app.include_router(registry_router)  # Authority registry API
    app.include_router(chat_router, prefix="/api/v1")  # Enhanced chat with synthesis linking

    # V3 Routes - Enhanced RAG and Synthesis with web research
    app.include_router(rag_v3_router)  # /api/rag/v3/* - Unified RAG with tri-modal processing
    app.include_router(synthesis_v3_router)  # /api/synthesis/v3/* - Enhanced synthesis with enrichment
    
    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ):
        """Handle validation errors with clear messages."""
        errors = []
        for error in exc.errors():
            loc = " -> ".join(str(l) for l in error["loc"])
            errors.append(f"{loc}: {error['msg']}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": errors,
                "code": "VALIDATION_ERROR"
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ):
        """Handle unexpected errors."""
        logger.exception(f"Unhandled error: {exc}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": str(exc) if settings.debug else "An error occurred",
                "code": "INTERNAL_ERROR"
            }
        )
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root redirect to docs."""
        return {
            "name": settings.api_title,
            "version": settings.api_version,
            "docs": "/docs",
            "health": "/api/v1/health"
        }

    # Image alias for /api/v1/images path (frontend compatibility)
    from fastapi.responses import FileResponse
    from pathlib import Path as FilePath
    import os

    IMAGE_DIR = FilePath(os.getenv("IMAGE_OUTPUT_DIR", "./output/images"))

    @app.get("/api/v1/images/{file_path:path}", tags=["Images"])
    async def serve_image_api_v1(file_path: str):
        """Serve images via /api/v1/images path (frontend compatibility)."""
        full_path = IMAGE_DIR / file_path
        full_path = full_path.resolve()

        if not full_path.is_relative_to(IMAGE_DIR.resolve()):
            from fastapi import HTTPException
            raise HTTPException(status_code=403, detail="Access denied")

        if not full_path.exists() or not full_path.is_file():
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Image not found")

        return FileResponse(full_path)

    return app


# Create app instance
app = create_app()


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="info"
    )
