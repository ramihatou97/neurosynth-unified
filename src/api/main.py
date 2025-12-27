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
    ingest_router
)
from src.api.routes.images import router as images_router

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
        # Allow startup even with degraded services
    
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
    app.include_router(health_router)
    app.include_router(search_router, prefix="/api/v1")
    app.include_router(rag_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(ingest_router)  # Has /api/v1/ingest prefix
    app.include_router(synthesis_router)
    app.include_router(images_router)  # Image serving with security
    
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
            "health": "/health"
        }
    
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
