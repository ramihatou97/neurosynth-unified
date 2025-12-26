"""
NeuroSynth Unified - Database Layer
====================================

PostgreSQL + pgvector database layer for the unified system.

Components:
- connection.py: Connection pool management
- repositories/: CRUD operations for each entity type

Usage:
    from src.database import init_database, get_repositories
    
    # Initialize
    db = await init_database("postgresql://...")
    
    # Get repositories
    repos = get_repositories(db)
    
    # Use repositories
    doc = await repos.documents.create_from_path("/path/to/file.pdf")
    await repos.chunks.create_many_for_document(doc['id'], chunks)
    
    results = await repos.chunks.semantic_search(query_embedding, top_k=10)
"""

from src.database.connection import (
    DatabaseConnection,
    init_database,
    close_database,
    get_database
)

from src.database.repositories import (
    DocumentRepository,
    ChunkRepository,
    ImageRepository,
    LinkRepository
)


class Repositories:
    """
    Container for all repository instances.
    
    Provides convenient access to all repositories with a single
    database connection.
    """
    
    def __init__(self, db: DatabaseConnection):
        self.db = db
        self._documents = None
        self._chunks = None
        self._images = None
        self._links = None
    
    @property
    def documents(self) -> DocumentRepository:
        if self._documents is None:
            self._documents = DocumentRepository(self.db)
        return self._documents
    
    @property
    def chunks(self) -> ChunkRepository:
        if self._chunks is None:
            self._chunks = ChunkRepository(self.db)
        return self._chunks
    
    @property
    def images(self) -> ImageRepository:
        if self._images is None:
            self._images = ImageRepository(self.db)
        return self._images
    
    @property
    def links(self) -> LinkRepository:
        if self._links is None:
            self._links = LinkRepository(self.db)
        return self._links


def get_repositories(db: DatabaseConnection = None) -> Repositories:
    """
    Get repository container.
    
    Args:
        db: Database connection. If None, uses singleton instance.
    
    Returns:
        Repositories container with all repository instances.
    """
    if db is None:
        db = DatabaseConnection.get_instance()
    return Repositories(db)


async def get_database_stats(db: DatabaseConnection = None) -> dict:
    """Get comprehensive database statistics."""
    repos = get_repositories(db)
    
    return {
        'connection': await (db or DatabaseConnection.get_instance()).get_stats(),
        'documents': await repos.documents.count(),
        'chunks': await repos.chunks.get_statistics(),
        'images': await repos.images.get_statistics(),
        'links': await repos.links.get_statistics()
    }


__all__ = [
    # Connection
    'DatabaseConnection',
    'init_database',
    'close_database',
    'get_database',
    
    # Repositories
    'DocumentRepository',
    'ChunkRepository',
    'ImageRepository',
    'LinkRepository',
    
    # Convenience
    'Repositories',
    'get_repositories',
    'get_database_stats'
]
