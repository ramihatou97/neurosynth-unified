"""
NeuroSynth Unified - Database Connection Manager
=================================================

Async connection pool management for PostgreSQL with pgvector.
"""

import logging
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg
from asyncpg import Pool, Connection
import numpy as np

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages PostgreSQL connection pool with pgvector support.
    
    Usage:
        db = DatabaseConnection(connection_string)
        await db.connect()
        
        async with db.acquire() as conn:
            result = await conn.fetch("SELECT * FROM documents")
        
        await db.close()
    """
    
    _instance: Optional['DatabaseConnection'] = None
    
    def __init__(
        self,
        connection_string: str,
        min_connections: int = 2,
        max_connections: int = 10
    ):
        self.connection_string = connection_string
        self.min_connections = min_connections
        self.max_connections = max_connections
        self._pool: Optional[Pool] = None
    
    @classmethod
    def get_instance(cls) -> 'DatabaseConnection':
        """Get singleton instance."""
        if cls._instance is None:
            raise RuntimeError("DatabaseConnection not initialized. Call initialize() first.")
        return cls._instance
    
    @classmethod
    async def initialize(
        cls,
        connection_string: str,
        **kwargs
    ) -> 'DatabaseConnection':
        """Initialize singleton instance."""
        if cls._instance is None:
            cls._instance = cls(connection_string, **kwargs)
            await cls._instance.connect()
        return cls._instance
    
    async def connect(self) -> None:
        """Initialize connection pool."""
        if self._pool is not None:
            return
        
        try:
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.min_connections,
                max_size=self.max_connections,
                init=self._init_connection,
                command_timeout=60
            )
            logger.info(
                f"Database pool created: {self.min_connections}-{self.max_connections} connections"
            )
        except Exception as e:
            logger.error(f"Failed to create database pool: {e}")
            raise
    
    async def _init_connection(self, conn: Connection) -> None:
        """Initialize each connection with pgvector codec."""
        await conn.set_type_codec(
            'vector',
            encoder=self._encode_vector,
            decoder=self._decode_vector,
            schema='public'
        )
    
    @staticmethod
    def _encode_vector(vector) -> str:
        """Encode numpy array/list to pgvector format."""
        if vector is None:
            return None
        if isinstance(vector, np.ndarray):
            return '[' + ','.join(str(float(x)) for x in vector) + ']'
        elif isinstance(vector, (list, tuple)):
            return '[' + ','.join(str(float(x)) for x in vector) + ']'
        return str(vector)
    
    @staticmethod
    def _decode_vector(value: str) -> Optional[np.ndarray]:
        """Decode pgvector string to numpy array."""
        if value is None:
            return None
        values = value.strip('[]').split(',')
        return np.array([float(x) for x in values], dtype=np.float32)
    
    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")
    
    @property
    def pool(self) -> Pool:
        """Get connection pool."""
        if not self._pool:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._pool
    
    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Connection, None]:
        """Acquire a connection from the pool."""
        async with self.pool.acquire() as conn:
            yield conn
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Connection, None]:
        """Acquire a connection with transaction."""
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args):
        """Fetch a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            result = await self.fetchval("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_stats(self) -> dict:
        """Get connection pool statistics."""
        if not self._pool:
            return {"status": "disconnected"}
        
        return {
            "status": "connected",
            "size": self._pool.get_size(),
            "free_size": self._pool.get_idle_size(),
            "min_size": self._pool.get_min_size(),
            "max_size": self._pool.get_max_size()
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def get_connection_string() -> str:
    """Get PostgreSQL connection string from environment."""
    import os
    return os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/neurosynth"
    )


async def get_database() -> DatabaseConnection:
    """Get database connection (for FastAPI dependency injection)."""
    return DatabaseConnection.get_instance()


async def init_database(connection_string: str, **kwargs) -> DatabaseConnection:
    """Initialize database connection."""
    return await DatabaseConnection.initialize(connection_string, **kwargs)


async def close_database() -> None:
    """Close database connection."""
    if DatabaseConnection._instance:
        await DatabaseConnection._instance.close()
        DatabaseConnection._instance = None
