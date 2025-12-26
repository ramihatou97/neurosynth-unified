#!/usr/bin/env python3
"""
NeuroSynth Unified - Database Initialization Script
====================================================

Initialize the PostgreSQL database with pgvector extension.

Usage:
    python scripts/init_database.py
    
    # Or with custom connection string
    DATABASE_URL=postgresql://... python scripts/init_database.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def init_database():
    """Initialize the database schema."""
    import asyncpg
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get connection string
    database_url = os.getenv("DATABASE_URL", "postgresql://neurosynth:neurosynth@localhost:5432/neurosynth")
    
    # Convert asyncpg format if needed
    if "+asyncpg" in database_url:
        database_url = database_url.replace("postgresql+asyncpg://", "postgresql://")
    
    print(f"Connecting to database...")
    
    try:
        conn = await asyncpg.connect(database_url)
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\nMake sure PostgreSQL is running and the database exists.")
        print("You can create the database with:")
        print("  createdb neurosynth")
        return False
    
    try:
        # Enable extensions
        print("Enabling extensions...")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        print("✓ Extensions enabled")
        
        # Create documents table
        print("Creating documents table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                source_path TEXT NOT NULL UNIQUE,
                title TEXT,
                total_pages INTEGER DEFAULT 0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source_path ON documents(source_path)")
        print("✓ Documents table created")
        
        # Create chunks table
        print("Creating chunks table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                page_number INTEGER,
                chunk_index INTEGER,
                start_char INTEGER,
                end_char INTEGER,
                chunk_type TEXT,
                specialty TEXT,
                cuis TEXT[] DEFAULT '{}',
                entities JSONB DEFAULT '{}',
                embedding VECTOR(1024),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks(chunk_type)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_cuis ON chunks USING GIN(cuis)")
        print("✓ Chunks table created")
        
        # Create images table
        print("Creating images table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                file_path TEXT,
                page_number INTEGER,
                image_index INTEGER,
                bbox JSONB,
                image_type TEXT,
                is_decorative BOOLEAN DEFAULT FALSE,
                triage_reason TEXT,
                vlm_caption TEXT,
                ocr_text TEXT,
                cuis TEXT[] DEFAULT '{}',
                embedding VECTOR(512),
                caption_embedding VECTOR(1024),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_images_document_id ON images(document_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_images_image_type ON images(image_type)")
        print("✓ Images table created")
        
        # Create links table
        print("Creating links table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS links (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                link_type TEXT NOT NULL,
                score FLOAT DEFAULT 0.0,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(chunk_id, image_id, link_type)
            )
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_links_chunk_id ON links(chunk_id)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_links_image_id ON links(image_id)")
        print("✓ Links table created")
        
        # Create vector indexes (if enough data exists)
        print("Creating vector indexes...")
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_images_embedding ON images 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 50)
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_images_caption_embedding ON images 
                USING ivfflat (caption_embedding vector_cosine_ops)
                WITH (lists = 50)
            """)
            print("✓ Vector indexes created")
        except Exception as e:
            print(f"⚠ Vector indexes deferred (need data first): {e}")
        
        # Create update trigger
        print("Creating triggers...")
        await conn.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql'
        """)
        await conn.execute("DROP TRIGGER IF EXISTS update_documents_updated_at ON documents")
        await conn.execute("""
            CREATE TRIGGER update_documents_updated_at
                BEFORE UPDATE ON documents
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column()
        """)
        print("✓ Triggers created")
        
        # Verify
        result = await conn.fetchval("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
        print(f"\n✅ Database initialized successfully!")
        print(f"   Tables created: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
        
    finally:
        await conn.close()


def main():
    """Main entry point."""
    print("=" * 50)
    print("NeuroSynth Database Initialization")
    print("=" * 50)
    print()
    
    success = asyncio.run(init_database())
    
    if success:
        print("\nNext steps:")
        print("1. Set your API keys in .env")
        print("2. Run: uvicorn src.api.main:app --reload")
        print("3. Visit: http://localhost:8000/docs")
    else:
        print("\nDatabase initialization failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
