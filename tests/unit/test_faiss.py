"""
NeuroSynth - FAISS Manager Unit Tests
======================================

Tests for FAISS index management and search.
"""

import pytest
import numpy as np
from pathlib import Path


# Skip all tests if FAISS not installed
pytest.importorskip("faiss")


# =============================================================================
# FAISSIndex Tests
# =============================================================================

class TestFAISSIndex:
    """Tests for FAISSIndex wrapper."""
    
    def test_build_index(self, sample_embeddings):
        """Build index from embeddings."""
        from src.retrieval.faiss_manager import FAISSIndex, FAISSIndexConfig
        
        config = FAISSIndexConfig(
            name="test",
            dimension=1024,
            index_type="Flat"
        )
        
        index = FAISSIndex(config)
        index.build(
            embeddings=sample_embeddings["chunk_embeddings"],
            ids=sample_embeddings["chunk_ids"]
        )
        
        assert index.size == 100
    
    def test_search_index(self, sample_embeddings):
        """Search returns results."""
        from src.retrieval.faiss_manager import FAISSIndex, FAISSIndexConfig
        
        config = FAISSIndexConfig(
            name="test",
            dimension=1024,
            index_type="Flat",
            metric="IP"
        )
        
        index = FAISSIndex(config)
        index.build(
            embeddings=sample_embeddings["chunk_embeddings"],
            ids=sample_embeddings["chunk_ids"]
        )
        
        # Search with first embedding
        query = sample_embeddings["chunk_embeddings"][0]
        results = index.search(query, k=5)
        
        assert len(results) == 5
        assert results[0][0] == "chunk_0"  # Should find itself
        assert all(isinstance(r[1], float) for r in results)
    
    def test_search_by_id(self, sample_embeddings):
        """Search by item ID."""
        from src.retrieval.faiss_manager import FAISSIndex, FAISSIndexConfig
        
        config = FAISSIndexConfig(
            name="test",
            dimension=1024,
            index_type="Flat"
        )
        
        index = FAISSIndex(config)
        index.build(
            embeddings=sample_embeddings["chunk_embeddings"],
            ids=sample_embeddings["chunk_ids"]
        )
        
        results = index.search_by_id("chunk_0", k=5)
        
        # Should return similar items, excluding itself
        assert len(results) == 5
        assert "chunk_0" not in [r[0] for r in results]
    
    def test_save_and_load(self, sample_embeddings, temp_index_dir):
        """Save and reload index."""
        from src.retrieval.faiss_manager import FAISSIndex, FAISSIndexConfig
        
        config = FAISSIndexConfig(
            name="test",
            dimension=1024,
            index_type="Flat"
        )
        
        # Build and save
        index = FAISSIndex(config)
        index.build(
            embeddings=sample_embeddings["chunk_embeddings"],
            ids=sample_embeddings["chunk_ids"]
        )
        index.save(temp_index_dir)
        
        # Load
        loaded = FAISSIndex.load(temp_index_dir, "test")
        
        assert loaded.size == 100
        
        # Search should work
        query = sample_embeddings["chunk_embeddings"][0]
        results = loaded.search(query, k=5)
        assert len(results) == 5
    
    def test_add_vectors(self, sample_embeddings):
        """Add vectors to existing index."""
        from src.retrieval.faiss_manager import FAISSIndex, FAISSIndexConfig
        
        config = FAISSIndexConfig(
            name="test",
            dimension=1024,
            index_type="Flat"
        )
        
        # Build with half
        index = FAISSIndex(config)
        index.build(
            embeddings=sample_embeddings["chunk_embeddings"][:50],
            ids=sample_embeddings["chunk_ids"][:50]
        )
        
        assert index.size == 50
        
        # Add rest
        added = index.add(
            embeddings=sample_embeddings["chunk_embeddings"][50:],
            ids=sample_embeddings["chunk_ids"][50:]
        )
        
        assert added == 50
        assert index.size == 100
    
    def test_ivf_index(self, sample_embeddings):
        """IVF index with training."""
        from src.retrieval.faiss_manager import FAISSIndex, FAISSIndexConfig
        
        config = FAISSIndexConfig(
            name="test",
            dimension=1024,
            index_type="IVFFlat",
            nlist=10,
            nprobe=3
        )
        
        index = FAISSIndex(config)
        index.build(
            embeddings=sample_embeddings["chunk_embeddings"],
            ids=sample_embeddings["chunk_ids"]
        )
        
        assert index.size == 100
        
        query = sample_embeddings["chunk_embeddings"][0]
        results = index.search(query, k=5)
        assert len(results) == 5


# =============================================================================
# FAISSManager Tests
# =============================================================================

class TestFAISSManager:
    """Tests for FAISSManager."""
    
    def test_build_text_index(self, sample_embeddings, temp_index_dir):
        """Build text chunk index."""
        from src.retrieval.faiss_manager import FAISSManager
        
        manager = FAISSManager(temp_index_dir)
        size = manager.build_text_index(
            embeddings=sample_embeddings["chunk_embeddings"],
            ids=sample_embeddings["chunk_ids"]
        )
        
        assert size == 100
    
    def test_build_image_index(self, sample_embeddings, temp_index_dir):
        """Build image visual index."""
        from src.retrieval.faiss_manager import FAISSManager
        
        manager = FAISSManager(temp_index_dir)
        size = manager.build_image_index(
            embeddings=sample_embeddings["image_embeddings"],
            ids=sample_embeddings["image_ids"]
        )
        
        assert size == 50
    
    def test_build_caption_index(self, sample_embeddings, temp_index_dir):
        """Build caption text index."""
        from src.retrieval.faiss_manager import FAISSManager
        
        manager = FAISSManager(temp_index_dir)
        size = manager.build_caption_index(
            embeddings=sample_embeddings["caption_embeddings"],
            ids=sample_embeddings["image_ids"]
        )
        
        assert size == 50
    
    def test_search_text(self, sample_embeddings, temp_index_dir):
        """Search text index."""
        from src.retrieval.faiss_manager import FAISSManager
        
        manager = FAISSManager(temp_index_dir)
        manager.build_text_index(
            embeddings=sample_embeddings["chunk_embeddings"],
            ids=sample_embeddings["chunk_ids"]
        )
        
        query = sample_embeddings["chunk_embeddings"][5]
        results = manager.search_text(query, k=10)
        
        assert len(results) == 10
        assert results[0][0] == "chunk_5"
    
    def test_search_by_text_for_images(self, sample_embeddings, temp_index_dir):
        """Search images using text query."""
        from src.retrieval.faiss_manager import FAISSManager
        
        manager = FAISSManager(temp_index_dir)
        manager.build_caption_index(
            embeddings=sample_embeddings["caption_embeddings"],
            ids=sample_embeddings["image_ids"]
        )
        
        # Use a caption embedding as query (simulating text-to-image)
        query = sample_embeddings["caption_embeddings"][0]
        results = manager.search_by_text_for_images(query, k=5)
        
        assert len(results) == 5
    
    def test_search_hybrid(self, sample_embeddings, temp_index_dir):
        """Hybrid search combining text and images."""
        from src.retrieval.faiss_manager import FAISSManager
        
        manager = FAISSManager(temp_index_dir)
        manager.build_text_index(
            embeddings=sample_embeddings["chunk_embeddings"],
            ids=sample_embeddings["chunk_ids"]
        )
        manager.build_caption_index(
            embeddings=sample_embeddings["caption_embeddings"],
            ids=sample_embeddings["image_ids"]
        )
        
        query = sample_embeddings["chunk_embeddings"][0]
        results = manager.search_hybrid(query, k=10, text_weight=0.7)
        
        assert len(results) > 0
        # Results should be unique IDs with combined scores
        ids = [r[0] for r in results]
        assert len(ids) == len(set(ids))
    
    def test_save_and_load(self, sample_embeddings, temp_index_dir):
        """Save and reload all indexes."""
        from src.retrieval.faiss_manager import FAISSManager
        
        # Build
        manager1 = FAISSManager(temp_index_dir)
        manager1.build_text_index(
            sample_embeddings["chunk_embeddings"],
            sample_embeddings["chunk_ids"]
        )
        manager1.build_image_index(
            sample_embeddings["image_embeddings"],
            sample_embeddings["image_ids"]
        )
        manager1.save()
        
        # Load
        manager2 = FAISSManager(temp_index_dir)
        stats = manager2.load()
        
        assert stats["text"] == 100
        assert stats["image"] == 50
        
        # Search should work
        query = sample_embeddings["chunk_embeddings"][0]
        results = manager2.search_text(query, k=5)
        assert len(results) == 5
    
    def test_get_stats(self, sample_embeddings, temp_index_dir):
        """Get index statistics."""
        from src.retrieval.faiss_manager import FAISSManager
        
        manager = FAISSManager(temp_index_dir)
        manager.build_text_index(
            sample_embeddings["chunk_embeddings"],
            sample_embeddings["chunk_ids"]
        )
        
        stats = manager.get_stats()
        
        assert stats["text"]["size"] == 100
        assert stats["text"]["dimension"] == 1024
        assert stats["image"]["size"] == 0  # Not built
    
    def test_search_similar_chunks(self, sample_embeddings, temp_index_dir):
        """Find similar chunks by ID."""
        from src.retrieval.faiss_manager import FAISSManager
        
        manager = FAISSManager(temp_index_dir)
        manager.build_text_index(
            sample_embeddings["chunk_embeddings"],
            sample_embeddings["chunk_ids"]
        )
        
        results = manager.search_similar_chunks("chunk_10", k=5)
        
        assert len(results) == 5
        assert "chunk_10" not in [r[0] for r in results]
    
    def test_empty_search(self, temp_index_dir):
        """Search with no index returns empty."""
        from src.retrieval.faiss_manager import FAISSManager
        
        manager = FAISSManager(temp_index_dir)
        
        query = np.random.randn(1024).astype(np.float32)
        results = manager.search_text(query, k=5)
        
        assert results == []


# =============================================================================
# Index Configuration Tests
# =============================================================================

class TestFAISSIndexConfig:
    """Tests for FAISSIndexConfig."""
    
    def test_default_config(self):
        """Default configuration values."""
        from src.retrieval.faiss_manager import FAISSIndexConfig

        config = FAISSIndexConfig(name="test", dimension=1024)

        assert config.index_type == "IVFFlat"
        assert config.nlist == 100
        assert config.nprobe == 5  # Optimized from 10 → 5 for 40% latency improvement
        assert config.metric == "IP"
    
    def test_custom_config(self):
        """Custom configuration values."""
        from src.retrieval.faiss_manager import FAISSIndexConfig
        
        config = FAISSIndexConfig(
            name="custom",
            dimension=512,
            index_type="HNSW",
            hnsw_m=64,
            hnsw_ef_search=128
        )
        
        assert config.dimension == 512
        assert config.index_type == "HNSW"
        assert config.hnsw_m == 64
    
    def test_preset_configs(self):
        """Preset configurations."""
        from src.retrieval.faiss_manager import TEXT_CONFIG, IMAGE_CONFIG, CAPTION_CONFIG
        
        assert TEXT_CONFIG.dimension == 1024
        assert IMAGE_CONFIG.dimension == 512
        assert CAPTION_CONFIG.dimension == 1024
