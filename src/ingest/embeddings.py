"""
NeuroSynth v2.0 - Embedding Providers (Enhanced)
=================================================

Abstract embedding interfaces with implementations for:
- OpenAI (text-embedding-3-small/large)
- Voyage AI (voyage-3, voyage-3-lite)
- CLIP (image embeddings)
- BiomedCLIP (medical image embeddings)

Enhancements:
- Retry logic with exponential backoff
- Timeout handling
- Dimension validation
- Rate limit handling
- Batch optimization
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable
import numpy as np

# Load environment variables from .env file early
# This ensures API keys are available when embedders are created
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system environment

from src.utils.circuit_breaker import CircuitBreaker, CircuitOpenError

logger = logging.getLogger(__name__)

# =============================================================================
# CIRCUIT BREAKERS FOR EXTERNAL APIS
# =============================================================================

# Circuit breaker for Voyage AI API
voyage_breaker = CircuitBreaker(
    name="voyage",
    failure_threshold=5,      # Open after 5 consecutive failures
    success_threshold=2,      # Close after 2 successes in half-open
    reset_timeout=60.0,       # Try recovery after 60s
    timeout=30.0              # Per-request timeout
)

# Circuit breaker for OpenAI Embeddings API
openai_embeddings_breaker = CircuitBreaker(
    name="openai_embeddings",
    failure_threshold=5,
    success_threshold=2,
    reset_timeout=60.0,
    timeout=30.0
)


class EmbeddingError(Exception):
    """Base embedding error."""
    pass


class RateLimitError(EmbeddingError):
    """API rate limit hit."""
    pass


class TimeoutError(EmbeddingError):
    """Embedding request timed out."""
    pass


class DimensionError(EmbeddingError):
    """Unexpected embedding dimension."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    timeout: float = 60.0


@dataclass
class EmbeddingStats:
    """Statistics for embedding operations."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retry_count: int = 0
    partial_responses: int = 0  # API returned fewer embeddings than requested
    total_tokens: int = 0
    total_time_ms: float = 0.0
    total_latency_seconds: float = 0.0


class TextEmbedder(ABC):
    """Abstract interface for text embedding models."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier."""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts efficiently."""
        pass


class ImageEmbedder(ABC):
    """Abstract interface for image embedding models."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model identifier."""
        pass
    
    @abstractmethod
    async def embed(self, image_path: Path) -> np.ndarray:
        """Embed a single image."""
        pass
    
    @abstractmethod
    async def embed_batch(self, image_paths: List[Path]) -> List[np.ndarray]:
        """Embed multiple images efficiently."""
        pass


async def retry_with_backoff(
    func: Callable,
    config: RetryConfig = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Execute async function with exponential backoff retry.
    
    Args:
        func: Async function to execute
        config: Retry configuration
        on_retry: Callback on retry (attempt_num, exception)
    
    Returns:
        Function result
        
    Raises:
        Last exception if all retries fail
    """
    config = config or RetryConfig()
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            async with asyncio.timeout(config.timeout):
                return await func()
        except asyncio.TimeoutError as e:
            last_exception = TimeoutError(f"Request timed out after {config.timeout}s")
            delay = min(
                config.initial_delay * (config.exponential_base ** attempt),
                config.max_delay
            )
        except Exception as e:
            last_exception = e
            # Check if rate limit error
            error_msg = str(e).lower()
            if 'rate' in error_msg or '429' in error_msg:
                delay = min(
                    config.initial_delay * (config.exponential_base ** (attempt + 1)),
                    config.max_delay
                )
            else:
                delay = config.initial_delay * (config.exponential_base ** attempt)
        
        if attempt < config.max_attempts - 1:
            if on_retry:
                on_retry(attempt + 1, last_exception)
            logger.warning(
                f"Embedding attempt {attempt + 1} failed: {last_exception}. "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)
    
    raise last_exception


# =============================================================================
# TEXT EMBEDDER IMPLEMENTATIONS
# =============================================================================

class OpenAITextEmbedder(TextEmbedder):
    """
    OpenAI text embeddings with retry logic.
    
    Models:
    - text-embedding-3-small (1536 dim, faster, cheaper)
    - text-embedding-3-large (3072 dim, better quality)
    - text-embedding-ada-002 (1536 dim, legacy)
    """
    
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        retry_config: RetryConfig = None
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required: pip install openai")
        
        self._model = model
        self._dimension = self.DIMENSIONS.get(model, 1536)
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self._retry_config = retry_config or RetryConfig()
        self._stats = EmbeddingStats()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def stats(self) -> EmbeddingStats:
        return self._stats
    
    async def embed(self, text: str) -> np.ndarray:
        """Embed single text with retry and circuit breaker protection."""
        self._stats.total_requests += 1

        async def _do_embed():
            response = await self._client.embeddings.create(
                model=self._model,
                input=text
            )
            return np.array(response.data[0].embedding, dtype=np.float32)

        def _on_retry(attempt, exc):
            self._stats.retry_count += 1

        try:
            # Circuit breaker prevents hammering a down service
            async with openai_embeddings_breaker:
                result = await retry_with_backoff(
                    _do_embed,
                    self._retry_config,
                    _on_retry
                )
            self._stats.successful_requests += 1
            return result
        except CircuitOpenError as e:
            self._stats.failed_requests += 1
            logger.warning(f"OpenAI Embeddings API circuit open, failing fast: {e}")
            raise EmbeddingError(f"OpenAI Embeddings API unavailable (circuit open): {e}") from e
        except Exception as e:
            self._stats.failed_requests += 1
            raise EmbeddingError(f"Failed to embed text: {e}") from e

    async def embed_batch(
        self,
        texts: List[str],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[np.ndarray]:
        """
        Embed multiple texts with retry, circuit breaker, and progress tracking.

        OpenAI supports batching natively (up to 2048 inputs).
        """
        if not texts:
            return []

        BATCH_SIZE = 2000
        all_embeddings = []
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx, i in enumerate(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i:i + BATCH_SIZE]
            self._stats.total_requests += 1

            # Create closure with batch captured
            current_batch = batch  # Capture for closure

            async def _do_batch():
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=current_batch
                )
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [
                    np.array(d.embedding, dtype=np.float32)
                    for d in sorted_data
                ]

            def _on_retry(attempt, exc):
                self._stats.retry_count += 1

            try:
                # Circuit breaker prevents hammering a down service
                async with openai_embeddings_breaker:
                    embeddings = await retry_with_backoff(
                        _do_batch,
                        self._retry_config,
                        _on_retry
                    )
                all_embeddings.extend(embeddings)
                self._stats.successful_requests += 1

                if on_progress:
                    on_progress(batch_idx + 1, total_batches)

            except CircuitOpenError as e:
                self._stats.failed_requests += 1
                logger.warning(f"OpenAI Embeddings API circuit open at batch {batch_idx}, failing fast: {e}")
                # Return zeros for remaining batches when circuit is open
                remaining_texts = len(texts) - len(all_embeddings)
                all_embeddings.extend([
                    np.zeros(self._dimension, dtype=np.float32)
                    for _ in range(remaining_texts)
                ])
                break  # Don't try more batches when circuit is open

            except Exception as e:
                self._stats.failed_requests += 1
                logger.error(f"Batch {batch_idx} failed: {e}")
                # Return zeros for failed batch to maintain alignment
                all_embeddings.extend([
                    np.zeros(self._dimension, dtype=np.float32)
                    for _ in batch
                ])

        return all_embeddings
    
    async def embed_with_validation(
        self,
        text: str,
        expected_dim: Optional[int] = None
    ) -> np.ndarray:
        """Embed with dimension validation."""
        emb = await self.embed(text)
        expected = expected_dim or self._dimension
        if len(emb) != expected:
            raise DimensionError(
                f"Expected {expected} dimensions, got {len(emb)}. "
                f"Model may have changed or schema needs update."
            )
        return emb


class VoyageTextEmbedder(TextEmbedder):
    """
    Voyage AI text embeddings with retry logic.
    
    Models:
    - voyage-3 (1024 dim, best quality)
    - voyage-3-lite (512 dim, faster)
    - voyage-2 (1024 dim, legacy)
    """
    
    DIMENSIONS = {
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-2": 1024,
    }
    
    def __init__(
        self,
        model: str = "voyage-3",
        api_key: Optional[str] = None,
        retry_config: RetryConfig = None
    ):
        try:
            import voyageai
        except ImportError:
            raise ImportError("voyageai package required: pip install voyageai")
        
        self._model = model
        self._dimension = self.DIMENSIONS.get(model, 1024)
        self._client = voyageai.AsyncClient(api_key=api_key)
        self._retry_config = retry_config or RetryConfig()
        self._stats = EmbeddingStats()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def stats(self) -> EmbeddingStats:
        return self._stats
    
    async def embed(self, text: str) -> np.ndarray:
        """Embed single text with retry and circuit breaker protection."""
        self._stats.total_requests += 1

        async def _do_embed():
            result = await self._client.embed(
                texts=[text],
                model=self._model,
                input_type="document"
            )
            return np.array(result.embeddings[0], dtype=np.float32)

        def _on_retry(attempt, exc):
            self._stats.retry_count += 1

        try:
            # Circuit breaker prevents hammering a down service
            async with voyage_breaker:
                result = await retry_with_backoff(
                    _do_embed,
                    self._retry_config,
                    _on_retry
                )
            self._stats.successful_requests += 1
            return result
        except CircuitOpenError as e:
            self._stats.failed_requests += 1
            logger.warning(f"Voyage API circuit open, failing fast: {e}")
            raise EmbeddingError(f"Voyage API unavailable (circuit open): {e}") from e
        except Exception as e:
            self._stats.failed_requests += 1
            raise EmbeddingError(f"Failed to embed text: {e}") from e
    
    async def embed_batch(
        self,
        texts: List[str],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[np.ndarray]:
        """
        Embed multiple texts with retry and circuit breaker protection.

        Voyage supports up to 128 texts per request.
        """
        if not texts:
            return []

        BATCH_SIZE = 128
        all_embeddings = []
        total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_idx, i in enumerate(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i:i + BATCH_SIZE]
            self._stats.total_requests += 1

            # Create closure with batch captured
            current_batch = batch  # Capture for closure

            async def _do_batch():
                result = await self._client.embed(
                    texts=current_batch,
                    model=self._model,
                    input_type="document"
                )
                return [
                    np.array(e, dtype=np.float32)
                    for e in result.embeddings
                ]

            def _on_retry(attempt, exc):
                self._stats.retry_count += 1

            try:
                # Circuit breaker prevents hammering a down service
                async with voyage_breaker:
                    embeddings = await retry_with_backoff(
                        _do_batch,
                        self._retry_config,
                        _on_retry
                    )

                # Validate embedding count matches input count
                # This prevents silent data loss when API returns partial results
                expected_count = len(current_batch)
                actual_count = len(embeddings)

                if actual_count != expected_count:
                    self._stats.partial_responses += 1
                    logger.warning(
                        f"Voyage API returned {actual_count} embeddings for {expected_count} texts "
                        f"in batch {batch_idx}. {'Padding' if actual_count < expected_count else 'Truncating'} to match."
                    )

                    if actual_count < expected_count:
                        # Pad with zero vectors for missing embeddings
                        embeddings.extend([
                            np.zeros(self._dimension, dtype=np.float32)
                            for _ in range(expected_count - actual_count)
                        ])
                    else:
                        # Truncate if API returned too many (unlikely but defensive)
                        embeddings = embeddings[:expected_count]

                all_embeddings.extend(embeddings)
                self._stats.successful_requests += 1

                if on_progress:
                    on_progress(batch_idx + 1, total_batches)

            except CircuitOpenError as e:
                self._stats.failed_requests += 1
                logger.warning(f"Voyage API circuit open at batch {batch_idx}, failing fast: {e}")
                # Return zeros for remaining batches when circuit is open
                remaining_texts = len(texts) - len(all_embeddings)
                all_embeddings.extend([
                    np.zeros(self._dimension, dtype=np.float32)
                    for _ in range(remaining_texts)
                ])
                break  # Don't try more batches when circuit is open

            except Exception as e:
                self._stats.failed_requests += 1
                logger.error(f"Batch {batch_idx} failed: {e}")
                all_embeddings.extend([
                    np.zeros(self._dimension, dtype=np.float32)
                    for _ in batch
                ])

        return all_embeddings


class LocalTextEmbedder(TextEmbedder):
    """
    Local text embeddings using sentence-transformers.
    
    Models:
    - all-MiniLM-L6-v2 (384 dim, fast)
    - all-mpnet-base-v2 (768 dim, better quality)
    - BAAI/bge-large-en-v1.5 (1024 dim, high quality)
    """
    
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        try:
            from sentence_transformers import SentenceTransformer
            from src.core.device_utils import get_optimal_device
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")

        self._model_name = model
        self._device = get_optimal_device(device)
        self._model = SentenceTransformer(model, device=self._device)
        self._dimension = self._model.get_sentence_embedding_dimension()
        self._stats = EmbeddingStats()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def stats(self) -> EmbeddingStats:
        return self._stats
    
    async def embed(self, text: str) -> np.ndarray:
        """Embed single text."""
        self._stats.total_requests += 1
        
        loop = asyncio.get_event_loop()
        try:
            embedding = await loop.run_in_executor(
                None,
                lambda: self._model.encode(text, convert_to_numpy=True)
            )
            self._stats.successful_requests += 1
            return embedding.astype(np.float32)
        except Exception as e:
            self._stats.failed_requests += 1
            raise EmbeddingError(f"Failed to embed text: {e}") from e
    
    async def embed_batch(
        self,
        texts: List[str],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[np.ndarray]:
        """Embed multiple texts."""
        if not texts:
            return []
        
        self._stats.total_requests += 1
        
        loop = asyncio.get_event_loop()
        try:
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            )
            self._stats.successful_requests += 1
            
            if on_progress:
                on_progress(1, 1)
            
            return [e.astype(np.float32) for e in embeddings]
        except Exception as e:
            self._stats.failed_requests += 1
            raise EmbeddingError(f"Failed to embed batch: {e}") from e


# =============================================================================
# IMAGE EMBEDDER IMPLEMENTATIONS
# =============================================================================

class CLIPImageEmbedder(ImageEmbedder):
    """
    OpenAI CLIP for image embeddings.
    
    Good for general images, not domain-specific.
    """
    
    def __init__(
        self,
        model: str = "ViT-L/14",
        device: Optional[str] = None
    ):
        try:
            import clip
            import torch
            from src.core.device_utils import get_optimal_device
        except ImportError:
            raise ImportError("clip and torch required: pip install git+https://github.com/openai/CLIP.git torch")

        self._model_name = model
        self._device = get_optimal_device(device)
        self._model, self._preprocess = clip.load(model, device=self._device)
        self._dimension = 768 if "ViT-L" in model else 512
        self._stats = EmbeddingStats()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return f"CLIP-{self._model_name}"
    
    @property
    def stats(self) -> EmbeddingStats:
        return self._stats
    
    async def embed(self, image_path: Path) -> np.ndarray:
        """Embed single image."""
        embeddings = await self.embed_batch([image_path])
        return embeddings[0]
    
    async def embed_batch(
        self,
        image_paths: List[Path],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[np.ndarray]:
        """Embed multiple images."""
        if not image_paths:
            return []
        
        import torch
        from PIL import Image
        
        self._stats.total_requests += 1
        
        def _process():
            images = []
            valid_indices = []
            
            for i, path in enumerate(image_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self._preprocess(img))
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
                    continue
            
            if not images:
                return [np.zeros(self._dimension, dtype=np.float32) for _ in image_paths]
            
            image_input = torch.stack(images).to(self._device)
            
            with torch.no_grad():
                features = self._model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
            
            # Map back to original indices
            results = [np.zeros(self._dimension, dtype=np.float32) for _ in image_paths]
            for i, idx in enumerate(valid_indices):
                results[idx] = features[i].cpu().numpy().astype(np.float32)
            
            return results
        
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, _process)
            self._stats.successful_requests += 1
            
            if on_progress:
                on_progress(1, 1)
            
            return result
        except Exception as e:
            self._stats.failed_requests += 1
            raise EmbeddingError(f"Failed to embed images: {e}") from e


class BiomedCLIPImageEmbedder(ImageEmbedder):
    """
    BiomedCLIP for medical image embeddings.
    
    Trained on medical images - better for surgical photos, scans.
    """
    
    def __init__(self, device: Optional[str] = None):
        try:
            from open_clip import create_model_and_transforms
            import torch
            from src.core.device_utils import get_optimal_device
        except ImportError:
            raise ImportError("open_clip and torch required: pip install open-clip-torch torch")

        self._device = get_optimal_device(device)
        self._model, _, self._preprocess = create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self._model = self._model.to(self._device)
        self._model.eval()
        self._dimension = 512
        self._stats = EmbeddingStats()
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return "BiomedCLIP"
    
    @property
    def stats(self) -> EmbeddingStats:
        return self._stats
    
    async def embed(self, image_path: Path) -> np.ndarray:
        """Embed single image."""
        embeddings = await self.embed_batch([image_path])
        return embeddings[0]
    
    async def embed_batch(
        self,
        image_paths: List[Path],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[np.ndarray]:
        """Embed multiple images."""
        if not image_paths:
            return []
        
        import torch
        from PIL import Image
        
        self._stats.total_requests += 1
        
        def _process():
            images = []
            valid_indices = []
            
            for i, path in enumerate(image_paths):
                try:
                    img = Image.open(path).convert("RGB")
                    images.append(self._preprocess(img))
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")
                    continue
            
            if not images:
                return [np.zeros(self._dimension, dtype=np.float32) for _ in image_paths]
            
            image_input = torch.stack(images).to(self._device)
            
            with torch.no_grad():
                features = self._model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
            
            results = [np.zeros(self._dimension, dtype=np.float32) for _ in image_paths]
            for i, idx in enumerate(valid_indices):
                results[idx] = features[i].cpu().numpy().astype(np.float32)
            
            return results
        
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, _process)
            self._stats.successful_requests += 1
            
            if on_progress:
                on_progress(1, 1)
            
            return result
        except Exception as e:
            self._stats.failed_requests += 1
            raise EmbeddingError(f"Failed to embed images: {e}") from e


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_text_embedder(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    retry_config: RetryConfig = None
) -> TextEmbedder:
    """
    Factory function to create text embedder.
    
    Args:
        provider: "openai", "voyage", or "local"
        model: Model name (optional, uses default)
        api_key: API key for cloud providers
        retry_config: Retry configuration
        
    Returns:
        TextEmbedder instance
    """
    if provider == "openai":
        return OpenAITextEmbedder(
            model=model or "text-embedding-3-small",
            api_key=api_key,
            retry_config=retry_config
        )
    elif provider == "voyage":
        return VoyageTextEmbedder(
            model=model or "voyage-3",
            api_key=api_key,
            retry_config=retry_config
        )
    elif provider == "local":
        return LocalTextEmbedder(
            model=model or "all-MiniLM-L6-v2"
        )
    else:
        raise ValueError(f"Unknown text embedder provider: {provider}")


class SubprocessBiomedCLIPEmbedder(ImageEmbedder):
    """
    BiomedCLIP via subprocess for memory isolation.

    Prevents segfaults when running SciSpacy + BiomedCLIP in same process.
    Uses subprocess that dies after batch → OS reclaims 100% memory.
    """

    def __init__(self):
        from src.ingest.subprocess_embedder import SubprocessEmbedder, EmbedderConfig

        self._embedder = SubprocessEmbedder(
            config=EmbedderConfig(
                max_retries=2,
                timeout_seconds=300,
                batch_size=10,
            )
        )
        self._dimension = 512
        self._stats = EmbeddingStats()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "BiomedCLIP-Subprocess"

    @property
    def stats(self) -> EmbeddingStats:
        return self._stats

    async def embed(self, image_path: Path) -> np.ndarray:
        """Embed single image."""
        embeddings = await self.embed_batch([image_path])
        return embeddings[0]

    async def embed_batch(
        self,
        image_paths: List[Path],
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[np.ndarray]:
        """Embed multiple images via subprocess."""
        import asyncio

        if not image_paths:
            return []

        self._stats.total_requests += 1
        start_time = asyncio.get_event_loop().time()

        try:
            # Convert to dicts with image bytes
            images = []
            for path in image_paths:
                path = Path(path)
                if path.exists():
                    with open(path, 'rb') as f:
                        images.append({
                            'image_bytes': f.read(),
                            'ext': path.suffix.lstrip('.'),
                            'path': str(path),
                        })

            if not images:
                return [np.zeros(self._dimension, dtype=np.float32) for _ in image_paths]

            # Run in thread pool (subprocess_embedder is sync)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._embedder.embed_images,
                images
            )

            # Extract embeddings
            embeddings = []
            for img in result:
                emb = img.get('embedding')
                if emb:
                    embeddings.append(np.array(emb, dtype=np.float32))
                else:
                    embeddings.append(np.zeros(self._dimension, dtype=np.float32))

            self._stats.total_tokens += len(embeddings)
            self._stats.total_latency_seconds += asyncio.get_event_loop().time() - start_time

            return embeddings

        except Exception as e:
            self._stats.failed_requests += 1
            raise EmbeddingError(f"Subprocess embedding failed: {e}") from e


def create_image_embedder(
    provider: str = "clip",
    model: Optional[str] = None,
    device: Optional[str] = None,
    use_subprocess: bool = False
) -> ImageEmbedder:
    """
    Factory function to create image embedder.

    Args:
        provider: "clip" or "biomedclip"
        model: Model name (optional, uses default)
        device: Device to use ("cuda" or "cpu")
        use_subprocess: Use subprocess isolation for biomedclip (recommended)

    Returns:
        ImageEmbedder instance
    """
    if provider == "clip":
        return CLIPImageEmbedder(
            model=model or "ViT-L/14",
            device=device
        )
    elif provider == "biomedclip":
        if use_subprocess:
            return SubprocessBiomedCLIPEmbedder()
        return BiomedCLIPImageEmbedder(device=device)
    elif provider == "biomedclip-subprocess":
        return SubprocessBiomedCLIPEmbedder()
    else:
        raise ValueError(f"Unknown image embedder provider: {provider}")


# =============================================================================
# CIRCUIT BREAKER HEALTH
# =============================================================================

def get_embedding_circuit_health() -> dict:
    """
    Get health status of embedding circuit breakers.

    Use in health endpoint:
        from src.ingest.embeddings import get_embedding_circuit_health
        @router.get("/health")
        async def health():
            return {
                "status": "healthy",
                "embedding_circuits": get_embedding_circuit_health()
            }
    """
    from src.utils.circuit_breaker import get_circuit_health
    return get_circuit_health({
        "voyage": voyage_breaker,
        "openai_embeddings": openai_embeddings_breaker,
    })
