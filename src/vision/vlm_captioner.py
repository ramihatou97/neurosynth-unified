"""
NeuroSynth Phase 1 - VLM Image Captioner

Claude Vision integration for generating medical image captions.

Features:
- Robust retry with exponential backoff
- Timeout handling
- Failure metrics tracking
- Batch rate limiting
- Context-aware prompting
"""

import asyncio
import base64
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from enum import Enum
import time

logger = logging.getLogger("neurosynth.vision.vlm")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class VLMError(Exception):
    """Base VLM error."""
    pass


class VLMTimeoutError(VLMError):
    """VLM request timed out."""
    pass


class VLMRateLimitError(VLMError):
    """VLM rate limit exceeded."""
    pass


class VLMAPIError(VLMError):
    """VLM API error."""
    pass


class FailureType(Enum):
    """Types of VLM failures."""
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    CONNECTION = "connection"
    API_ERROR = "api_error"
    INVALID_IMAGE = "invalid_image"
    UNKNOWN = "unknown"


@dataclass
class VLMConfig:
    """VLM captioner configuration."""
    
    # Model
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 500
    
    # Timeouts (seconds)
    request_timeout: float = 60.0
    total_timeout: float = 120.0
    
    # Retry settings
    max_retries: int = 3
    initial_delay: float = 2.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    
    # Rate limiting
    min_request_interval: float = 0.5  # seconds between requests
    
    # Image settings
    max_image_size: int = 5 * 1024 * 1024  # 5MB
    supported_formats: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".gif", ".webp")


@dataclass
class VLMStats:
    """VLM usage statistics."""
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    retried: int = 0
    
    # Failure breakdown
    failures_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Timing
    total_time_seconds: float = 0.0
    avg_latency_seconds: float = 0.0
    
    # Tokens
    total_output_tokens: int = 0
    
    def record_success(self, latency: float, tokens: int = 0):
        """Record a successful request."""
        self.total_requests += 1
        self.successful += 1
        self.total_time_seconds += latency
        self.total_output_tokens += tokens
        self._update_avg_latency()
    
    def record_failure(self, failure_type: FailureType, retried: bool = False):
        """Record a failed request."""
        self.total_requests += 1
        self.failed += 1
        if retried:
            self.retried += 1
        
        type_str = failure_type.value
        self.failures_by_type[type_str] = self.failures_by_type.get(type_str, 0) + 1
    
    def _update_avg_latency(self):
        """Update average latency."""
        if self.successful > 0:
            self.avg_latency_seconds = self.total_time_seconds / self.successful
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful / self.total_requests) * 100
    
    def to_dict(self) -> Dict:
        """Export as dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "retried": self.retried,
            "success_rate_pct": round(self.get_success_rate(), 1),
            "failures_by_type": dict(self.failures_by_type),
            "avg_latency_seconds": round(self.avg_latency_seconds, 2),
            "total_output_tokens": self.total_output_tokens
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

MEDICAL_CAPTION_PROMPT = """You are a neurosurgical imaging expert. Describe this medical image with precision.

Include:
1. Image type (MRI, CT, surgical photo, diagram, histology, etc.)
2. Anatomical structures visible
3. Key findings or abnormalities (if any)
4. Surgical relevance (approach, technique, landmarks)

Context from surrounding text:
{context}

Provide a concise, accurate caption (2-4 sentences) suitable for a medical textbook.
Focus on what a neurosurgeon would find clinically relevant."""

ANATOMICAL_DIAGRAM_PROMPT = """Describe this anatomical diagram for a neurosurgical reference.

Include:
1. Region/structures depicted
2. Labeled anatomical landmarks
3. Surgical corridors or approaches shown
4. Relationships between structures

Context: {context}

Provide a precise caption (2-4 sentences) emphasizing surgical anatomy."""


# ═══════════════════════════════════════════════════════════════════════════════
# VLM CAPTIONER
# ═══════════════════════════════════════════════════════════════════════════════

class VLMCaptioner:
    """
    Claude Vision captioner for medical images.
    
    Features:
    - Robust retry with exponential backoff
    - Timeout handling
    - Failure metrics
    - Rate limiting
    """
    
    def __init__(self, config: VLMConfig = None, api_key: str = None):
        """
        Initialize VLM captioner.
        
        Args:
            config: VLM configuration
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env if not provided)
        """
        self.config = config or VLMConfig()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        # Initialize client lazily
        self._client = None
        
        # Statistics
        self.stats = VLMStats()
        
        # Rate limiting
        self._last_request_time = 0.0
        
        logger.info(f"VLM captioner initialized: model={self.config.model}")
    
    @property
    def client(self):
        """Lazy-init Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        return self._client
    
    async def caption(
        self,
        image_path: Path = None,
        image_bytes: bytes = None,
        context: str = "",
        prompt_type: str = "medical"
    ) -> str:
        """
        Generate caption for a medical image.
        
        Args:
            image_path: Path to image file
            image_bytes: Raw image bytes
            context: Surrounding text context
            prompt_type: "medical" or "diagram"
        
        Returns:
            Generated caption string
        
        Raises:
            VLMError: On unrecoverable failure
        """
        # Validate input
        if image_path is None and image_bytes is None:
            raise ValueError("Either image_path or image_bytes required")
        
        # Load image if path provided
        if image_path:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Check file size
            if image_path.stat().st_size > self.config.max_image_size:
                logger.warning(f"Image too large: {image_path} ({image_path.stat().st_size} bytes)")
                return ""
            
            # Check format
            if image_path.suffix.lower() not in self.config.supported_formats:
                logger.warning(f"Unsupported format: {image_path.suffix}")
                return ""
            
            image_bytes = image_path.read_bytes()
        
        # Determine media type
        media_type = self._detect_media_type(image_bytes, image_path)
        
        # Select prompt
        if prompt_type == "diagram":
            prompt = ANATOMICAL_DIAGRAM_PROMPT.format(context=context[:500] if context else "None")
        else:
            prompt = MEDICAL_CAPTION_PROMPT.format(context=context[:500] if context else "None")
        
        # Execute with retry
        return await self._execute_with_retry(image_bytes, media_type, prompt)
    
    async def _execute_with_retry(
        self,
        image_bytes: bytes,
        media_type: str,
        prompt: str
    ) -> str:
        """Execute request with retry logic."""
        
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Rate limiting
                await self._rate_limit()
                
                # Execute request
                t0 = time.time()
                result = await self._make_request(image_bytes, media_type, prompt)
                latency = time.time() - t0
                
                # Record success
                self.stats.record_success(latency)
                
                return result
                
            except VLMRateLimitError as e:
                last_error = e
                failure_type = FailureType.RATE_LIMIT
                
            except VLMTimeoutError as e:
                last_error = e
                failure_type = FailureType.TIMEOUT
                
            except ConnectionError as e:
                last_error = e
                failure_type = FailureType.CONNECTION
                
            except VLMAPIError as e:
                last_error = e
                failure_type = FailureType.API_ERROR
                
            except Exception as e:
                last_error = e
                failure_type = FailureType.UNKNOWN
            
            # Should we retry?
            if attempt < self.config.max_retries:
                delay = min(
                    self.config.initial_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )
                
                logger.warning(
                    f"VLM request failed (attempt {attempt + 1}/{self.config.max_retries + 1}): "
                    f"{failure_type.value} - {last_error}. Retrying in {delay:.1f}s..."
                )
                
                self.stats.record_failure(failure_type, retried=True)
                await asyncio.sleep(delay)
            else:
                # Final failure
                self.stats.record_failure(failure_type, retried=False)
                logger.error(f"VLM request failed after {self.config.max_retries + 1} attempts: {last_error}")
        
        return ""  # Return empty caption on failure
    
    async def _make_request(
        self,
        image_bytes: bytes,
        media_type: str,
        prompt: str
    ) -> str:
        """Make single VLM request."""
        
        # Encode image
        image_b64 = base64.standard_b64encode(image_bytes).decode()
        
        # Build message
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
        
        # Execute with timeout
        try:
            async with asyncio.timeout(self.config.request_timeout):
                loop = asyncio.get_event_loop()
                
                def _call():
                    return self.client.messages.create(
                        model=self.config.model,
                        max_tokens=self.config.max_tokens,
                        messages=[message]
                    )
                
                response = await loop.run_in_executor(None, _call)
                
        except asyncio.TimeoutError:
            raise VLMTimeoutError(f"Request timed out after {self.config.request_timeout}s")
        
        except Exception as e:
            error_str = str(e).lower()
            
            if "rate" in error_str or "429" in error_str:
                raise VLMRateLimitError(str(e))
            elif "connection" in error_str or "network" in error_str:
                raise ConnectionError(str(e))
            else:
                raise VLMAPIError(str(e))
        
        # Extract text
        if response.content and len(response.content) > 0:
            return response.content[0].text
        
        return ""
    
    async def _rate_limit(self):
        """Enforce minimum interval between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        
        if elapsed < self.config.min_request_interval:
            wait = self.config.min_request_interval - elapsed
            await asyncio.sleep(wait)
        
        self._last_request_time = time.time()
    
    def _detect_media_type(self, image_bytes: bytes, image_path: Path = None) -> str:
        """Detect image media type."""
        
        # Check magic bytes first
        if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        elif image_bytes[:2] == b'\xff\xd8':
            return "image/jpeg"
        elif image_bytes[:6] in (b'GIF87a', b'GIF89a'):
            return "image/gif"
        elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
            return "image/webp"
        
        # Fall back to extension
        if image_path:
            ext = image_path.suffix.lower()
            if ext in ('.jpg', '.jpeg'):
                return "image/jpeg"
            elif ext == '.png':
                return "image/png"
            elif ext == '.gif':
                return "image/gif"
            elif ext == '.webp':
                return "image/webp"
        
        # Default to JPEG
        return "image/jpeg"
    
    def get_stats(self) -> Dict:
        """Get captioning statistics."""
        return self.stats.to_dict()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = VLMStats()


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH CAPTIONER
# ═══════════════════════════════════════════════════════════════════════════════

class BatchVLMCaptioner:
    """
    Batch VLM captioner with concurrency control.
    
    Processes multiple images with controlled parallelism.
    """
    
    def __init__(
        self,
        captioner: VLMCaptioner = None,
        max_concurrent: int = 3,
        progress_callback: Callable[[int, int], None] = None
    ):
        """
        Initialize batch captioner.
        
        Args:
            captioner: VLM captioner instance
            max_concurrent: Maximum concurrent requests
            progress_callback: Called with (completed, total) after each image
        """
        self.captioner = captioner or VLMCaptioner()
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback
        
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def caption_batch(
        self,
        images: List[Tuple[Path, str]],  # (path, context) pairs
        prompt_type: str = "medical"
    ) -> List[Tuple[Path, str]]:
        """
        Caption multiple images.
        
        Args:
            images: List of (image_path, context) tuples
            prompt_type: Prompt type for all images
        
        Returns:
            List of (image_path, caption) tuples
        """
        completed = 0
        total = len(images)
        results = []
        
        async def process_one(path: Path, context: str) -> Tuple[Path, str]:
            nonlocal completed
            
            async with self._semaphore:
                caption = await self.captioner.caption(
                    image_path=path,
                    context=context,
                    prompt_type=prompt_type
                )
                
                completed += 1
                if self.progress_callback:
                    self.progress_callback(completed, total)
                
                return (path, caption)
        
        # Process all images
        tasks = [process_one(path, ctx) for path, ctx in images]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        final_results = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Batch caption error: {r}")
                final_results.append((None, ""))
            else:
                final_results.append(r)
        
        return final_results
    
    def get_stats(self) -> Dict:
        """Get captioner statistics."""
        return self.captioner.get_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def create_vlm_captioner(
    model: str = None,
    api_key: str = None,
    **kwargs
) -> VLMCaptioner:
    """
    Factory function for VLM captioner.
    
    Args:
        model: Model name (default: claude-sonnet-4-20250514)
        api_key: API key (default: from environment)
        **kwargs: Additional VLMConfig options
    
    Returns:
        Configured VLMCaptioner
    """
    config = VLMConfig(
        model=model or "claude-sonnet-4-20250514",
        **kwargs
    )
    
    return VLMCaptioner(config=config, api_key=api_key)
