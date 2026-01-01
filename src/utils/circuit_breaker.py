"""
Circuit Breaker Pattern for External Services
==============================================

Provides resilience for external API calls (Voyage, Claude).
Prevents cascade failures by failing fast when services are unavailable.

Usage:
    breaker = CircuitBreaker(name="claude", failure_threshold=3)

    async with breaker:
        result = await call_claude_api()
"""

import asyncio
import time
import logging
from typing import Optional, Callable, Any, TypeVar
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# CIRCUIT BREAKER STATES
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Failing, reject requests immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


# =============================================================================
# CIRCUIT BREAKER IMPLEMENTATION
# =============================================================================

class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    States:
    - CLOSED: Normal operation
    - OPEN: Service failing, reject calls immediately
    - HALF_OPEN: Testing recovery, allow limited calls
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0,
        reset_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Service name for logging
            failure_threshold: Failures before opening circuit
            success_threshold: Successes in half-open before closing
            timeout: Request timeout in seconds
            reset_timeout: Time before attempting recovery
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for automatic transitions."""
        if self._state == CircuitState.OPEN:
            # Check if reset timeout has passed
            if self._stats.last_failure_time:
                elapsed = time.time() - self._stats.last_failure_time
                if elapsed >= self.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN (reset timeout)")

        return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def stats(self) -> CircuitStats:
        return self._stats

    async def __aenter__(self):
        """Check if call should proceed."""
        async with self._lock:
            state = self.state

            if state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                raise CircuitOpenError(
                    f"Circuit {self.name} is OPEN. "
                    f"Service unavailable, try again in {self.reset_timeout}s"
                )

            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self._stats.rejected_calls += 1
                    raise CircuitOpenError(
                        f"Circuit {self.name} is HALF_OPEN. "
                        f"Max test calls ({self.half_open_max_calls}) reached."
                    )
                self._half_open_calls += 1

            self._stats.total_calls += 1

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Record result and update state."""
        async with self._lock:
            if exc_type is None:
                # Success
                await self._on_success()
            else:
                # Failure
                await self._on_failure(exc_val)

        return False  # Don't suppress exception

    async def _on_success(self):
        """Handle successful call."""
        self._stats.successful_calls += 1
        self._stats.last_success_time = time.time()
        self._stats.consecutive_successes += 1
        self._stats.consecutive_failures = 0

        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.success_threshold:
                self._state = CircuitState.CLOSED
                logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED (recovered)")

    async def _on_failure(self, error: Exception):
        """Handle failed call."""
        self._stats.failed_calls += 1
        self._stats.last_failure_time = time.time()
        self._stats.consecutive_failures += 1
        self._stats.consecutive_successes = 0

        logger.warning(f"Circuit {self.name}: failure #{self._stats.consecutive_failures}: {error}")

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens the circuit
            self._state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN (failure during test)")

        elif self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit {self.name}: CLOSED -> OPEN (threshold reached)")

    def reset(self):
        """Manually reset circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_calls = 0
        logger.info(f"Circuit {self.name}: manually reset to CLOSED")


class CircuitOpenError(Exception):
    """Raised when circuit is open and rejecting calls."""
    pass


# =============================================================================
# RETRY WITH BACKOFF
# =============================================================================

async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential: bool = True,
    retryable_exceptions: tuple = (Exception,)
) -> Any:
    """
    Retry async function with exponential backoff.

    Args:
        func: Async function to call
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        exponential: Use exponential backoff
        retryable_exceptions: Exceptions that trigger retry

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()

        except retryable_exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(f"All {max_retries + 1} attempts failed: {e}")
                raise

            # Calculate delay
            if exponential:
                delay = min(base_delay * (2 ** attempt), max_delay)
            else:
                delay = base_delay

            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s")
            await asyncio.sleep(delay)

    raise last_exception


# =============================================================================
# DECORATOR VERSION
# =============================================================================

def with_circuit_breaker(
    breaker: CircuitBreaker,
    fallback: Callable = None,
    timeout: float = None
):
    """
    Decorator to wrap async function with circuit breaker.

    Usage:
        voyage_breaker = CircuitBreaker("voyage")

        @with_circuit_breaker(voyage_breaker, fallback=cached_embedding)
        async def get_embedding(text: str) -> List[float]:
            return await voyage_api.embed(text)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                async with breaker:
                    if timeout:
                        return await asyncio.wait_for(
                            func(*args, **kwargs),
                            timeout=timeout
                        )
                    return await func(*args, **kwargs)

            except CircuitOpenError as e:
                if fallback:
                    logger.warning(f"Circuit open, using fallback for {func.__name__}")
                    return await fallback(*args, **kwargs) if asyncio.iscoroutinefunction(fallback) else fallback(*args, **kwargs)
                raise

            except asyncio.TimeoutError:
                logger.error(f"{func.__name__} timed out after {timeout}s")
                if fallback:
                    return await fallback(*args, **kwargs) if asyncio.iscoroutinefunction(fallback) else fallback(*args, **kwargs)
                raise

        return wrapper
    return decorator


# =============================================================================
# HEALTH CHECK INTEGRATION
# =============================================================================

def get_circuit_health(breakers: dict) -> dict:
    """
    Get health status of all circuit breakers.

    Usage in health endpoint:
        @router.get("/health")
        async def health():
            return {
                "status": "healthy",
                "circuits": get_circuit_health({
                    "claude": rag_engine.claude_breaker,
                    "voyage": embedder.breaker
                })
            }
    """
    return {
        name: {
            "state": breaker.state.value,
            "stats": {
                "total_calls": breaker.stats.total_calls,
                "successful": breaker.stats.successful_calls,
                "failed": breaker.stats.failed_calls,
                "rejected": breaker.stats.rejected_calls,
                "consecutive_failures": breaker.stats.consecutive_failures
            },
            "healthy": breaker.is_closed
        }
        for name, breaker in breakers.items()
    }
