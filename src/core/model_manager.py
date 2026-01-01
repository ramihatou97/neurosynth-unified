# src/core/model_manager.py
"""
Thread-safe Model Manager with lazy loading, reference counting, and automatic cleanup.
Prevents OOM errors by enforcing memory budgets and ensuring models are unloaded when idle.
"""
import gc
import logging
import threading
import psutil
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelType(Enum):
    SCISPACY = "scispacy"
    BIOMEDCLIP = "biomedclip"
    SENTENCE_TRANSFORMER = "sentence_transformer"


@dataclass
class ModelInfo:
    name: str
    memory_mb: int
    loaded: bool = False
    instance: Any = None
    load_count: int = 0
    ref_count: int = 0  # Track active users to prevent premature unloading


class ModelManager:
    """
    Singleton model manager with lazy loading, reference counting, and automatic cleanup.

    Usage:
        with model_manager.load(ModelType.SCISPACY) as nlp:
            doc = nlp(text)
    """

    _instance = None
    _lock = threading.Lock()

    # Memory budget (MB) - conservative for dev environments
    MEMORY_BUDGET_MB = 4000

    MODEL_CONFIGS = {
        ModelType.SCISPACY: ModelInfo("en_core_sci_lg", memory_mb=400),
        ModelType.BIOMEDCLIP: ModelInfo("BiomedCLIP", memory_mb=2000),
        ModelType.SENTENCE_TRANSFORMER: ModelInfo("all-MiniLM-L6-v2", memory_mb=200),
    }

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        # Deep copy configs to allow state tracking
        self._models: Dict[ModelType, ModelInfo] = {
            k: ModelInfo(v.name, v.memory_mb)
            for k, v in self.MODEL_CONFIGS.items()
        }
        self._current_memory_mb = 0
        self._model_lock = threading.Lock()
        self._initialized = True
        logger.info(f"ModelManager initialized with {self.MEMORY_BUDGET_MB}MB budget")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage stats."""
        process = psutil.Process()
        return {
            "process_rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "managed_models_mb": self._current_memory_mb,
            "memory_budget_mb": self.MEMORY_BUDGET_MB,
            "system_available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
            "loaded_models": [
                {"name": k.value, "ref_count": v.ref_count, "load_count": v.load_count}
                for k, v in self._models.items() if v.loaded
            ],
        }

    def _check_memory_budget(self, needed_mb: int):
        """Ensure we have space for the new model."""
        projected = self._current_memory_mb + needed_mb

        if projected > self.MEMORY_BUDGET_MB:
            logger.warning(
                f"Memory budget exceeded: {projected}MB > {self.MEMORY_BUDGET_MB}MB. "
                f"Attempting to unload unused models."
            )
            self._unload_unused()

            # Re-check after cleanup
            projected = self._current_memory_mb + needed_mb
            if projected > self.MEMORY_BUDGET_MB:
                logger.error(
                    f"Cannot load model: would exceed budget even after cleanup. "
                    f"Current: {self._current_memory_mb}MB, Needed: {needed_mb}MB"
                )
                raise MemoryError(f"Insufficient memory budget for model ({needed_mb}MB needed)")

    def _unload_unused(self):
        """Unload models that are loaded but have 0 active users."""
        for model_type, info in self._models.items():
            if info.loaded and info.ref_count == 0:
                self._unload_unsafe(model_type)

    def _load_model_instance(self, model_type: ModelType) -> Any:
        """Internal method to perform the heavy lifting of loading."""
        logger.info(f"Loading model into memory: {model_type.value}")

        if model_type == ModelType.SCISPACY:
            import spacy
            # Disable unneeded components for speed
            nlp = spacy.load("en_core_sci_lg", disable=["parser"])
            logger.info(f"SciSpacy loaded: {nlp.meta['name']}")
            return nlp

        elif model_type == ModelType.BIOMEDCLIP:
            import open_clip
            import torch
            # Check for CUDA availability inside the loader
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading BiomedCLIP on device: {device}")
            model, preprocess = open_clip.create_model_from_pretrained(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                device=device
            )
            model.eval()
            logger.info("BiomedCLIP loaded successfully")
            return {"model": model, "preprocess": preprocess, "device": device}

        elif model_type == ModelType.SENTENCE_TRANSFORMER:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SentenceTransformer loaded")
            return model

        raise ValueError(f"Unknown model type: {model_type}")

    def _unload_unsafe(self, model_type: ModelType):
        """Unload model without locking (caller must hold lock or ensure safety)."""
        model_info = self._models[model_type]
        if not model_info.loaded:
            return

        logger.info(f"Unloading model: {model_type.value}")

        # Delete the instance
        model_info.instance = None
        model_info.loaded = False
        self._current_memory_mb -= model_info.memory_mb

        # Force garbage collection
        gc.collect()

        # PyTorch-specific cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
        except ImportError:
            pass

        logger.info(f"Model {model_type.value} unloaded. Current memory: {self._current_memory_mb}MB")

    @contextmanager
    def load(self, model_type: ModelType):
        """
        Context manager for thread-safe model loading.
        Increments ref_count on enter, decrements on exit.
        Only unloads if ref_count drops to 0.

        Usage:
            with model_manager.load(ModelType.SCISPACY) as nlp:
                doc = nlp(text)
        """
        with self._model_lock:
            model_info = self._models[model_type]

            # If not loaded, load it
            if not model_info.loaded:
                self._check_memory_budget(model_info.memory_mb)
                try:
                    model_info.instance = self._load_model_instance(model_type)
                    model_info.loaded = True
                    self._current_memory_mb += model_info.memory_mb
                    logger.info(f"Current managed memory: {self._current_memory_mb}MB")
                except Exception as e:
                    logger.error(f"Failed to load {model_type.value}: {e}")
                    raise

            # Increment active user count
            model_info.ref_count += 1
            model_info.load_count += 1

        try:
            yield model_info.instance
        finally:
            with self._model_lock:
                model_info.ref_count -= 1
                # If no one is using it, unload immediately (Aggressive GC strategy)
                if model_info.ref_count <= 0:
                    self._unload_unsafe(model_type)

    def is_loaded(self, model_type: ModelType) -> bool:
        """Check if a model is currently loaded."""
        return self._models[model_type].loaded

    def unload_all(self):
        """Force unload all models regardless of ref_count."""
        with self._model_lock:
            for model_type in self._models:
                self._unload_unsafe(model_type)
                self._models[model_type].ref_count = 0
        logger.info("All models unloaded")


# Global singleton instance
model_manager = ModelManager()
