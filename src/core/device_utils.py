"""Device selection utilities for PyTorch models."""

import torch
from typing import Optional


def get_optimal_device(device: Optional[str] = None) -> str:
    """
    Get optimal PyTorch device with MPS, CUDA, and CPU fallback.

    Priority order:
    1. Explicitly specified device (if provided)
    2. MPS (Apple Silicon GPU) if available
    3. CUDA (NVIDIA GPU) if available
    4. CPU (fallback)

    Args:
        device: Optional device override ("mps", "cuda", "cpu", "cuda:0", etc.)
               If None, auto-detects best available device.

    Returns:
        Device string: "mps", "cuda", "cuda:0", or "cpu"

    Examples:
        >>> get_optimal_device()  # Auto-detect
        'mps'  # on Apple Silicon Mac

        >>> get_optimal_device("cpu")  # Force CPU
        'cpu'

        >>> get_optimal_device("cuda:1")  # Specific GPU
        'cuda:1'
    """
    # Explicit device specified
    if device is not None and device != "auto":
        return device

    # Check MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"

    # Check CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return "cuda"

    # Fallback to CPU
    return "cpu"


def print_device_info(device: str) -> None:
    """Print information about the selected device."""
    if device == "mps":
        print("Using Apple Silicon GPU (MPS)")
    elif device.startswith("cuda"):
        gpu_id = int(device.split(":")[1]) if ":" in device else 0
        if torch.cuda.is_available():
            print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(gpu_id)}")
    elif device == "cpu":
        print("Using CPU (no GPU acceleration)")
