"""
NeuroSynth 2.0 - Surgical Reasoning Engine
==========================================

A physics-aware surgical reasoning and simulation module that provides:
- Clinical risk assessment based on anatomical principles
- Step-by-step surgical simulation with state tracking
- Anatomical extraction from medical imaging (MRI/CTA)
- Graph neural networks for implicit pattern learning

This module is designed as an isolated "plugin" to the main NeuroSynth
application, sharing the same database connection but maintaining
separate code organization for safety.

Usage:
    from src.neurosynth2.dependencies import (
        initialize_ns2_dependencies,
        get_clinical_reasoner,
        get_surgical_simulator
    )
"""

__version__ = "2.0.0"
__all__ = [
    "dependencies",
    "reasoning",
    "vision",
    "gnn",
]
