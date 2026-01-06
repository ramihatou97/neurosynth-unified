"""
NeuroSynth 2.0 - Dependency Injection
=====================================

FastAPI dependency factories for surgical reasoning components.
Follows the NPRSS pattern for isolated module dependencies.

Usage in routes:
    from src.neurosynth2.dependencies import (
        get_clinical_reasoner,
        get_surgical_simulator
    )

    @router.post("/simulate")
    async def simulate(
        simulator = Depends(get_surgical_simulator)
    ):
        ...
"""

import logging
from typing import Optional
from asyncpg import Pool

logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL SINGLETONS
# =============================================================================

_db_pool: Optional[Pool] = None
_reasoner = None
_simulator = None


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def initialize_ns2_dependencies(db_pool: Pool):
    """
    Initialize NeuroSynth 2.0 dependencies.

    Called by src/api/main.py during startup.
    Injects the main application's database pool into this module.

    Args:
        db_pool: AsyncPG connection pool from the main application
    """
    global _db_pool
    _db_pool = db_pool
    logger.info("NeuroSynth 2.0: Database pool injected.")


def get_ns2_db_pool() -> Pool:
    """
    Get the injected database connection pool.

    Returns:
        AsyncPG connection pool

    Raises:
        RuntimeError: If dependencies not initialized
    """
    if _db_pool is None:
        raise RuntimeError(
            "NeuroSynth 2.0 is not initialized. "
            "Call initialize_ns2_dependencies() at application startup."
        )
    return _db_pool


# =============================================================================
# SERVICE DEPENDENCIES (Lazy-loaded Singletons)
# =============================================================================

async def get_clinical_reasoner():
    """
    Get or create ClinicalReasoner singleton.

    Lazy-loads the reasoner on first access to avoid startup delays.

    Returns:
        ClinicalReasoner instance (initialized)
    """
    global _reasoner
    if _reasoner is None:
        from src.neurosynth2.reasoning.clinical_reasoner import ClinicalReasoner
        logger.info("NeuroSynth 2.0: Initializing Clinical Reasoner...")
        _reasoner = ClinicalReasoner(get_ns2_db_pool())
        await _reasoner.initialize()
        logger.info("NeuroSynth 2.0: Clinical Reasoner ready.")
    return _reasoner


async def get_surgical_simulator():
    """
    Get or create SurgicalSimulator singleton.

    Lazy-loads the simulator on first access. Depends on ClinicalReasoner.

    Returns:
        SurgicalSimulator instance (initialized)
    """
    global _simulator
    if _simulator is None:
        from src.neurosynth2.reasoning.surgical_simulator import SurgicalSimulator
        reasoner = await get_clinical_reasoner()
        logger.info("NeuroSynth 2.0: Initializing Surgical Simulator...")
        _simulator = SurgicalSimulator(reasoner, get_ns2_db_pool())
        await _simulator.initialize()
        logger.info("NeuroSynth 2.0: Surgical Simulator ready.")
    return _simulator


# =============================================================================
# OPTIONAL DEPENDENCIES (Graceful Degradation)
# =============================================================================

async def get_anatomical_extractor():
    """
    Get AnatomicalExtractor for vision processing.

    Returns None if vision dependencies (torch, monai) not available.

    Returns:
        AnatomicalExtractor instance or None
    """
    try:
        from src.neurosynth2.vision.anatomical_extractor import AnatomicalExtractor
        return AnatomicalExtractor()
    except ImportError as e:
        logger.warning(f"NeuroSynth 2.0: Vision module not available: {e}")
        return None


async def get_neuro_gat():
    """
    Get NeuroGAT graph neural network.

    Returns None if GNN dependencies (torch, torch_geometric) not available.

    Returns:
        NeuroGAT instance or None
    """
    try:
        from src.neurosynth2.gnn.neuro_gat import NeuroGAT
        return NeuroGAT()
    except ImportError as e:
        logger.warning(f"NeuroSynth 2.0: GNN module not available: {e}")
        return None


# =============================================================================
# CLEANUP
# =============================================================================

async def cleanup_ns2_dependencies():
    """
    Cleanup NeuroSynth 2.0 dependencies.

    Called during application shutdown.
    """
    global _reasoner, _simulator
    logger.info("NeuroSynth 2.0: Cleaning up dependencies...")
    _reasoner = None
    _simulator = None
