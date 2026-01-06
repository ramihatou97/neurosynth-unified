"""
NeuroSynth 2.0 - Reasoning Module
==================================

Physics-aware surgical reasoning and simulation.

This module provides:
- ClinicalReasoner: Rule-based deductive reasoning engine
- SurgicalSimulator: Step-by-step surgical simulation with dynamic state
- PhysicsBootstrapper: Automated population of anatomical physics data

Core concepts:
- EntityPhysics: Physical properties of anatomical structures
- ClinicalPrinciple: IF-THEN axioms for surgical reasoning
- PatientState: Dynamic state tracking during simulation
- RiskAssessment: Evaluated risks with mechanisms and mitigations

Usage:
    from src.reasoning import ClinicalReasoner, SurgicalSimulator
    
    # Initialize reasoner
    reasoner = ClinicalReasoner(database)
    await reasoner.initialize()
    
    # Assess risks
    risks = await reasoner.assess_action(
        action="retract",
        structure="temporal_lobe",
        context=SurgicalContext(approach="pterional", ...)
    )
    
    # Run simulation
    simulator = SurgicalSimulator(reasoner, database)
    await simulator.initialize()
    
    result = await simulator.simulate_approach(
        approach="pterional",
        target="MCA_aneurysm"
    )

Schema Requirements:
    Run migration 007_physics_schema.sql to create required tables:
    - anatomical_entities
    - surgical_corridors
    - clinical_principles
    - causal_edges
    - simulation_sessions
"""

from src.neurosynth2.reasoning.models import (
    # Enums
    RiskLevel,
    PatientCondition,
    SimulationVerdict,
    ComplicationType,
    SurgicalAction,
    
    # Core models
    EntityPhysics,
    RiskAssessment,
    SurgicalContext,
    PatientState,
    SimulationStep,
    SimulationResult,
    SurgicalCorridor,
    ClinicalPrinciple,
)

from src.neurosynth2.reasoning.clinical_reasoner import (
    ClinicalReasoner,
    BuiltInRules,
)

from src.neurosynth2.reasoning.surgical_simulator import (
    SurgicalSimulator,
    ActionInference,
    ComplicationSimulator,
)

from src.neurosynth2.reasoning.physics_bootstrapper import (
    PhysicsBootstrapper,
    ExtractionResult,
    KNOWN_END_ARTERIES,
    KNOWN_TETHERED_STRUCTURES,
    KNOWN_ELOQUENT_STRUCTURES,
)

__all__ = [
    # Enums
    "RiskLevel",
    "PatientCondition",
    "SimulationVerdict",
    "ComplicationType",
    "SurgicalAction",
    
    # Models
    "EntityPhysics",
    "RiskAssessment",
    "SurgicalContext",
    "PatientState",
    "SimulationStep",
    "SimulationResult",
    "SurgicalCorridor",
    "ClinicalPrinciple",
    
    # Reasoner
    "ClinicalReasoner",
    "BuiltInRules",
    
    # Simulator
    "SurgicalSimulator",
    "ActionInference",
    "ComplicationSimulator",
    
    # Bootstrapper
    "PhysicsBootstrapper",
    "ExtractionResult",
    "KNOWN_END_ARTERIES",
    "KNOWN_TETHERED_STRUCTURES",
    "KNOWN_ELOQUENT_STRUCTURES",
]

__version__ = "2.0.0"
