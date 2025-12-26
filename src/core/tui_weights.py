"""
UMLS Semantic Type (TUI) Weights for Neurosurgical Domain

Higher weights for anatomical and pathological concepts
that are most relevant for neurosurgical literature.

Reference: UMLS Semantic Types
https://www.nlm.nih.gov/research/umls/META3_current_semantic_types.html
"""

from typing import Dict

# =============================================================================
# SEMANTIC TYPE WEIGHTS
# =============================================================================
# Weights range from 0.1 (low importance) to 1.0 (highest importance)
# for neurosurgical domain relevance.

TUI_WEIGHTS: Dict[str, float] = {
    # ─────────────────────────────────────────────────────────────────────────
    # ANATOMICAL (Highest - 1.0)
    # Critical for neurosurgical localization and approach planning
    # ─────────────────────────────────────────────────────────────────────────
    "T023": 1.0,    # Body Part, Organ, or Organ Component
    "T024": 1.0,    # Tissue
    "T025": 0.9,    # Cell
    "T026": 0.85,   # Cell Component
    "T029": 1.0,    # Body Location or Region
    "T030": 0.95,   # Body Space or Junction
    "T031": 0.85,   # Body Substance

    # ─────────────────────────────────────────────────────────────────────────
    # PATHOLOGICAL (Highest - 1.0)
    # Core diagnostic and treatment targets
    # ─────────────────────────────────────────────────────────────────────────
    "T047": 1.0,    # Disease or Syndrome
    "T048": 0.95,   # Mental or Behavioral Dysfunction
    "T049": 0.85,   # Cell or Molecular Dysfunction
    "T046": 0.95,   # Pathologic Function
    "T191": 1.0,    # Neoplastic Process (tumors - critical for neurosurg)
    "T020": 0.9,    # Acquired Abnormality
    "T019": 0.85,   # Congenital Abnormality
    "T190": 0.85,   # Anatomical Abnormality

    # ─────────────────────────────────────────────────────────────────────────
    # PROCEDURAL (High - 0.9+)
    # Surgical techniques and diagnostic procedures
    # ─────────────────────────────────────────────────────────────────────────
    "T061": 0.95,   # Therapeutic or Preventive Procedure
    "T060": 0.9,    # Diagnostic Procedure
    "T059": 0.8,    # Laboratory Procedure
    "T058": 0.8,    # Health Care Activity
    "T063": 0.7,    # Molecular Biology Research Technique

    # ─────────────────────────────────────────────────────────────────────────
    # CLINICAL FINDINGS (High - 0.8+)
    # Signs, symptoms, and examination findings
    # ─────────────────────────────────────────────────────────────────────────
    "T033": 0.85,   # Finding
    "T034": 0.75,   # Laboratory or Test Result
    "T184": 0.85,   # Sign or Symptom
    "T201": 0.7,    # Clinical Attribute
    "T037": 0.8,    # Injury or Poisoning

    # ─────────────────────────────────────────────────────────────────────────
    # PHARMACOLOGICAL (Medium - 0.7)
    # Medications and drugs used in neurosurgery
    # ─────────────────────────────────────────────────────────────────────────
    "T121": 0.75,   # Pharmacologic Substance
    "T200": 0.7,    # Clinical Drug
    "T195": 0.65,   # Antibiotic
    "T109": 0.6,    # Organic Chemical
    "T123": 0.65,   # Biologically Active Substance
    "T116": 0.55,   # Amino Acid, Peptide, or Protein
    "T126": 0.55,   # Enzyme

    # ─────────────────────────────────────────────────────────────────────────
    # DEVICES & EQUIPMENT (Medium - 0.6)
    # Surgical instruments and medical devices
    # ─────────────────────────────────────────────────────────────────────────
    "T074": 0.7,    # Medical Device
    "T075": 0.6,    # Research Device
    "T073": 0.5,    # Manufactured Object

    # ─────────────────────────────────────────────────────────────────────────
    # ORGANISMS (Medium - 0.5)
    # Pathogens relevant to infections
    # ─────────────────────────────────────────────────────────────────────────
    "T007": 0.55,   # Bacterium
    "T005": 0.55,   # Virus
    "T004": 0.5,    # Fungus
    "T204": 0.45,   # Eukaryote

    # ─────────────────────────────────────────────────────────────────────────
    # CONCEPTUAL (Low - 0.3)
    # Abstract concepts, less directly relevant
    # ─────────────────────────────────────────────────────────────────────────
    "T170": 0.3,    # Intellectual Product
    "T062": 0.25,   # Research Activity
    "T169": 0.25,   # Functional Concept
    "T078": 0.2,    # Idea or Concept
    "T080": 0.2,    # Qualitative Concept
    "T081": 0.25,   # Quantitative Concept
    "T089": 0.15,   # Regulation or Law

    # ─────────────────────────────────────────────────────────────────────────
    # GEOGRAPHIC/TEMPORAL (Lowest - 0.1)
    # Rarely relevant for clinical neurosurgery
    # ─────────────────────────────────────────────────────────────────────────
    "T083": 0.1,    # Geographic Area
    "T079": 0.15,   # Temporal Concept
}


# Semantic type names for logging/debugging
TUI_NAMES: Dict[str, str] = {
    # Anatomical
    "T023": "Body Part/Organ",
    "T024": "Tissue",
    "T025": "Cell",
    "T029": "Body Location",
    "T030": "Body Space/Junction",
    "T031": "Body Substance",

    # Pathological
    "T047": "Disease/Syndrome",
    "T048": "Mental Dysfunction",
    "T191": "Neoplasm",
    "T046": "Pathologic Function",
    "T020": "Acquired Abnormality",
    "T019": "Congenital Abnormality",

    # Procedural
    "T061": "Therapeutic Procedure",
    "T060": "Diagnostic Procedure",
    "T059": "Laboratory Procedure",

    # Clinical
    "T033": "Finding",
    "T034": "Lab Result",
    "T184": "Sign/Symptom",
    "T037": "Injury/Poisoning",

    # Pharmacological
    "T121": "Pharmacologic Substance",
    "T200": "Clinical Drug",

    # Devices
    "T074": "Medical Device",
}


def get_tui_weight(tui: str) -> float:
    """
    Get weight for a TUI, defaulting to 0.5 for unknown types.

    Args:
        tui: Semantic Type ID (e.g., "T047")

    Returns:
        Weight between 0.0 and 1.0
    """
    return TUI_WEIGHTS.get(tui, 0.5)


def get_tui_name(tui: str) -> str:
    """
    Get human-readable name for a TUI.

    Args:
        tui: Semantic Type ID (e.g., "T047")

    Returns:
        Human-readable name or the TUI itself if unknown
    """
    return TUI_NAMES.get(tui, tui)


def is_anatomical_tui(tui: str) -> bool:
    """Check if TUI represents an anatomical concept."""
    return tui in {"T023", "T024", "T025", "T026", "T029", "T030", "T031"}


def is_pathological_tui(tui: str) -> bool:
    """Check if TUI represents a pathological concept."""
    return tui in {"T047", "T048", "T049", "T046", "T191", "T020", "T019", "T190"}


def is_procedural_tui(tui: str) -> bool:
    """Check if TUI represents a procedural concept."""
    return tui in {"T061", "T060", "T059", "T058", "T063"}
