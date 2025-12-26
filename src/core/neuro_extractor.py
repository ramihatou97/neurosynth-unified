"""
NeuroSynth v2.0 - Neurosurgical Entity Extractor (Enhanced)
============================================================

Expert system for extracting and classifying neurosurgical entities.

Key features:
1. 100+ domain-specific regex patterns
2. Ambiguity resolution (C5 bone vs C5 artery)
3. Entity normalization to standard terms
4. Confidence scoring
5. External YAML configuration for maintainability

Enhancements:
- Patterns loadable from external YAML config
- Extended ambiguity resolution
- Extraction caching for performance
- Improved context window
"""

import re
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
import hashlib

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from src.shared.models import NeuroEntity, EntityRelation

logger = logging.getLogger(__name__)

# Default context window size (chars before/after entity)
DEFAULT_CONTEXT_CHARS = 100


# ═══════════════════════════════════════════════════════════════════════════════
# EXPANDED AMBIGUITY RULES - Context-Based Entity Resolution
# ═══════════════════════════════════════════════════════════════════════════════

EXPANDED_AMBIGUITY_RULES = {
    # ─────────────────────────────────────────────────────────────────────────
    # M1 - Motor cortex vs MCA segment
    # ─────────────────────────────────────────────────────────────────────────
    "M1": {
        "contexts": {
            "ANATOMY_VASCULAR_ARTERIAL": [
                "mca", "segment", "artery", "bifurcation", "aneurysm", "occlusion",
                "thrombus", "thrombectomy", "a1", "a2", "m2", "m3", "ica"
            ],
            "ANATOMY_BRAIN_CORTICAL": [
                "cortex", "motor", "gyrus", "precentral", "homunculus", "somatotopic",
                "hand knob", "face", "leg", "primary motor"
            ]
        },
        "default": "ANATOMY_VASCULAR_ARTERIAL"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # DBS - Procedure vs functional target
    # ─────────────────────────────────────────────────────────────────────────
    "DBS": {
        "contexts": {
            "PROCEDURE_ACTION": [
                "placement", "implant", "lead", "electrode", "programming",
                "battery", "generator", "revision", "surgery"
            ],
            "FUNCTIONAL_NEURO": [
                "target", "stn", "gpi", "vim", "parkinson", "tremor", "dystonia",
                "stimulation", "parameters"
            ]
        },
        "default": "FUNCTIONAL_NEURO"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # NPH - Normal pressure hydrocephalus
    # ─────────────────────────────────────────────────────────────────────────
    "NPH": {
        "contexts": {
            "PATHOLOGY_HYDROCEPHALUS": [
                "hydrocephalus", "ventriculomegaly", "gait", "dementia", "urinary",
                "triad", "shunt", "lumbar drain", "tap test", "evan's"
            ]
        },
        "default": "PATHOLOGY_HYDROCEPHALUS"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # CPP - Cerebral perfusion pressure vs Choroid plexus papilloma
    # ─────────────────────────────────────────────────────────────────────────
    "CPP": {
        "contexts": {
            "PHYSIOLOGY_CRITICAL_CARE": [
                "pressure", "icp", "map", "mmhg", "perfusion", "cerebral",
                "goal", "maintain", "above", "below", "icu"
            ],
            "PATHOLOGY_TUMOR_CLASSIFICATION": [
                "tumor", "papilloma", "choroid plexus", "ventricle", "lateral",
                "fourth", "pediatric", "cpc"
            ]
        },
        "default": "PHYSIOLOGY_CRITICAL_CARE"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # STN - Subthalamic nucleus (DBS target)
    # ─────────────────────────────────────────────────────────────────────────
    "STN": {
        "contexts": {
            "FUNCTIONAL_NEURO": [
                "dbs", "target", "parkinson", "stimulation", "lead", "electrode",
                "programming", "tremor", "dyskinesia"
            ],
            "ANATOMY_BRAIN_SUBCORTICAL": [
                "nucleus", "basal ganglia", "anatomical", "boundaries", "zona incerta"
            ]
        },
        "default": "FUNCTIONAL_NEURO"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # GPi - Globus pallidus internus (DBS target)
    # ─────────────────────────────────────────────────────────────────────────
    "GPI": {
        "contexts": {
            "FUNCTIONAL_NEURO": [
                "dbs", "target", "dystonia", "parkinson", "stimulation", "pallidotomy"
            ],
            "ANATOMY_BRAIN_SUBCORTICAL": [
                "nucleus", "basal ganglia", "anatomical", "gpe"
            ]
        },
        "default": "FUNCTIONAL_NEURO"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # VIM - Ventral intermediate nucleus (DBS target for tremor)
    # ─────────────────────────────────────────────────────────────────────────
    "VIM": {
        "contexts": {
            "FUNCTIONAL_NEURO": [
                "dbs", "target", "tremor", "essential tremor", "thalamotomy", "stimulation"
            ],
            "ANATOMY_BRAIN_SUBCORTICAL": [
                "thalamus", "nucleus", "vpl", "vpm"
            ]
        },
        "default": "FUNCTIONAL_NEURO"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # EDH - Epidural hematoma (always trauma)
    # ─────────────────────────────────────────────────────────────────────────
    "EDH": {
        "contexts": {
            "PATHOLOGY_TRAUMA": [
                "hematoma", "epidural", "trauma", "skull fracture", "middle meningeal",
                "lucid interval", "herniation", "evacuation"
            ]
        },
        "default": "PATHOLOGY_TRAUMA"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # SDH - Subdural hematoma (trauma)
    # ─────────────────────────────────────────────────────────────────────────
    "SDH": {
        "contexts": {
            "PATHOLOGY_TRAUMA": [
                "hematoma", "subdural", "trauma", "bridging vein", "chronic", "acute",
                "subacute", "midline shift", "evacuation", "burr hole", "craniectomy"
            ]
        },
        "default": "PATHOLOGY_TRAUMA"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TBI - Traumatic brain injury
    # ─────────────────────────────────────────────────────────────────────────
    "TBI": {
        "contexts": {
            "PATHOLOGY_TRAUMA": [
                "trauma", "injury", "concussion", "contusion", "dai", "gcs",
                "mild", "moderate", "severe"
            ]
        },
        "default": "PATHOLOGY_TRAUMA"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TICI - Thrombolysis in cerebral infarction (endovascular score)
    # ─────────────────────────────────────────────────────────────────────────
    "TICI": {
        "contexts": {
            "SCORE_ENDOVASCULAR": [
                "score", "reperfusion", "thrombectomy", "2a", "2b", "2c", "3",
                "recanalization", "angiography", "post"
            ]
        },
        "default": "SCORE_ENDOVASCULAR"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # mRS - Modified Rankin Scale
    # ─────────────────────────────────────────────────────────────────────────
    "MRS": {
        "contexts": {
            "MEASUREMENT": [
                "score", "outcome", "discharge", "90-day", "functional",
                "0", "1", "2", "3", "4", "5", "6"
            ]
        },
        "default": "MEASUREMENT"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # MVD - Microvascular decompression (pain procedure)
    # ─────────────────────────────────────────────────────────────────────────
    "MVD": {
        "contexts": {
            "PROCEDURE_PAIN": [
                "decompression", "trigeminal", "neuralgia", "tn", "teflon",
                "sca", "aica", "retrosigmoid", "nerve", "vessel"
            ]
        },
        "default": "PROCEDURE_PAIN"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # SCS - Spinal cord stimulation (pain procedure)
    # ─────────────────────────────────────────────────────────────────────────
    "SCS": {
        "contexts": {
            "PROCEDURE_PAIN": [
                "stimulation", "spinal cord", "paddle", "percutaneous", "trial",
                "fbss", "crps", "pain", "dorsal column"
            ]
        },
        "default": "PROCEDURE_PAIN"
    },
}

EXPANDED_AMBIGUITY_RULES_CONTINUED = {
    # ─────────────────────────────────────────────────────────────────────────
    # WHO - World Health Organization (tumor grading context)
    # ─────────────────────────────────────────────────────────────────────────
    "WHO": {
        "contexts": {
            "PATHOLOGY_TUMOR_MOLECULAR": [
                "grade", "Grade", "I", "II", "III", "IV", "1", "2", "3", "4",
                "classification", "2016", "2021", "CNS", "tumor"
            ]
        },
        "default": "PATHOLOGY_TUMOR_MOLECULAR"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # IDH - Isocitrate dehydrogenase
    # ─────────────────────────────────────────────────────────────────────────
    "IDH": {
        "contexts": {
            "PATHOLOGY_TUMOR_MOLECULAR": [
                "mutant", "mutation", "wildtype", "wild-type", "wt", "mut",
                "status", "R132H", "IDH1", "IDH2", "glioma", "astrocytoma"
            ]
        },
        "default": "PATHOLOGY_TUMOR_MOLECULAR"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # MGMT - Methylation status
    # ─────────────────────────────────────────────────────────────────────────
    "MGMT": {
        "contexts": {
            "PATHOLOGY_TUMOR_MOLECULAR": [
                "methylated", "unmethylated", "methylation", "promoter",
                "status", "temozolomide", "GBM", "glioblastoma"
            ]
        },
        "default": "PATHOLOGY_TUMOR_MOLECULAR"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # PI - Pelvic incidence (spine) vs other
    # ─────────────────────────────────────────────────────────────────────────
    "PI": {
        "contexts": {
            "PATHOLOGY_SPINE_DEFORMITY": [
                "pelvic", "incidence", "LL", "mismatch", "sagittal",
                "spinopelvic", "parameter", "degree"
            ]
        },
        "default": "PATHOLOGY_SPINE_DEFORMITY"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # SVA - Sagittal vertical axis
    # ─────────────────────────────────────────────────────────────────────────
    "SVA": {
        "contexts": {
            "PATHOLOGY_SPINE_DEFORMITY": [
                "sagittal", "vertical", "axis", "balance", "alignment",
                "C7", "plumb", "mm", "cm", "positive", "negative"
            ]
        },
        "default": "PATHOLOGY_SPINE_DEFORMITY"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # ASPECTS - Alberta Stroke Program Early CT Score
    # ─────────────────────────────────────────────────────────────────────────
    "ASPECTS": {
        "contexts": {
            "SCORE_ENDOVASCULAR": [
                "score", "CT", "stroke", "MCA", "infarct", "thrombectomy",
                "selection", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
            ]
        },
        "default": "SCORE_ENDOVASCULAR"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # TN - Trigeminal neuralgia
    # ─────────────────────────────────────────────────────────────────────────
    "TN": {
        "contexts": {
            "PATHOLOGY_PAIN": [
                "neuralgia", "trigeminal", "pain", "tic", "MVD", "rhizotomy",
                "type 1", "type 2", "classical", "Gamma Knife"
            ]
        },
        "default": "PATHOLOGY_PAIN"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # FBSS - Failed back surgery syndrome
    # ─────────────────────────────────────────────────────────────────────────
    "FBSS": {
        "contexts": {
            "PATHOLOGY_PAIN": [
                "failed back", "back surgery", "syndrome", "SCS", "stimulation",
                "pain", "revision", "laminectomy"
            ]
        },
        "default": "PATHOLOGY_PAIN"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # CRPS - Complex regional pain syndrome
    # ─────────────────────────────────────────────────────────────────────────
    "CRPS": {
        "contexts": {
            "PATHOLOGY_PAIN": [
                "complex regional", "pain syndrome", "RSD", "type 1", "type 2",
                "SCS", "stimulation", "allodynia", "sympathetic"
            ]
        },
        "default": "PATHOLOGY_PAIN"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # ICP - Intracranial pressure
    # ─────────────────────────────────────────────────────────────────────────
    "ICP": {
        "contexts": {
            "PHYSIOLOGY_CRITICAL_CARE": [
                "pressure", "intracranial", "monitor", "bolt", "evd",
                "mmhg", "elevated", "herniation", "treatment"
            ]
        },
        "default": "PHYSIOLOGY_CRITICAL_CARE"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # EVD - External ventricular drain
    # ─────────────────────────────────────────────────────────────────────────
    "EVD": {
        "contexts": {
            "INSTRUMENT": [
                "drain", "ventricular", "placement", "csf", "icp",
                "kocher", "frazier", "catheter"
            ]
        },
        "default": "INSTRUMENT"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # AVM - Arteriovenous malformation
    # ─────────────────────────────────────────────────────────────────────────
    "AVM": {
        "contexts": {
            "PATHOLOGY_AVM": [
                "malformation", "arteriovenous", "nidus", "feeder", "draining",
                "spetzler", "martin", "hemorrhage", "embolization", "radiosurgery"
            ]
        },
        "default": "PATHOLOGY_AVM"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # GKS/GKRS - Gamma Knife (Radio)Surgery
    # ─────────────────────────────────────────────────────────────────────────
    "GKS": {
        "contexts": {
            "PROCEDURE_RADIOSURGERY": [
                "gamma knife", "radiosurgery", "srs", "dose", "gy",
                "margin", "isodose", "frame"
            ]
        },
        "default": "PROCEDURE_RADIOSURGERY"
    },
    "GKRS": {
        "contexts": {
            "PROCEDURE_RADIOSURGERY": [
                "gamma knife", "radiosurgery", "srs", "dose", "gy",
                "margin", "isodose", "frame"
            ]
        },
        "default": "PROCEDURE_RADIOSURGERY"
    },
}


def resolve_entity_ambiguity(
    term: str,
    context: str,
    candidate_categories: list,
    ambiguity_rules: dict = None
) -> str:
    """
    Resolve ambiguous entity categorization based on surrounding context.

    Args:
        term: The extracted term/entity (e.g., "M1", "DBS", "NPH")
        context: The surrounding text (typically ±50-100 characters)
        candidate_categories: List of possible category matches
        ambiguity_rules: Dictionary of rules (uses EXPANDED_AMBIGUITY_RULES if None)

    Returns:
        The most appropriate category string

    Example:
        >>> resolve_entity_ambiguity(
        ...     term="M1",
        ...     context="The tumor involved the M1 segment of the MCA",
        ...     candidate_categories=["ANATOMY_BRAIN_CORTICAL", "ANATOMY_VASCULAR_ARTERIAL"]
        ... )
        'ANATOMY_VASCULAR_ARTERIAL'
    """
    # Merge rule dictionaries
    if ambiguity_rules is None:
        rules = {**EXPANDED_AMBIGUITY_RULES, **EXPANDED_AMBIGUITY_RULES_CONTINUED}
    else:
        rules = ambiguity_rules

    # Normalize term for lookup
    term_upper = term.upper().strip()

    # Check if we have rules for this term
    if term_upper not in rules:
        # No specific rule - return first candidate or generic
        return candidate_categories[0] if candidate_categories else "UNKNOWN"

    rule = rules[term_upper]
    context_lower = context.lower()

    # Score each possible category based on context matches
    category_scores = {}

    for category, keywords in rule.get("contexts", {}).items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in context_lower:
                score += 1
                # Bonus for exact phrase matches
                if f" {keyword.lower()} " in f" {context_lower} ":
                    score += 0.5
        category_scores[category] = score

    # Return highest scoring category, or default if no matches
    if category_scores:
        best_category = max(category_scores, key=category_scores.get)
        if category_scores[best_category] > 0:
            return best_category

    # Fall back to default
    return rule.get("default", candidate_categories[0] if candidate_categories else "UNKNOWN")


def get_context_window(text: str, match_start: int, match_end: int, window_size: int = 100) -> str:
    """
    Extract context window around a match position.

    Args:
        text: Full text
        match_start: Start position of match
        match_end: End position of match
        window_size: Characters to include on each side

    Returns:
        Context string
    """
    start = max(0, match_start - window_size)
    end = min(len(text), match_end + window_size)
    return text[start:end]


class NeuroExpertPatterns:
    """
    Exhaustive regex patterns for the neurosurgical domain.
    
    Can be loaded from external YAML config for maintainability.
    Falls back to hardcoded defaults if config unavailable.
    
    Total: 100+ patterns covering all major areas.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize patterns.
        
        Args:
            config_path: Path to YAML config file (optional)
        """
        self.PATTERNS: Dict[str, List[str]] = {}
        self.NORMALIZATIONS: Dict[str, str] = {}
        self._config_path = config_path
        
        # Try loading from config first
        if config_path and config_path.exists() and HAS_YAML:
            self._load_from_yaml(config_path)
        else:
            # Use hardcoded defaults
            self.PATTERNS = self._get_default_patterns()
            self.NORMALIZATIONS = self._get_default_normalizations()
    
    def _load_from_yaml(self, config_path: Path):
        """Load patterns from YAML configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Load patterns
            self.PATTERNS = config.get("patterns", {})
            
            # Load normalizations
            self.NORMALIZATIONS = config.get("normalizations", {})
            
            logger.info(
                f"Loaded {len(self.PATTERNS)} pattern categories, "
                f"{len(self.NORMALIZATIONS)} normalizations from {config_path}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.warning("Falling back to default patterns")
            self.PATTERNS = self._get_default_patterns()
            self.NORMALIZATIONS = self._get_default_normalizations()
    
    def save_to_yaml(self, config_path: Path):
        """
        Save current patterns to YAML file.
        
        Useful for exporting defaults to editable config.
        """
        if not HAS_YAML:
            raise ImportError("PyYAML required: pip install pyyaml")
        
        config = {
            "patterns": self.PATTERNS,
            "normalizations": self.NORMALIZATIONS
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Saved patterns to {config_path}")
    
    def _get_default_patterns(self) -> Dict[str, List[str]]:
        """Return hardcoded default patterns."""
        return {
            # =================================================================
            # VASCULAR NEUROSURGERY
            # =================================================================
            
            "ANATOMY_VASCULAR_ARTERIAL": [
                # ICA segments
                r"\b(C[1-7])\s*(segment)?\b",
                # Circle of Willis segments
                r"\b(A|M|P)[1-4]\s*(segment)?\b",
                # Named arteries (abbreviated)
                r"\b(ICA|MCA|ACA|PCA|SCA|AICA|PICA|VA|BA)\b",
                # Communicating arteries
                r"\b(anterior|posterior)\s+communicating\s+artery\b",
                r"\b(ACoA|AComA|ACOM|PCoA|PComA|PCOM)\b",
                # Ophthalmic and choroidal
                r"\b(ophthalmic|anterior\s+choroidal|posterior\s+choroidal)\s+artery\b",
                # Perforators
                r"\b(lenticulostriate|thalamoperforat|heubner|recurrent\s+artery)\b",
                r"\b(perforator|perforating\s+arter)\w*\b",
                # Hypophyseal
                r"\b(superior|inferior)\s+hypophyseal\s+artery\b",
                # Full names
                r"\b(internal\s+carotid|middle\s+cerebral|anterior\s+cerebral)\s+artery\b",
                r"\b(posterior\s+cerebral|superior\s+cerebellar)\s+artery\b",
                r"\b(anterior\s+inferior\s+cerebellar|posterior\s+inferior\s+cerebellar)\s+artery\b",
                r"\b(vertebral|basilar)\s+artery\b",
            ],
            
            "ANATOMY_VASCULAR_VENOUS": [
                # Major sinuses
                r"\b(SSS|ISS|TS|SS)\b",
                r"\b(superior\s+sagittal|inferior\s+sagittal)\s+sinus\b",
                r"\b(transverse|sigmoid|cavernous|petrosal|straight)\s+sinus\b",
                r"\b(torcula|confluence\s+of\s+sinuses?)\b",
                # Named veins
                r"\b(vein\s+of\s+(?:Galen|Labbé|Labbe|Trolard|Rosenthal))\b",
                r"\b(basal\s+vein|internal\s+cerebral\s+vein)\b",
                r"\b(superficial\s+middle\s+cerebral\s+vein|SMCV)\b",
                # Cortical veins
                r"\b(bridging\s+vein|cortical\s+vein)\b",
                # Deep venous
                r"\b(deep\s+venous\s+system|galenic\s+system)\b",
            ],
            
            "PATHOLOGY_VASCULAR": [
                # Aneurysms
                r"\b(aneurysm|aneurysmal)\b",
                r"\b(saccular|fusiform|dissecting|mycotic|traumatic)\s+aneurysm\b",
                r"\b(giant|large|small|micro)\s*aneurysm\b",
                # AVMs
                r"\b(AVM|arteriovenous\s+malformation)\b",
                r"\b(dural\s+AV\s*fistula|dAVF|DAVF)\b",
                r"\b(cavernoma|cavernous\s+malformation|CCM)\b",
                # Other vascular
                r"\b(moyamoya)\b",
                r"\b(vasospasm|DCI|delayed\s+cerebral\s+ischemia)\b",
                r"\b(subarachnoid\s+hemorrhage|SAH)\b",
                r"\b(intracerebral\s+hemorrhage|ICH)\b",
                r"\b(infarction|ischemia|stroke)\b",
            ],
            
            # =================================================================
            # SKULL BASE & CRANIAL NERVES
            # =================================================================
            
            "ANATOMY_SKULL_BASE": [
                # Anterior skull base
                r"\b(planum\s+sphenoidale|tuberculum\s+sellae|dorsum\s+sellae)\b",
                r"\b(crista\s+galli|cribriform\s+plate|olfactory\s+groove)\b",
                r"\b(optic\s+strut|optic\s+canal)\b",
                # Parasellar
                r"\b(sella\s+turcica|pituitary\s+fossa|diaphragma\s+sellae)\b",
                r"\b(anterior|posterior)\s+clinoid\b",
                r"\b(cavernous\s+sinus)\b",
                # Petrous/Posterior
                r"\b(clivus|petrous\s+apex|tegmen\s+tympani)\b",
                r"\b(IAM|IAC|internal\s+acoustic\s+(?:meatus|canal))\b",
                r"\b(jugular\s+foramen|jugular\s+tubercle)\b",
                r"\b(cerebellopontine\s+angle|CPA)\b",
                # Foramina
                r"\b(foramen\s+(?:ovale|rotundum|spinosum|lacerum|magnum))\b",
                r"\b(superior\s+orbital\s+fissure|SOF)\b",
                r"\b(inferior\s+orbital\s+fissure|IOF)\b",
            ],
            
            "ANATOMY_CRANIAL_NERVES": [
                # Roman numeral format
                r"\b(CN\s*[IVX]+)\b",
                r"\b(cranial\s+nerve\s*[IVX]+)\b",
                # Named nerves
                r"\b(olfactory|optic|oculomotor|trochlear|trigeminal)\s+nerve\b",
                r"\b(abducens|abducent|facial|vestibulocochlear)\s+nerve\b",
                r"\b(glossopharyngeal|vagus|accessory|hypoglossal)\s+nerve\b",
                r"\b(acoustic\s+nerve|cochlear\s+nerve|vestibular\s+nerve)\b",
                # Ganglia
                r"\b(gasserian\s+ganglion|trigeminal\s+ganglion)\b",
                r"\b(geniculate\s+ganglion)\b",
                # Specific structures
                r"\b(optic\s+chiasm|chiasmatic)\b",
                r"\b(Meckel'?s?\s+cave)\b",
            ],
            
            # =================================================================
            # SPINE
            # =================================================================
            
            "ANATOMY_SPINE_BONE": [
                # Vertebral levels (context-dependent)
                r"\b(C|T|L|S)[1-8]\b(?!\s*segment)",
                r"\b(C|T|L)[1-8][-–](C|T|L|S)[1-8]\b",
                # Special vertebrae
                r"\b(atlas|axis|dens|odontoid)\b",
                r"\b(sacrum|coccyx)\b",
                # Structures
                r"\b(pedicle|lamina|facet|pars\s+interarticularis)\b",
                r"\b(spinous\s+process|transverse\s+process)\b",
                r"\b(vertebral\s+body|disc\s+space|intervertebral)\b",
                r"\b(foramen|foraminal|neuroforamen)\b",
                # Regions
                r"\b(cervical|thoracic|lumbar|sacral|lumbosacral)\s+spine\b",
            ],
            
            "ANATOMY_SPINE_NEURAL": [
                # Spinal cord
                r"\b(spinal\s+cord|medulla\s+spinalis)\b",
                r"\b(conus\s+medullaris|cauda\s+equina|filum\s+terminale)\b",
                # Nerve roots
                r"\b(nerve\s+root|dorsal\s+root\s+ganglion|DRG)\b",
                r"\b(ventral\s+root|dorsal\s+root)\b",
                # Meninges
                r"\b(thecal\s+sac|dura|dural)\b",
                r"\b(epidural|subdural|intradural)\b",
                # Ligaments
                r"\b(ligamentum\s+flavum|posterior\s+longitudinal\s+ligament|PLL)\b",
                r"\b(anterior\s+longitudinal\s+ligament|ALL)\b",
            ],
            
            # =================================================================
            # TUMOR
            # =================================================================
            
            "PATHOLOGY_TUMOR": [
                # Gliomas
                r"\b(glioma|glioblastoma|GBM)\b",
                r"\b(astrocytoma|oligodendroglioma|ependymoma)\b",
                r"\b(diffuse\s+intrinsic\s+pontine\s+glioma|DIPG)\b",
                # Extra-axial
                r"\b(meningioma)\b",
                r"\b(schwannoma|neurofibroma|vestibular\s+schwannoma)\b",
                r"\b(acoustic\s+neuroma)\b",
                # Sellar/Parasellar
                r"\b(pituitary\s+adenoma|macroadenoma|microadenoma)\b",
                r"\b(craniopharyngioma)\b",
                r"\b(Rathke'?s?\s+cleft\s+cyst)\b",
                # Other
                r"\b(metastasis|metastases|metastatic)\b",
                r"\b(hemangioblastoma|chordoma|chondrosarcoma)\b",
                r"\b(medulloblastoma|PNET)\b",
                r"\b(epidermoid|dermoid|teratoma)\b",
                # Grades
                r"\b(WHO\s+grade\s+[IVX]+|grade\s+[IVX]+\s+tumor)\b",
                r"\b(low[-\s]grade|high[-\s]grade)\s+(glioma|tumor)\b",
            ],
            
            # =================================================================
            # FUNCTIONAL & EPILEPSY
            # =================================================================
            
            "PATHOLOGY_FUNCTIONAL": [
                # Movement disorders
                r"\b(Parkinson'?s?|parkinsonian)\b",
                r"\b(essential\s+tremor|dystonia|chorea)\b",
                # Epilepsy
                r"\b(epilepsy|seizure|ictal|interictal)\b",
                r"\b(mesial\s+temporal\s+sclerosis|MTS)\b",
                r"\b(focal\s+cortical\s+dysplasia|FCD)\b",
                # Pain
                r"\b(trigeminal\s+neuralgia|TN)\b",
                r"\b(glossopharyngeal\s+neuralgia)\b",
            ],
            
            # =================================================================
            # SURGICAL PROCEDURES
            # =================================================================
            
            "PROCEDURE_APPROACH": [
                # Supratentorial
                r"\b(pterional|frontotemporal)\b",
                r"\b(orbitozygomatic|OZ)\b",
                r"\b(interhemispheric|transcallosal)\b",
                r"\b(subfrontal|bifrontal)\b",
                r"\b(subtemporal)\b",
                # Posterior fossa
                r"\b(suboccipital|retrosigmoid|retromastoid)\b",
                r"\b(far\s+lateral|extreme\s+lateral)\b",
                r"\b(telovelar|trans[-]?vermian)\b",
                # Skull base
                r"\b(transsphenoidal|endoscopic\s+endonasal)\b",
                r"\b(presigmoid|transpetrosal)\b",
                r"\b(transmastoid|translabyrinthine)\b",
                # Spine
                r"\b(ACDF|anterior\s+cervical\s+discectomy)\b",
                r"\b(PLIF|TLIF|ALIF|XLIF|OLIF)\b",
                r"\b(laminoplasty|laminectomy)\b",
            ],
            
            "PROCEDURE_ACTION": [
                # Cranial
                r"\b(craniotomy|craniectomy|cranioplasty)\b",
                r"\b(burr\s+hole|trephination)\b",
                # Tumor
                r"\b(resection|debulking|biopsy)\b",
                r"\b(gross\s+total\s+resection|GTR)\b",
                r"\b(subtotal\s+resection|STR)\b",
                # Vascular
                r"\b(clipping|clip\s+application)\b",
                r"\b(coiling|embolization|embolisation)\b",
                r"\b(bypass|EC[-]?IC\s+bypass)\b",
                r"\b(trapping|wrapping)\b",
                # Spine
                r"\b(discectomy|corpectomy|foraminotomy)\b",
                r"\b(fusion|arthrodesis)\b",
                r"\b(decompression|laminectomy)\b",
                # General surgical actions
                r"\b(dissection|retraction|coagulation)\b",
                r"\b(incision|exposure|closure)\b",
                r"\b(hemostasis|irrigation)\b",
            ],
            
            "PROCEDURE_MODALITY": [
                # DBS
                r"\b(DBS|deep\s+brain\s+stimulation)\b",
                r"\b(VIM|STN|GPi)\s+(target|stimulation)\b",
                # Radiosurgery
                r"\b(gamma\s+knife|GKS|radiosurgery)\b",
                r"\b(SRS|stereotactic\s+radiosurgery)\b",
                r"\b(CyberKnife|LINAC)\b",
                # Neuromodulation
                r"\b(SCS|spinal\s+cord\s+stimulation)\b",
                r"\b(RNS|responsive\s+neurostimulation)\b",
                r"\b(VNS|vagus\s+nerve\s+stimulation)\b",
            ],
            
            # =================================================================
            # INSTRUMENTS
            # =================================================================
            
            "INSTRUMENT": [
                # Electrosurgical
                r"\b(bipolar|monopolar)\s*(forceps|coagulation)?\b",
                r"\b(bovie|electrocautery)\b",
                # Suction/Aspiration
                r"\b(suction|aspirator)\b",
                r"\b(CUSA|cavitron|ultrasonic\s+aspirator)\b",
                # Dissectors
                r"\b(dissector|penfield|rhoton)\b",
                r"\b(micro[-]?dissector|micro[-]?hook)\b",
                # Rongeurs
                r"\b(kerrison|rongeur|leksell)\b",
                # Clips
                r"\b(aneurysm\s+clip|yasargil\s+clip|sugita\s+clip)\b",
                r"\b(fenestrated\s+clip|temporary\s+clip|permanent\s+clip)\b",
                # Spine hardware (specific multi-word patterns first to avoid "lateral" being matched as MEASUREMENT)
                r"\b(lateral\s+mass\s+screws?)\b",
                r"\b(pedicle\s+screws?)\b",
                r"\b(transarticular\s+screws?)\b",
                r"\b(odontoid\s+screws?)\b",
                r"\b(polyaxial\s+screws?)\b",
                r"\b(monoaxial\s+screws?)\b",
                r"\b(set\s+screws?)\b",
                r"\b(titanium\s+rods?)\b",
                r"\b(cervical\s+plates?)\b",
                r"\b(locking\s+plates?)\b",
                r"\b(thoracolumbar\s+rods?)\b",
                r"\b(interbody\s+cages?)\b",
                r"\b(cross[-\s]?links?)\b",
                r"\b(pedicle\s+screw|rod|cage|plate)\b",
                r"\b(lateral\s+mass\s+screw|interbody\s+cage)\b",
                # Visualization
                r"\b(microscope|operating\s+microscope)\b",
                r"\b(endoscope|neuroendoscope)\b",
                r"\b(exoscope)\b",
                # Navigation
                r"\b(neuronavigation|stereotactic\s+frame|frameless)\b",
                r"\b(BrainLab|StealthStation|Medtronic)\b",
                # Retractors
                r"\b(retractor|leyla|greenberg|yasargil)\s*retractor?\b",
                r"\b(self[-]?retaining\s+retractor)\b",
            ],
            
            # =================================================================
            # MEASUREMENTS & SCALES
            # =================================================================
            
            "MEASUREMENT": [
                # Dimensions
                r"\b(\d+(?:\.\d+)?)\s*(mm|cm|m)\b",
                # Angles
                r"\b(\d+(?:\.\d+)?)\s*(degrees?|°)\b",
                # Spatial relationships
                r"\b(proximal|distal|superior|inferior|medial|lateral)\b",
                r"\b(anterior|posterior|rostral|caudal|dorsal|ventral)\b",
                # Grading scales
                r"\b(Spetzler[-\s]?Martin|SM\s+grade)\b",
                r"\b(Hunt[-\s]?Hess|H[-]?H\s+grade)\b",
                r"\b(Fisher\s+grade|modified\s+Fisher)\b",
                r"\b(WFNS\s+grade)\b",
                r"\b(Karnofsky|KPS)\b",
                r"\b(mRS|modified\s+Rankin)\b",
                r"\b(GCS|Glasgow\s+Coma\s+Scale?)\b",
            ],
            
            # =================================================================
            # IMAGING
            # =================================================================
            
            "IMAGING": [
                r"\b(MRI|MR\s+imaging|magnetic\s+resonance)\b",
                r"\b(CT|computed\s+tomography)\b",
                r"\b(CTA|CT\s+angiography)\b",
                r"\b(MRA|MR\s+angiography)\b",
                r"\b(DSA|digital\s+subtraction\s+angiography)\b",
                r"\b(T1[-\s]?weighted|T2[-\s]?weighted|FLAIR)\b",
                r"\b(diffusion[-\s]?weighted|DWI|ADC)\b",
                r"\b(contrast[-\s]?enhanced|gadolinium)\b",
                r"\b(X[-]?ray|fluoroscopy)\b",
                r"\b(PET|positron\s+emission)\b",
            ],
        }
    
    def _get_default_normalizations(self) -> Dict[str, str]:
        """Return hardcoded default normalizations."""
        return {
            # Cranial nerves
            "CN I": "Olfactory Nerve (CN I)",
            "CN II": "Optic Nerve (CN II)",
            "CN III": "Oculomotor Nerve (CN III)",
            "CN IV": "Trochlear Nerve (CN IV)",
            "CN V": "Trigeminal Nerve (CN V)",
            "CN VI": "Abducens Nerve (CN VI)",
            "CN VII": "Facial Nerve (CN VII)",
            "CN VIII": "Vestibulocochlear Nerve (CN VIII)",
            "CN IX": "Glossopharyngeal Nerve (CN IX)",
            "CN X": "Vagus Nerve (CN X)",
            "CN XI": "Accessory Nerve (CN XI)",
            "CN XII": "Hypoglossal Nerve (CN XII)",
            
            # Vascular abbreviations
            "ICA": "Internal Carotid Artery",
            "MCA": "Middle Cerebral Artery",
            "ACA": "Anterior Cerebral Artery",
            "PCA": "Posterior Cerebral Artery",
            "VA": "Vertebral Artery",
            "BA": "Basilar Artery",
            "SCA": "Superior Cerebellar Artery",
            "AICA": "Anterior Inferior Cerebellar Artery",
            "PICA": "Posterior Inferior Cerebellar Artery",
            "ACOM": "Anterior Communicating Artery",
            "PCOM": "Posterior Communicating Artery",
            "ACoA": "Anterior Communicating Artery",
            "PCoA": "Posterior Communicating Artery",
            
            # Segments
            "M1": "M1 Segment (MCA)",
            "M2": "M2 Segment (MCA)",
            "M3": "M3 Segment (MCA)",
            "M4": "M4 Segment (MCA)",
            "A1": "A1 Segment (ACA)",
            "A2": "A2 Segment (ACA)",
            "P1": "P1 Segment (PCA)",
            "P2": "P2 Segment (PCA)",
            
            # Pathology
            "AVM": "Arteriovenous Malformation",
            "SAH": "Subarachnoid Hemorrhage",
            "ICH": "Intracerebral Hemorrhage",
            "GBM": "Glioblastoma Multiforme",
            "DBS": "Deep Brain Stimulation",
            
            # Venous
            "SSS": "Superior Sagittal Sinus",
            "ISS": "Inferior Sagittal Sinus",
            "TS": "Transverse Sinus",
            "SS": "Sigmoid Sinus",
            
            # Procedures
            "GTR": "Gross Total Resection",
            "STR": "Subtotal Resection",
            "ACDF": "Anterior Cervical Discectomy and Fusion",
            
            # Additional normalizations
            "CPA": "Cerebellopontine Angle",
            "IAC": "Internal Auditory Canal",
            "IAM": "Internal Auditory Meatus",
            "DCI": "Delayed Cerebral Ischemia",
        }


class NeuroExpertTextExtractor:
    """
    Expert system for extracting and classifying neurosurgical entities.
    
    Features:
    1. Pattern-based entity detection
    2. Context-aware ambiguity resolution
    3. Entity normalization
    4. Confidence scoring
    5. Extraction caching for performance
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        context_chars: int = DEFAULT_CONTEXT_CHARS
    ):
        """
        Initialize the extractor.
        
        Args:
            config_path: Path to YAML patterns config (optional)
            context_chars: Characters of context to capture around entities
        """
        self._patterns_source = NeuroExpertPatterns(config_path)
        self.patterns = self._patterns_source.PATTERNS
        self.normalizations = self._patterns_source.NORMALIZATIONS
        self.context_chars = context_chars
        
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile all regex patterns for performance."""
        for category, pattern_list in self.patterns.items():
            self._compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in pattern_list
            ]
    
    def reload_patterns(self, config_path: Path):
        """
        Reload patterns from a new config file.
        
        Allows hot-reloading without restarting.
        """
        self._patterns_source = NeuroExpertPatterns(config_path)
        self.patterns = self._patterns_source.PATTERNS
        self.normalizations = self._patterns_source.NORMALIZATIONS
        self._compiled_patterns.clear()
        self._compile_patterns()
        self._extract_cached.cache_clear()
        logger.info(f"Reloaded patterns from {config_path}")
    
    def extract(self, text: str) -> List[NeuroEntity]:
        """
        Main entry point: Extract, resolve ambiguity, and deduplicate entities.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of extracted entities with resolved categories
        """
        # Use cached extraction for repeated texts
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return list(self._extract_cached(text_hash, text))
    
    @lru_cache(maxsize=1000)
    def _extract_cached(self, text_hash: str, text: str) -> Tuple[NeuroEntity, ...]:
        """
        Cached extraction implementation.
        
        Returns tuple for hashability.
        """
        raw_entities = []
        
        for category, compiled_list in self._compiled_patterns.items():
            for pattern in compiled_list:
                for match in pattern.finditer(text):
                    entity_text = match.group()
                    start, end = match.span()
                    
                    # Get surrounding context (expanded window)
                    context_start = max(0, start - self.context_chars)
                    context_end = min(len(text), end + self.context_chars)
                    context = text[context_start:context_end]
                    
                    # Create entity
                    entity = NeuroEntity(
                        text=entity_text.strip(),
                        category=category,
                        normalized=self._normalize_entity(entity_text),
                        start=start,
                        end=end,
                        confidence=0.9,
                        context_snippet=context
                    )
                    
                    # Resolve ambiguities based on context
                    self._resolve_ambiguity(entity, text)
                    raw_entities.append(entity)
        
        # Deduplicate overlapping entities
        return tuple(self._deduplicate(raw_entities))
    
    def _normalize_entity(self, text: str) -> str:
        """Normalize entity to standard form."""
        text_clean = text.strip()
        text_upper = text_clean.upper()
        
        # Check direct mappings
        if text_upper in self.normalizations:
            return self.normalizations[text_upper]
        
        # Check with spaces removed
        text_no_space = re.sub(r'\s+', '', text_upper)
        for key, value in self.normalizations.items():
            if re.sub(r'\s+', '', key) == text_no_space:
                return value
        
        return text_clean
    
    def _resolve_ambiguity(self, entity: NeuroEntity, full_text: str):
        """
        Adjust category based on context.
        
        Critical ambiguities:
        - C1-C7: Vertebral level vs ICA segment
        - Clip: Skin clip vs Aneurysm clip
        - Root: Nerve root vs Mathematical root
        - Foramen: Skull base vs Spine
        - M1: MCA segment vs Primary motor cortex
        - SCA: Superior cerebellar artery vs Spinocerebellar ataxia
        """
        context = entity.context_snippet.lower()
        
        # Ambiguity 1: C-Levels (C1-C7)
        if re.match(r"^[Cc][1-7]$", entity.text):
            vascular_evidence = [
                "ica", "carotid", "artery", "aneurysm", "segment",
                "clinoid", "ophthalmic", "cavernous", "petrous"
            ]
            spine_evidence = [
                "vertebra", "bone", "fracture", "spine", "spinal",
                "cervical", "root", "disc", "pedicle", "level",
                "fusion", "laminectomy", "foraminotomy"
            ]
            
            vascular_score = sum(1 for w in vascular_evidence if w in context)
            spine_score = sum(1 for w in spine_evidence if w in context)
            
            if vascular_score > spine_score:
                entity.category = "ANATOMY_VASCULAR_ARTERIAL"
                entity.confidence = 0.95
                entity.normalized = f"C{entity.text[1]} Segment (ICA)"
            elif spine_score > vascular_score:
                entity.category = "ANATOMY_SPINE_BONE"
                entity.confidence = 0.95
                entity.normalized = f"C{entity.text[1]} Vertebra"
            else:
                entity.category = "ANATOMY_SPINE_BONE"
                entity.confidence = 0.6
        
        # Ambiguity 2: "Clip"
        if "clip" in entity.text.lower():
            aneurysm_evidence = [
                "aneurysm", "neck", "dome", "temporary", "permanent",
                "yasargil", "sugita", "fenestrated"
            ]
            skin_evidence = ["scalp", "drape", "raney", "skin", "closure"]
            
            if any(w in context for w in aneurysm_evidence):
                entity.category = "INSTRUMENT"
                entity.confidence = 0.95
            elif any(w in context for w in skin_evidence):
                entity.category = "INSTRUMENT"
                entity.confidence = 0.9
        
        # Ambiguity 3: "Root"
        if "root" in entity.text.lower():
            nerve_evidence = ["nerve", "dorsal", "ventral", "spinal", "compression"]
            math_evidence = ["square", "equation", "calculation"]
            
            if any(w in context for w in nerve_evidence):
                entity.category = "ANATOMY_SPINE_NEURAL"
                entity.confidence = 0.95
            elif any(w in context for w in math_evidence):
                entity.confidence = 0.1
        
        # Ambiguity 4: "Foramen"
        if "foramen" in entity.text.lower():
            skull_base_evidence = [
                "skull base", "middle fossa", "trigeminal", "ovale",
                "rotundum", "spinosum", "lacerum", "magnum"
            ]
            spine_evidence = ["vertebra", "intervertebral", "nerve root", "neural"]
            
            if any(w in context for w in skull_base_evidence):
                entity.category = "ANATOMY_SKULL_BASE"
                entity.confidence = 0.95
            elif any(w in context for w in spine_evidence):
                entity.category = "ANATOMY_SPINE_BONE"
                entity.confidence = 0.9
        
        # Ambiguity 5: "M1"
        if entity.text.upper() == "M1":
            vascular_evidence = ["mca", "artery", "segment", "bifurcation", "aneurysm"]
            cortex_evidence = ["cortex", "motor", "gyrus", "precentral", "homunculus"]
            
            if any(w in context for w in vascular_evidence):
                entity.category = "ANATOMY_VASCULAR_ARTERIAL"
                entity.confidence = 0.95
            elif any(w in context for w in cortex_evidence):
                entity.category = "ANATOMY_NEURAL"
                entity.confidence = 0.9
                entity.normalized = "Primary Motor Cortex (M1)"
        
        # Ambiguity 6: "SCA"
        if entity.text.upper() == "SCA":
            artery_evidence = ["artery", "cerebellar", "aneurysm", "vessel"]
            ataxia_evidence = ["ataxia", "spinocerebellar", "genetic", "hereditary"]

            if any(w in context for w in artery_evidence):
                entity.category = "ANATOMY_VASCULAR_ARTERIAL"
                entity.confidence = 0.95
            elif any(w in context for w in ataxia_evidence):
                entity.category = "PATHOLOGY"
                entity.confidence = 0.9
                entity.normalized = "Spinocerebellar Ataxia"

        # ─────────────────────────────────────────────────────────────────────────
        # EXPANDED AMBIGUITY RESOLUTION (from EXPANDED_AMBIGUITY_RULES)
        # Uses the module-level resolve_entity_ambiguity function
        # ─────────────────────────────────────────────────────────────────────────

        term_upper = entity.text.upper().strip()
        all_rules = {**EXPANDED_AMBIGUITY_RULES, **EXPANDED_AMBIGUITY_RULES_CONTINUED}

        # Check if this term has expanded ambiguity rules
        if term_upper in all_rules:
            resolved = resolve_entity_ambiguity(
                term=entity.text,
                context=context,
                candidate_categories=[entity.category]
            )
            if resolved != entity.category and resolved != "UNKNOWN":
                entity.category = resolved
                entity.confidence = 0.95

    def _deduplicate(self, entities: List[NeuroEntity]) -> List[NeuroEntity]:
        """
        Remove overlapping entities, preferring longer/more specific matches.
        """
        if not entities:
            return []
        
        # Sort by start position, then by length (descending)
        sorted_entities = sorted(
            entities,
            key=lambda e: (e.start, -(e.end - e.start))
        )
        
        unique = []
        last_end = -1
        
        for entity in sorted_entities:
            if entity.confidence < 0.3:
                continue
            
            if entity.start >= last_end:
                unique.append(entity)
                last_end = entity.end
            elif entity.category != unique[-1].category and entity.confidence > 0.8:
                pass
        
        return unique
    
    def extract_relations(
        self,
        text: str,
        entities: List[NeuroEntity],
        chunk_id: str = "",
        document_id: str = ""
    ) -> List[EntityRelation]:
        """
        Extract relationships between entities with full context.

        Returns structured EntityRelation objects for Knowledge Graph construction.

        Args:
            text: Source text to extract relations from
            entities: Entities already extracted from this text
            chunk_id: ID of the source chunk (for provenance tracking)
            document_id: ID of the source document

        Returns:
            List of EntityRelation objects with typed relationships
        """
        relations = []

        # More flexible patterns allowing 1-4 word entity names and passive voice
        relation_patterns = [
            # Supply relationships (active and passive voice)
            (r"(\w+(?:\s+\w+){0,3})\s+(?:supplies?|supply|supplying|provides?\s+blood\s+to)\s+(?:the\s+)?(?:blood\s+(?:to|supply)\s+(?:the\s+)?)?(\w+(?:\s+\w+){0,3})", "supplies"),
            (r"(\w+(?:\s+\w+){0,3})\s+(?:is|are)\s+supplied\s+by\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})", "supplies"),

            # Innervation (active and passive voice)
            (r"(\w+(?:\s+\w+){0,3})\s+(?:innervates?|innervating|provides?\s+innervation\s+to)\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})", "innervates"),
            (r"(\w+(?:\s+\w+){0,3})\s+(?:is|are)\s+innervated\s+by\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})", "innervates"),

            # Spatial relationships
            (r"(\w+(?:\s+\w+){0,3})\s+(?:is\s+)?(?:located\s+)?(?:adjacent|lateral|medial|superior|inferior|anterior|posterior)\s+to\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})", "spatial"),

            # Causation
            (r"(?:injury|damage|lesion|compression)\s+(?:to|of)\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})\s+(?:causes?|results?\s+in|leads?\s+to|produces?)\s+(\w+(?:\s+\w+){0,3})", "causes"),

            # Traversal/course
            (r"(\w+(?:\s+\w+){0,3})\s+(?:passes?|courses?|traverses?|travels?|runs?)\s+(?:through|within|along|across|over)\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})", "traverses"),

            # Origin
            (r"(\w+(?:\s+\w+){0,3})\s+(?:originates?|arises?|comes?)\s+from\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})", "originates"),

            # Branching
            (r"(\w+(?:\s+\w+){0,3})\s+(?:branches?\s+(?:off|from)|gives?\s+(?:off|rise\s+to))\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})", "branches"),

            # Connection
            (r"(\w+(?:\s+\w+){0,3})\s+(?:connects?|joins?)\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})\s+(?:to|with)\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})", "connects"),

            # Insertion/attachment (spine-specific)
            (r"(\w+(?:\s+\w+){0,3})\s+(?:inserts?\s+(?:on|into|at)|attaches?\s+to)\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})", "attaches"),

            # Articulation (spine-specific)
            (r"(\w+(?:\s+\w+){0,3})\s+(?:articulates?\s+with)\s+(?:the\s+)?(\w+(?:\s+\w+){0,3})", "articulates"),
        ]

        for pattern_str, relation_type in relation_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            for match in pattern.finditer(text):
                source_text = match.group(1)
                target_text = match.group(2)

                source_entity = self._find_entity_match(source_text, entities)
                target_entity = self._find_entity_match(target_text, entities)

                if source_entity and target_entity:
                    # Calculate confidence based on entity match quality
                    confidence = min(source_entity.confidence, target_entity.confidence)

                    # Extract context snippet (50 chars before and after)
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text), match.end() + 50)
                    context_snippet = text[context_start:context_end]

                    relations.append(EntityRelation(
                        source_entity=source_entity.normalized,
                        target_entity=target_entity.normalized,
                        relation_type=relation_type,
                        source_category=source_entity.category,
                        target_category=target_entity.category,
                        confidence=confidence,
                        chunk_id=chunk_id,
                        document_id=document_id,
                        context_snippet=context_snippet
                    ))

        return relations

    def extract_relations_legacy(
        self,
        text: str,
        entities: List[NeuroEntity]
    ) -> List[Dict]:
        """Legacy method returning dicts for backward compatibility."""
        relations = self.extract_relations(text, entities)
        return [r.to_dict() for r in relations]
    
    def _find_entity_match(
        self,
        text: str,
        entities: List[NeuroEntity]
    ) -> Optional[NeuroEntity]:
        """Find entity matching text."""
        text_lower = text.lower().strip()
        
        for entity in entities:
            if entity.text.lower() == text_lower:
                return entity
        
        for entity in entities:
            if text_lower in entity.text.lower() or entity.text.lower() in text_lower:
                return entity
        
        return None
    
    def extract_figure_references(self, text: str) -> List[str]:
        """Extract references to figures in the text."""
        patterns = [
            r"(?:Figure|Fig\.?)\s*(\d+[A-Za-z]?(?:[-–]\d+[A-Za-z]?)?)",
            r"(?:see|shown\s+in|as\s+in)\s+(?:Figure|Fig\.?)\s*(\d+)",
            r"\((?:Figure|Fig\.?)\s*(\d+[A-Za-z]?)\)",
        ]
        
        refs = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                refs.append(f"Figure {match.group(1)}")
        
        return list(set(refs))
    
    def extract_table_references(self, text: str) -> List[str]:
        """Extract references to tables in the text."""
        patterns = [
            r"(?:Table|Tbl\.?)\s*(\d+[A-Za-z]?)",
            r"(?:see|shown\s+in)\s+(?:Table|Tbl\.?)\s*(\d+)",
        ]
        
        refs = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                refs.append(f"Table {match.group(1)}")
        
        return list(set(refs))
