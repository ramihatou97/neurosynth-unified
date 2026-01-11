"""
Comprehensive Neurosurgical Ontology
====================================

Expert neurosurgical knowledge base for gap detection, covering 7 board certification domains:

1. NEUROANATOMY: Foramina, triangles, arterial segments, venous sinuses
2. NEUROPHYSIOLOGY: BTF 4th Edition guidelines, cerebral hemodynamics, ICP
3. NEUROPHARMACOLOGY: Exact dosing for osmotherapy, steroids, AEDs, sedation
4. NEUROPATHOLOGY: WHO 2021 CNS tumor classification
5. LANDMARK_EVIDENCE: Critical trials (ISAT, BRAT, ARUBA, CRASH, etc.)
6. DANGER_ZONES: Subspecialty-specific critical structures
7. PROCEDURAL_TEMPLATES: Step-by-step operative procedures

This ontology enables detection of missing critical information in synthesized content.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set


# =============================================================================
# DOMAIN 1: NEUROANATOMY
# =============================================================================

CRANIAL_FORAMINA: Dict[str, Dict[str, Any]] = {
    "foramen_magnum": {
        "contents": [
            "medulla oblongata",
            "vertebral arteries",
            "spinal accessory nerve (CN XI)",
            "anterior spinal artery",
            "posterior spinal arteries",
            "tectorial membrane",
            "alar ligaments",
        ],
        "clinical_significance": "Chiari malformation, foramen magnum decompression",
    },
    "foramen_ovale": {
        "contents": [
            "mandibular nerve (V3)",
            "accessory meningeal artery",
            "lesser petrosal nerve (occasionally)",
            "emissary veins",
        ],
        "clinical_significance": "Trigeminal rhizotomy, skull base tumors",
    },
    "foramen_spinosum": {
        "contents": [
            "middle meningeal artery",
            "meningeal branch of V3 (nervus spinosus)",
            "middle meningeal vein",
        ],
        "clinical_significance": "Epidural hematoma, dural AVM",
    },
    "foramen_rotundum": {
        "contents": ["maxillary nerve (V2)"],
        "clinical_significance": "V2 schwannoma, skull base approaches",
    },
    "superior_orbital_fissure": {
        "contents": [
            "oculomotor nerve (CN III)",
            "trochlear nerve (CN IV)",
            "ophthalmic nerve (V1)",
            "abducens nerve (CN VI)",
            "superior ophthalmic vein",
            "inferior ophthalmic vein",
            "sympathetic fibers",
        ],
        "clinical_significance": "Cavernous sinus syndrome, orbital apex syndrome",
    },
    "inferior_orbital_fissure": {
        "contents": [
            "maxillary nerve (V2)",
            "infraorbital artery",
            "infraorbital vein",
            "zygomatic nerve",
        ],
        "clinical_significance": "Orbital tumors, blow-out fractures",
    },
    "optic_canal": {
        "contents": [
            "optic nerve (CN II)",
            "ophthalmic artery",
            "sympathetic fibers",
        ],
        "clinical_significance": "Optic nerve decompression, pituitary surgery",
    },
    "internal_acoustic_meatus": {
        "contents": [
            "facial nerve (CN VII)",
            "vestibulocochlear nerve (CN VIII)",
            "labyrinthine artery (AICA branch)",
            "nervus intermedius",
        ],
        "clinical_significance": "Vestibular schwannoma, translabyrinthine approach",
    },
    "jugular_foramen": {
        "contents": [
            "glossopharyngeal nerve (CN IX)",
            "vagus nerve (CN X)",
            "spinal accessory nerve (CN XI)",
            "sigmoid sinus → internal jugular vein",
            "inferior petrosal sinus",
            "posterior meningeal artery",
        ],
        "clinical_significance": "Jugular foramen tumors, glomus jugulare",
    },
    "hypoglossal_canal": {
        "contents": [
            "hypoglossal nerve (CN XII)",
            "meningeal branch of ascending pharyngeal artery",
            "emissary vein",
        ],
        "clinical_significance": "Far lateral approach, hypoglossal schwannoma",
    },
    "stylomastoid_foramen": {
        "contents": ["facial nerve (CN VII)", "stylomastoid artery"],
        "clinical_significance": "Facial nerve decompression, parotid surgery",
    },
    "carotid_canal": {
        "contents": [
            "internal carotid artery",
            "internal carotid venous plexus",
            "sympathetic plexus",
        ],
        "clinical_significance": "Skull base approaches, carotid dissection",
    },
    "foramen_lacerum": {
        "contents": [
            "cartilage (in life)",
            "greater petrosal nerve (traverses above)",
            "deep petrosal nerve",
            "artery of pterygoid canal",
        ],
        "clinical_significance": "Skull base tumors, vidian nerve",
    },
    "cribriform_plate": {
        "contents": [
            "olfactory nerve fibers (CN I)",
            "anterior ethmoidal artery",
            "anterior ethmoidal nerve",
        ],
        "clinical_significance": "CSF rhinorrhea, esthesioneuroblastoma",
    },
}


SKULL_BASE_TRIANGLES: Dict[str, Dict[str, Any]] = {
    "anterolateral_triangle": {
        "borders": ["V3", "greater petrosal nerve", "foramen spinosum"],
        "contents": ["middle meningeal artery", "mandibular nerve"],
        "approach_relevance": "Middle fossa approach",
    },
    "anteromedial_triangle": {
        "borders": ["V3", "petrous ICA (horizontal)", "greater petrosal nerve"],
        "contents": ["horizontal petrous ICA"],
        "approach_relevance": "Kawase approach, petrous apex",
    },
    "posterolateral_triangle": {
        "borders": ["sigmoid sinus", "jugular bulb", "posterior semicircular canal"],
        "contents": ["jugular bulb", "endolymphatic sac"],
        "approach_relevance": "Retrosigmoid approach",
    },
    "posteromedial_triangle": {
        "borders": ["IAC", "jugular bulb", "cochlea"],
        "contents": ["inferior petrosal sinus"],
        "approach_relevance": "Translabyrinthine approach",
    },
    "kawase_triangle": {
        "borders": ["trigeminal impression", "arcuate eminence", "greater petrosal nerve"],
        "contents": ["petrous apex bone"],
        "approach_relevance": "Kawase approach, anterior petrosectomy",
    },
    "glasscock_triangle": {
        "borders": ["foramen spinosum", "foramen ovale", "arcuate eminence"],
        "contents": ["middle meningeal artery", "greater petrosal nerve"],
        "approach_relevance": "Middle fossa approach, GSPN identification",
    },
    "trautmann_triangle": {
        "borders": ["sigmoid sinus", "superior petrosal sinus", "labyrinth block"],
        "contents": ["presigmoid dura"],
        "approach_relevance": "Retrolabyrinthine approach",
    },
}


ARTERIAL_SEGMENTS: Dict[str, Dict[str, Any]] = {
    "ICA": {
        "segments": [
            {"name": "C1 (cervical)", "description": "From bifurcation to carotid canal"},
            {"name": "C2 (petrous)", "description": "Within petrous bone, vertical then horizontal"},
            {"name": "C3 (lacerum)", "description": "Above foramen lacerum"},
            {"name": "C4 (cavernous)", "description": "Within cavernous sinus, S-shaped"},
            {"name": "C5 (clinoid)", "description": "Between proximal/distal dural rings"},
            {"name": "C6 (ophthalmic)", "description": "From distal ring to PCoA"},
            {"name": "C7 (communicating)", "description": "PCoA to bifurcation"},
        ],
        "major_branches": [
            "ophthalmic artery",
            "posterior communicating artery",
            "anterior choroidal artery",
            "superior hypophyseal artery",
            "meningohypophyseal trunk",
            "inferolateral trunk",
        ],
    },
    "MCA": {
        "segments": [
            {"name": "M1 (sphenoidal)", "description": "From ICA bifurcation to genu"},
            {"name": "M2 (insular)", "description": "On insula, within sylvian fissure"},
            {"name": "M3 (opercular)", "description": "On opercula, within sylvian fissure"},
            {"name": "M4 (cortical)", "description": "On cortical surface"},
        ],
        "perforators": [
            "lenticulostriate arteries (lateral group)",
            "early temporal branches",
            "insular perforators",
        ],
    },
    "ACA": {
        "segments": [
            {"name": "A1 (precommunicating)", "description": "From ICA to ACoA"},
            {"name": "A2 (postcommunicating)", "description": "From ACoA to genu of corpus callosum"},
            {"name": "A3 (precallosal)", "description": "Around genu of corpus callosum"},
            {"name": "A4 (supracallosal)", "description": "Above corpus callosum"},
            {"name": "A5 (postcallosal)", "description": "Behind splenium"},
        ],
        "perforators": [
            "medial lenticulostriate arteries",
            "recurrent artery of Heubner",
            "orbitofrontal artery",
            "frontopolar artery",
        ],
    },
    "vertebral": {
        "segments": [
            {"name": "V1 (preforaminal)", "description": "Origin to C6 transverse foramen"},
            {"name": "V2 (foraminal)", "description": "Within transverse foramina C6-C2"},
            {"name": "V3 (atlantic)", "description": "C2 to dura (around atlas)"},
            {"name": "V4 (intradural)", "description": "From dura to vertebrobasilar junction"},
        ],
        "major_branches": [
            "PICA",
            "anterior spinal artery",
            "posterior spinal artery",
            "posterior meningeal artery",
        ],
    },
    "basilar": {
        "segments": [
            {"name": "lower third", "description": "AICA territory"},
            {"name": "middle third", "description": "Pontine perforators"},
            {"name": "upper third", "description": "SCA, P1, thalamoperforators"},
        ],
        "major_branches": [
            "AICA",
            "pontine perforators",
            "SCA",
            "PCA (P1 segment)",
        ],
    },
    "PCA": {
        "segments": [
            {"name": "P1 (precommunicating)", "description": "From basilar to PCoA"},
            {"name": "P2 (ambient)", "description": "Around midbrain in ambient cistern"},
            {"name": "P3 (quadrigeminal)", "description": "In quadrigeminal cistern"},
            {"name": "P4 (calcarine)", "description": "Terminal cortical branches"},
        ],
        "perforators": [
            "thalamoperforators",
            "posterior choroidal arteries",
            "peduncular perforators",
        ],
    },
}


VENOUS_SINUSES: Dict[str, Dict[str, Any]] = {
    "superior_sagittal_sinus": {
        "course": "Midline, from crista galli to torcula",
        "drainage": "Torcula Herophili → transverse sinus",
        "tributaries": ["superior cerebral veins", "diploic veins", "emissary veins"],
        "clinical_significance": "Parasagittal meningioma, sinus thrombosis",
    },
    "inferior_sagittal_sinus": {
        "course": "In free edge of falx cerebri",
        "drainage": "Straight sinus",
        "tributaries": ["medial cerebral veins", "falcine veins"],
    },
    "straight_sinus": {
        "course": "Junction of falx and tentorium",
        "drainage": "Torcula Herophili",
        "tributaries": ["great vein of Galen", "inferior sagittal sinus"],
    },
    "transverse_sinus": {
        "course": "Along tentorial attachment to skull",
        "drainage": "Sigmoid sinus",
        "dominance": "Right > Left in 60% of cases",
    },
    "sigmoid_sinus": {
        "course": "S-shaped, from transverse to jugular foramen",
        "drainage": "Internal jugular vein",
        "clinical_significance": "Sigmoid sinus thrombosis, retrosigmoid approach",
    },
    "cavernous_sinus": {
        "contents": [
            "ICA (cavernous segment)",
            "CN III (superior wall)",
            "CN IV (lateral wall)",
            "CN V1 (lateral wall)",
            "CN V2 (lateral wall, lower)",
            "CN VI (within sinus, lateral to ICA)",
        ],
        "clinical_significance": "Cavernous sinus syndrome, CCF, pituitary adenoma",
    },
    "superior_petrosal_sinus": {
        "course": "Tentorial attachment on petrous ridge",
        "drainage": "Transverse-sigmoid junction",
        "clinical_significance": "Petrosal approach, venous injury",
    },
    "inferior_petrosal_sinus": {
        "course": "Petro-occipital fissure",
        "drainage": "Internal jugular vein (or jugular bulb)",
        "clinical_significance": "Petrosal sinus sampling, jugular foramen tumors",
    },
}


CORTICAL_LOCALIZATION: Dict[str, Dict[str, Any]] = {
    "primary_motor_cortex": {
        "location": "Precentral gyrus",
        "brodmann_area": 4,
        "function": "Voluntary motor control",
        "somatotopy": "Homunculus (leg medial, face lateral)",
    },
    "primary_sensory_cortex": {
        "location": "Postcentral gyrus",
        "brodmann_area": "1, 2, 3",
        "function": "Somatosensory processing",
        "somatotopy": "Homunculus pattern",
    },
    "broca_area": {
        "location": "Inferior frontal gyrus (pars opercularis, triangularis)",
        "brodmann_area": "44, 45",
        "function": "Speech production",
        "dominance": "Left hemisphere (>95%)",
    },
    "wernicke_area": {
        "location": "Superior temporal gyrus (posterior)",
        "brodmann_area": 22,
        "function": "Language comprehension",
        "dominance": "Left hemisphere",
    },
    "primary_visual_cortex": {
        "location": "Calcarine sulcus banks",
        "brodmann_area": 17,
        "function": "Visual processing",
        "retinotopy": "Upper field inferior, lower field superior",
    },
    "primary_auditory_cortex": {
        "location": "Superior temporal gyrus (Heschl gyrus)",
        "brodmann_area": "41, 42",
        "function": "Auditory processing",
    },
}


# =============================================================================
# DOMAIN 2: NEUROPHYSIOLOGY (BTF 4th Edition)
# =============================================================================

ICP_PARAMETERS: Dict[str, Any] = {
    "treatment_threshold": {
        "value": "22 mmHg",
        "source": "BTF 4th Edition 2016",
        "note": "Changed from 20 mmHg in 3rd edition",
    },
    "monitoring_indication": {
        "value": "GCS 3-8 with abnormal CT",
        "alternatives": [
            "GCS 3-8 with normal CT + 2 of: age >40, posturing, SBP <90",
        ],
    },
    "normal_ICP": {
        "value": "7-15 mmHg",
        "note": "Supine adult",
    },
    "CPP_target": {
        "value": "60-70 mmHg",
        "source": "BTF 4th Edition",
        "note": "Avoid aggressive CPP >70 (respiratory complications)",
    },
    "brain_tissue_O2": {
        "threshold": "< 15 mmHg triggers intervention",
        "optimal": "25-35 mmHg",
    },
}


CEREBRAL_HEMODYNAMICS: Dict[str, Any] = {
    "CBF_normal": {
        "value": "50 ml/100g/min",
        "gray_matter": "80 ml/100g/min",
        "white_matter": "20 ml/100g/min",
    },
    "CBF_ischemic_threshold": {
        "value": "< 20 ml/100g/min",
        "note": "Electrical dysfunction",
    },
    "CBF_infarction_threshold": {
        "value": "< 10 ml/100g/min",
        "note": "Membrane failure, irreversible injury",
    },
    "autoregulation_range": {
        "MAP_range": "60-150 mmHg",
        "note": "May be impaired in TBI",
    },
    "CMRO2": {
        "value": "3.5 ml O2/100g/min",
        "note": "Cerebral metabolic rate of oxygen",
    },
}


CSF_DYNAMICS: Dict[str, Any] = {
    "production_rate": {
        "value": "0.3-0.4 ml/min (500 ml/day)",
        "source": "Choroid plexus (70%), ependyma (30%)",
    },
    "total_volume": {
        "value": "150 ml",
        "distribution": "Ventricles 25ml, subarachnoid space 125ml",
    },
    "absorption": {
        "primary": "Arachnoid granulations",
        "secondary": "Lymphatics, nerve root sleeves",
    },
    "turnover": {
        "value": "3-4 times per day",
    },
}


NEUROMONITORING: Dict[str, Dict[str, Any]] = {
    "MEP": {
        "full_name": "Motor Evoked Potentials",
        "stimulus": "Transcranial electrical stimulation",
        "recording": "Muscle (compound muscle action potential)",
        "significance": "Corticospinal tract integrity",
        "warning_criteria": {
            "amplitude": ">50% decrease",
            "threshold": ">100V increase",
        },
    },
    "SSEP": {
        "full_name": "Somatosensory Evoked Potentials",
        "stimulus": "Peripheral nerve (median, ulnar, tibial)",
        "recording": "Cortex (scalp electrodes)",
        "significance": "Dorsal column-medial lemniscus integrity",
        "warning_criteria": {
            "amplitude": ">50% decrease",
            "latency": ">10% increase",
        },
    },
    "BAEP": {
        "full_name": "Brainstem Auditory Evoked Potentials",
        "stimulus": "Click stimulus to ear",
        "recording": "Scalp electrodes",
        "waves": {
            "I": "Auditory nerve",
            "II": "Cochlear nucleus",
            "III": "Superior olivary complex",
            "IV": "Lateral lemniscus",
            "V": "Inferior colliculus",
        },
        "significance": "Posterior fossa surgery, vestibular schwannoma",
    },
    "EMG": {
        "full_name": "Electromyography",
        "types": ["Free-running", "Triggered"],
        "applications": ["Cranial nerve monitoring", "Nerve root monitoring"],
    },
    "EEG": {
        "full_name": "Electroencephalography",
        "applications": ["Burst suppression", "Seizure detection", "Depth of anesthesia"],
    },
}


# =============================================================================
# DOMAIN 3: NEUROPHARMACOLOGY (Exact Dosing)
# =============================================================================

OSMOTHERAPY: Dict[str, Dict[str, Any]] = {
    "mannitol": {
        "class": "Osmotic diuretic",
        "dose": "0.25-1 g/kg IV",
        "onset": "15-30 minutes",
        "duration": "3-6 hours",
        "max_serum_osm": "320 mOsm/L",
        "contraindications": ["Renal failure", "Hypovolemia", "CHF"],
        "monitoring": "Serum osmolality, renal function, volume status",
        "mechanism": "Osmotic gradient, reduces brain water content",
    },
    "hypertonic_saline_3%": {
        "class": "Hypertonic crystalloid",
        "dose": "150-500 ml bolus",
        "target_Na": "145-155 mEq/L",
        "max_Na": "160 mEq/L",
        "rate_correction": "Max 10-12 mEq/L per 24h (risk of ODS)",
        "advantages": ["No osmotic diuresis", "Volume expansion"],
    },
    "hypertonic_saline_23.4%": {
        "class": "Hypertonic crystalloid (concentrated)",
        "dose": "30 ml bolus via central line",
        "indication": "Impending herniation, acute ICP crisis",
        "onset": "Minutes",
        "caution": "Central line required, severe phlebitis if peripheral",
    },
}


STEROIDS: Dict[str, Dict[str, Any]] = {
    "dexamethasone": {
        "class": "Glucocorticoid",
        "loading_dose": "10 mg IV",
        "maintenance_dose": "4 mg IV/PO q6h",
        "tumor_edema_dose": "4-24 mg/day depending on severity",
        "indications": [
            "Vasogenic edema (tumors, abscesses)",
            "Spinal cord injury (controversial)",
            "Bacterial meningitis (before antibiotics)",
        ],
        "contraindication": "Primary brain injury (TBI) - CRASH trial",
        "taper": "Gradual taper over 7-14 days to prevent rebound",
    },
    "methylprednisolone": {
        "class": "Glucocorticoid",
        "NASCIS_protocol": {
            "note": "No longer recommended by most guidelines",
            "historical_dose": "30 mg/kg bolus, 5.4 mg/kg/hr x 23h",
        },
        "current_use": "Largely abandoned for SCI",
    },
}


ANTIEPILEPTICS: Dict[str, Dict[str, Any]] = {
    "levetiracetam": {
        "class": "SV2A modulator",
        "loading_dose": "20-60 mg/kg IV (max 4500 mg)",
        "maintenance_dose": "500-1500 mg BID",
        "max_daily": "3000 mg",
        "advantages": [
            "No hepatic metabolism",
            "No drug interactions",
            "No level monitoring required",
            "IV formulation",
        ],
        "side_effects": ["Irritability", "Behavioral changes"],
    },
    "phenytoin": {
        "class": "Sodium channel blocker",
        "loading_dose": "15-20 mg/kg IV (max rate 50 mg/min)",
        "maintenance_dose": "100 mg TID or 300 mg daily",
        "target_level": "10-20 mcg/ml (total), 1-2 mcg/ml (free)",
        "monitoring": "Levels at 48h, then q2-4 weeks",
        "interactions": "Multiple CYP450 interactions",
        "cautions": ["Arrhythmias during loading", "Purple glove syndrome"],
    },
    "fosphenytoin": {
        "class": "Phenytoin prodrug",
        "dose": "Same as phenytoin in PE (phenytoin equivalents)",
        "max_rate": "150 mg PE/min",
        "advantage": "IM compatible, less tissue necrosis",
    },
    "valproic_acid": {
        "class": "Multiple mechanisms",
        "loading_dose": "20-40 mg/kg IV",
        "maintenance_dose": "15-45 mg/kg/day divided BID-TID",
        "target_level": "50-100 mcg/ml",
        "cautions": ["Hepatotoxicity", "Thrombocytopenia", "Teratogenicity"],
    },
}


SEDATION_ANALGESIA: Dict[str, Dict[str, Any]] = {
    "propofol": {
        "class": "GABA agonist",
        "induction": "1-2 mg/kg IV",
        "maintenance": "25-75 mcg/kg/min (burst suppression: 100-200 mcg/kg/min)",
        "ICP_effect": "Reduces ICP, reduces CMRO2",
        "caution": "Propofol infusion syndrome (>48h, >5 mg/kg/hr)",
        "advantages": ["Rapid onset/offset", "Antiemetic", "Anticonvulsant"],
    },
    "midazolam": {
        "class": "Benzodiazepine",
        "bolus": "0.02-0.08 mg/kg IV",
        "infusion": "0.04-0.2 mg/kg/hr",
        "ICP_effect": "Reduces ICP, reduces CMRO2",
        "reversal": "Flumazenil 0.2 mg IV",
        "caution": "Accumulation with prolonged use",
    },
    "fentanyl": {
        "class": "Opioid",
        "bolus": "1-2 mcg/kg IV",
        "infusion": "0.5-2 mcg/kg/hr",
        "ICP_effect": "Neutral if ventilation maintained",
        "reversal": "Naloxone 0.04-0.4 mg IV",
    },
    "dexmedetomidine": {
        "class": "Alpha-2 agonist",
        "loading": "1 mcg/kg over 10 min (optional)",
        "infusion": "0.2-0.7 mcg/kg/hr",
        "advantages": ["Cooperative sedation", "Minimal respiratory depression"],
        "caution": "Bradycardia, hypotension",
    },
}


REVERSAL_AGENTS: Dict[str, Dict[str, Any]] = {
    "protamine": {
        "reverses": "Heparin",
        "dose": "1 mg per 100 units heparin",
        "caution": "Anaphylaxis (fish allergy, prior protamine)",
    },
    "vitamin_K": {
        "reverses": "Warfarin",
        "dose": "10 mg IV slow infusion",
        "onset": "6-12 hours",
    },
    "idarucizumab": {
        "reverses": "Dabigatran",
        "dose": "5 g IV (2 x 2.5g vials)",
        "onset": "Minutes",
    },
    "andexanet_alfa": {
        "reverses": "Factor Xa inhibitors (rivaroxaban, apixaban)",
        "dose": "Varies by drug and timing",
    },
    "4_factor_PCC": {
        "reverses": "Warfarin, Factor Xa inhibitors (off-label)",
        "dose": "25-50 units/kg",
        "contents": ["II", "VII", "IX", "X", "Protein C", "Protein S"],
    },
}


# =============================================================================
# DOMAIN 4: NEUROPATHOLOGY (WHO 2021)
# =============================================================================

WHO_2021_CNS_TUMORS: Dict[str, Dict[str, Any]] = {
    "glioblastoma_IDH_wildtype": {
        "grade": 4,
        "required": "IDH wildtype status",
        "molecular_features": [
            "TERT promoter mutation",
            "+7/-10 chromosome changes",
            "EGFR amplification",
        ],
        "note": "Any one molecular feature sufficient for grade 4 if IDH wildtype",
    },
    "astrocytoma_IDH_mutant": {
        "grade_2": "Without high-grade features",
        "grade_3": "With significant mitotic activity",
        "grade_4": "With CDKN2A/B homozygous deletion",
        "required": "IDH mutant, ATRX loss (usually)",
        "note": "CDKN2A/B deletion = grade 4 regardless of histology",
    },
    "oligodendroglioma_IDH_mutant_1p19q_codeleted": {
        "required": ["IDH mutation", "1p/19q codeletion"],
        "grade_2": "Without anaplastic features",
        "grade_3": "With anaplastic features (brisk mitoses, necrosis, MVP)",
    },
    "meningioma": {
        "grade_1": "Typical (9 subtypes)",
        "grade_2": [
            "Atypical (4-19 mitoses/10 HPF, brain invasion, or 3+ atypical features)",
            "Clear cell",
            "Chordoid",
        ],
        "grade_3": [
            "Anaplastic (20+ mitoses/10 HPF)",
            "Papillary",
            "Rhabdoid",
        ],
        "molecular": "TERT promoter mutation = higher grade behavior",
    },
    "medulloblastoma": {
        "molecular_groups": [
            {"name": "WNT-activated", "prognosis": "Excellent"},
            {"name": "SHH-activated (TP53 wildtype)", "prognosis": "Intermediate"},
            {"name": "SHH-activated (TP53 mutant)", "prognosis": "Poor"},
            {"name": "Group 3", "prognosis": "Poor (MYC amplification)"},
            {"name": "Group 4", "prognosis": "Intermediate"},
        ],
    },
    "ependymoma": {
        "supratentorial": "ZFTA fusion (previously RELA) or YAP1 fusion",
        "posterior_fossa": "PFA (poor prognosis) vs PFB (better prognosis)",
        "spinal": "Myxopapillary ependymoma now grade 2",
    },
}


# =============================================================================
# DOMAIN 5: LANDMARK EVIDENCE
# =============================================================================

LANDMARK_TRIALS: Dict[str, Dict[str, Any]] = {
    "ISAT": {
        "full_name": "International Subarachnoid Aneurysm Trial",
        "year": 2002,
        "finding": "Coiling superior to clipping for ruptured aneurysms at 1 year",
        "outcome": "23.7% vs 30.6% death/dependency",
        "limitation": "Selection bias, predominantly anterior circulation",
    },
    "BRAT": {
        "full_name": "Barrow Ruptured Aneurysm Trial",
        "year": 2015,
        "finding": "Similar outcomes coil vs clip at 6 years follow-up",
        "outcome": "No significant difference in mRS",
        "note": "Higher retreatment rates with coiling",
    },
    "ARUBA": {
        "full_name": "A Randomized Trial of Unruptured Brain AVMs",
        "year": 2014,
        "finding": "Medical management superior to intervention for unruptured AVMs",
        "outcome": "30.7% (intervention) vs 10.1% (medical) stroke/death at 33mo",
        "criticism": "Short follow-up, heterogeneous interventions",
    },
    "STICH": {
        "full_name": "Surgical Trial in Intracerebral Haemorrhage",
        "year": 2005,
        "finding": "Early surgery not beneficial for supratentorial ICH",
        "outcome": "26% vs 24% favorable outcome",
    },
    "STICH_II": {
        "full_name": "STICH II",
        "year": 2013,
        "finding": "Early surgery not beneficial for lobar ICH without IVH",
        "patient_population": "Lobar hemorrhage, 10-100ml, within 1cm of surface",
    },
    "DECRA": {
        "full_name": "Decompressive Craniectomy Trial",
        "year": 2011,
        "finding": "Bifrontal DC reduces ICP but not 6-month outcomes",
        "outcome": "More unfavorable outcomes with surgery",
        "criticism": "Early surgery, low ICP threshold (>20mmHg for 15min)",
    },
    "RESCUEicp": {
        "full_name": "Randomised Evaluation of Surgery with Craniectomy for Uncontrollable Elevation of ICP",
        "year": 2016,
        "finding": "DC reduces mortality but increases severe disability",
        "outcome": "26.9% mortality (surgery) vs 48.9% (medical)",
        "conclusion": "Lower mortality traded for higher vegetative state",
    },
    "CRASH": {
        "full_name": "Corticosteroid Randomisation After Significant Head Injury",
        "year": 2004,
        "finding": "Steroids harmful in TBI",
        "outcome": "21.1% vs 17.9% mortality at 2 weeks",
        "implication": "Steroids contraindicated in TBI",
    },
    "NASCIS": {
        "full_name": "National Acute Spinal Cord Injury Study",
        "years": "1984 (I), 1990 (II), 1997 (III)",
        "historical_protocol": "Methylprednisolone 30mg/kg + 5.4mg/kg/hr x 23-48h",
        "current_status": "No longer recommended by most guidelines",
        "criticism": "Post-hoc analysis, marginal benefit, significant complications",
    },
    "INTERACT2": {
        "full_name": "Intensive Blood Pressure Reduction in Acute Cerebral Haemorrhage Trial",
        "year": 2013,
        "finding": "Intensive BP lowering (<140mmHg) safe, improved functional outcomes",
        "protocol": "SBP <140 within 1 hour",
    },
    "ATACH2": {
        "full_name": "Antihypertensive Treatment of Acute Cerebral Hemorrhage II",
        "year": 2016,
        "finding": "Intensive BP reduction (110-139) not superior to standard (140-179)",
        "outcome": "No benefit, more renal adverse events",
    },
    "CAST": {
        "full_name": "Chinese Acute Stroke Trial",
        "year": 1997,
        "finding": "Aspirin reduces recurrence/death in acute ischemic stroke",
        "dose": "160 mg started within 48h",
    },
    "IST": {
        "full_name": "International Stroke Trial",
        "year": 1997,
        "finding": "Aspirin reduces death/dependency in acute ischemic stroke",
        "additional": "Heparin showed no net benefit",
    },
}


# =============================================================================
# DOMAIN 6: DANGER ZONES (by Subspecialty)
# =============================================================================

DANGER_ZONES: Dict[str, List[Dict[str, str]]] = {
    "skull_base": [
        {"structure": "petrous ICA", "consequence": "Catastrophic hemorrhage, stroke"},
        {"structure": "cavernous sinus", "consequence": "CCF, cranial neuropathies"},
        {"structure": "vidian nerve", "consequence": "Dry eye (loss of lacrimation)"},
        {"structure": "greater petrosal nerve", "consequence": "Dry eye"},
        {"structure": "cochlea", "consequence": "Sensorineural hearing loss"},
        {"structure": "semicircular canals", "consequence": "Vertigo, hearing loss"},
        {"structure": "facial nerve", "consequence": "Facial paralysis"},
        {"structure": "jugular bulb", "consequence": "Hemorrhage, air embolism"},
    ],
    "vascular": [
        {"structure": "lenticulostriate arteries", "consequence": "Basal ganglia stroke"},
        {"structure": "thalamoperforators", "consequence": "Thalamic stroke"},
        {"structure": "recurrent artery of Heubner", "consequence": "Caudate infarct, abulia"},
        {"structure": "anterior choroidal artery", "consequence": "Hemiplegia, hemianopia"},
        {"structure": "PICA origin", "consequence": "Lateral medullary syndrome"},
        {"structure": "M1 perforators", "consequence": "Internal capsule stroke"},
        {"structure": "basilar perforators", "consequence": "Pontine stroke"},
    ],
    "spine": [
        {"structure": "vertebral artery V2", "consequence": "Stroke, dissection"},
        {"structure": "C5 nerve root", "consequence": "Deltoid weakness (C5 palsy)"},
        {"structure": "artery of Adamkiewicz", "consequence": "Anterior spinal artery syndrome"},
        {"structure": "conus medullaris", "consequence": "Bowel/bladder dysfunction"},
        {"structure": "cauda equina", "consequence": "Radiculopathy, incontinence"},
        {"structure": "dura (at T12-L2)", "consequence": "CSF leak, meningitis"},
    ],
    "tumor": [
        {"structure": "optic apparatus", "consequence": "Blindness"},
        {"structure": "hypothalamus", "consequence": "DI, panhypopituitarism, hyperphagia"},
        {"structure": "brainstem", "consequence": "Cranial neuropathies, death"},
        {"structure": "eloquent cortex", "consequence": "Motor/speech deficits"},
        {"structure": "pericallosal arteries", "consequence": "Supplementary motor syndrome"},
        {"structure": "vein of Labbe", "consequence": "Venous infarct"},
        {"structure": "vein of Trolard", "consequence": "Venous infarct"},
    ],
    "functional": [
        {"structure": "internal capsule", "consequence": "Hemiplegia"},
        {"structure": "thalamus", "consequence": "Sensory loss, pain syndrome"},
        {"structure": "basal ganglia", "consequence": "Movement disorders"},
        {"structure": "subthalamic nucleus", "consequence": "Hemiballismus"},
    ],
    "pediatric": [
        {"structure": "growing skull", "consequence": "Craniofacial deformity"},
        {"structure": "unfused sutures", "consequence": "Skull base instability"},
        {"structure": "developing brain", "consequence": "Cognitive deficits"},
    ],
}


# =============================================================================
# DOMAIN 7: PROCEDURAL TEMPLATES
# =============================================================================

PROCEDURAL_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "pterional_craniotomy": {
        "indication": "Anterior circulation aneurysms, suprasellar tumors, orbitofrontal lesions",
        "positioning": [
            "Supine",
            "Head rotated 30-45 degrees contralateral",
            "Vertex down 15-20 degrees",
            "Malar eminence highest point",
            "Mayfield pins",
        ],
        "incision": [
            "Start 1cm anterior to tragus",
            "Curve posteriorly behind hairline",
            "End at midline or contralateral",
            "Stay behind temporal branch of facial nerve",
        ],
        "key_steps": [
            "Interfascial dissection to protect frontalis branch",
            "Subperiosteal elevation of temporalis muscle",
            "MacCarty keyhole placement",
            "Single burr hole at keyhole",
            "Craniotomy with footplate",
            "Sphenoid wing drilling flush with orbital roof",
            "Anterior clinoidectomy if needed",
            "Sylvian fissure opening",
            "Arachnoid dissection under microscope",
        ],
        "danger_zones": [
            "Frontalis branch CN VII (temporal fat pad)",
            "Middle meningeal artery",
            "Sphenoid wing veins",
            "Optic nerve",
            "ICA",
        ],
        "instruments": [
            "Midas Rex (M8 burr)",
            "Perforator",
            "Craniotome with footplate",
            "Kerrison rongeurs",
            "High-speed drill",
            "Bipolar forceps",
            "Microsurgical instruments",
        ],
        "bailout": [
            "Premature aneurysm rupture: temporary clips, suction, proximal control",
            "Brain swelling: mannitol, CSF drainage, hyperventilation",
            "Dural venous sinus injury: hemostatic agents, repair",
        ],
    },
    "retrosigmoid_craniotomy": {
        "indication": "CPA tumors, MVD, petroclival meningioma",
        "positioning": [
            "Lateral decubitus or park bench",
            "Head flexed, vertex tilted down",
            "Mastoid highest point",
            "Shoulder pulled down",
        ],
        "key_steps": [
            "Linear or curvilinear incision behind mastoid",
            "Suboccipital muscle dissection",
            "Single burr hole at asterion",
            "Craniotomy exposing transverse-sigmoid junction",
            "Dural opening C-shaped based on sinuses",
            "CSF release from cisterna magna",
            "Retractor-free technique preferred",
        ],
        "danger_zones": [
            "Sigmoid sinus",
            "Transverse sinus",
            "AICA loop",
            "Facial nerve (CN VII)",
            "Cochlear nerve (CN VIII)",
            "Lower cranial nerves (IX, X, XI)",
        ],
        "instruments": [
            "Midas Rex",
            "High-speed drill",
            "CUSA",
            "Facial nerve stimulator",
            "Neuromonitoring (BAEP, facial EMG)",
        ],
        "bailout": [
            "Sigmoid sinus injury: packing, surgicel, repair",
            "Air embolism: flood field, Durant position, aspirate RA",
            "AICA injury: temporary clip, repair vs sacrifice if collateral flow",
        ],
    },
    "lumbar_microdiscectomy": {
        "indication": "Herniated lumbar disc with radiculopathy failing conservative treatment",
        "positioning": [
            "Prone on Wilson frame or Andrews table",
            "Abdomen free",
            "Hips and knees flexed",
            "Arms tucked or on armboards",
        ],
        "key_steps": [
            "Fluoroscopic level confirmation",
            "Midline incision (2-3 cm)",
            "Subperiosteal dissection to lamina",
            "Laminotomy (superior edge of inferior lamina)",
            "Ligamentum flavum removal",
            "Nerve root identification and gentle retraction",
            "Annulotomy and discectomy",
            "Exploration of foramen",
            "Hemostasis and closure",
        ],
        "danger_zones": [
            "Nerve root (traction injury)",
            "Dura (CSF leak)",
            "Wrong level",
            "Iliac vessels (anterior perforation - rare)",
        ],
        "instruments": [
            "Tubular retractor system",
            "Operating microscope or loupes",
            "Kerrison rongeurs",
            "Pituitary rongeurs",
            "Nerve root retractor",
            "Penfield dissectors",
        ],
        "bailout": [
            "Durotomy: primary repair, fibrin glue, fat graft",
            "Nerve root injury: avoid further manipulation, steroids",
            "Persistent symptoms: consider foraminal stenosis, recurrent disc",
        ],
    },
    "ACDF": {
        "indication": "Cervical radiculopathy/myelopathy from disc herniation or spondylosis",
        "positioning": [
            "Supine with shoulder roll",
            "Slight neck extension",
            "Head on horseshoe or pins",
            "Tape shoulders down for lower cervical",
        ],
        "key_steps": [
            "Transverse skin incision at appropriate level",
            "Platysma divided along fibers",
            "Develop plane medial to carotid sheath",
            "Longus colli dissection and retractor placement",
            "Fluoroscopic confirmation",
            "Complete discectomy to PLL",
            "Decompression of neural elements",
            "Endplate preparation",
            "Cage/graft placement",
            "Plate and screw fixation",
        ],
        "danger_zones": [
            "Recurrent laryngeal nerve",
            "Esophagus",
            "Carotid artery",
            "Vertebral artery (uncovertebral joint)",
            "Spinal cord",
        ],
        "instruments": [
            "Caspar retractor system",
            "Operating microscope",
            "High-speed drill",
            "Kerrison rongeurs",
            "Curettes",
            "Cervical plate system",
        ],
        "bailout": [
            "Esophageal injury: primary repair, drainage, antibiotics",
            "Vertebral artery injury: hemostatic agents, possible sacrifice",
            "CSF leak: primary repair, lumbar drain",
            "Recurrent laryngeal nerve injury: observation, ENT consult",
        ],
    },
    "EVD_placement": {
        "indication": "Hydrocephalus, ICP monitoring, intraventricular hemorrhage",
        "positioning": [
            "Supine with head of bed elevated 30 degrees",
            "Head in neutral position",
        ],
        "key_steps": [
            "Identify Kocher's point (11cm posterior to nasion, 3cm lateral to midline)",
            "Prep and drape",
            "Local anesthesia",
            "Twist drill or burr hole",
            "Dural incision",
            "Catheter trajectory: toward medial canthus (coronal), toward tragus (sagittal)",
            "Target depth: 6-7 cm (adjusted for ventricle size)",
            "Confirm CSF flow",
            "Tunnel catheter and connect to drainage system",
        ],
        "danger_zones": [
            "Motor cortex",
            "Superior sagittal sinus (midline)",
            "Intracerebral hemorrhage track",
        ],
        "bailout": [
            "Hemorrhage: maintain trajectory, ICP monitoring",
            "Failed cannulation: reposition, consider image guidance",
            "Infection: antibiotics, possible removal/replacement",
        ],
    },
    "decompressive_craniectomy": {
        "indication": "Malignant MCA infarction, severe TBI with refractory ICP",
        "positioning": [
            "Supine with head turned",
            "May use lateral position",
        ],
        "key_steps": [
            "Large question-mark incision",
            "Wide bone flap (at least 12x15 cm)",
            "Temporal decompression to middle fossa floor",
            "Preserve bridging veins if possible",
            "Duraplasty with augmentation (pericranium, dural substitute)",
            "Bone flap storage (abdomen, freezer)",
        ],
        "danger_zones": [
            "Middle meningeal artery",
            "Cortical veins",
            "Sagittal sinus (if bilateral)",
        ],
        "bailout": [
            "Significant brain herniation through craniectomy: staged closure",
            "Hemorrhage: meticulous hemostasis, possible return to OR",
        ],
    },
}


# =============================================================================
# TEMPLATE-SPECIFIC REQUIRED ELEMENTS
# =============================================================================

DISORDER_REQUIRED_ELEMENTS: Dict[str, List[str]] = {
    "epidemiology": ["incidence", "prevalence", "demographics", "risk factors"],
    "pathophysiology": ["mechanism", "etiology", "pathogenesis", "natural history"],
    "clinical_presentation": ["symptoms", "signs", "clinical features", "syndromes"],
    "diagnosis": ["imaging", "laboratory", "workup", "differential diagnosis"],
    "treatment": ["medical management", "surgical indications", "surgical options"],
    "prognosis": ["outcomes", "survival", "recurrence", "follow-up"],
}

ANATOMY_REQUIRED_ELEMENTS: Dict[str, List[str]] = {
    "boundaries": ["borders", "limits", "extent"],
    "contents": ["structures", "nerves", "vessels", "organs"],
    "relations": ["adjacent structures", "relationships", "proximity"],
    "blood_supply": ["arterial supply", "venous drainage"],
    "innervation": ["nerve supply", "autonomic innervation"],
    "clinical_significance": ["surgical approaches", "clinical correlations"],
}

CONCEPT_REQUIRED_ELEMENTS: Dict[str, List[str]] = {
    "definition": ["defined as", "refers to", "meaning"],
    "classification": ["types", "categories", "subtypes", "grades"],
    "mechanism": ["how it works", "mechanism of action", "pathway"],
    "clinical_relevance": ["importance", "significance", "implications"],
}


# =============================================================================
# HELPER CLASS: Ontology Accessor
# =============================================================================

@dataclass
class NeurosurgicalOntology:
    """
    Unified accessor for the neurosurgical ontology.

    Provides structured access to all 7 domains for gap detection.
    """

    @staticmethod
    def get_danger_zones(subspecialty: str) -> List[Dict[str, str]]:
        """Get danger zones for a subspecialty."""
        return DANGER_ZONES.get(subspecialty, [])

    @staticmethod
    def get_procedural_template(procedure: str) -> Optional[Dict[str, Any]]:
        """Get procedural template by name."""
        return PROCEDURAL_TEMPLATES.get(procedure)

    @staticmethod
    def get_landmark_trial(trial_name: str) -> Optional[Dict[str, Any]]:
        """Get landmark trial information."""
        return LANDMARK_TRIALS.get(trial_name)

    @staticmethod
    def get_drug_info(drug_name: str) -> Optional[Dict[str, Any]]:
        """Get drug information from pharmacology ontology."""
        for category in [OSMOTHERAPY, STEROIDS, ANTIEPILEPTICS, SEDATION_ANALGESIA]:
            if drug_name in category:
                return category[drug_name]
        return None

    @staticmethod
    def get_arterial_segments(vessel: str) -> Optional[Dict[str, Any]]:
        """Get arterial segment information."""
        return ARTERIAL_SEGMENTS.get(vessel)

    @staticmethod
    def get_foramen_contents(foramen: str) -> Optional[Dict[str, Any]]:
        """Get foramen contents."""
        return CRANIAL_FORAMINA.get(foramen)

    @staticmethod
    def get_who_tumor_classification(tumor_type: str) -> Optional[Dict[str, Any]]:
        """Get WHO 2021 tumor classification."""
        return WHO_2021_CNS_TUMORS.get(tumor_type)

    @staticmethod
    def get_icp_parameters() -> Dict[str, Any]:
        """Get ICP management parameters (BTF 4th Edition)."""
        return ICP_PARAMETERS

    @staticmethod
    def list_procedures() -> List[str]:
        """List all available procedural templates."""
        return list(PROCEDURAL_TEMPLATES.keys())

    @staticmethod
    def list_subspecialties() -> List[str]:
        """List all subspecialties with danger zones."""
        return list(DANGER_ZONES.keys())

    @staticmethod
    def list_landmark_trials() -> List[str]:
        """List all landmark trials."""
        return list(LANDMARK_TRIALS.keys())

    @staticmethod
    def get_required_elements(template_type: str) -> Dict[str, List[str]]:
        """Get required elements for a template type."""
        mapping = {
            "DISORDER": DISORDER_REQUIRED_ELEMENTS,
            "ANATOMY": ANATOMY_REQUIRED_ELEMENTS,
            "CONCEPT": CONCEPT_REQUIRED_ELEMENTS,
        }
        return mapping.get(template_type, {})
