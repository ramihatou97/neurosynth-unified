"""
NeuroSynth - Reference Library Scanner
======================================

Scans PDF library to extract rich metadata for browsing and filtering
BEFORE ingestion. Enables users to:
- Browse available references
- Filter by specialty, subject, document type
- Search chapter titles and content previews
- Select specific documents or chapters for ingestion

This is a lightweight pre-ingestion tool that extracts metadata only,
without full chunking or embedding.
"""

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Tuple, Callable
from uuid import uuid4

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

logger = logging.getLogger(__name__)


# =============================================================================
# Enums & Constants
# =============================================================================

class DocumentType(str, Enum):
    """Type of reference document."""
    TEXTBOOK = "textbook"
    ATLAS = "atlas"
    HANDBOOK = "handbook"
    JOURNAL_ARTICLE = "journal_article"
    REVIEW = "review"
    CASE_SERIES = "case_series"
    GUIDELINES = "guidelines"
    EXAM_QUESTIONS = "exam_questions"
    COURSE_MATERIAL = "course_material"
    LECTURE_NOTES = "lecture_notes"
    OPERATIVE_VIDEO_COMPANION = "operative_video_companion"
    CHAPTER = "chapter"
    UNKNOWN = "unknown"


class Specialty(str, Enum):
    """Neurosurgical subspecialties."""
    VASCULAR = "vascular"
    TUMOR = "tumor"
    SKULL_BASE = "skull_base"
    SPINE = "spine"
    FUNCTIONAL = "functional"
    PEDIATRIC = "pediatric"
    TRAUMA = "trauma"
    PERIPHERAL_NERVE = "peripheral_nerve"
    NEURORADIOLOGY = "neuroradiology"
    NEUROANATOMY = "neuroanatomy"
    GENERAL = "general"


# Authority detection patterns - Tier 1 = Masters, Tier 2 = Major, etc.
AUTHORITY_PATTERNS: Dict[str, Tuple[float, List[str]]] = {
    # Tier 1: Definitive Masters (1.0)
    "RHOTON": (1.00, ["rhoton", "microsurgical anatomy"]),
    "LAWTON": (1.00, ["lawton", "seven aneurysms", "7 aneurysms"]),
    "SAMII": (1.00, ["samii"]),
    "SPETZLER": (1.00, ["spetzler"]),
    "AL_MEFTY": (1.00, ["al-mefty", "almefty", "al mefty"]),
    # Tier 2: Major Textbooks (0.9)
    "YOUMANS": (0.90, ["youmans", "neurological surgery"]),
    "SCHMIDEK": (0.90, ["schmidek", "operative neurosurgical"]),
    "CONNOLLY": (0.90, ["connolly"]),
    "SEKHAR": (0.90, ["sekhar"]),
    # Tier 3: Standard References (0.8-0.85)
    "GREENBERG": (0.85, ["greenberg", "handbook of neurosurgery"]),
    "BENZEL": (0.80, ["benzel", "spine surgery"]),
    "OSBORN": (0.85, ["osborn", "diagnostic imaging"]),
    "WINN": (0.85, ["winn"]),
    "ROTHMAN": (0.85, ["rothman", "simeone"]),
    # Tier 4: Other
    "JOURNAL": (0.75, ["journal", "j neurosurg", "neurosurgery"]),
    "AO_SPINE": (0.80, ["ao spine", "aospine"]),
}

# Specialty detection keywords
SPECIALTY_KEYWORDS: Dict[Specialty, List[str]] = {
    Specialty.VASCULAR: [
        "aneurysm", "avm", "bypass", "carotid", "stroke", "hemorrhage",
        "vasospasm", "moyamoya", "dural fistula", "subarachnoid", "sah",
        "arteriovenous", "ica", "mca", "aca", "pca", "basilar", "vascular"
    ],
    Specialty.TUMOR: [
        "glioma", "meningioma", "resection", "tumor", "oncology",
        "glioblastoma", "schwannoma", "metastasis", "craniopharyngioma",
        "pituitary adenoma", "vestibular schwannoma", "astrocytoma"
    ],
    Specialty.SPINE: [
        "cervical", "lumbar", "thoracic", "fusion", "disc", "laminectomy",
        "spondylosis", "myelopathy", "stenosis", "scoliosis", "kyphosis",
        "vertebral", "spinal cord", "foraminotomy", "corpectomy", "spine"
    ],
    Specialty.FUNCTIONAL: [
        "dbs", "epilepsy", "parkinson", "stimulation", "tremor",
        "movement disorder", "dystonia", "ablation", "seizure",
        "deep brain", "vagus nerve", "responsive neurostimulation"
    ],
    Specialty.SKULL_BASE: [
        "skull base", "pituitary", "acoustic", "petroclival",
        "transsphenoidal", "endoscopic endonasal", "chordoma",
        "jugular foramen", "cavernous sinus", "petrous", "clivus"
    ],
    Specialty.PEDIATRIC: [
        "pediatric", "child", "congenital", "shunt", "craniosynostosis",
        "chiari", "myelomeningocele", "hydrocephalus", "tethered cord",
        "encephalocele", "arachnoid cyst"
    ],
    Specialty.TRAUMA: [
        "trauma", "tbi", "subdural", "epidural", "contusion",
        "decompressive", "icp", "herniation", "skull fracture",
        "penetrating", "blast injury", "diffuse axonal"
    ],
    Specialty.PERIPHERAL_NERVE: [
        "peripheral nerve", "brachial plexus", "carpal tunnel",
        "ulnar", "nerve repair", "neuroma", "entrapment"
    ],
    Specialty.NEURORADIOLOGY: [
        "imaging", "mri", "ct scan", "angiography", "radiological",
        "neuroradiology", "diffusion", "perfusion"
    ],
    Specialty.NEUROANATOMY: [
        "anatomy", "neuroanatomy", "dissection", "cadaver",
        "microsurgical anatomy", "topography", "morphology"
    ],
}

# Subspecialty detection keywords (nested under specialties)
SUBSPECIALTY_KEYWORDS: Dict[Specialty, Dict[str, List[str]]] = {
    Specialty.VASCULAR: {
        "aneurysm": ["aneurysm", "clipping", "coiling", "sah", "subarachnoid", "aneurysmal"],
        "avm": ["avm", "arteriovenous malformation", "nidus", "spetzler", "embolization"],
        "cavernoma": ["cavernoma", "cavernous malformation", "ccm", "developmental venous"],
        "stroke": ["stroke", "ischemic", "thrombolysis", "thrombectomy", "penumbra"],
        "moyamoya": ["moyamoya", "revascularization", "encephaloduroarteriosynangiosis"],
        "bypass": ["bypass", "ec-ic", "sta-mca", "revascularization", "extracranial"],
        "dural_fistula": ["dural fistula", "davf", "dural arteriovenous", "carotid cavernous"],
        "hemorrhage": ["hemorrhage", "ich", "intracerebral", "hematoma evacuation"],
    },
    Specialty.TUMOR: {
        "glioma": ["glioma", "glioblastoma", "astrocytoma", "oligodendroglioma", "gbm", "who grade"],
        "meningioma": ["meningioma", "convexity", "parasagittal", "falcine", "sphenoid wing"],
        "metastasis": ["metastasis", "metastatic", "brain met", "stereotactic radiosurgery"],
        "pituitary": ["pituitary", "adenoma", "prolactinoma", "cushing", "acromegaly", "transsphenoidal"],
        "schwannoma": ["schwannoma", "vestibular", "acoustic", "facial nerve"],
        "pediatric_tumor": ["medulloblastoma", "ependymoma", "pilocytic", "pediatric brain tumor"],
        "skull_base_tumor": ["chordoma", "craniopharyngioma", "petroclival", "olfactory groove"],
        "lymphoma": ["lymphoma", "pcnsl", "primary cns lymphoma"],
    },
    Specialty.SPINE: {
        "cervical": ["cervical", "acdf", "corpectomy", "laminoplasty", "odontoid", "c1-c2"],
        "lumbar": ["lumbar", "discectomy", "laminectomy", "fusion", "l4-l5", "l5-s1", "sciatica"],
        "thoracic": ["thoracic", "thoracolumbar", "kyphosis", "t-spine"],
        "deformity": ["scoliosis", "kyphosis", "deformity", "spinal deformity", "correction"],
        "spinal_tumor": ["spinal tumor", "intradural", "intramedullary", "extradural", "spinal metastasis"],
        "trauma_spine": ["spinal trauma", "burst fracture", "chance fracture", "spinal cord injury"],
        "degenerative": ["stenosis", "spondylosis", "spondylolisthesis", "degenerative", "myelopathy"],
        "minimally_invasive": ["mis", "minimally invasive", "tubular", "percutaneous", "endoscopic spine"],
    },
    Specialty.SKULL_BASE: {
        "anterior_skull_base": ["anterior skull base", "olfactory groove", "planum", "cribriform"],
        "middle_fossa": ["middle fossa", "meckel cave", "trigeminal", "cavernous sinus"],
        "posterior_fossa": ["posterior fossa", "cerebellopontine", "cpa", "foramen magnum"],
        "pituitary_sb": ["sellar", "suprasellar", "transsphenoidal", "endoscopic endonasal"],
        "lateral_skull_base": ["petrous", "jugular foramen", "infratemporal", "far lateral"],
        "clival": ["clivus", "clival", "chordoma", "petroclival"],
    },
    Specialty.FUNCTIONAL: {
        "dbs": ["dbs", "deep brain stimulation", "subthalamic", "stn", "gpi", "vim"],
        "epilepsy": ["epilepsy", "seizure", "temporal lobe", "amygdalohippocampectomy", "laser ablation"],
        "movement_disorder": ["parkinson", "tremor", "dystonia", "movement disorder", "essential tremor"],
        "pain": ["pain", "trigeminal neuralgia", "spinal cord stimulation", "rhizotomy"],
        "spasticity": ["spasticity", "baclofen", "intrathecal", "selective dorsal rhizotomy"],
        "psychiatric": ["ocd", "depression", "psychiatric", "cingulotomy"],
    },
    Specialty.PEDIATRIC: {
        "hydrocephalus": ["hydrocephalus", "shunt", "ventriculoperitoneal", "etv", "aqueductal"],
        "craniosynostosis": ["craniosynostosis", "sagittal", "metopic", "coronal", "cranial vault"],
        "chiari": ["chiari", "tonsillar herniation", "syringomyelia", "tethered cord"],
        "myelomeningocele": ["myelomeningocele", "spina bifida", "neural tube", "fetal surgery"],
        "pediatric_vascular": ["moyamoya pediatric", "vein of galen", "pediatric avm"],
        "congenital": ["congenital", "encephalocele", "dermoid", "arachnoid cyst"],
    },
    Specialty.TRAUMA: {
        "tbi": ["tbi", "traumatic brain injury", "concussion", "diffuse axonal"],
        "subdural": ["subdural", "sdh", "chronic subdural", "acute subdural"],
        "epidural": ["epidural", "edh", "middle meningeal", "epidural hematoma"],
        "skull_fracture": ["skull fracture", "depressed fracture", "basal skull", "csf leak"],
        "penetrating": ["penetrating", "gunshot", "blast injury", "foreign body"],
        "icp_management": ["icp", "intracranial pressure", "decompressive craniectomy", "bolt"],
    },
    Specialty.PERIPHERAL_NERVE: {
        "brachial_plexus": ["brachial plexus", "erb palsy", "nerve transfer", "plexopathy"],
        "carpal_tunnel": ["carpal tunnel", "median nerve", "cts", "cubital tunnel"],
        "nerve_tumor": ["neuroma", "schwannoma", "mpnst", "neurofibromatosis"],
        "nerve_repair": ["nerve repair", "nerve graft", "conduit", "neurorraphy"],
        "entrapment": ["entrapment", "ulnar nerve", "peroneal", "tarsal tunnel"],
    },
    Specialty.NEURORADIOLOGY: {
        "angiography": ["angiography", "dsa", "cerebral angiogram", "catheter"],
        "ct_imaging": ["ct", "computed tomography", "ct angio", "cta", "ct perfusion"],
        "mri_imaging": ["mri", "magnetic resonance", "flair", "dwi", "diffusion weighted"],
        "interventional": ["interventional", "embolization", "coiling", "stenting", "flow diverter"],
        "nuclear_medicine": ["pet", "spect", "nuclear", "fdg"],
    },
    Specialty.NEUROANATOMY: {
        "microsurgical_anatomy": ["microsurgical anatomy", "rhoton", "cadaveric", "dissection"],
        "vascular_anatomy": ["circle of willis", "arterial", "venous anatomy", "dural sinus"],
        "cranial_nerves": ["cranial nerve", "cn", "facial nerve", "trigeminal anatomy"],
        "white_matter": ["white matter", "tract", "fiber", "connectome", "dti"],
        "surgical_approaches": ["approach", "corridor", "pterional", "retrosigmoid", "orbitozygomatic"],
    },
}

# Evidence level patterns (for journal articles)
EVIDENCE_LEVEL_PATTERNS: Dict[str, List[str]] = {
    "Ia": ["meta-analysis", "systematic review", "cochrane"],
    "Ib": ["randomized controlled trial", "rct", "double-blind", "placebo-controlled"],
    "IIa": ["prospective cohort", "controlled trial without randomization"],
    "IIb": ["cohort study", "case-control", "retrospective cohort"],
    "III": ["case series", "comparative study", "observational"],
    "IV": ["case report", "expert opinion", "editorial", "review article"],
}

# Document type detection patterns
DOCTYPE_PATTERNS: Dict[DocumentType, List[str]] = {
    DocumentType.TEXTBOOK: ["textbook", "comprehensive", "principles of"],
    DocumentType.ATLAS: ["atlas", "illustrated", "pictorial"],
    DocumentType.HANDBOOK: ["handbook", "manual", "pocket", "quick reference"],
    DocumentType.JOURNAL_ARTICLE: ["journal", "article", "et al.", "doi:"],
    DocumentType.REVIEW: ["review", "systematic review", "meta-analysis"],
    DocumentType.GUIDELINES: ["guidelines", "recommendations", "consensus", "protocol"],
    DocumentType.EXAM_QUESTIONS: ["questions", "mcq", "board review", "self-assessment", "exam"],
    DocumentType.COURSE_MATERIAL: ["course", "curriculum", "lecture", "syllabus"],
    DocumentType.LECTURE_NOTES: ["notes", "summary", "overview"],
    DocumentType.OPERATIVE_VIDEO_COMPANION: ["video", "operative", "surgical technique"],
    DocumentType.CHAPTER: ["chapter"],
}


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ChapterInfo:
    """Metadata for a single chapter/section."""
    id: str
    title: str
    level: int  # 1 = main chapter, 2 = section, 3 = subsection
    page_start: int
    page_end: int
    page_count: int
    word_count_estimate: int
    has_images: bool
    image_count_estimate: int
    specialties: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    preview: str = ""  # First ~500 chars of content

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReferenceDocument:
    """Full metadata for a reference document."""
    # Identity
    id: str
    file_path: str
    file_name: str
    file_size_mb: float
    content_hash: str

    # Bibliographic
    title: str
    authors: Optional[str] = None
    year: Optional[int] = None
    publisher: Optional[str] = None
    edition: Optional[str] = None
    isbn: Optional[str] = None
    series: Optional[str] = None
    volume: Optional[int] = None

    # Classification
    document_type: DocumentType = DocumentType.UNKNOWN
    primary_specialty: Specialty = Specialty.GENERAL
    specialties: List[str] = field(default_factory=list)
    subspecialties: List[str] = field(default_factory=list)  # Nested under primary_specialty
    evidence_level: Optional[str] = None  # Ia, Ib, IIa, IIb, III, IV (for journal articles)
    authority_source: str = "GENERAL"
    authority_score: float = 0.70

    # Structure
    page_count: int = 0
    chapter_count: int = 0
    chapters: List[ChapterInfo] = field(default_factory=list)

    # Content indicators
    has_toc: bool = False
    has_index: bool = False
    has_images: bool = False
    image_count_estimate: int = 0
    has_tables: bool = False
    word_count_estimate: int = 0

    # Processing status
    scan_date: str = field(default_factory=lambda: datetime.now().isoformat())
    first_seen_date: Optional[str] = None  # Date document was first discovered
    is_new: bool = False  # True if discovered in most recent scan
    is_ingested: bool = False
    ingested_date: Optional[str] = None
    ingested_document_id: Optional[str] = None  # FK to documents table

    # Hierarchy (for book→chapter grouping)
    parent_id: Optional[str] = None  # Parent document ID (book for chapters)
    sort_order: int = 0  # Order within parent
    section_type: str = "document"  # document|book|chapter|section|appendix
    grouping_confidence: float = 0.0  # Confidence of grouping algorithm

    # Search helpers
    all_keywords: List[str] = field(default_factory=list)
    chapter_titles_text: str = ""  # Concatenated for full-text search

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["document_type"] = self.document_type.value
        data["primary_specialty"] = self.primary_specialty.value
        data["chapters"] = [c.to_dict() if hasattr(c, 'to_dict') else c for c in self.chapters]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceDocument":
        """Reconstruct from dictionary."""
        data["document_type"] = DocumentType(data.get("document_type", "unknown"))
        data["primary_specialty"] = Specialty(data.get("primary_specialty", "general"))
        chapters = data.get("chapters", [])
        data["chapters"] = [
            ChapterInfo(**c) if isinstance(c, dict) else c
            for c in chapters
        ]
        return cls(**data)


@dataclass
class LibraryCatalog:
    """Complete catalog of scanned reference library."""
    documents: List[ReferenceDocument] = field(default_factory=list)
    scan_date: str = field(default_factory=lambda: datetime.now().isoformat())
    library_path: str = ""
    total_documents: int = 0
    total_pages: int = 0
    total_chapters: int = 0

    # Scan statistics (populated after scan)
    scan_stats: Dict[str, int] = field(default_factory=dict)  # scanned_count, cached_count, etc.

    # Indexes for fast lookup
    _by_specialty: Dict[str, List[str]] = field(default_factory=dict)
    _by_type: Dict[str, List[str]] = field(default_factory=dict)
    _by_authority: Dict[str, List[str]] = field(default_factory=dict)
    _by_hash: Dict[str, str] = field(default_factory=dict)  # content_hash -> doc_id

    def add_document(self, doc: ReferenceDocument):
        """Add document and update indexes."""
        self.documents.append(doc)
        self.total_documents += 1
        self.total_pages += doc.page_count
        self.total_chapters += doc.chapter_count

        # Hash index for duplicate detection
        self._by_hash[doc.content_hash] = doc.id

        # Update specialty index
        for spec in doc.specialties:
            if spec not in self._by_specialty:
                self._by_specialty[spec] = []
            self._by_specialty[spec].append(doc.id)

        # Update type index
        doc_type = doc.document_type.value
        if doc_type not in self._by_type:
            self._by_type[doc_type] = []
        self._by_type[doc_type].append(doc.id)

        # Update authority index
        if doc.authority_source not in self._by_authority:
            self._by_authority[doc.authority_source] = []
        self._by_authority[doc.authority_source].append(doc.id)

    def get_document(self, doc_id: str) -> Optional[ReferenceDocument]:
        """Get document by ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def get_by_hash(self, content_hash: str) -> Optional[ReferenceDocument]:
        """Get document by content hash."""
        doc_id = self._by_hash.get(content_hash)
        if doc_id:
            return self.get_document(doc_id)
        return None

    def search(
        self,
        query: Optional[str] = None,
        specialty: Optional[str] = None,
        subspecialty: Optional[str] = None,
        document_type: Optional[str] = None,
        authority_source: Optional[str] = None,
        evidence_level: Optional[str] = None,
        min_authority_score: float = 0.0,
        has_images: Optional[bool] = None,
        is_ingested: Optional[bool] = None,
        is_new: Optional[bool] = None,
        min_pages: int = 0,
        max_pages: int = 999999,
    ) -> List[ReferenceDocument]:
        """
        Search and filter the catalog.

        Returns:
            List of matching documents, sorted by authority score
        """
        results = []

        for doc in self.documents:
            # Apply filters
            if specialty and specialty.lower() not in [s.lower() for s in doc.specialties]:
                continue

            # Subspecialty filter - check if the subspecialty matches any in document
            if subspecialty:
                doc_subspecialties = getattr(doc, 'subspecialties', [])
                if subspecialty.lower() not in [s.lower() for s in doc_subspecialties]:
                    continue

            if document_type and doc.document_type.value != document_type:
                continue

            if authority_source and doc.authority_source != authority_source:
                continue

            # Evidence level filter
            if evidence_level:
                doc_evidence = getattr(doc, 'evidence_level', None)
                if doc_evidence != evidence_level:
                    continue

            if doc.authority_score < min_authority_score:
                continue

            if has_images is not None and doc.has_images != has_images:
                continue

            if is_ingested is not None and doc.is_ingested != is_ingested:
                continue

            if is_new is not None and doc.is_new != is_new:
                continue

            if doc.page_count < min_pages or doc.page_count > max_pages:
                continue

            # Text search
            if query:
                query_lower = query.lower()
                searchable = (
                    doc.title.lower() + " " +
                    doc.chapter_titles_text.lower() + " " +
                    " ".join(doc.all_keywords).lower() + " " +
                    (doc.authors or "").lower()
                )
                if query_lower not in searchable:
                    continue

            results.append(doc)

        # Sort by authority score descending
        results.sort(key=lambda d: d.authority_score, reverse=True)
        return results

    def search_chapters(
        self,
        query: str,
        specialty: Optional[str] = None,
    ) -> List[Tuple[ReferenceDocument, ChapterInfo]]:
        """
        Search within chapter titles and content previews.

        Returns:
            List of (document, chapter) tuples matching the query
        """
        results = []
        query_lower = query.lower()

        for doc in self.documents:
            if specialty and specialty.lower() not in [s.lower() for s in doc.specialties]:
                continue

            for chapter in doc.chapters:
                searchable = (
                    chapter.title.lower() + " " +
                    chapter.preview.lower() + " " +
                    " ".join(chapter.keywords).lower()
                )
                if query_lower in searchable:
                    results.append((doc, chapter))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics for dashboard."""
        return {
            "total_documents": self.total_documents,
            "total_pages": self.total_pages,
            "total_chapters": self.total_chapters,
            "ingested_count": sum(1 for d in self.documents if d.is_ingested),
            "not_ingested_count": sum(1 for d in self.documents if not d.is_ingested),
            "new_count": sum(1 for d in self.documents if getattr(d, 'is_new', False)),
            "scanned_count": self.scan_stats.get('scanned_count', self.total_documents),
            "cached_count": self.scan_stats.get('cached_count', 0),
            "by_specialty": {k: len(v) for k, v in self._by_specialty.items()},
            "by_type": {k: len(v) for k, v in self._by_type.items()},
            "by_authority": {k: len(v) for k, v in self._by_authority.items()},
            "scan_date": self.scan_date,
        }

    def get_filter_options(self) -> Dict[str, Any]:
        """Get available filter values for UI dropdowns."""
        # Collect subspecialties organized by specialty
        subspecialties_by_specialty: Dict[str, Set[str]] = {}
        evidence_levels_found: Set[str] = set()

        for doc in self.documents:
            # Get the primary specialty
            primary_spec = doc.primary_specialty.value if hasattr(doc.primary_specialty, 'value') else str(doc.primary_specialty)

            # Add subspecialties
            doc_subspecialties = getattr(doc, 'subspecialties', [])
            if doc_subspecialties:
                if primary_spec not in subspecialties_by_specialty:
                    subspecialties_by_specialty[primary_spec] = set()
                subspecialties_by_specialty[primary_spec].update(doc_subspecialties)

            # Collect evidence levels
            evidence_level = getattr(doc, 'evidence_level', None)
            if evidence_level:
                evidence_levels_found.add(evidence_level)

        # Convert sets to sorted lists
        subspecialties_dict = {
            spec: sorted(list(subs))
            for spec, subs in subspecialties_by_specialty.items()
        }

        # Order evidence levels properly
        evidence_order = ["Ia", "Ib", "IIa", "IIb", "III", "IV"]
        evidence_levels = [e for e in evidence_order if e in evidence_levels_found]

        return {
            "specialties": list(self._by_specialty.keys()),
            "subspecialties": subspecialties_dict,
            "document_types": list(self._by_type.keys()),
            "authority_sources": list(self._by_authority.keys()),
            "evidence_levels": evidence_levels,
        }

    def to_json(self, path: str):
        """Export catalog to JSON file."""
        data = {
            "scan_date": self.scan_date,
            "library_path": self.library_path,
            "statistics": self.get_statistics(),
            "documents": [doc.to_dict() for doc in self.documents],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "LibraryCatalog":
        """Load catalog from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        catalog = cls(
            scan_date=data.get("scan_date", ""),
            library_path=data.get("library_path", ""),
        )

        for doc_data in data.get("documents", []):
            doc = ReferenceDocument.from_dict(doc_data)
            catalog.add_document(doc)

        return catalog

    def merge_with_previous(self, previous_catalog: "LibraryCatalog") -> int:
        """
        Merge with previous catalog to detect new documents.

        Sets is_new=True for documents not in previous catalog.
        Preserves first_seen_date from previous catalog for existing docs.

        Returns:
            Number of new documents detected
        """
        # Build lookup of previous documents by content_hash
        previous_by_hash = {doc.content_hash: doc for doc in previous_catalog.documents}
        previous_by_path = {doc.file_path: doc for doc in previous_catalog.documents}

        new_count = 0
        current_date = datetime.now().isoformat()

        for doc in self.documents:
            # Check if document existed before (by hash or path)
            prev_doc = previous_by_hash.get(doc.content_hash) or previous_by_path.get(doc.file_path)

            if prev_doc:
                # Existing document - preserve first_seen_date
                doc.first_seen_date = prev_doc.first_seen_date or prev_doc.scan_date
                doc.is_new = False
                # Preserve ingestion status
                doc.is_ingested = prev_doc.is_ingested
                doc.ingested_date = prev_doc.ingested_date
                doc.ingested_document_id = prev_doc.ingested_document_id
            else:
                # New document!
                doc.first_seen_date = current_date
                doc.is_new = True
                new_count += 1

        return new_count

    def get_new_documents(self) -> List[ReferenceDocument]:
        """Get all documents marked as new."""
        return [doc for doc in self.documents if doc.is_new]

    def apply_grouping(self, grouping_result: Dict[str, Any]) -> int:
        """
        Apply book grouping results to catalog documents.

        Args:
            grouping_result: Result from IntelligentBookGrouper.group_documents()

        Returns:
            Number of documents organized into book hierarchies
        """
        from uuid import uuid4

        organized_count = 0
        doc_by_id = {doc.id: doc for doc in self.documents}

        # Process each book group
        for book in grouping_result.get('books', []):
            # Create a virtual "book" parent document
            book_id = str(uuid4())
            book_doc = ReferenceDocument(
                id=book_id,
                file_path="",  # Virtual document has no file
                file_name=book['book_title'],
                file_size_mb=0,
                content_hash=f"book_{book['book_signature']}",
                title=book['book_title'],
                document_type=DocumentType.TEXTBOOK,
                primary_specialty=Specialty.GENERAL,
                authority_source=book.get('authority_tier', 'GENERAL'),
                authority_score=0.9 if book.get('authority_tier') else 0.7,
                section_type="book",
                grouping_confidence=book['confidence'],
            )

            # Sum up book stats from chapters
            total_pages = 0
            total_chapters = 0
            all_specialties = set()

            # Update chapter documents with parent reference
            for idx, chapter_data in enumerate(book.get('chapters', [])):
                chapter_id = chapter_data['id']
                if chapter_id in doc_by_id:
                    chapter_doc = doc_by_id[chapter_id]
                    chapter_doc.parent_id = book_id
                    chapter_doc.sort_order = idx
                    chapter_doc.section_type = "chapter"
                    chapter_doc.grouping_confidence = book['confidence']

                    total_pages += chapter_doc.page_count
                    total_chapters += 1
                    all_specialties.update(chapter_doc.specialties)
                    organized_count += 1

            # Update book document with aggregated stats
            book_doc.page_count = total_pages
            book_doc.chapter_count = total_chapters
            book_doc.specialties = list(all_specialties)
            if all_specialties:
                book_doc.primary_specialty = Specialty(list(all_specialties)[0])

            # Add book to catalog
            self.documents.append(book_doc)

        logger.info(f"Applied grouping: {organized_count} chapters organized into {len(grouping_result.get('books', []))} books")
        return organized_count

    def get_book_hierarchy(self) -> Dict[str, Any]:
        """
        Get documents organized by book hierarchy.

        Returns:
            Dict with 'books' (parent docs with children) and 'standalone' (ungrouped docs)
        """
        books = []
        standalone = []
        children_by_parent = {}

        # Group children by parent_id
        for doc in self.documents:
            if doc.parent_id:
                if doc.parent_id not in children_by_parent:
                    children_by_parent[doc.parent_id] = []
                children_by_parent[doc.parent_id].append(doc)
            elif doc.section_type == "book":
                books.append(doc)
            else:
                standalone.append(doc)

        # Build book hierarchy
        book_hierarchy = []
        for book in books:
            children = children_by_parent.get(book.id, [])
            children.sort(key=lambda c: c.sort_order)
            book_hierarchy.append({
                'book': book.to_dict(),
                'chapters': [c.to_dict() for c in children],
                'chapter_count': len(children),
            })

        return {
            'books': book_hierarchy,
            'standalone': [d.to_dict() for d in standalone],
            'total_books': len(book_hierarchy),
            'total_standalone': len(standalone),
        }


# =============================================================================
# Scanner Implementation
# =============================================================================

class LibraryScanner:
    """
    Scans a PDF library to extract metadata for browsing.

    This is a lightweight scanner that extracts:
    - PDF metadata (title, author, etc.)
    - Table of contents / outline (or visual detection fallback)
    - Page count and image estimates
    - Specialty and document type classification
    - Content previews for each chapter

    It does NOT perform:
    - Full text extraction
    - Chunking
    - Embedding generation
    - Database storage
    """

    def __init__(
        self,
        library_path: str,
        preview_length: int = 500,
        max_chapters: int = 100,
    ):
        """
        Initialize the scanner.

        Args:
            library_path: Root directory containing PDF files
            preview_length: Characters to extract for chapter previews
            max_chapters: Maximum chapters to extract per document
        """
        if not HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) is required for library scanning. Install with: pip install pymupdf")

        self.library_path = Path(library_path)
        self.preview_length = preview_length
        self.max_chapters = max_chapters

    async def scan_library(
        self,
        recursive: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_concurrency: int = 4,
        timeout_per_pdf: float = 30.0,
        auto_group: bool = True,
        grouping_confidence_threshold: float = 0.6,
        previous_catalog: Optional["LibraryCatalog"] = None,
        incremental: bool = True,
    ) -> LibraryCatalog:
        """
        Scan all PDFs in the library with parallel processing.

        Args:
            recursive: Whether to scan subdirectories
            progress_callback: Optional callback(current, total, filename)
            max_concurrency: Max parallel PDF scans (default 4)
            timeout_per_pdf: Timeout in seconds per PDF (default 30s)
            auto_group: Whether to auto-group chapters into books (default True)
            grouping_confidence_threshold: Min confidence for grouping (default 0.6)
            previous_catalog: Previous catalog for incremental scanning
            incremental: If True, skip unchanged files (based on path + file size)

        Returns:
            LibraryCatalog with all extracted metadata
        """
        catalog = LibraryCatalog(library_path=str(self.library_path))

        # Build lookup of previously scanned documents for incremental mode
        prev_by_path = {}
        if incremental and previous_catalog:
            for doc in previous_catalog.documents:
                prev_by_path[doc.file_path] = doc
            logger.info(f"Incremental mode: {len(prev_by_path)} previously scanned documents")

        # Find all PDFs
        if recursive:
            pdf_files = list(self.library_path.rglob("*.pdf"))
        else:
            pdf_files = list(self.library_path.glob("*.pdf"))

        total = len(pdf_files)
        logger.info(f"Found {total} PDF files to scan in {self.library_path}")

        # Use semaphore for controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        completed = [0]  # Mutable counter for progress
        cached_count = [0]  # Track how many were reused from cache
        scanned_count = [0]  # Track how many were actually scanned

        async def scan_with_timeout(pdf_path: Path) -> Optional[ReferenceDocument]:
            async with semaphore:
                try:
                    # INCREMENTAL MODE: Check if we can skip this file
                    path_str = str(pdf_path)
                    if incremental and path_str in prev_by_path:
                        prev_doc = prev_by_path[path_str]
                        # Check if file is unchanged (compare size in bytes)
                        try:
                            current_size_mb = pdf_path.stat().st_size / (1024 * 1024)
                            # Allow 1KB tolerance for file size comparison
                            if abs(current_size_mb - prev_doc.file_size_mb) < 0.001:
                                # File unchanged - reuse existing metadata
                                completed[0] += 1
                                cached_count[0] += 1
                                if progress_callback:
                                    progress_callback(completed[0], total, f"✓ {pdf_path.name}")
                                logger.debug(f"Cached: {pdf_path.name}")
                                return prev_doc
                        except OSError:
                            pass  # File might have been deleted, scan anyway

                    # Run full scan in executor to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    doc = await asyncio.wait_for(
                        loop.run_in_executor(None, self._scan_document_sync, pdf_path),
                        timeout=timeout_per_pdf
                    )
                    completed[0] += 1
                    scanned_count[0] += 1
                    if progress_callback:
                        progress_callback(completed[0], total, pdf_path.name)
                    return doc
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout scanning {pdf_path.name} (>{timeout_per_pdf}s)")
                    completed[0] += 1
                    scanned_count[0] += 1
                    if progress_callback:
                        progress_callback(completed[0], total, pdf_path.name)
                    # Return minimal fallback document
                    return self._create_fallback_document(pdf_path)
                except Exception as e:
                    logger.warning(f"Failed to scan {pdf_path.name}: {e}")
                    completed[0] += 1
                    scanned_count[0] += 1
                    if progress_callback:
                        progress_callback(completed[0], total, pdf_path.name)
                    return None

        # Scan all PDFs in parallel with controlled concurrency
        results = await asyncio.gather(*[scan_with_timeout(p) for p in pdf_files])

        # Add successful results to catalog
        for doc in results:
            if doc is not None:
                catalog.add_document(doc)

        # Log scan summary with incremental stats
        if incremental and cached_count[0] > 0:
            logger.info(
                f"Scan complete: {catalog.total_documents} documents "
                f"({scanned_count[0]} scanned, {cached_count[0]} cached), "
                f"{catalog.total_pages:,} pages, {catalog.total_chapters} chapters"
            )
        else:
            logger.info(f"Scan complete: {catalog.total_documents} documents, {catalog.total_pages:,} pages, {catalog.total_chapters} chapters")

        # Store scan statistics in catalog for frontend display
        catalog.scan_stats = {
            "total_files": total,
            "scanned_count": scanned_count[0],
            "cached_count": cached_count[0],
            "failed_count": total - catalog.total_documents,
        }

        # Auto-group chapters into books if enabled
        if auto_group and catalog.total_documents > 0:
            try:
                from src.library.grouper import group_library_documents

                # Convert catalog documents to dicts for grouper
                doc_dicts = []
                for doc in catalog.documents:
                    doc_dicts.append({
                        'id': doc.id,
                        'title': doc.title,
                        'filename': doc.file_name,
                        'file_path': doc.file_path,
                        'authority_tier': doc.authority_source,
                        'metadata': {}
                    })

                # Run grouping
                grouping_result = group_library_documents(
                    doc_dicts,
                    confidence_threshold=grouping_confidence_threshold
                )

                # Apply grouping to catalog
                organized = catalog.apply_grouping(grouping_result)
                logger.info(
                    f"Book grouping: {grouping_result['stats']['book_count']} books detected, "
                    f"{organized} chapters organized"
                )
            except ImportError as e:
                logger.warning(f"Could not import grouper: {e}")
            except Exception as e:
                logger.warning(f"Book grouping failed: {e}")

        return catalog

    def _scan_document_sync(self, pdf_path: Path) -> ReferenceDocument:
        """Synchronous version for executor - duplicates scan_document logic."""
        # Calculate content hash
        content_hash = self._compute_hash(pdf_path)

        # Open PDF
        doc = fitz.open(str(pdf_path))

        try:
            # Extract basic metadata
            metadata = doc.metadata or {}
            page_count = len(doc)

            # Extract title (from metadata or filename)
            title = self._extract_title(metadata, pdf_path)

            # Extract author/year from metadata
            authors = metadata.get("author")
            year = self._extract_year(metadata)

            # Detect authority source and score
            authority_source, authority_score = self._detect_authority(title, authors)

            # Extract table of contents (with visual fallback)
            toc = doc.get_toc()
            has_toc = len(toc) > 3

            # Build chapter list - try TOC first, fallback to visual detection
            if has_toc:
                chapters = self._extract_chapters_from_toc(doc, toc)
            else:
                chapters = self._extract_chapters_visual_fast(doc)  # Use fast version

            # Estimate images (sample only)
            image_count = self._estimate_images(doc)

            # Estimate word count (sample-based)
            word_count = self._estimate_word_count(doc)

            # Detect specialties from title and TOC
            all_text = title + " " + " ".join(c.title for c in chapters)
            specialties = self._detect_specialties(all_text)
            primary_specialty = specialties[0] if specialties else Specialty.GENERAL

            # Detect subspecialties within primary specialty
            subspecialties = self._detect_subspecialties(all_text, primary_specialty)

            # Detect document type
            document_type = self._detect_document_type(title, pdf_path, chapters)

            # Detect evidence level (for journal articles)
            evidence_level = self._detect_evidence_level(all_text, document_type)

            # Build keyword list
            keywords = self._extract_keywords(title, chapters, specialties)

            # Build chapter titles text for search
            chapter_titles = " | ".join(c.title for c in chapters)

            return ReferenceDocument(
                id=str(uuid4()),
                file_path=str(pdf_path),
                file_name=pdf_path.name,
                file_size_mb=pdf_path.stat().st_size / (1024 * 1024),
                content_hash=content_hash,
                title=title,
                authors=authors,
                year=year,
                publisher=metadata.get("producer"),
                document_type=document_type,
                primary_specialty=primary_specialty,
                specialties=[s.value for s in specialties] if specialties else [Specialty.GENERAL.value],
                subspecialties=subspecialties,
                evidence_level=evidence_level,
                authority_source=authority_source,
                authority_score=authority_score,
                page_count=page_count,
                chapter_count=len(chapters),
                chapters=chapters,
                has_toc=has_toc,
                has_images=image_count > 0,
                image_count_estimate=image_count,
                word_count_estimate=word_count,
                all_keywords=keywords,
                chapter_titles_text=chapter_titles,
            )
        finally:
            doc.close()

    def _create_fallback_document(self, pdf_path: Path) -> ReferenceDocument:
        """Create minimal document when scan times out."""
        return ReferenceDocument(
            id=str(uuid4()),
            file_path=str(pdf_path),
            file_name=pdf_path.name,
            file_size_mb=pdf_path.stat().st_size / (1024 * 1024),
            content_hash=self._compute_hash(pdf_path),
            title=pdf_path.stem.replace("_", " ").replace("-", " "),
            document_type=DocumentType.UNKNOWN,
            primary_specialty=Specialty.GENERAL,
            specialties=[Specialty.GENERAL.value],
            authority_source="GENERAL",
            authority_score=0.5,
            page_count=0,  # Unknown
            chapter_count=0,
        )

    async def scan_document(self, pdf_path: Path) -> ReferenceDocument:
        """
        Scan a single PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ReferenceDocument with extracted metadata
        """
        # Calculate content hash
        content_hash = self._compute_hash(pdf_path)

        # Open PDF
        doc = fitz.open(str(pdf_path))

        try:
            # Extract basic metadata
            metadata = doc.metadata or {}
            page_count = len(doc)

            # Extract title (from metadata or filename)
            title = self._extract_title(metadata, pdf_path)

            # Extract author/year from metadata
            authors = metadata.get("author")
            year = self._extract_year(metadata)

            # Detect authority source and score
            authority_source, authority_score = self._detect_authority(title, authors)

            # Extract table of contents (with visual fallback)
            toc = doc.get_toc()
            has_toc = len(toc) > 3

            # Build chapter list - try TOC first, fallback to visual detection
            if has_toc:
                chapters = self._extract_chapters_from_toc(doc, toc)
            else:
                chapters = self._extract_chapters_visual(doc)

            # Estimate images
            image_count = self._estimate_images(doc)

            # Estimate word count (sample-based)
            word_count = self._estimate_word_count(doc)

            # Detect specialties from title and TOC
            all_text = title + " " + " ".join(c.title for c in chapters)
            specialties = self._detect_specialties(all_text)
            primary_specialty = specialties[0] if specialties else Specialty.GENERAL

            # Detect subspecialties within primary specialty
            subspecialties = self._detect_subspecialties(all_text, primary_specialty)

            # Detect document type
            document_type = self._detect_document_type(title, pdf_path, chapters)

            # Detect evidence level (for journal articles)
            evidence_level = self._detect_evidence_level(all_text, document_type)

            # Build keyword list
            keywords = self._extract_keywords(title, chapters, specialties)

            # Build chapter titles text for search
            chapter_titles = " | ".join(c.title for c in chapters)

            return ReferenceDocument(
                id=str(uuid4()),
                file_path=str(pdf_path),
                file_name=pdf_path.name,
                file_size_mb=pdf_path.stat().st_size / (1024 * 1024),
                content_hash=content_hash,
                title=title,
                authors=authors,
                year=year,
                publisher=metadata.get("producer"),
                document_type=document_type,
                primary_specialty=primary_specialty,
                specialties=[s.value for s in specialties] if specialties else [Specialty.GENERAL.value],
                subspecialties=subspecialties,
                evidence_level=evidence_level,
                authority_source=authority_source,
                authority_score=authority_score,
                page_count=page_count,
                chapter_count=len(chapters),
                chapters=chapters,
                has_toc=has_toc,
                has_images=image_count > 0,
                image_count_estimate=image_count,
                word_count_estimate=word_count,
                all_keywords=keywords,
                chapter_titles_text=chapter_titles,
            )
        finally:
            doc.close()

    def _compute_hash(self, pdf_path: Path) -> str:
        """Compute SHA256 hash of file (first 64KB for speed)."""
        sha256 = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            # Hash first 64KB for speed (enough for uniqueness)
            sha256.update(f.read(65536))
        return sha256.hexdigest()[:16]

    def _extract_title(self, metadata: Dict, pdf_path: Path) -> str:
        """Extract title from metadata or filename."""
        title = metadata.get("title", "").strip()
        if title and len(title) > 3:
            return title

        # Clean filename
        name = pdf_path.stem
        # Remove common suffixes and clean up
        for suffix in ["_copy", " copy", "_ocr", " - ", "_", "-"]:
            name = name.replace(suffix, " ")
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def _extract_year(self, metadata: Dict) -> Optional[int]:
        """Extract publication year from metadata."""
        for key in ["creationDate", "modDate"]:
            date_str = metadata.get(key, "")
            if date_str:
                match = re.search(r"(19|20)\d{2}", date_str)
                if match:
                    return int(match.group())
        return None

    def _detect_authority(self, title: str, authors: Optional[str]) -> Tuple[str, float]:
        """Detect authority source and score."""
        searchable = (title + " " + (authors or "")).lower()

        for source, (score, keywords) in AUTHORITY_PATTERNS.items():
            if any(kw in searchable for kw in keywords):
                return source, score

        return "GENERAL", 0.70

    def _extract_chapters_from_toc(
        self,
        doc: "fitz.Document",
        toc: List[List],
    ) -> List[ChapterInfo]:
        """Extract chapter metadata from PDF's embedded TOC."""
        chapters = []

        for i, entry in enumerate(toc[:self.max_chapters]):
            level, title, page_num = entry[0], entry[1], entry[2]

            # Only include top-level chapters for main view
            if level > 2:
                continue

            # Determine page range
            if i + 1 < len(toc):
                next_page = toc[i + 1][2]
                page_end = max(page_num, next_page - 1)
            else:
                page_end = len(doc) - 1

            page_count = max(1, page_end - page_num + 1)

            # Extract preview
            preview = self._extract_preview(doc, page_num)

            # Detect specialties for this chapter
            specialties = self._detect_specialties(title + " " + preview)

            # Estimate images in this range
            image_count = self._count_images_in_range(doc, page_num, page_end)

            # Extract keywords
            keywords = self._extract_chapter_keywords(title, preview)

            chapters.append(ChapterInfo(
                id=str(uuid4()),
                title=title.strip(),
                level=level,
                page_start=page_num,
                page_end=page_end,
                page_count=page_count,
                word_count_estimate=page_count * 300,  # Rough estimate
                has_images=image_count > 0,
                image_count_estimate=image_count,
                specialties=[s.value for s in specialties],
                keywords=keywords,
                preview=preview,
            ))

        return chapters

    def _extract_chapters_visual_fast(self, doc: "fitz.Document", max_pages: int = 50) -> List[ChapterInfo]:
        """
        Fast visual chapter detection - only scans first N pages.
        Used for timeout-sensitive scanning.
        """
        # Limit scanning to first max_pages (most textbooks have chapters in front)
        total_pages = len(doc)
        scan_limit = min(max_pages, total_pages)

        # If document is small, just return a single chapter
        if total_pages <= 10:
            return [ChapterInfo(
                id=str(uuid4()),
                title="Full Document",
                level=1,
                page_start=0,
                page_end=total_pages - 1,
                page_count=total_pages,
                word_count_estimate=self._estimate_word_count(doc),
                has_images=self._estimate_images(doc) > 0,
                image_count_estimate=self._estimate_images(doc),
                preview=self._extract_preview(doc, 0),
            )]

        # Quick font analysis from first 10 pages only
        styles: Dict[float, int] = {}
        for page_idx in range(min(10, scan_limit)):
            try:
                page = doc[page_idx]
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                size = round(span["size"], 1)
                                text_len = len(span["text"].strip())
                                if text_len > 0:
                                    styles[size] = styles.get(size, 0) + text_len
            except Exception:
                continue

        if not styles:
            return [ChapterInfo(
                id=str(uuid4()),
                title="Full Document",
                level=1,
                page_start=0,
                page_end=total_pages - 1,
                page_count=total_pages,
                word_count_estimate=self._estimate_word_count(doc),
                has_images=self._estimate_images(doc) > 0,
                image_count_estimate=self._estimate_images(doc),
                preview=self._extract_preview(doc, 0),
            )]

        body_size = max(styles, key=styles.get)
        header_threshold = body_size + 4.0
        chapters = []

        # Only scan first max_pages for chapter headers
        for page_idx in range(scan_limit):
            try:
                page = doc[page_idx]
                blocks = page.get_text("dict")["blocks"]
                if not blocks:
                    continue

                for block in blocks[:3]:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        if not line["spans"]:
                            continue
                        span = line["spans"][0]
                        text = span["text"].strip()
                        size = span["size"]

                        if len(text) < 3:
                            continue

                        is_header = (
                            size > header_threshold and
                            (len(text) < 100 or "chapter" in text.lower() or re.match(r"^\d+[\.\s]", text))
                        )

                        if is_header:
                            if chapters:
                                chapters[-1].page_end = page_idx - 1
                                chapters[-1].page_count = max(1, chapters[-1].page_end - chapters[-1].page_start + 1)

                            chapters.append(ChapterInfo(
                                id=str(uuid4()),
                                title=text[:100],
                                level=1,
                                page_start=page_idx,
                                page_end=total_pages - 1,
                                page_count=1,
                                word_count_estimate=0,
                                has_images=False,
                                image_count_estimate=0,
                                specialties=[],
                                keywords=[],
                                preview=self._extract_preview(doc, page_idx),
                            ))
                            break
                    else:
                        continue
                    break
            except Exception:
                continue

        # Finalize
        if chapters:
            chapters[-1].page_end = total_pages - 1
            chapters[-1].page_count = max(1, chapters[-1].page_end - chapters[-1].page_start + 1)
            for ch in chapters:
                ch.word_count_estimate = ch.page_count * 300
        else:
            chapters = [ChapterInfo(
                id=str(uuid4()),
                title="Full Document",
                level=1,
                page_start=0,
                page_end=total_pages - 1,
                page_count=total_pages,
                word_count_estimate=self._estimate_word_count(doc),
                has_images=self._estimate_images(doc) > 0,
                image_count_estimate=self._estimate_images(doc),
                preview=self._extract_preview(doc, 0),
            )]

        return chapters

    def _extract_chapters_visual(self, doc: "fitz.Document") -> List[ChapterInfo]:
        """
        Detect chapters based on visual font analysis.
        Fallback when PDF has no embedded TOC.
        """
        chapters = []

        # 1. Analyze font styles from first 20 pages to find body text size
        styles: Dict[float, int] = {}
        sample_pages = min(20, len(doc))

        for page_idx in range(sample_pages):
            try:
                page = doc[page_idx]
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                size = round(span["size"], 1)
                                text_len = len(span["text"].strip())
                                if text_len > 0:
                                    styles[size] = styles.get(size, 0) + text_len
            except Exception:
                continue

        if not styles:
            # Can't analyze - return single chapter for whole doc
            return [ChapterInfo(
                id=str(uuid4()),
                title="Full Document",
                level=1,
                page_start=0,
                page_end=len(doc) - 1,
                page_count=len(doc),
                word_count_estimate=self._estimate_word_count(doc),
                has_images=self._estimate_images(doc) > 0,
                image_count_estimate=self._estimate_images(doc),
                preview=self._extract_preview(doc, 0),
            )]

        # Find body text size (most common)
        body_size = max(styles, key=styles.get)
        header_threshold = body_size + 4.0  # Headers must be significantly larger

        # 2. Scan all pages for potential chapter headings
        for page_idx in range(len(doc)):
            try:
                page = doc[page_idx]
                blocks = page.get_text("dict")["blocks"]

                if not blocks:
                    continue

                # Check first text block on page
                for block in blocks[:3]:  # Check first few blocks
                    if "lines" not in block:
                        continue

                    for line in block["lines"]:
                        if not line["spans"]:
                            continue

                        span = line["spans"][0]
                        text = span["text"].strip()
                        size = span["size"]

                        # Skip empty or very short text
                        if len(text) < 3:
                            continue

                        # Detect chapter heading: Large font + short text or contains "chapter"
                        is_header = (
                            size > header_threshold and
                            (len(text) < 100 or "chapter" in text.lower() or re.match(r"^\d+[\.\s]", text))
                        )

                        if is_header:
                            # Update previous chapter's end page
                            if chapters:
                                chapters[-1].page_end = page_idx - 1
                                chapters[-1].page_count = max(1, chapters[-1].page_end - chapters[-1].page_start + 1)

                            # Extract preview
                            preview = self._extract_preview(doc, page_idx)

                            chapters.append(ChapterInfo(
                                id=str(uuid4()),
                                title=text[:100],  # Truncate long titles
                                level=1,
                                page_start=page_idx,
                                page_end=len(doc) - 1,  # Will be updated
                                page_count=1,  # Will be updated
                                word_count_estimate=0,  # Will be calculated
                                has_images=False,  # Will be calculated
                                image_count_estimate=0,
                                specialties=[],
                                keywords=self._extract_chapter_keywords(text, preview),
                                preview=preview,
                            ))
                            break  # Found heading on this page, move to next
                    else:
                        continue
                    break

            except Exception as e:
                logger.debug(f"Error scanning page {page_idx}: {e}")
                continue

        # Update last chapter's end page
        if chapters:
            chapters[-1].page_end = len(doc) - 1
            chapters[-1].page_count = max(1, chapters[-1].page_end - chapters[-1].page_start + 1)

        # Calculate word counts and images for each chapter
        for chapter in chapters:
            chapter.word_count_estimate = chapter.page_count * 300
            chapter.image_count_estimate = self._count_images_in_range(doc, chapter.page_start, chapter.page_end)
            chapter.has_images = chapter.image_count_estimate > 0
            chapter.specialties = [s.value for s in self._detect_specialties(chapter.title + " " + chapter.preview)]

        # If no chapters detected, create one for whole document
        if not chapters:
            chapters = [ChapterInfo(
                id=str(uuid4()),
                title="Full Document",
                level=1,
                page_start=0,
                page_end=len(doc) - 1,
                page_count=len(doc),
                word_count_estimate=self._estimate_word_count(doc),
                has_images=self._estimate_images(doc) > 0,
                image_count_estimate=self._estimate_images(doc),
                preview=self._extract_preview(doc, 0),
            )]

        return chapters

    def _extract_preview(self, doc: "fitz.Document", page_num: int) -> str:
        """Extract text preview from a page."""
        try:
            if page_num >= len(doc):
                return ""
            page = doc[page_num]
            text = page.get_text("text")
            # Clean and truncate
            text = re.sub(r"\s+", " ", text).strip()
            return text[:self.preview_length]
        except Exception:
            return ""

    def _count_images_in_range(self, doc: "fitz.Document", start: int, end: int) -> int:
        """Count images in a page range."""
        count = 0
        for page_idx in range(start, min(end + 1, len(doc))):
            try:
                count += len(doc[page_idx].get_images())
            except Exception:
                pass
        return count

    def _estimate_images(self, doc: "fitz.Document") -> int:
        """Estimate total image count by sampling."""
        total = 0
        sample_pages = min(10, len(doc))
        for i in range(sample_pages):
            try:
                total += len(doc[i].get_images())
            except Exception:
                pass

        if sample_pages < len(doc):
            # Extrapolate
            total = int(total * len(doc) / sample_pages)

        return total

    def _estimate_word_count(self, doc: "fitz.Document") -> int:
        """Estimate word count by sampling."""
        total_words = 0
        sample_pages = min(5, len(doc))

        for i in range(sample_pages):
            try:
                text = doc[i].get_text("text")
                total_words += len(text.split())
            except Exception:
                pass

        if sample_pages < len(doc):
            avg_per_page = total_words / sample_pages if sample_pages > 0 else 0
            total_words = int(avg_per_page * len(doc))

        return total_words

    def _detect_specialties(self, text: str) -> List[Specialty]:
        """Detect specialties from text content."""
        text_lower = text.lower()
        scores: Dict[Specialty, int] = {}

        for specialty, keywords in SPECIALTY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[specialty] = score

        if not scores:
            return [Specialty.GENERAL]

        # Return top specialties
        sorted_specs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [s for s, _ in sorted_specs[:3]]

    def _detect_subspecialties(self, text: str, primary_specialty: Specialty) -> List[str]:
        """
        Detect subspecialties within a primary specialty from text content.

        Args:
            text: Text content to analyze
            primary_specialty: The detected primary specialty

        Returns:
            List of subspecialty names (max 3)
        """
        if primary_specialty not in SUBSPECIALTY_KEYWORDS:
            return []

        text_lower = text.lower()
        scores: Dict[str, int] = {}

        # Only search subspecialties for the detected primary specialty
        for subspecialty, keywords in SUBSPECIALTY_KEYWORDS[primary_specialty].items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[subspecialty] = score

        if not scores:
            return []

        # Return top 3 subspecialties by score
        sorted_subs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [sub for sub, _ in sorted_subs[:3]]

    def _detect_evidence_level(self, text: str, document_type: DocumentType) -> Optional[str]:
        """
        Detect evidence level for journal articles and reviews.

        Args:
            text: Text content to analyze (title, abstract, metadata)
            document_type: Type of document

        Returns:
            Evidence level (Ia, Ib, IIa, IIb, III, IV) or None
        """
        # Only detect for journal articles, reviews, and case series
        if document_type not in [DocumentType.JOURNAL_ARTICLE, DocumentType.REVIEW, DocumentType.CASE_SERIES]:
            return None

        text_lower = text.lower()

        # Check patterns in order of evidence strength (highest first)
        for level in ["Ia", "Ib", "IIa", "IIb", "III", "IV"]:
            patterns = EVIDENCE_LEVEL_PATTERNS.get(level, [])
            if any(pattern in text_lower for pattern in patterns):
                return level

        # Default to IV for journal articles without clear evidence markers
        if document_type == DocumentType.JOURNAL_ARTICLE:
            return "IV"

        return None

    def _detect_document_type(self, title: str, pdf_path: Path, chapters: List[ChapterInfo]) -> DocumentType:
        """Detect document type from title, filename, and structure."""
        searchable = (title + " " + pdf_path.stem + " " + " ".join(c.title for c in chapters)).lower()

        # Check for exam questions
        if any(kw in searchable for kw in ["question", "mcq", "exam", "self-assessment"]):
            return DocumentType.EXAM_QUESTIONS

        for doc_type, patterns in DOCTYPE_PATTERNS.items():
            if any(p in searchable for p in patterns):
                return doc_type

        # Heuristics based on structure
        page_count = sum(c.page_count for c in chapters) if chapters else 0
        if page_count < 30:
            return DocumentType.JOURNAL_ARTICLE
        elif page_count < 100:
            return DocumentType.CHAPTER

        return DocumentType.TEXTBOOK

    def _extract_keywords(
        self,
        title: str,
        chapters: List[ChapterInfo],
        specialties: List[Specialty],
    ) -> List[str]:
        """Extract keywords from document."""
        keywords = set()

        # Add words from title
        for word in re.findall(r"\b[a-zA-Z]{4,}\b", title):
            keywords.add(word.lower())

        # Add chapter keywords
        for chapter in chapters:
            keywords.update(chapter.keywords)

        # Add specialty names
        for spec in specialties:
            keywords.add(spec.value)

        return list(keywords)[:50]

    def _extract_chapter_keywords(self, title: str, preview: str) -> List[str]:
        """Extract keywords from chapter title and preview."""
        text = title + " " + preview
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

        # Filter common words
        stopwords = {
            "this", "that", "with", "from", "have", "been", "will", "were",
            "their", "which", "when", "where", "what", "there", "than",
            "then", "them", "these", "those", "your", "about", "would",
            "could", "should", "other", "after", "before", "between"
        }
        keywords = [w for w in words if w not in stopwords]

        # Count and return top keywords
        from collections import Counter
        counts = Counter(keywords)
        return [w for w, _ in counts.most_common(10)]
