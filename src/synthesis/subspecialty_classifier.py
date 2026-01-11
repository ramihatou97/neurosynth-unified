"""
Neurosurgical Subspecialty Classifier
=====================================

Semantic classification of neurosurgical topics into subspecialties.
Uses vector similarity against subspecialty prototypes for accurate routing.

Subspecialties:
- skull_base: Skull base surgery, approaches, tumors
- vascular: Aneurysms, AVMs, cavernomas, bypass
- spine: Degenerative, deformity, trauma, tumors
- tumor: Brain tumors (non-skull base)
- functional: Movement disorders, epilepsy, pain
- pediatric: Pediatric-specific conditions
- trauma: Neurotrauma, TBI, SCI
- peripheral_nerve: Peripheral nerve surgery

This classifier is used by Stage 1 of the 14-stage gap detection algorithm.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Subspecialty(Enum):
    """Neurosurgical subspecialties."""

    SKULL_BASE = "skull_base"
    VASCULAR = "vascular"
    SPINE = "spine"
    TUMOR = "tumor"
    FUNCTIONAL = "functional"
    PEDIATRIC = "pediatric"
    TRAUMA = "trauma"
    PERIPHERAL_NERVE = "peripheral_nerve"
    GENERAL = "general"  # Fallback for unclassified topics


class EmbeddingService(Protocol):
    """Protocol for embedding service interface."""

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single string."""
        ...


@dataclass
class ClassificationResult:
    """Result of subspecialty classification."""

    subspecialty: Subspecialty
    confidence_score: float
    all_scores: Dict[str, float]
    is_confident: bool


# Gold-standard descriptions for each subspecialty
SUBSPECIALTY_PROTOTYPES: Dict[Subspecialty, str] = {
    Subspecialty.SKULL_BASE: """
        Skull base surgery including anterior, middle, and posterior fossa approaches.
        Endoscopic endonasal surgery, transsphenoidal approaches to pituitary tumors.
        Acoustic neuroma (vestibular schwannoma), meningioma of skull base,
        petroclival tumors, jugular foramen tumors, glomus tumors.
        Cranial base triangles, cavernous sinus, petrous bone surgery.
        Facial nerve preservation, hearing preservation surgery.
        Translabyrinthine, retrosigmoid, middle fossa approaches.
    """,
    Subspecialty.VASCULAR: """
        Cerebrovascular surgery including aneurysm clipping and coiling.
        Arteriovenous malformation resection and embolization.
        Cavernous malformation, developmental venous anomaly.
        EC-IC bypass surgery, STA-MCA bypass, moyamoya disease.
        Intracranial hemorrhage including ICH, SAH, IVH.
        Carotid endarterectomy, carotid stenosis.
        Dural arteriovenous fistula, CCF.
        Stroke intervention, thrombectomy.
    """,
    Subspecialty.SPINE: """
        Spinal surgery including cervical, thoracic, lumbar, sacral spine.
        Degenerative disc disease, herniated disc, spinal stenosis.
        Cervical myelopathy, lumbar radiculopathy.
        ACDF, posterior cervical fusion, lumbar fusion, laminectomy.
        Spinal deformity, scoliosis, kyphosis correction.
        Spinal cord tumors, intradural extramedullary, intramedullary tumors.
        Spinal instrumentation, pedicle screws, rods.
        Minimally invasive spine surgery, tubular discectomy.
    """,
    Subspecialty.TUMOR: """
        Brain tumor surgery including glioma, glioblastoma, astrocytoma.
        Metastatic brain tumors, brain metastases.
        Convexity meningioma, falcine meningioma, parasagittal meningioma.
        Intraventricular tumors, colloid cyst, choroid plexus tumors.
        Pineal region tumors, posterior fossa tumors in adults.
        Awake craniotomy, brain mapping, eloquent cortex surgery.
        Fluorescence-guided surgery, 5-ALA.
        Neuro-oncology, chemotherapy, radiation therapy for brain tumors.
    """,
    Subspecialty.FUNCTIONAL: """
        Functional neurosurgery including deep brain stimulation DBS.
        Movement disorders, Parkinson disease, essential tremor, dystonia.
        Epilepsy surgery, temporal lobectomy, amygdalohippocampectomy.
        Responsive neurostimulation RNS, VNS vagus nerve stimulation.
        Stereotactic radiosurgery, Gamma Knife, CyberKnife.
        Pain surgery, spinal cord stimulation, intrathecal pumps.
        Trigeminal neuralgia, microvascular decompression MVD.
        Hemifacial spasm, glossopharyngeal neuralgia.
    """,
    Subspecialty.PEDIATRIC: """
        Pediatric neurosurgery including congenital malformations.
        Hydrocephalus in children, VP shunt, ETV endoscopic third ventriculostomy.
        Chiari malformation, myelomeningocele, spina bifida.
        Craniosynostosis, plagiocephaly, scaphocephaly.
        Pediatric brain tumors, medulloblastoma, pilocytic astrocytoma.
        Encephalocele, Dandy-Walker malformation.
        Tethered cord syndrome, lipomyelomeningocele.
        Moyamoya in children, pediatric stroke.
    """,
    Subspecialty.TRAUMA: """
        Neurotrauma including traumatic brain injury TBI.
        Epidural hematoma, subdural hematoma, traumatic SAH.
        ICP management, intracranial pressure monitoring, EVD placement.
        Decompressive craniectomy, DC for malignant edema.
        Skull fracture, depressed skull fracture, basilar skull fracture.
        Penetrating brain injury, gunshot wound to head.
        Spinal cord injury SCI, ASIA classification, methylprednisolone.
        Cervical spine clearance, spinal trauma, burst fracture.
        BTF guidelines, CPP management, osmotherapy.
    """,
    Subspecialty.PERIPHERAL_NERVE: """
        Peripheral nerve surgery including nerve repair and grafting.
        Brachial plexus injury, birth brachial plexus palsy.
        Carpal tunnel syndrome, cubital tunnel syndrome.
        Nerve tumor, schwannoma of peripheral nerve, neurofibroma.
        Nerve transfer surgery, nerve reconstruction.
        Traumatic nerve injury, nerve laceration.
        Thoracic outlet syndrome TOS.
        Peripheral nerve entrapment syndromes.
    """,
}


# Keyword-based fallback for when embeddings unavailable
SUBSPECIALTY_KEYWORDS: Dict[Subspecialty, List[str]] = {
    Subspecialty.SKULL_BASE: [
        "skull base", "transsphenoidal", "pituitary", "acoustic neuroma",
        "vestibular schwannoma", "petroclival", "petrous", "cavernous sinus",
        "translabyrinthine", "middle fossa", "endoscopic endonasal",
        "jugular foramen", "glomus", "CPA", "cerebellopontine angle",
    ],
    Subspecialty.VASCULAR: [
        "aneurysm", "clipping", "coiling", "AVM", "arteriovenous",
        "cavernoma", "bypass", "STA-MCA", "moyamoya", "SAH",
        "subarachnoid hemorrhage", "ICH", "intracerebral hemorrhage",
        "carotid", "endarterectomy", "thrombectomy", "stroke", "DAVF",
    ],
    Subspecialty.SPINE: [
        "spine", "spinal", "cervical", "thoracic", "lumbar", "sacral",
        "disc", "discectomy", "laminectomy", "fusion", "ACDF",
        "stenosis", "myelopathy", "radiculopathy", "scoliosis",
        "pedicle screw", "decompression", "foraminotomy",
    ],
    Subspecialty.TUMOR: [
        "glioma", "glioblastoma", "GBM", "astrocytoma", "oligodendroglioma",
        "meningioma", "metastasis", "metastatic", "brain tumor",
        "awake craniotomy", "brain mapping", "5-ALA", "neuro-oncology",
        "convexity", "falcine", "parasagittal", "intraventricular",
    ],
    Subspecialty.FUNCTIONAL: [
        "DBS", "deep brain stimulation", "Parkinson", "essential tremor",
        "dystonia", "epilepsy", "seizure", "temporal lobectomy",
        "RNS", "VNS", "Gamma Knife", "CyberKnife", "radiosurgery",
        "trigeminal neuralgia", "MVD", "microvascular decompression",
        "spinal cord stimulator", "pain pump",
    ],
    Subspecialty.PEDIATRIC: [
        "pediatric", "child", "infant", "neonatal", "congenital",
        "hydrocephalus", "VP shunt", "ETV", "Chiari", "myelomeningocele",
        "spina bifida", "craniosynostosis", "plagiocephaly",
        "medulloblastoma", "pilocytic astrocytoma", "Dandy-Walker",
        "tethered cord", "encephalocele",
    ],
    Subspecialty.TRAUMA: [
        "TBI", "traumatic brain injury", "trauma", "EDH", "epidural",
        "SDH", "subdural", "ICP", "intracranial pressure", "EVD",
        "decompressive craniectomy", "skull fracture", "penetrating",
        "SCI", "spinal cord injury", "BTF", "CPP", "herniation",
    ],
    Subspecialty.PERIPHERAL_NERVE: [
        "peripheral nerve", "brachial plexus", "carpal tunnel",
        "cubital tunnel", "nerve repair", "nerve graft", "nerve transfer",
        "entrapment", "thoracic outlet", "schwannoma", "neurofibroma",
    ],
}


class SubspecialtyClassifier:
    """
    Classifies neurosurgical topics into subspecialties.

    Uses semantic similarity when embedding service is available,
    falls back to keyword matching otherwise.

    Usage:
        classifier = SubspecialtyClassifier(embedding_service)
        await classifier.initialize()

        result = await classifier.classify("MCA aneurysm clipping technique")
        print(f"Subspecialty: {result.subspecialty.value}")  # "vascular"
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        confidence_threshold: float = 0.70,
    ):
        """
        Initialize the classifier.

        Args:
            embedding_service: Service for generating embeddings (optional)
            confidence_threshold: Minimum confidence for classification
        """
        self.embedder = embedding_service
        self.threshold = confidence_threshold
        self._prototype_embeddings: Dict[Subspecialty, np.ndarray] = {}
        self._is_initialized = False

    async def initialize(self) -> None:
        """Pre-compute embeddings for subspecialty prototypes."""
        if self._is_initialized or self.embedder is None:
            return

        logger.info("Initializing SubspecialtyClassifier with embeddings")

        for subspecialty, description in SUBSPECIALTY_PROTOTYPES.items():
            try:
                # Clean up whitespace in prototype description
                clean_description = " ".join(description.split())
                vector = await self.embedder.embed_text(clean_description)
                self._prototype_embeddings[subspecialty] = np.array(vector)
                logger.debug(f"Embedded prototype for {subspecialty.value}")
            except Exception as e:
                logger.error(f"Failed to embed prototype for {subspecialty.value}: {e}")
                raise

        self._is_initialized = True
        logger.info(f"SubspecialtyClassifier initialized with {len(self._prototype_embeddings)} prototypes")

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    async def classify(
        self,
        topic: str,
        topic_embedding: Optional[List[float]] = None,
    ) -> ClassificationResult:
        """
        Classify a topic into a neurosurgical subspecialty.

        Args:
            topic: The topic/query to classify
            topic_embedding: Pre-computed embedding (optional)

        Returns:
            ClassificationResult with subspecialty and confidence
        """
        # Use semantic classification if embeddings available
        if self.embedder is not None:
            if not self._is_initialized:
                await self.initialize()

            return await self._classify_semantic(topic, topic_embedding)

        # Fall back to keyword classification
        return self._classify_keywords(topic)

    async def _classify_semantic(
        self,
        topic: str,
        topic_embedding: Optional[List[float]] = None,
    ) -> ClassificationResult:
        """Classify using vector similarity."""
        # Get embedding for topic
        if topic_embedding is not None:
            target_vec = np.array(topic_embedding)
        else:
            target_vec = np.array(await self.embedder.embed_text(topic))

        # Compare against all prototypes
        scores: Dict[str, float] = {}
        best_subspecialty = Subspecialty.GENERAL
        best_score = -1.0

        for subspecialty, proto_vec in self._prototype_embeddings.items():
            score = self._cosine_similarity(target_vec, proto_vec)
            scores[subspecialty.value] = score

            if score > best_score:
                best_score = score
                best_subspecialty = subspecialty

        return ClassificationResult(
            subspecialty=best_subspecialty,
            confidence_score=best_score,
            all_scores=scores,
            is_confident=(best_score >= self.threshold),
        )

    def _classify_keywords(self, topic: str) -> ClassificationResult:
        """Classify using keyword matching (fallback)."""
        topic_lower = topic.lower()
        scores: Dict[str, float] = {}

        for subspecialty, keywords in SUBSPECIALTY_KEYWORDS.items():
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in topic_lower)
            score = matches / len(keywords) if keywords else 0.0
            scores[subspecialty.value] = score

        # Find best match
        best_subspecialty = Subspecialty.GENERAL
        best_score = 0.0

        for subspecialty in Subspecialty:
            if subspecialty.value in scores and scores[subspecialty.value] > best_score:
                best_score = scores[subspecialty.value]
                best_subspecialty = subspecialty

        return ClassificationResult(
            subspecialty=best_subspecialty,
            confidence_score=best_score,
            all_scores=scores,
            is_confident=(best_score >= 0.15),  # Lower threshold for keywords
        )

    def get_related_subspecialties(self, primary: Subspecialty) -> List[Subspecialty]:
        """
        Get subspecialties related to the primary classification.

        Some topics span multiple subspecialties (e.g., pediatric spine).
        """
        relations: Dict[Subspecialty, List[Subspecialty]] = {
            Subspecialty.SKULL_BASE: [Subspecialty.TUMOR, Subspecialty.VASCULAR],
            Subspecialty.VASCULAR: [Subspecialty.SKULL_BASE, Subspecialty.TRAUMA],
            Subspecialty.SPINE: [Subspecialty.TUMOR, Subspecialty.TRAUMA, Subspecialty.PERIPHERAL_NERVE],
            Subspecialty.TUMOR: [Subspecialty.SKULL_BASE, Subspecialty.SPINE, Subspecialty.FUNCTIONAL],
            Subspecialty.FUNCTIONAL: [Subspecialty.TUMOR],
            Subspecialty.PEDIATRIC: [Subspecialty.TUMOR, Subspecialty.VASCULAR, Subspecialty.SPINE],
            Subspecialty.TRAUMA: [Subspecialty.SPINE, Subspecialty.VASCULAR],
            Subspecialty.PERIPHERAL_NERVE: [Subspecialty.SPINE],
        }
        return relations.get(primary, [])


# Convenience function for quick classification
async def classify_subspecialty(
    topic: str,
    embedding_service: Optional[EmbeddingService] = None,
) -> str:
    """
    Convenience function to classify a topic.

    Returns subspecialty name as string.
    """
    classifier = SubspecialtyClassifier(embedding_service)
    result = await classifier.classify(topic)
    return result.subspecialty.value
