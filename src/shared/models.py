"""
NeuroSynth Unified Models
=========================

Merged Phase 1 + Phase 2 data models with database support.

Key design principles:
1. Phase 1 models as canonical (rich medical domain knowledge)
2. Database serialization (to_db, from_db) for Phase 2
3. Full backward compatibility with Phase 1 pickle exports
4. Support for embeddings, UMLS CUIs, knowledge graph relationships
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
import numpy as np
import json


# =============================================================================
# ENUMS (Phase 1)
# =============================================================================

class DocumentStatus(Enum):
    """Processing state of a document."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    ERROR = "error"


class ChunkType(Enum):
    """Semantic classification of chunk content."""
    PROCEDURE = "procedure"
    ANATOMY = "anatomy"
    PATHOLOGY = "pathology"
    CLINICAL = "clinical"
    CASE = "case"
    GENERAL = "general"
    FRONT_MATTER = "front_matter"  # Titles, authors, affiliations - deprioritized in search


class ImageType(Enum):
    """Classification of extracted images."""
    SURGICAL_PHOTO = "surgical_photo"
    ANATOMY_DIAGRAM = "anatomy_diagram"
    IMAGING_SCAN = "imaging_scan"
    FLOWCHART = "flowchart"
    ILLUSTRATION = "illustration"
    UNKNOWN = "unknown"


class EntityCategory(Enum):
    """High-level entity category."""
    ANATOMY_VASCULAR = "anatomy_vascular"
    ANATOMY_NEURAL = "anatomy_neural"
    ANATOMY_BONE = "anatomy_bone"
    PATHOLOGY = "pathology"
    PROCEDURE = "procedure"
    INSTRUMENT = "instrument"
    MEASUREMENT = "measurement"
    UNKNOWN = "unknown"


class LinkMatchType(Enum):
    """How an image-chunk link was established."""
    DETERMINISTIC = "deterministic"
    FUSION_SEMANTIC = "fusion:semantic"
    FUSION_CUI = "fusion:cui"
    CUI_ONLY = "cui_only"
    PROXIMITY = "proximity"
    NONE = "none"


# =============================================================================
# ENTITY MODELS (Phase 1)
# =============================================================================

@dataclass
class NeuroEntity:
    """A specific medical entity identified in text."""
    text: str
    category: str
    normalized: str
    start: int
    end: int
    confidence: float
    context_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "category": self.category,
            "normalized": self.normalized,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "context_snippet": self.context_snippet
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NeuroEntity":
        return cls(
            text=data["text"],
            category=data["category"],
            normalized=data.get("normalized", data["text"]),
            start=data.get("start", 0),
            end=data.get("end", 0),
            confidence=data.get("confidence", 0.9),
            context_snippet=data.get("context_snippet", "")
        )


@dataclass
class EntityRelation:
    """Relationship between two medical entities for knowledge graph."""
    source_entity: str
    target_entity: str
    relation_type: str
    source_category: str
    target_category: str
    confidence: float
    chunk_id: str
    document_id: str
    context_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_entity,
            "target": self.target_entity,
            "type": self.relation_type,
            "source_category": self.source_category,
            "target_category": self.target_category,
            "confidence": self.confidence,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "context": self.context_snippet[:200] if self.context_snippet else ""
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityRelation":
        return cls(
            source_entity=data["source"],
            target_entity=data["target"],
            relation_type=data["type"],
            source_category=data.get("source_category", "UNKNOWN"),
            target_category=data.get("target_category", "UNKNOWN"),
            confidence=data.get("confidence", 0.8),
            chunk_id=data.get("chunk_id", ""),
            document_id=data.get("document_id", ""),
            context_snippet=data.get("context", "")
        )


@dataclass
class UMLSEntity:
    """UMLS concept extracted via SciSpacy."""
    cui: str
    name: str
    score: float
    tui: str
    semantic_type: str
    weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cui": self.cui,
            "name": self.name,
            "score": self.score,
            "tui": self.tui,
            "semantic_type": self.semantic_type,
            "weight": self.weight
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UMLSEntity":
        return cls(
            cui=data["cui"],
            name=data["name"],
            score=data.get("score", 0.8),
            tui=data.get("tui", "T000"),
            semantic_type=data.get("semantic_type", "Unknown"),
            weight=data.get("weight", 0.5)
        )


@dataclass
class LinkResult:
    """Result of tri-pass image-chunk linking."""
    chunk_id: str
    image_id: str
    strength: float
    match_type: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "image_id": self.image_id,
            "strength": round(self.strength, 4),
            "match_type": self.match_type,
            "details": self.details
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinkResult":
        return cls(
            chunk_id=data["chunk_id"],
            image_id=data["image_id"],
            strength=data["strength"],
            match_type=data["match_type"],
            details=data.get("details", {})
        )


@dataclass
class ProcessingManifest:
    """Metadata about processed document for Phase 2 consumption."""
    document_id: str
    source_path: str
    text_embedding_dim: int
    image_embedding_dim: int
    text_embedding_provider: str
    chunk_count: int
    image_count: int
    link_count: int
    chunks_with_cuis: int = 0
    images_with_cuis: int = 0
    files: Dict[str, str] = field(default_factory=dict)
    processing_time_seconds: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "embedding_config": {
                "text_dimension": self.text_embedding_dim,
                "image_dimension": self.image_embedding_dim,
                "text_provider": self.text_embedding_provider
            },
            "counts": {
                "chunks": self.chunk_count,
                "images": self.image_count,
                "links": self.link_count
            },
            "statistics": {
                "chunks_with_cuis": self.chunks_with_cuis,
                "images_with_cuis": self.images_with_cuis
            },
            "files": self.files,
            "processing_time_seconds": self.processing_time_seconds,
            "created_at": self.created_at
        }


# =============================================================================
# DOCUMENT MODELS (Phase 1 + DB)
# =============================================================================

@dataclass
class Document:
    """Source document metadata."""
    id: str
    file_path: Path
    content_hash: str
    title: str
    series: Optional[str] = None
    volume: Optional[int] = None
    chapter: Optional[int] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    specialty: str = "general"
    authority_score: float = 1.0
    status: DocumentStatus = DocumentStatus.PENDING
    page_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    indexed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    organization_id: Optional[str] = None
    extraction_metrics: Optional[Dict[str, Any]] = None
    # Phase 2: Database fields
    db_id: Optional[UUID] = None
    processing_time_seconds: Optional[float] = None

    @classmethod
    def create(cls, file_path: Path, content_hash: str, title: str) -> "Document":
        return cls(
            id=str(uuid4()),
            file_path=file_path,
            content_hash=content_hash,
            title=title
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "file_path": str(self.file_path),
            "content_hash": self.content_hash,
            "title": self.title,
            "series": self.series,
            "volume": self.volume,
            "chapter": self.chapter,
            "authors": self.authors,
            "year": self.year,
            "specialty": self.specialty,
            "authority_score": self.authority_score,
            "status": self.status.value,
            "page_count": self.page_count,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "extraction_metrics": self.extraction_metrics
        }

    def to_db(self) -> Dict[str, Any]:
        """Convert to database row format."""
        return {
            "id": self.db_id or uuid4(),
            "source_path": str(self.file_path),
            "title": self.title,
            "total_pages": self.page_count,
            "authority_score": self.authority_score,
            "processing_time_seconds": self.processing_time_seconds,
            "metadata": {
                "series": self.series,
                "volume": self.volume,
                "chapter": self.chapter,
                "authors": self.authors,
                "year": self.year,
                "specialty": self.specialty,
                "status": self.status.value,
                "created_by": self.created_by,
                "organization_id": self.organization_id,
                "extraction_metrics": self.extraction_metrics
            }
        }

    @classmethod
    def from_db(cls, row: Dict[str, Any]) -> "Document":
        """Create from database row."""
        metadata = row.get("metadata", {})
        return cls(
            id=str(row.get("source_path", "unknown")),
            db_id=row.get("id"),
            file_path=Path(row["source_path"]),
            content_hash=row.get("content_hash", ""),
            title=row["title"],
            series=metadata.get("series"),
            volume=metadata.get("volume"),
            chapter=metadata.get("chapter"),
            authors=metadata.get("authors"),
            year=metadata.get("year"),
            specialty=metadata.get("specialty", "general"),
            authority_score=row.get("authority_score", 1.0),
            page_count=row.get("total_pages", 0),
            processing_time_seconds=row.get("processing_time_seconds")
        )


@dataclass
class Page:
    """Raw page content with layout metadata."""
    document_id: str
    page_number: int
    content: str
    has_images: bool = False
    has_tables: bool = False
    word_count: int = 0
    used_ocr: bool = False
    sections: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.word_count == 0:
            self.word_count = len(self.content.split())


@dataclass
class Section:
    """Document structure element."""
    title: str
    level: int
    page_start: int
    page_end: int
    content: str = ""

    def __repr__(self) -> str:
        return f"Section('{self.title}', level={self.level}, pages={self.page_start}-{self.page_end})"


# =============================================================================
# CHUNK MODELS (Phase 1 + DB)
# =============================================================================

@dataclass
class SemanticChunk:
    """Core retrieval unit - semantically complete medical knowledge."""
    id: str
    document_id: str
    content: str
    title: str
    section_path: List[str]
    page_start: int
    page_end: int
    chunk_type: ChunkType
    specialty_tags: List[str] = field(default_factory=list)
    entities: List[NeuroEntity] = field(default_factory=list)
    entity_names: List[str] = field(default_factory=list)
    figure_refs: List[str] = field(default_factory=list)
    table_refs: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)
    text_embedding: Optional[np.ndarray] = None
    fused_embedding: Optional[np.ndarray] = None
    keywords: List[str] = field(default_factory=list)
    readability_score: float = 0.0
    coherence_score: float = 0.0
    completeness_score: float = 0.0
    type_specific_score: float = 0.0  # v2.2: Fourth quality dimension
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    cuis: List[str] = field(default_factory=list)
    umls_entities: List[UMLSEntity] = field(default_factory=list)
    # Human-readable summary (Stage 4.5)
    summary: Optional[str] = None
    # Phase 2: Database fields
    db_id: Optional[UUID] = None
    contextual_content: Optional[str] = None
    created_at: Optional[datetime] = None

    # ==========================================================================
    # v2.2 Enhanced Metadata Fields
    # ==========================================================================

    # Procedural metadata
    surgical_phase: Optional[str] = None
    """Phase of surgical procedure (positioning, exposure, approach, etc.)"""

    step_number: Optional[int] = None
    """Explicit step number if present in content (e.g., Step 3)"""

    step_sequence: Optional[str] = None
    """Position in sequence (e.g., '3_of_8' for ordering during synthesis)"""

    # High-value content flags
    has_pitfall: bool = False
    """Contains surgical pitfall, pearl, or critical warning"""

    has_teaching_point: bool = False
    """Contains explicit teaching point or key concept"""

    has_key_measurement: bool = False
    """Contains critical measurements (distances, angles, etc.)"""

    # Pathology-specific metadata
    grading_scale: Optional[str] = None
    """Grading scale used (e.g., 'spetzler_martin', 'who', 'hunt_hess')"""

    grade_value: Optional[str] = None
    """Specific grade if mentioned (e.g., 'III', '4')"""

    molecular_markers: List[str] = field(default_factory=list)
    """Molecular markers mentioned (IDH, MGMT, 1p/19q, etc.)"""

    # Anatomy-specific metadata
    anatomical_region: Optional[str] = None
    """Broad anatomical region (skull_base, spine, vascular, etc.)"""

    spatial_relationships: List[str] = field(default_factory=list)
    """Key spatial relationships mentioned (e.g., 'lateral_to:optic_nerve')"""

    has_variation: bool = False
    """Describes anatomical variation"""

    # Clinical-specific metadata
    has_decision_point: bool = False
    """Contains clinical decision point or algorithm branch"""

    has_evidence_citation: bool = False
    """Contains reference to study or evidence"""

    # v2.2 Orphan status
    is_orphan: bool = False
    """Chunk appears to be mid-sequence (starts with Step 2+, Then, etc.)"""

    # Imaging-specific metadata
    imaging_modality: Optional[str] = None
    """Primary imaging modality discussed (MRI, CT, etc.)"""

    imaging_sequences: List[str] = field(default_factory=list)
    """Specific sequences mentioned (T1, T2, FLAIR, etc.)"""

    # ==========================================================================
    # v2.2 Computed Properties
    # ==========================================================================

    @property
    def quality_score(self) -> float:
        """Weighted aggregate quality score (v2.2: 4 dimensions + orphan penalty)."""
        base_score = (
            self.readability_score * 0.20 +
            self.coherence_score * 0.25 +
            self.completeness_score * 0.35 +
            self.type_specific_score * 0.20
        )
        # Apply orphan penalty (v2.2)
        if self.is_orphan:
            base_score = max(0.0, base_score - 0.20)
        return base_score

    @property
    def is_high_value(self) -> bool:
        """Check if chunk contains high-value content."""
        return self.has_pitfall or self.has_teaching_point or self.has_key_measurement

    @property
    def has_step_context(self) -> bool:
        """Check if chunk has procedural step information."""
        return self.step_number is not None or self.surgical_phase is not None

    @classmethod
    def create(
        cls,
        document_id: str,
        content: str,
        section_title: str,
        page: int,
        chunk_type: ChunkType,
        entities: List[NeuroEntity] = None
    ) -> "SemanticChunk":
        entities = entities or []
        return cls(
            id=str(uuid4()),
            document_id=document_id,
            content=content,
            title=section_title,
            section_path=[section_title],
            page_start=page,
            page_end=page,
            chunk_type=chunk_type,
            entities=entities,
            entity_names=[e.text for e in entities]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "content": self.content,
            "title": self.title,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "chunk_type": self.chunk_type.value,
            "specialty_tags": self.specialty_tags,
            "entity_names": self.entity_names,
            "entities": [e.to_dict() for e in self.entities],
            "figure_refs": self.figure_refs,
            "image_ids": self.image_ids,
            "keywords": self.keywords,
            "cuis": self.cuis,
            "umls_entities": [e.to_dict() for e in self.umls_entities]
        }

    def to_db(self) -> Dict[str, Any]:
        """Convert to database row format."""
        # Handle numpy arrays - convert to lists for JSON serialization
        text_embedding_list = None
        if self.text_embedding is not None:
            text_embedding_list = self.text_embedding.tolist() if isinstance(self.text_embedding, np.ndarray) else self.text_embedding

        return {
            "id": self.db_id or uuid4(),
            "document_id": self.document_id,
            "content": self.content,
            "page_number": self.page_start,
            "chunk_index": 0,  # Could be calculated
            "chunk_type": self.chunk_type.value,
            "specialty": self.specialty_tags[0] if self.specialty_tags else None,
            "embedding": text_embedding_list,
            "cuis": self.cuis,
            "entities": [e.to_dict() for e in self.entities],
            "contextual_content": self.contextual_content,
            "metadata": {
                "title": self.title,
                "section_path": self.section_path,
                "entity_names": self.entity_names,
                "figure_refs": self.figure_refs,
                "table_refs": self.table_refs,
                "image_ids": self.image_ids,
                "keywords": self.keywords,
                "readability_score": self.readability_score,
                "coherence_score": self.coherence_score,
                "completeness_score": self.completeness_score,
                "embedding_model": self.embedding_model,
                "umls_entities": [e.to_dict() for e in self.umls_entities]
            }
        }

    @classmethod
    def from_db(cls, row: Dict[str, Any]) -> "SemanticChunk":
        """Create from database row."""
        metadata = row.get("metadata", {})

        # Handle embedding
        embedding = None
        if row.get("embedding"):
            embedding = np.array(row["embedding"]) if isinstance(row["embedding"], list) else row["embedding"]

        # Parse entities
        entities = []
        if row.get("entities"):
            for e in row["entities"]:
                entities.append(NeuroEntity.from_dict(e))

        # Parse UMLS entities
        umls_entities = []
        if metadata.get("umls_entities"):
            for e in metadata["umls_entities"]:
                umls_entities.append(UMLSEntity.from_dict(e))

        # Infer chunk type
        chunk_type = ChunkType(row.get("chunk_type", "general")) if row.get("chunk_type") else ChunkType.GENERAL

        return cls(
            id=str(row.get("id", "unknown")),
            db_id=row.get("id"),
            document_id=row["document_id"],
            content=row["content"],
            title=metadata.get("title", ""),
            section_path=metadata.get("section_path", []),
            page_start=row.get("page_number", 0),
            page_end=row.get("page_number", 0),
            chunk_type=chunk_type,
            entity_names=metadata.get("entity_names", []),
            entities=entities,
            figure_refs=metadata.get("figure_refs", []),
            table_refs=metadata.get("table_refs", []),
            image_ids=metadata.get("image_ids", []),
            text_embedding=embedding,
            keywords=metadata.get("keywords", []),
            cuis=row.get("cuis", []),
            umls_entities=umls_entities,
            contextual_content=row.get("contextual_content"),
            created_at=row.get("created_at")
        )

    @property
    def is_valid(self) -> bool:
        """Chunk is valid only if text embedding exists."""
        return (
            self.text_embedding is not None and
            len(self.content.strip()) > 0
        )


# =============================================================================
# IMAGE MODELS (Phase 1 + DB)
# =============================================================================

@dataclass
class ExtractedImage:
    """Visual element with full context."""
    id: str
    document_id: str
    page_number: int
    file_path: Path
    width: int
    height: int
    format: str = "png"
    content_hash: str = ""
    image_type: ImageType = ImageType.UNKNOWN
    is_decorative: bool = False
    quality_score: float = 0.0
    caption: Optional[str] = None
    caption_confidence: float = 0.0
    figure_id: Optional[str] = None
    surrounding_text: str = ""
    sequence_id: Optional[str] = None
    sequence_position: Optional[int] = None
    chunk_ids: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    embedding_model: Optional[str] = None
    vlm_caption: Optional[str] = None
    vlm_image_type: Optional[str] = None
    caption_embedding: Optional[np.ndarray] = None
    # Human-readable caption summary (Stage 8.5)
    caption_summary: Optional[str] = None
    cuis: List[str] = field(default_factory=list)
    umls_entities: List[UMLSEntity] = field(default_factory=list)
    link_strengths: Dict[str, float] = field(default_factory=dict)
    # Phase 2: Database fields
    db_id: Optional[UUID] = None
    triage_tier: Optional[int] = None
    triage_skipped: bool = False
    triage_reason: Optional[str] = None
    created_at: Optional[datetime] = None

    @classmethod
    def create(
        cls,
        document_id: str,
        page_number: int,
        file_path: Path,
        width: int,
        height: int,
        content_hash: str
    ) -> "ExtractedImage":
        return cls(
            id=str(uuid4()),
            document_id=document_id,
            page_number=page_number,
            file_path=file_path,
            width=width,
            height=height,
            content_hash=content_hash
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "file_path": str(self.file_path),
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "content_hash": self.content_hash,
            "image_type": self.image_type.value,
            "is_decorative": self.is_decorative,
            "quality_score": self.quality_score,
            "caption": self.caption,
            "caption_confidence": self.caption_confidence,
            "figure_id": self.figure_id,
            "surrounding_text": self.surrounding_text[:500] if self.surrounding_text else "",
            "chunk_ids": self.chunk_ids,
            "vlm_caption": self.vlm_caption,
            "vlm_image_type": self.vlm_image_type,
            "cuis": self.cuis,
            "umls_entities": [e.to_dict() for e in self.umls_entities],
            "link_strengths": self.link_strengths
        }

    def to_db(self) -> Dict[str, Any]:
        """Convert to database row format."""
        # Handle numpy arrays
        embedding_list = None
        if self.embedding is not None:
            embedding_list = self.embedding.tolist() if isinstance(self.embedding, np.ndarray) else self.embedding

        caption_embedding_list = None
        if self.caption_embedding is not None:
            caption_embedding_list = self.caption_embedding.tolist() if isinstance(self.caption_embedding, np.ndarray) else self.caption_embedding

        return {
            "id": self.db_id or uuid4(),
            "document_id": self.document_id,
            "file_path": str(self.file_path),
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "image_type": self.image_type.value,
            "is_decorative": self.is_decorative,
            "vlm_caption": self.vlm_caption,
            "vlm_confidence": self.caption_confidence,
            "embedding": embedding_list,
            "caption_embedding": caption_embedding_list,
            "cuis": self.cuis,
            "triage_tier": self.triage_tier,
            "triage_skipped": self.triage_skipped,
            "triage_reason": self.triage_reason,
            "metadata": {
                "content_hash": self.content_hash,
                "caption": self.caption,
                "figure_id": self.figure_id,
                "surrounding_text": self.surrounding_text[:500] if self.surrounding_text else "",
                "chunk_ids": self.chunk_ids,
                "vlm_image_type": self.vlm_image_type,
                "quality_score": self.quality_score,
                "embedding_model": self.embedding_model,
                "umls_entities": [e.to_dict() for e in self.umls_entities],
                "link_strengths": self.link_strengths
            }
        }

    @classmethod
    def from_db(cls, row: Dict[str, Any]) -> "ExtractedImage":
        """Create from database row."""
        metadata = row.get("metadata", {})

        # Handle embeddings
        embedding = None
        if row.get("embedding"):
            embedding = np.array(row["embedding"]) if isinstance(row["embedding"], list) else row["embedding"]

        caption_embedding = None
        if row.get("caption_embedding"):
            caption_embedding = np.array(row["caption_embedding"]) if isinstance(row["caption_embedding"], list) else row["caption_embedding"]

        # Parse UMLS entities
        umls_entities = []
        if metadata.get("umls_entities"):
            for e in metadata["umls_entities"]:
                umls_entities.append(UMLSEntity.from_dict(e))

        # Infer image type
        image_type = ImageType(row.get("image_type", "unknown")) if row.get("image_type") else ImageType.UNKNOWN

        return cls(
            id=str(row.get("id", "unknown")),
            db_id=row.get("id"),
            document_id=row["document_id"],
            page_number=row["page_number"],
            file_path=Path(row["file_path"]),
            width=row.get("width", 0),
            height=row.get("height", 0),
            format=row.get("format", "png"),
            image_type=image_type,
            is_decorative=row.get("is_decorative", False),
            vlm_caption=row.get("vlm_caption"),
            embedding=embedding,
            caption_embedding=caption_embedding,
            cuis=row.get("cuis", []),
            triage_tier=row.get("triage_tier"),
            triage_skipped=row.get("triage_skipped", False),
            triage_reason=row.get("triage_reason"),
            caption=metadata.get("caption"),
            figure_id=metadata.get("figure_id"),
            surrounding_text=metadata.get("surrounding_text", ""),
            chunk_ids=metadata.get("chunk_ids", []),
            vlm_image_type=metadata.get("vlm_image_type"),
            quality_score=metadata.get("quality_score", 0.0),
            embedding_model=metadata.get("embedding_model"),
            umls_entities=umls_entities,
            link_strengths=metadata.get("link_strengths", {}),
            created_at=row.get("created_at")
        )

    @property
    def embeddable_text(self) -> str:
        """Combined text for caption embedding."""
        parts = []
        if self.vlm_caption:
            parts.append(self.vlm_caption)
        if self.caption:
            parts.append(f"Figure caption: {self.caption}")
        if self.surrounding_text:
            parts.append(f"Context: {self.surrounding_text[:300]}")
        return " | ".join(parts) if parts else ""

    @property
    def is_valid(self) -> bool:
        """Image valid only if both embeddings exist (or is decorative)."""
        if self.is_decorative:
            return True
        return (
            self.embedding is not None and
            self.caption_embedding is not None
        )


# =============================================================================
# TABLE MODELS
# =============================================================================

@dataclass
class ExtractedTable:
    """Extracted table with structure preserved as Markdown and HTML."""
    id: str
    document_id: str
    page_number: int
    markdown_content: str
    html_content: str = ""
    raw_text: str = ""
    table_type: str = "general"
    title: Optional[str] = None
    chunk_ids: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        document_id: str,
        page_number: int,
        markdown_content: str,
        raw_text: str,
        html_content: str = ""
    ) -> "ExtractedTable":
        return cls(
            id=str(uuid4()),
            document_id=document_id,
            page_number=page_number,
            markdown_content=markdown_content,
            html_content=html_content,
            raw_text=raw_text
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "markdown_content": self.markdown_content,
            "html_content": self.html_content,
            "raw_text": self.raw_text,
            "table_type": self.table_type,
            "title": self.title,
            "chunk_ids": self.chunk_ids
        }


# =============================================================================
# SEARCH RESULT MODELS
# =============================================================================

@dataclass
class SearchResult:
    """Result from hybrid search with quality metrics."""
    chunk_id: str
    document_id: str
    content: str
    title: str
    chunk_type: ChunkType
    page_start: int
    entity_names: List[str]
    image_ids: List[str]
    authority_score: float
    keyword_score: float
    semantic_score: float
    final_score: float
    document_title: Optional[str] = None
    cuis: List[str] = field(default_factory=list)  # UMLS Concept Unique Identifiers
    images: List[ExtractedImage] = field(default_factory=list)
    # Quality scores (populated from chunks table)
    readability_score: float = 0.0
    coherence_score: float = 0.0
    completeness_score: float = 0.0
    # Embedding vector for MMR diversity reranking
    embedding: Optional[np.ndarray] = None

    @property
    def quality_score(self) -> float:
        """Computed aggregate quality score (weighted average)."""
        return (
            self.readability_score * 0.25 +
            self.coherence_score * 0.40 +
            self.completeness_score * 0.35
        )


@dataclass
class ImageSearchResult:
    """Result from image search."""
    image_id: str
    document_id: str
    file_path: Path
    caption: Optional[str]
    figure_id: Optional[str]
    image_type: ImageType
    similarity_score: float
    linked_chunks: List[str] = field(default_factory=list)


# =============================================================================
# EXTRACTION METRICS
# =============================================================================

@dataclass
class ExtractionMetrics:
    """Metrics for entity extraction quality."""
    total_entities: int = 0
    ambiguous_entities: int = 0
    unresolved_entities: int = 0
    normalized_entities: int = 0
    novel_entities: int = 0
    ocr_pages: int = 0
    total_pages: int = 0
    processing_time_ms: float = 0.0

    @property
    def specificity_score(self) -> float:
        if self.total_entities == 0:
            return 1.0
        confident = self.total_entities - self.unresolved_entities
        return confident / self.total_entities

    @property
    def normalization_rate(self) -> float:
        if self.total_entities == 0:
            return 1.0
        return self.normalized_entities / self.total_entities

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entities": self.total_entities,
            "ambiguous_entities": self.ambiguous_entities,
            "unresolved_entities": self.unresolved_entities,
            "normalized_entities": self.normalized_entities,
            "novel_entities": self.novel_entities,
            "ocr_pages": self.ocr_pages,
            "total_pages": self.total_pages,
            "processing_time_ms": self.processing_time_ms,
            "specificity_score": self.specificity_score,
            "normalization_rate": self.normalization_rate
        }
