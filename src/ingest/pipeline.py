"""
NeuroSynth v2.0 - Ingest Pipeline (Enhanced)
=============================================

Complete ingestion pipeline for neurosurgical documents.

Pipeline stages:
1. Open PDF (single open)
2. Extract structure (sections)
3. Extract pages + images + tables (with OCR fallback)
4. Chunk semantically
5. Link images to chunks
6. Generate embeddings
7. Store atomically

Enhancements:
- OCR fallback for scanned PDFs
- Granular progress tracking
- Streaming extraction option
- Checkpoint-based recovery
- Expanded authority scores
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable, Tuple, Set, AsyncIterator, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Load environment variables from .env file
# This ensures VOYAGE_API_KEY and other secrets are available
# even when running outside the API context
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system environment

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# OCR support
try:
    import pytesseract
    from PIL import Image
    import io
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

import pickle
import time

from src.shared.models import (
    Document, Page, Section, SemanticChunk, ExtractedImage, ExtractedTable,
    DocumentStatus, ChunkType, ProcessingManifest, LinkResult, NeuroEntity
)
from src.core.database import NeuroDatabase
from src.core.neuro_chunker import NeuroSemanticChunker, TableAwareChunker
from src.core.quality_scorer import ChunkQualityScorer, get_quality_scorer
from src.ingest.section_detector import SectionDetector, OutlineBasedSectionDetector
from src.ingest.image_extractor import NeuroImageExtractor
from src.ingest.table_extractor import TableExtractor
from src.ingest.fusion import TriPassLinker, EmbeddingFuser
from src.ingest.embeddings import TextEmbedder, ImageEmbedder, create_text_embedder, create_image_embedder
from src.core.metrics import get_metrics_collector

# UMLS extraction for standardized medical concepts
try:
    from src.core.umls_extractor import UMLSExtractor, get_default_extractor
    HAS_UMLS = True
except ImportError:
    HAS_UMLS = False

# Contextual preprocessing for improved retrieval
try:
    from src.ingest.contextual_preprocessor import (
        ContextualPreprocessor, ContextConfig, ContextMode
    )
    HAS_CONTEXTUAL = True
except ImportError:
    HAS_CONTEXTUAL = False

# Knowledge Graph for entity relationships (GraphRAG)
try:
    from src.retrieval.knowledge_graph import NeurosurgicalKnowledgeGraph
    from src.core.neuro_extractor import NeuroExpertTextExtractor as NeuroExtractor
    HAS_KNOWLEDGE_GRAPH = True
except ImportError:
    HAS_KNOWLEDGE_GRAPH = False

# Relation Extraction Pipeline (spaCy NLP-based)
try:
    from src.ingest.relation_pipeline import RelationExtractionPipeline
    HAS_RELATION_EXTRACTION = True
except Exception as e:
    # ImportError, pydantic.ConfigError (Python 3.14+ compat), or other
    HAS_RELATION_EXTRACTION = False
    import logging
    logging.getLogger(__name__).warning(f"Relation extraction unavailable: {e}")

# VLM Image Captioning (Claude Vision)
try:
    from src.retrieval.vlm_captioner import VLMImageCaptioner, ImageInput, ImageType
    HAS_VLM = True
except ImportError:
    HAS_VLM = False

# Visual Triage imports (Phase 1 Enhancement)
try:
    from src.vision.visual_triage import VisualTriage, TriageAwareVLMCaptioner
    HAS_TRIAGE = True
except ImportError:
    HAS_TRIAGE = False

# Monitoring imports (Phase 1 Enhancement)
try:
    from src.monitoring import record_vlm_request, record_document_processed
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Pipeline processing stages for progress tracking."""
    INIT = "init"
    STRUCTURE = "structure"
    PAGES = "pages"
    IMAGES = "images"
    TABLES = "tables"
    CHUNKING = "chunking"
    CHUNK_SUMMARIZATION = "chunk_summarization"  # Stage 4.5: Brief summaries
    UMLS_EXTRACTION = "umls_extraction"  # UMLS CUI extraction
    LINKING = "linking"
    TEXT_EMBEDDING = "text_embedding"
    IMAGE_EMBEDDING = "image_embedding"
    CAPTION_EMBEDDING = "caption_embedding"  # VLM caption embedding
    VLM_CAPTION = "vlm_caption"
    CAPTION_SUMMARIZATION = "caption_summarization"  # Stage 8.5: Caption summaries
    STORAGE = "storage"
    COMPLETE = "complete"


@dataclass
class PipelineConfig:
    """Configuration for the ingestion pipeline."""
    output_dir: Path
    enable_images: bool = True
    enable_tables: bool = True
    enable_embeddings: bool = True
    enable_ocr: bool = True
    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 800

    # OCR settings
    ocr_dpi: int = 300
    ocr_min_text_threshold: int = 50  # chars below which OCR triggers
    ocr_language: str = "eng"

    # Progress tracking
    enable_progress: bool = True

    # Contextual embeddings (Anthropic approach - 49% improvement)
    enable_contextual: bool = True
    contextual_mode: str = "llm_full"  # template | llm_light | llm_full
    contextual_model: str = "claude-sonnet-4-20250514"
    contextual_api_key: Optional[str] = None  # Falls back to ANTHROPIC_API_KEY env var

    # Knowledge Graph (GraphRAG - +25% entity recall)
    enable_knowledge_graph: bool = True
    knowledge_graph_path: Optional[Path] = None  # Falls back to output_dir/knowledge_graph.json

    # Relation Extraction (spaCy NLP - 16 relation types)
    enable_relation_extraction: bool = True
    relation_extraction_model: str = "en_core_web_lg"  # or scispacy model
    relation_min_confidence: float = 0.5

    # VLM Image Captioning (Claude Vision - +30% image retrieval)
    enable_vlm_captions: bool = True
    vlm_model: str = "claude-sonnet-4-20250514"
    vlm_batch_size: int = 10
    vlm_api_key: Optional[str] = None  # Falls back to ANTHROPIC_API_KEY env var

    # UMLS Extraction (SciSpacy - dual extraction with regex)
    enable_umls: bool = True
    umls_model: str = "en_core_sci_lg"
    umls_threshold: float = 0.80

    # Auto-embedders (auto-create if not provided)
    text_embedding_provider: str = "voyage"
    text_embedding_model: str = "voyage-3"  # 1024d
    image_embedding_provider: str = "biomedclip"  # 512d

    # Caption embedding (embed VLM captions with text embedder)
    enable_caption_embedding: bool = True

    # Content summarization (brief human-readable summaries)
    # IMPORTANT: Always keep enabled - summaries are critical for UX
    enable_summaries: bool = True  # DO NOT DISABLE - fallbacks ensure 100% coverage
    summary_model: str = "claude-sonnet-4-20250514"
    summary_max_concurrent: int = 10  # Concurrent API calls for batch efficiency


@dataclass
class PipelineResult:
    """Result of processing a document."""
    document: Document
    page_count: int
    chunk_count: int
    image_count: int
    table_count: int
    duration_seconds: float
    ocr_pages: int = 0  # Pages processed with OCR
    error: Optional[str] = None

    # Full extracted data for Phase 2 export
    chunks: List[SemanticChunk] = field(default_factory=list)
    images: List[ExtractedImage] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    links: List[LinkResult] = field(default_factory=list)  # Image-chunk links

    # Detailed metrics
    extraction_metrics: dict = field(default_factory=dict)


@dataclass
class ProgressInfo:
    """Progress information for callbacks."""
    stage: ProcessingStage
    stage_progress: float  # 0.0 - 1.0 within stage
    overall_progress: float  # 0.0 - 1.0 overall
    message: str
    details: Optional[dict] = None


class ProgressTracker:
    """
    Granular progress tracking for pipeline stages.
    
    Maps each stage to a portion of overall progress.
    """
    
    STAGE_WEIGHTS = {
        ProcessingStage.INIT: (0.0, 0.02),
        ProcessingStage.STRUCTURE: (0.02, 0.06),
        ProcessingStage.PAGES: (0.06, 0.15),
        ProcessingStage.IMAGES: (0.15, 0.20),
        ProcessingStage.TABLES: (0.20, 0.24),
        ProcessingStage.CHUNKING: (0.24, 0.28),
        ProcessingStage.CHUNK_SUMMARIZATION: (0.28, 0.32),
        ProcessingStage.UMLS_EXTRACTION: (0.32, 0.36),
        ProcessingStage.LINKING: (0.36, 0.40),
        ProcessingStage.TEXT_EMBEDDING: (0.40, 0.48),
        ProcessingStage.IMAGE_EMBEDDING: (0.48, 0.52),
        ProcessingStage.VLM_CAPTION: (0.52, 0.85),  # 33% - VLM is the bottleneck
        ProcessingStage.CAPTION_SUMMARIZATION: (0.85, 0.88),
        ProcessingStage.CAPTION_EMBEDDING: (0.88, 0.92),
        ProcessingStage.STORAGE: (0.92, 0.98),
        ProcessingStage.COMPLETE: (0.98, 1.0),
    }
    
    def __init__(self, callback: Optional[Callable[[ProgressInfo], None]] = None):
        self.callback = callback
        self.current_stage = ProcessingStage.INIT
    
    def update(
        self,
        stage: ProcessingStage,
        stage_progress: float = 0.0,
        message: str = "",
        details: Optional[dict] = None
    ):
        """Update progress and notify callback."""
        self.current_stage = stage
        
        start, end = self.STAGE_WEIGHTS.get(stage, (0.0, 1.0))
        overall = start + (end - start) * min(stage_progress, 1.0)
        
        if self.callback:
            info = ProgressInfo(
                stage=stage,
                stage_progress=stage_progress,
                overall_progress=overall,
                message=message,
                details=details
            )
            self.callback(info)


class NeuroIngestPipeline:
    """
    Complete ingestion pipeline for neurosurgical documents.
    
    Orchestrates all extraction, chunking, and storage operations.
    
    Enhancements:
    - OCR fallback for scanned documents
    - Granular progress tracking
    - Expanded authority scores
    """
    
    # Authority scores for known sources (expanded)
    AUTHORITY_SCORES = {
        # Tier 1: Gold standard references
        "rhoton": 3.0,
        "seven aneurysms": 2.8,
        "lawton": 2.8,
        
        # Tier 2: Major textbooks
        "youmans": 2.5,
        "sekhar": 2.5,
        "spetzler": 2.5,
        "bambakidis": 2.3,
        "quinones": 2.3,
        "alfredo": 2.3,  # Quinones-Hinojosa Atlas
        
        # Tier 3: Standard references
        "schmidek": 2.0,
        "winn": 2.0,
        "greenberg": 2.0,
        "principles of neurosurgery": 2.0,
        "rengachary": 2.0,
        
        # Tier 4: Specialty references
        "osborn": 1.8,  # Neuroradiology
        "harbaugh": 1.8,
        "benzel": 1.8,  # Spine
        "vaccaro": 1.8,
        
        # Tier 5: Guidelines/Reviews
        "congress of neurological surgeons": 1.5,
        "aans": 1.5,
        "journal of neurosurgery": 1.3,
        "neurosurgery": 1.3,
    }
    
    # Specialty detection keywords
    SPECIALTY_KEYWORDS = {
        "vascular": [
            "aneurysm", "avm", "bypass", "carotid", "stroke", "hemorrhage",
            "vasospasm", "moyamoya", "dural fistula", "subarachnoid"
        ],
        "tumor": [
            "glioma", "meningioma", "resection", "tumor", "oncology",
            "glioblastoma", "schwannoma", "metastasis", "craniopharyngioma"
        ],
        "spine": [
            "cervical", "lumbar", "fusion", "disc", "laminectomy",
            "spondylosis", "myelopathy", "stenosis", "scoliosis"
        ],
        "functional": [
            "dbs", "epilepsy", "parkinson", "stimulation", "tremor",
            "movement disorder", "dystonia", "ablation"
        ],
        "skull_base": [
            "skull base", "pituitary", "acoustic", "petroclival",
            "transsphenoidal", "endoscopic endonasal", "chordoma"
        ],
        "pediatric": [
            "pediatric", "child", "congenital", "shunt", "craniosynostosis",
            "chiari", "myelomeningocele", "hydrocephalus"
        ],
        "trauma": [
            "trauma", "tbi", "subdural", "epidural", "contusion",
            "decompressive", "icp", "herniation"
        ],
    }
    
    def __init__(
        self,
        config: PipelineConfig,
        text_embedder: Optional[TextEmbedder] = None,
        image_embedder: Optional[ImageEmbedder] = None,
        database: Optional[NeuroDatabase] = None,
        enable_triage: bool = True,      # NEW - Phase 1 Enhancement
        enable_metrics: bool = False,    # NEW - Phase 1 Enhancement
    ):
        """
        Initialize the pipeline.
        
        Args:
            config: Pipeline configuration
            text_embedder: Text embedding provider
            image_embedder: Image embedding provider
            database: Database connection
        """
        if not HAS_FITZ:
            raise ImportError("PyMuPDF (fitz) required: pip install PyMuPDF")

        self.config = config
        self.database = database

        # Auto-create embedders if not provided and embeddings are enabled
        if config.enable_embeddings:
            if text_embedder is None:
                try:
                    self.text_embedder = create_text_embedder(
                        provider=config.text_embedding_provider
                    )
                    logger.info(
                        f"Auto-created text embedder: {config.text_embedding_provider} "
                        f"({self.text_embedder.dimension}d)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to auto-create text embedder: {e}")
                    self.text_embedder = None
            else:
                self.text_embedder = text_embedder

            if image_embedder is None:
                try:
                    # Use subprocess mode for biomedclip to avoid memory issues
                    use_subprocess = config.image_embedding_provider == "biomedclip"
                    self.image_embedder = create_image_embedder(
                        provider=config.image_embedding_provider,
                        use_subprocess=use_subprocess
                    )
                    mode_str = " (subprocess)" if use_subprocess else ""
                    logger.info(
                        f"Auto-created image embedder: {config.image_embedding_provider}{mode_str} "
                        f"({self.image_embedder.dimension}d)"
                    )
                except Exception as e:
                    logger.warning(f"Image embeddings disabled: {e}")
                    self.image_embedder = None
            else:
                self.image_embedder = image_embedder
        else:
            self.text_embedder = text_embedder
            self.image_embedder = image_embedder

        # Log device usage for embedders
        if self.text_embedder and hasattr(self.text_embedder, '_device'):
            logger.info(f"Text embedder using device: {self.text_embedder._device}")

        if self.image_embedder and hasattr(self.image_embedder, '_device'):
            logger.info(f"Image embedder using device: {self.image_embedder._device}")

        # Initialize components
        self.section_detector = OutlineBasedSectionDetector()
        self.chunker = TableAwareChunker()
        self.linker = TriPassLinker()
        self.fuser = EmbeddingFuser()
        
        # Per-document components (initialized during processing)
        self._image_extractor: Optional[NeuroImageExtractor] = None
        self._table_extractor: Optional[TableExtractor] = None

        # Contextual preprocessor for improved retrieval
        self._contextual_preprocessor: Optional["ContextualPreprocessor"] = None
        if config.enable_contextual and HAS_CONTEXTUAL:
            mode_map = {
                "template": ContextMode.TEMPLATE,
                "llm_light": ContextMode.LLM_LIGHT,
                "llm_full": ContextMode.LLM_FULL
            }
            ctx_config = ContextConfig(
                mode=mode_map.get(config.contextual_mode, ContextMode.LLM_FULL),
                llm_model=config.contextual_model,
                llm_api_key=config.contextual_api_key,
                include_entities=True,
                include_chunk_type=True
            )
            self._contextual_preprocessor = ContextualPreprocessor(ctx_config)
            logger.info(
                f"Contextual preprocessing enabled: mode={config.contextual_mode}, "
                f"model={config.contextual_model}"
            )
        elif config.enable_contextual and not HAS_CONTEXTUAL:
            logger.warning(
                "Contextual preprocessing requested but contextual_preprocessor.py not found. "
                "Embeddings will be generated without context enrichment."
            )

        # OCR availability check
        if config.enable_ocr and not HAS_OCR:
            logger.warning(
                "OCR requested but pytesseract not available. "
                "Install with: pip install pytesseract pillow"
            )

        # Knowledge Graph for GraphRAG queries
        self._knowledge_graph: Optional["NeurosurgicalKnowledgeGraph"] = None
        self._entity_extractor: Optional["NeuroExtractor"] = None
        if config.enable_knowledge_graph and HAS_KNOWLEDGE_GRAPH:
            self._knowledge_graph = NeurosurgicalKnowledgeGraph()
            self._entity_extractor = NeuroExtractor()
            self._kg_path = config.knowledge_graph_path or (config.output_dir / "knowledge_graph.json")

            # Load existing graph if present
            if self._kg_path.exists():
                try:
                    self._knowledge_graph.load(self._kg_path)
                    logger.info(f"Loaded existing knowledge graph: {self._knowledge_graph.stats()}")
                except Exception as e:
                    logger.warning(f"Failed to load existing knowledge graph: {e}")

            logger.info("Knowledge graph enabled for GraphRAG traversal")
        elif config.enable_knowledge_graph and not HAS_KNOWLEDGE_GRAPH:
            logger.warning(
                "Knowledge graph requested but knowledge_graph.py not found. "
                "Entity relationships will not be captured for GraphRAG."
            )

        # VLM Image Captioning with Optional Triage Wrapper (Phase 1 Enhancement)
        self._vlm_captioner: Optional["VLMImageCaptioner"] = None
        self._triage: Optional["VisualTriage"] = None

        if config.enable_vlm_captions and HAS_VLM:
            # Initialize base VLM captioner
            base_vlm = VLMImageCaptioner(
                model=config.vlm_model,
                api_key=config.vlm_api_key,
                batch_size=config.vlm_batch_size
            )

            # Wrap with triage if enabled
            if enable_triage and HAS_TRIAGE:
                self._triage = VisualTriage()
                self._vlm_captioner = TriageAwareVLMCaptioner(
                    vlm_captioner=base_vlm,
                    triage=self._triage
                )
                logger.info(f"VLM captioning with triage enabled: model={config.vlm_model}")
                logger.info("Visual triage active - expect 60-70% VLM cost savings")
            else:
                self._vlm_captioner = base_vlm
                logger.info(f"VLM captioning enabled (no triage): model={config.vlm_model}")

        elif config.enable_vlm_captions and not HAS_VLM:
            logger.warning(
                "VLM captioning requested but vlm_captioner.py not found or anthropic not installed. "
                "Images will not have semantic captions."
            )

        # Metrics server (optional)
        if enable_metrics and HAS_MONITORING:
            try:
                from src.monitoring import start_metrics_server
                start_metrics_server(port=9090)
                logger.info("Metrics server started on http://localhost:9090/metrics")
            except Exception as e:
                logger.warning(f"Failed to start metrics server: {e}")

        # UMLS Extractor (SciSpacy - complementary to regex entities)
        # Uses cached singleton to save ~1.2GB RAM for concurrent documents
        self._umls_extractor: Optional["UMLSExtractor"] = None
        if config.enable_umls and HAS_UMLS:
            try:
                self._umls_extractor = get_default_extractor(
                    model=config.umls_model,
                    threshold=config.umls_threshold
                )
                logger.info(
                    f"UMLS extraction enabled (cached): model={config.umls_model}, "
                    f"threshold={config.umls_threshold}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize UMLS extractor: {e}")
        elif config.enable_umls and not HAS_UMLS:
            logger.warning(
                "UMLS extraction requested but scispacy not installed. "
                "Install with: pip install scispacy && pip install en_core_sci_lg model"
            )

        # Relation Extraction Pipeline (spaCy NLP - 16 relation types)
        self._relation_pipeline: Optional["RelationExtractionPipeline"] = None
        if config.enable_relation_extraction and HAS_RELATION_EXTRACTION:
            try:
                from src.core.relation_extractor import NeuroRelationExtractor
                extractor = NeuroRelationExtractor(model=config.relation_extraction_model)
                self._relation_pipeline = RelationExtractionPipeline(
                    db_pool=database.pool if database else None,
                    extractor=extractor,
                    batch_size=50,
                    min_confidence=config.relation_min_confidence,
                )
                logger.info(
                    f"Relation extraction enabled: model={config.relation_extraction_model}, "
                    f"min_confidence={config.relation_min_confidence}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize relation extractor: {e}")
        elif config.enable_relation_extraction and not HAS_RELATION_EXTRACTION:
            logger.warning(
                "Relation extraction requested but relation_pipeline.py not found. "
                "Entity relationships will not be extracted."
            )

    async def process_document(
        self,
        pdf_path: Path,
        on_progress: Optional[Callable[[ProgressInfo], None]] = None
    ) -> PipelineResult:
        """
        Process a single PDF through the complete pipeline.
        
        Args:
            pdf_path: Path to PDF file
            on_progress: Progress callback
            
        Returns:
            PipelineResult with statistics
        """
        start_time = datetime.now()
        pdf_path = Path(pdf_path)
        tracker = ProgressTracker(on_progress)
        ocr_pages = 0
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        tracker.update(ProcessingStage.INIT, 0.0, "Computing content hash")
        
        # Compute content hash
        content_hash = self._compute_hash(pdf_path)
        
        # Check for existing document
        if self.database:
            existing = await self.database.get_document_by_hash(content_hash)
            if existing and existing.status == DocumentStatus.READY:
                tracker.update(ProcessingStage.COMPLETE, 1.0, "Document already indexed")
                duration = (datetime.now() - start_time).total_seconds()
                return PipelineResult(
                    document=existing,
                    page_count=existing.page_count,
                    chunk_count=0,
                    image_count=0,
                    table_count=0,
                    duration_seconds=duration
                )
        
        tracker.update(ProcessingStage.INIT, 0.5, "Opening PDF")
        
        # Open PDF
        doc = fitz.open(pdf_path)
        document = None
        
        try:
            # Create document record with title tracking
            title, title_info = self._extract_title(doc, pdf_path)
            ingestion_warnings = title_info.get("warnings", [])

            document = Document.create(
                file_path=pdf_path,
                content_hash=content_hash,
                title=title
            )
            # Store title extraction metadata
            document.extraction_metrics = {
                "title_source": title_info.get("source"),
                "metadata_title": title_info.get("metadata_title"),
            }
            document.page_count = len(doc)
            document.status = DocumentStatus.EXTRACTING
            
            tracker.update(
                ProcessingStage.STRUCTURE, 0.0,
                "Detecting document structure",
                {"pages": len(doc)}
            )
            
            # Stage 1: Detect sections
            sections = self.section_detector.detect_sections(doc)
            
            tracker.update(
                ProcessingStage.STRUCTURE, 1.0,
                f"Found {len(sections)} sections"
            )
            
            # Stage 2: Extract pages, images, tables (with OCR)
            pages, images, tables, ocr_pages = await self._extract_content_with_ocr(
                doc, document.id, tracker
            )
            
            # Update document metadata
            full_text = "\n".join(p.content for p in pages)
            document.specialty = self._detect_specialty(full_text)
            document.authority_score = self._compute_authority(pdf_path, document.title)

            # Validate title against content (detect mislabeling)
            content_warnings = self._validate_title_against_content(document.title, full_text)
            ingestion_warnings.extend(content_warnings)

            # Log any content validation warnings
            for warning in content_warnings:
                severity = warning.get("severity", "warning")
                msg = warning.get("message", "Unknown validation issue")
                if severity == "critical":
                    logger.error(f"MISLABELING DETECTED: {msg}")
                else:
                    logger.warning(f"Content validation: {msg}")

            # Store warnings in extraction metrics
            if ingestion_warnings:
                document.extraction_metrics["ingestion_warnings"] = ingestion_warnings

            tracker.update(ProcessingStage.CHUNKING, 0.0, "Semantic chunking")

            # Stage 3: Semantic chunking
            document.status = DocumentStatus.CHUNKING
            chunks = self._chunk_sections(sections, tables, document.id, tracker)

            # Stage 4.5: Generate brief summaries for chunks
            await self._summarize_chunks(chunks, tracker)

            # Stage 3.5: UMLS extraction (complements regex entities)
            if self._umls_extractor:
                await self._extract_umls_cuis(chunks, tracker)

            # Stage 4: Generate embeddings BEFORE linking
            # CRITICAL FIX: TriPassLinker Pass 3 requires caption_embedding to exist
            # Moving this BEFORE linking ensures semantic similarity works correctly
            if self.config.enable_embeddings:
                document.status = DocumentStatus.EMBEDDING
                await self._generate_embeddings(chunks, images, tracker, document)

            tracker.update(ProcessingStage.LINKING, 0.0, "Linking images to chunks")

            # Stage 5: Link images to chunks (TriPassLinker returns chunks, images, and links)
            # Now that embeddings exist, Pass 3 (semantic similarity) will work correctly
            chunks, images, links = self.linker.link(chunks, images)

            tracker.update(ProcessingStage.LINKING, 1.0, f"Linked {len(images)} images to {len(links)} link relationships")

            # Stage 5.5: Build Knowledge Graph (GraphRAG)
            if self._knowledge_graph and self._entity_extractor:
                await self._build_knowledge_graph(chunks, document, tracker)

            # Stage 5.6: Extract Relations (spaCy NLP)
            if self._relation_pipeline:
                await self._extract_relations(chunks, tracker)

            # Stage 6: Fuse embeddings (uses link scores from TriPassLinker)
            # Note: Embedding generation moved BEFORE linking (Stage 4)
            # This step only does fusion with link weights now
            if self.config.enable_embeddings:
                chunks = self.fuser.fuse_embeddings(chunks, images)

            # Stage 8.5: Generate brief summaries for image captions
            await self._summarize_captions(images, tracker)

            tracker.update(ProcessingStage.STORAGE, 0.0, "Storing to database")
            
            # Stage 6: Store atomically
            if self.database:
                async with self.database.transaction() as conn:
                    await self.database.insert_document(document, conn)
                    await self.database.insert_pages(pages, conn)
                    await self.database.insert_chunks(chunks, conn)
                    await self.database.insert_images(images, conn)
                    if tables:
                        await self.database.insert_tables(tables, conn)
                    if links:
                        await self.database.insert_links(links, conn)
            
            # Mark complete
            document.status = DocumentStatus.READY
            document.indexed_at = datetime.now()
            
            if self.database:
                await self.database.update_document_status(
                    document.id,
                    DocumentStatus.READY
                )
            
            duration = (datetime.now() - start_time).total_seconds()

            # Collect metrics for monitoring
            # Build extraction metrics including any warnings
            pipeline_extraction_metrics = {
                "sections": len(sections),
                "ocr_pages": ocr_pages,
                "specialty": document.specialty,
                "authority_score": document.authority_score,
                "title_source": document.extraction_metrics.get("title_source") if document.extraction_metrics else None,
            }
            # Include ingestion warnings if any were collected
            if ingestion_warnings:
                pipeline_extraction_metrics["ingestion_warnings"] = ingestion_warnings

            result = PipelineResult(
                document=document,
                page_count=len(pages),
                chunk_count=len(chunks),
                image_count=len(images),
                table_count=len(tables),
                duration_seconds=duration,
                ocr_pages=ocr_pages,
                chunks=chunks,
                images=images,
                tables=tables,
                links=links if 'links' in dir() else [],
                extraction_metrics=pipeline_extraction_metrics
            )

            metrics = get_metrics_collector()
            metrics.collect_from_pipeline(result)
            if self.text_embedder:
                metrics.collect_from_embedder(self.text_embedder,
                    getattr(self.text_embedder, 'provider', 'unknown'))

            tracker.update(
                ProcessingStage.COMPLETE, 1.0,
                f"Complete: {len(chunks)} chunks, {len(images)} images"
            )

            # Log triage savings (Phase 1 Enhancement)
            if hasattr(self._vlm_captioner, 'get_savings'):
                try:
                    savings = self._vlm_captioner.get_savings()
                    logger.info(
                        f"📊 VLM Triage Savings: {savings['savings_pct']:.0f}% "
                        f"({savings['vlm_calls_avoided']}/{savings['total_images']} calls avoided)"
                    )

                    # Optional: Record metrics
                    if HAS_MONITORING:
                        try:
                            from src.monitoring import record_document_processed
                            record_document_processed(
                                document_id=result.document.id,
                                images_total=savings['total_images'],
                                images_skipped=savings['vlm_calls_avoided']
                            )
                        except Exception as e:
                            logger.debug(f"Metrics recording failed: {e}")
                except Exception as e:
                    logger.debug(f"Failed to log triage savings: {e}")

            return result

        except Exception as e:
            logger.error(f"Pipeline error for {pdf_path}: {e}", exc_info=True)
            
            if document and self.database:
                await self.database.update_document_status(
                    document.id,
                    DocumentStatus.ERROR,
                    str(e)
                )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return PipelineResult(
                document=document or Document.create(
                    file_path=pdf_path,
                    content_hash=content_hash,
                    title=pdf_path.stem
                ),
                page_count=0,
                chunk_count=0,
                image_count=0,
                table_count=0,
                duration_seconds=duration,
                error=str(e)
            )
        finally:
            doc.close()
    
    async def _extract_content_with_ocr(
        self,
        doc: "fitz.Document",
        document_id: str,
        tracker: ProgressTracker
    ) -> Tuple[List[Page], List[ExtractedImage], List[ExtractedTable], int]:
        """
        Extract pages, images, and tables with OCR fallback.
        
        Returns:
            Tuple of (pages, images, tables, ocr_page_count)
        """
        # Initialize extractors
        images_dir = self.config.output_dir / "images" / document_id
        self._image_extractor = NeuroImageExtractor(output_dir=images_dir)
        self._table_extractor = TableExtractor()
        
        pages = []
        images = []
        tables = []
        seen_hashes: Set[str] = set()
        ocr_pages = 0
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # Progress update
            progress = page_num / total_pages
            tracker.update(
                ProcessingStage.PAGES,
                progress,
                f"Extracting page {page_num + 1}/{total_pages}"
            )
            
            # Extract text (with OCR fallback)
            text = page.get_text("text")
            used_ocr = False
            
            # OCR fallback: If page has < threshold chars, likely a scan
            if (
                self.config.enable_ocr
                and HAS_OCR
                and len(text.strip()) < self.config.ocr_min_text_threshold
            ):
                ocr_text = self._perform_ocr(page)
                if ocr_text and len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text
                    used_ocr = True
                    ocr_pages += 1
                    logger.info(f"Page {page_num}: Used OCR ({len(text)} chars)")
            
            # Extract tables
            page_tables = []
            if self.config.enable_tables:
                page_tables = self._table_extractor.extract_tables(
                    page, page_num, document_id
                )
                tables.extend(page_tables)
            
            # Create page record
            pages.append(Page(
                document_id=document_id,
                page_number=page_num,
                content=text,
                has_tables=len(page_tables) > 0,
                word_count=len(text.split())
            ))
            
            # Extract images
            if self.config.enable_images:
                tracker.update(
                    ProcessingStage.IMAGES,
                    progress,
                    f"Extracting images from page {page_num + 1}"
                )
                
                page_images = self._image_extractor.extract_from_page(
                    doc=doc,
                    page=page,
                    page_num=page_num,
                    document_id=document_id,
                    seen_hashes=seen_hashes
                )
                
                if page_images:
                    pages[-1].has_images = True
                    images.extend(page_images)
        
        tracker.update(
            ProcessingStage.TABLES,
            1.0,
            f"Extracted {len(tables)} tables"
        )
        
        return pages, images, tables, ocr_pages
    
    def _perform_ocr(self, page: "fitz.Page") -> Optional[str]:
        """
        Perform OCR on a page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            OCR text or None if failed
        """
        if not HAS_OCR:
            return None
        
        try:
            # Render page to image at configured DPI
            pix = page.get_pixmap(dpi=self.config.ocr_dpi)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Perform OCR with configured language
            ocr_text = pytesseract.image_to_string(
                image,
                lang=self.config.ocr_language
            )
            
            return ocr_text
            
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return None
    
    def _chunk_sections(
        self,
        sections: List[Section],
        tables: List[ExtractedTable],
        document_id: str,
        tracker: ProgressTracker
    ) -> List[SemanticChunk]:
        """Chunk all sections using medical-aware chunker."""
        all_chunks = []
        
        # Build table lookup by page
        tables_by_page = {}
        for table in tables:
            if table.page_number not in tables_by_page:
                tables_by_page[table.page_number] = []
            tables_by_page[table.page_number].append(table)
        
        total_sections = len(sections)
        
        for idx, section in enumerate(sections):
            progress = idx / max(total_sections, 1)
            tracker.update(
                ProcessingStage.CHUNKING,
                progress,
                f"Chunking section: {section.title[:50]}..."
            )
            
            # Get tables for this section's pages
            section_tables = []
            for page_num in range(section.page_start, section.page_end + 1):
                section_tables.extend(tables_by_page.get(page_num, []))
            
            # Chunk with table awareness
            section_chunks = self.chunker.chunk_section(
                section_text=section.content,
                section_title=section.title,
                page_num=section.page_start,
                doc_id=document_id,
                tables=section_tables
            )
            
            # Update page_end for multi-page sections
            for chunk in section_chunks:
                chunk.page_end = section.page_end
            
            all_chunks.extend(section_chunks)
        
        tracker.update(
            ProcessingStage.CHUNKING,
            1.0,
            f"Created {len(all_chunks)} chunks"
        )

        # Apply quality scoring to all chunks
        quality_scorer = get_quality_scorer()
        for chunk in all_chunks:
            quality_scorer.score_chunk(chunk)

        return all_chunks

    async def _extract_umls_cuis(
        self,
        chunks: List[SemanticChunk],
        tracker: ProgressTracker
    ) -> None:
        """
        Extract UMLS CUIs from chunks using SciSpacy.

        Complements the regex-based entity extraction with standardized
        UMLS Concept Unique Identifiers for improved semantic matching.

        Args:
            chunks: List of semantic chunks to process
            tracker: Progress tracker for updates
        """
        if not self._umls_extractor:
            return

        tracker.update(
            ProcessingStage.UMLS_EXTRACTION,
            0.0,
            f"Extracting UMLS concepts from {len(chunks)} chunks"
        )

        total_chunks = len(chunks)
        total_cuis = 0
        chunks_with_cuis = 0

        # Batch extract for efficiency
        try:
            texts = [chunk.content for chunk in chunks]
            # Use multiple CPU cores for parallel UMLS extraction (4-6x speedup)
            n_cores = max(1, (os.cpu_count() or 1) - 1)
            entities_batch = self._umls_extractor.extract_batch(
                texts,
                batch_size=32,
                n_process=n_cores
            )

            for idx, (chunk, entities) in enumerate(zip(chunks, entities_batch)):
                if entities:
                    chunk.cuis = [e.cui for e in entities]
                    # Store full entity objects if model supports it
                    if hasattr(chunk, 'umls_entities'):
                        chunk.umls_entities = entities
                    total_cuis += len(entities)
                    chunks_with_cuis += 1

                # Progress update every 20 chunks
                if (idx + 1) % 20 == 0 or idx == total_chunks - 1:
                    progress = (idx + 1) / total_chunks
                    tracker.update(
                        ProcessingStage.UMLS_EXTRACTION,
                        progress,
                        f"UMLS: {total_cuis} concepts from {chunks_with_cuis}/{idx+1} chunks"
                    )

            logger.info(
                f"UMLS extraction complete: {total_cuis} concepts from "
                f"{chunks_with_cuis}/{total_chunks} chunks"
            )

        except Exception as e:
            logger.error(f"UMLS extraction failed: {e}")
            tracker.update(
                ProcessingStage.UMLS_EXTRACTION,
                1.0,
                f"UMLS extraction failed: {e}"
            )

    async def _build_knowledge_graph(
        self,
        chunks: List[SemanticChunk],
        document: Document,
        tracker: ProgressTracker
    ):
        """
        Build knowledge graph from chunk entities and relationships.

        Extracts entities and relationships from each chunk and adds them
        to the persistent knowledge graph for GraphRAG queries.

        Expected improvement: +25% entity query recall
        """
        if not self._knowledge_graph or not self._entity_extractor:
            return

        total_chunks = len(chunks)
        total_entities = 0
        total_relations = 0

        # Collect entities for database persistence
        entities_to_persist = set()  # Use set to dedupe by normalized name

        for idx, chunk in enumerate(chunks):
            # Extract entities if not already present
            if not chunk.entities:
                chunk.entities = self._entity_extractor.extract(chunk.content)
                chunk.entity_names = [e.normalized for e in chunk.entities]

            # Add regex entities to graph
            for entity in chunk.entities:
                self._knowledge_graph.add_entity(entity, chunk.id, document.id)
                total_entities += 1
                # Collect for DB persistence
                entities_to_persist.add((entity.normalized, entity.category))

            # Phase 4.1: Add UMLS entities to knowledge graph
            # Creates additional nodes for UMLS-linked concepts with CUIs
            umls_entities = getattr(chunk, 'umls_entities', None)
            if umls_entities:
                for umls_ent in umls_entities:
                    # Convert UMLSEntity to NeuroEntity for graph compatibility
                    graph_entity = NeuroEntity(
                        text=umls_ent.name,
                        category=f"UMLS_{umls_ent.semantic_type.upper().replace(' ', '_')}",
                        normalized=f"{umls_ent.name} [{umls_ent.cui}]",
                        start=umls_ent.start_char,
                        end=umls_ent.end_char,
                        confidence=umls_ent.score,
                        context_snippet=""
                    )
                    self._knowledge_graph.add_entity(graph_entity, chunk.id, document.id)
                    total_entities += 1

            # Extract and add relationships
            relations = self._entity_extractor.extract_relations(
                text=chunk.content,
                entities=chunk.entities,
                chunk_id=chunk.id,
                document_id=document.id
            )

            for relation in relations:
                self._knowledge_graph.add_relation(relation)
                total_relations += 1

            # Progress update every 10 chunks
            if (idx + 1) % 10 == 0 or idx == total_chunks - 1:
                progress = (idx + 1) / total_chunks
                tracker.update(
                    ProcessingStage.LINKING,  # Reuse linking stage progress
                    0.5 + progress * 0.5,  # Second half of linking stage
                    f"Knowledge graph: {total_entities} entities, {total_relations} relations"
                )

        # Persist entities to PostgreSQL for Knowledge Graph API
        logger.info(f"Entity persistence check: database={self.database is not None}, entities={len(entities_to_persist)}")
        if self.database and entities_to_persist:
            logger.info(f"Persisting {len(entities_to_persist)} entities to PostgreSQL...")
            try:
                async with self.database.pool.acquire() as conn:
                    persisted_count = 0
                    # Batch insert entities, checking for existing by name
                    for name, category in entities_to_persist:
                        # Check if entity already exists by name
                        existing = await conn.fetchrow(
                            "SELECT id FROM entities WHERE name = $1 LIMIT 1", name
                        )
                        if existing:
                            # Update mention count
                            await conn.execute("""
                                UPDATE entities SET
                                    mention_count = mention_count + 1,
                                    updated_at = NOW()
                                WHERE id = $1
                            """, existing['id'])
                        else:
                            # Insert new entity
                            await conn.execute("""
                                INSERT INTO entities (name, category, source)
                                VALUES ($1, $2, 'knowledge_graph')
                            """, name, category or 'extracted')
                            persisted_count += 1
                logger.info(f"Persisted {persisted_count} new entities to database (total: {len(entities_to_persist)})")
            except Exception as e:
                logger.warning(f"Failed to persist entities to database: {e}")

        # Save knowledge graph
        try:
            self._knowledge_graph.save(self._kg_path)
            stats = self._knowledge_graph.stats()
            logger.info(
                f"Knowledge graph updated: {stats['total_nodes']} nodes, "
                f"{stats['total_edges']} edges (added {total_entities} entities, "
                f"{total_relations} relations from this document)"
            )
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")

    async def _extract_relations(
        self,
        chunks: List[SemanticChunk],
        tracker: ProgressTracker
    ) -> None:
        """
        Extract entity relations from chunks using spaCy NLP.

        Complements the regex-based knowledge graph with NLP-based relation
        extraction for 16 neurosurgical relation types.

        Args:
            chunks: List of semantic chunks to process
            tracker: Progress tracker for updates
        """
        if not self._relation_pipeline:
            return

        total_chunks = len(chunks)
        total_relations = 0

        tracker.update(
            ProcessingStage.LINKING,
            0.6,
            f"Extracting relations from {total_chunks} chunks"
        )

        for idx, chunk in enumerate(chunks):
            try:
                relations = await self._relation_pipeline.process_chunk(
                    chunk_id=chunk.id,
                    chunk_text=chunk.content,
                )
                total_relations += len(relations)

                # Progress update every 10 chunks
                if (idx + 1) % 10 == 0 or idx == total_chunks - 1:
                    progress = 0.6 + (idx + 1) / total_chunks * 0.3
                    tracker.update(
                        ProcessingStage.LINKING,
                        progress,
                        f"Relation extraction: {total_relations} relations from {idx+1}/{total_chunks} chunks"
                    )

            except Exception as e:
                logger.warning(f"Relation extraction failed for chunk {chunk.id}: {e}")

        # Flush remaining relations
        try:
            await self._relation_pipeline.flush()
            stats = self._relation_pipeline.get_stats()
            logger.info(
                f"Relation extraction complete: {stats['relations']} relations, "
                f"{stats['entities']} entities from {stats['chunks']} chunks"
            )
        except Exception as e:
            logger.error(f"Failed to flush relations: {e}")

        tracker.update(
            ProcessingStage.LINKING,
            0.95,
            f"Relation extraction complete: {total_relations} relations"
        )

    async def _generate_embeddings(
        self,
        chunks: List[SemanticChunk],
        images: List[ExtractedImage],
        tracker: ProgressTracker,
        document: Optional[Document] = None
    ):
        """Generate text and image embeddings with progress tracking.

        If contextual preprocessing is enabled, chunks are enriched with
        document/section context before embedding (49% retrieval improvement).
        """

        # Text embeddings
        if self.text_embedder and chunks:
            tracker.update(
                ProcessingStage.TEXT_EMBEDDING,
                0.0,
                f"Embedding {len(chunks)} chunks"
            )

            # Use contextual preprocessing if available
            if self._contextual_preprocessor and document:
                tracker.update(
                    ProcessingStage.TEXT_EMBEDDING,
                    0.0,
                    f"Adding context to {len(chunks)} chunks (mode: {self.config.contextual_mode})"
                )

                def on_context_progress(processed, total):
                    progress = processed / total * 0.3  # Context is 30% of embedding work
                    tracker.update(
                        ProcessingStage.TEXT_EMBEDDING,
                        progress,
                        f"Context enrichment {processed}/{total}"
                    )

                # Get contextually enriched texts
                texts = await self._contextual_preprocessor.process_chunks(
                    chunks, document, on_progress=on_context_progress
                )

                # Log preprocessor stats
                stats = self._contextual_preprocessor.get_stats()
                logger.info(
                    f"Contextual preprocessing complete: "
                    f"{stats['total_processed']} chunks, "
                    f"{stats['llm_calls']} LLM calls, "
                    f"{stats['template_fallbacks']} fallbacks"
                )
            else:
                # No context - use raw content
                texts = [c.content for c in chunks]

            def on_batch_progress(batch_num, total_batches):
                # Embedding is 70% of work (after 30% context)
                base_progress = 0.3 if self._contextual_preprocessor else 0.0
                progress = base_progress + (batch_num / total_batches) * (1.0 - base_progress)
                tracker.update(
                    ProcessingStage.TEXT_EMBEDDING,
                    progress,
                    f"Text batch {batch_num}/{total_batches}"
                )

            embeddings = await self.text_embedder.embed_batch(
                texts,
                on_progress=on_batch_progress
            )

            for chunk, emb in zip(chunks, embeddings):
                chunk.text_embedding = emb

        # Summary embeddings (for multi-index retrieval)
        if self.text_embedder and chunks:
            chunks_with_summaries = [c for c in chunks if c.summary]
            if chunks_with_summaries:
                tracker.update(
                    ProcessingStage.TEXT_EMBEDDING,
                    0.85,  # At 85% progress
                    f"Embedding {len(chunks_with_summaries)} summaries"
                )

                summary_texts = [c.summary for c in chunks_with_summaries]
                summary_embeddings = await self.text_embedder.embed_batch(summary_texts)

                for chunk, emb in zip(chunks_with_summaries, summary_embeddings):
                    chunk.summary_embedding = emb

                logger.info(f"Summary embeddings complete: {len(chunks_with_summaries)} summaries embedded")

        # Image embeddings
        if self.image_embedder and images:
            # Filter to meaningful images
            real_images = [
                img for img in images
                if not img.is_decorative and img.file_path.exists()
            ]
            
            if real_images:
                tracker.update(
                    ProcessingStage.IMAGE_EMBEDDING,
                    0.0,
                    f"Embedding {len(real_images)} images"
                )
                
                paths = [img.file_path for img in real_images]
                
                def on_image_progress(batch_num, total_batches):
                    progress = batch_num / total_batches
                    tracker.update(
                        ProcessingStage.IMAGE_EMBEDDING,
                        progress,
                        f"Image batch {batch_num}/{total_batches}"
                    )
                
                img_embeddings = await self.image_embedder.embed_batch(
                    paths,
                    on_progress=on_image_progress
                )
                
                for img, emb in zip(real_images, img_embeddings):
                    img.embedding = emb
        
        tracker.update(
            ProcessingStage.IMAGE_EMBEDDING,
            1.0,
            "Embeddings complete"
        )

        # VLM captioning with triage awareness (Phase 1 Enhancement)
        if self._vlm_captioner and images:
            # Reset triage for new document (clears duplicate cache)
            if self._triage:
                self._triage.reset()

            real_images = [
                img for img in images
                if not img.is_decorative and img.file_path and img.file_path.exists()
            ]

            if real_images:
                tracker.update(
                    ProcessingStage.VLM_CAPTION,
                    0.0,
                    f"Captioning {len(real_images)} images with Claude Vision"
                )

                # Convert to ImageInput objects
                image_inputs = []
                for img in real_images:
                    img_input = ImageInput(
                        id=img.content_hash or str(img.file_path.stem),
                        file_path=img.file_path,
                        width=img.width,
                        height=img.height,
                        image_type=ImageType.UNKNOWN,
                        surrounding_text=img.surrounding_text or "",
                        quality_score=img.quality_score
                    )
                    image_inputs.append(img_input)

                def on_vlm_progress(processed, total):
                    tracker.update(
                        ProcessingStage.VLM_CAPTION,
                        processed / total,
                        f"Captioning image {processed}/{total}"
                    )

                caption_results = await self._vlm_captioner.caption_batch(
                    image_inputs,
                    on_progress=on_vlm_progress
                )

                # Store captions (handle both tuple and direct responses - Phase 1 Enhancement)
                for img, result in zip(real_images, caption_results):
                    # Handle TriageAwareVLMCaptioner response (tuple: result, triage_info)
                    if isinstance(result, tuple):
                        actual_result, triage_info = result
                        if triage_info.get('skipped'):
                            logger.debug(f"Triage skipped {img.id}: {triage_info['reason']}")
                        result = actual_result

                    # Store caption if successful
                    if result.success:
                        img.vlm_caption = result.caption
                        img.vlm_image_type = result.image_type.value
                        logger.debug(f"VLM caption for {img.content_hash}: {result.caption[:100]}...")

                        # Extract figure ID from VLM caption if not already set
                        if not img.figure_id and result.caption:
                            fig_id = self._extract_figure_id_from_caption(result.caption)
                            if fig_id:
                                img.figure_id = fig_id
                                logger.debug(f"Extracted figure_id '{fig_id}' from VLM caption")

                vlm_stats = self._vlm_captioner.get_stats()
                logger.info(
                    f"VLM captioning complete: {vlm_stats['successful']}/{vlm_stats['images_processed']} "
                    f"successful, {vlm_stats['total_tokens']} tokens used"
                )

                # Auto-export captions to JSON for viewing
                # Get images_dir from first image's file path
                if real_images and real_images[0].file_path:
                    images_dir = real_images[0].file_path.parent
                    self._export_vlm_captions(real_images, images_dir)

        tracker.update(
            ProcessingStage.VLM_CAPTION,
            1.0,
            "VLM captioning complete"
        )

        # Caption embedding (embed VLM captions with text embedder for cross-modal search)
        if (
            self.config.enable_caption_embedding
            and self.text_embedder
            and images
        ):
            await self._embed_captions(images, tracker)

    async def _embed_captions(
        self,
        images: List[ExtractedImage],
        tracker: ProgressTracker
    ) -> None:
        """
        Embed VLM captions using the text embedder.

        Enables cross-modal semantic search by creating text embeddings
        of VLM-generated captions, using the same dimension as chunk text.

        Args:
            images: List of images (with vlm_caption populated)
            tracker: Progress tracker
        """
        # Filter to images with VLM captions
        images_with_captions = [
            img for img in images
            if img.vlm_caption and not img.is_decorative
        ]

        if not images_with_captions:
            tracker.update(
                ProcessingStage.CAPTION_EMBEDDING,
                1.0,
                "No captions to embed"
            )
            return

        tracker.update(
            ProcessingStage.CAPTION_EMBEDDING,
            0.0,
            f"Embedding {len(images_with_captions)} VLM captions"
        )

        # Prepare embeddable texts (use full context if available)
        texts = []
        for img in images_with_captions:
            if hasattr(img, 'embeddable_text') and img.embeddable_text:
                texts.append(img.embeddable_text)
            else:
                texts.append(img.vlm_caption)

        try:
            def on_caption_embed_progress(batch_num, total_batches):
                progress = batch_num / total_batches
                tracker.update(
                    ProcessingStage.CAPTION_EMBEDDING,
                    progress,
                    f"Caption embedding batch {batch_num}/{total_batches}"
                )

            embeddings = await self.text_embedder.embed_batch(
                texts,
                on_progress=on_caption_embed_progress
            )

            # Store embeddings on images
            for img, emb in zip(images_with_captions, embeddings):
                img.caption_embedding = emb

            # Also extract UMLS CUIs from captions if extractor available
            if self._umls_extractor:
                for img in images_with_captions:
                    if img.vlm_caption:
                        try:
                            entities = self._umls_extractor.extract(img.vlm_caption)
                            img.cuis = [e.cui for e in entities]
                            if hasattr(img, 'umls_entities'):
                                img.umls_entities = entities
                        except Exception as e:
                            logger.debug(f"UMLS extraction failed for image caption: {e}")

            logger.info(
                f"Caption embedding complete: {len(embeddings)} captions embedded "
                f"({self.text_embedder.dimension}d)"
            )

        except Exception as e:
            logger.error(f"Caption embedding failed: {e}")

        tracker.update(
            ProcessingStage.CAPTION_EMBEDDING,
            1.0,
            f"Embedded {len(images_with_captions)} captions"
        )

    def _export_vlm_captions(self, images: List[ExtractedImage], images_dir: Path) -> None:
        """Export VLM captions to JSON and text files for viewing."""
        captions_data = []
        for img in images:
            captions_data.append({
                "image_file": img.file_path.name if img.file_path else None,
                "page": img.page_number,
                "size": f"{img.width}x{img.height}",
                "vlm_type": img.vlm_image_type,
                "vlm_caption": img.vlm_caption,
                "pdf_caption": img.caption,
                "figure_id": img.figure_id,
            })

        # Export JSON
        json_file = images_dir / "vlm_captions.json"
        with open(json_file, "w") as f:
            json.dump(captions_data, f, indent=2)

        # Export readable text
        txt_file = images_dir / "vlm_captions.txt"
        with open(txt_file, "w") as f:
            for i, item in enumerate(captions_data, 1):
                f.write(f"{'='*70}\n")
                f.write(f"IMAGE {i}: {item['image_file']}\n")
                f.write(f"{'='*70}\n")
                f.write(f"Page: {item['page']}\n")
                f.write(f"Size: {item['size']}\n")
                f.write(f"Type: {item['vlm_type']}\n")
                f.write(f"Figure ID: {item['figure_id']}\n")
                f.write(f"\nVLM Caption:\n{item['vlm_caption']}\n\n")

        logger.info(f"VLM captions exported to: {json_file}")

    def _extract_figure_id_from_caption(self, caption: str) -> Optional[str]:
        """
        Extract figure ID from VLM caption text.

        Looks for patterns like "Figure 6.3", "Fig. 6A", "Plate 3.2" etc.
        Returns normalized figure ID like 'fig_6.3' or 'fig_6a'.
        """
        if not caption:
            return None

        # Figure ID extraction patterns (ordered by specificity)
        patterns = [
            re.compile(r'(?:Figure|Fig\.?)\s*(\d+\.\d+[a-zA-Z]?)', re.I),  # "Figure 6.3A"
            re.compile(r'(?:Figure|Fig\.?)\s*(\d+[a-zA-Z])', re.I),         # "Figure 6A"
            re.compile(r'(?:Figure|Fig\.?)\s*(\d+)', re.I),                  # "Figure 6"
            re.compile(r'(?:Plate|Panel)\s*(\d+(?:\.\d+)?[a-zA-Z]?)', re.I), # "Plate 3.2"
        ]

        for pattern in patterns:
            match = pattern.search(caption)
            if match:
                fig_num = match.group(1).lower()
                return f"fig_{fig_num}"

        return None

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _extract_title(self, doc: "fitz.Document", path: Path) -> Tuple[str, Dict[str, Any]]:
        """
        Extract document title with source tracking.

        Returns:
            Tuple of (title_string, title_info_dict)
            title_info contains: source, metadata_title, warnings
        """
        metadata_title = doc.metadata.get("title", "").strip()
        title_info = {
            "source": None,
            "metadata_title": metadata_title or None,
            "warnings": []
        }

        # Try PDF metadata first
        if metadata_title and len(metadata_title) > 5:
            title_info["source"] = "pdf_metadata"
            logger.info(f"Using PDF metadata title: '{metadata_title[:50]}{'...' if len(metadata_title) > 50 else ''}'")
            return metadata_title[:200], title_info

        # Fall back to filename - log the reason
        if not metadata_title:
            warning_msg = f"PDF has no metadata title, using filename: {path.name}"
            warning = {"type": "no_metadata_title", "message": warning_msg}
        else:
            warning_msg = f"PDF title too short ({len(metadata_title)} chars: '{metadata_title}'), using filename: {path.name}"
            warning = {"type": "metadata_title_too_short", "message": warning_msg}

        title_info["source"] = "filename"
        title_info["warnings"].append(warning)
        logger.warning(warning_msg)

        name = path.stem
        name = name.replace("_", " ").replace("-", " ")
        return name, title_info
    
    def _detect_specialty(self, text: str) -> str:
        """Detect neurosurgical subspecialty from text."""
        text_lower = text[:50000].lower()
        
        scores = {}
        for specialty, keywords in self.SPECIALTY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[specialty] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "general"
    
    def _compute_authority(self, path: Path, title: str) -> float:
        """Compute source authority score."""
        path_str = str(path).lower()
        title_lower = title.lower()

        for key, score in self.AUTHORITY_SCORES.items():
            if key in path_str or key in title_lower:
                return score

        return 1.0

    def _validate_title_against_content(
        self,
        title: str,
        full_text: str
    ) -> List[Dict[str, Any]]:
        """
        Validate that title keywords appear in document content.
        Detect chapter number mismatches via figure references.

        Returns:
            List of warning dictionaries with type, severity, message, details
        """
        warnings = []

        # 1. Title keyword validation
        # Extract meaningful words (4+ chars) from title
        title_words = set(re.findall(r'\b[A-Za-z]{4,}\b', title.lower()))
        # Remove common/generic words
        stop_words = {'chapter', 'section', 'part', 'the', 'and', 'for', 'with', 'from'}
        title_words -= stop_words

        if title_words:
            text_lower = full_text.lower()
            matches = sum(1 for w in title_words if w in text_lower)
            match_ratio = matches / len(title_words)

            if match_ratio < 0.3:  # Less than 30% of title keywords found
                warnings.append({
                    "type": "title_content_mismatch",
                    "severity": "high",
                    "message": f"Title keywords not found in content ({match_ratio:.0%} match). "
                              f"Document may be mislabeled.",
                    "details": {
                        "title_words": list(title_words),
                        "matches": matches,
                        "match_ratio": round(match_ratio, 2)
                    }
                })

        # 2. Chapter number validation via figure references
        # Extract chapter number from title (e.g., "Chapter 5" -> 5)
        title_chapter_match = re.search(r'Chapter\s*(\d+)', title, re.IGNORECASE)
        if title_chapter_match:
            expected_chapter = int(title_chapter_match.group(1))

            # Find figure references in content (Fig 6.1, Figure 5.2, Figs. 6.3, etc.)
            fig_refs = re.findall(r'Fig(?:ure|s)?\.?\s*(\d+)\.\d+', full_text)
            if fig_refs:
                fig_chapters = [int(c) for c in fig_refs]
                # Find most common chapter in figure references
                chapter_counts = {}
                for c in fig_chapters:
                    chapter_counts[c] = chapter_counts.get(c, 0) + 1
                most_common_chapter = max(chapter_counts, key=chapter_counts.get)
                most_common_count = chapter_counts[most_common_chapter]

                if most_common_chapter != expected_chapter and most_common_count >= 3:
                    warnings.append({
                        "type": "chapter_mismatch",
                        "severity": "critical",
                        "message": f"Title indicates Chapter {expected_chapter} but content contains "
                                  f"Figure {most_common_chapter}.x references ({most_common_count} times). "
                                  f"Document appears to be Chapter {most_common_chapter}.",
                        "details": {
                            "title_chapter": expected_chapter,
                            "detected_chapter": most_common_chapter,
                            "figure_refs_count": len(fig_refs),
                            "chapter_ref_counts": chapter_counts
                        }
                    })

        return warnings

    def _generate_fallback_summary(self, content: str, title: Optional[str] = None) -> str:
        """
        Generate fallback summary when API fails.

        Strategy:
        1. Extract first meaningful sentence (ends with period)
        2. If too short, use title
        3. If no title, truncate content
        """
        if not content:
            return title or "Content unavailable"

        # Try first sentence
        sentences = content.split('.')
        if sentences and len(sentences[0].strip()) >= 20:
            first_sentence = sentences[0].strip()
            # Clean and truncate
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:97] + "..."
            return first_sentence

        # Fallback to title if available
        if title and len(title) >= 10:
            return title[:100] if len(title) > 100 else title

        # Last resort: truncate content
        clean_content = content.strip()[:97] + "..." if len(content) > 100 else content.strip()
        return clean_content

    def _generate_fallback_caption_summary(
        self,
        vlm_caption: Optional[str],
        caption: Optional[str],
        page_number: int
    ) -> str:
        """
        Generate fallback caption summary when API fails.

        Strategy:
        1. Truncate VLM caption if available
        2. Use simple caption if no VLM
        3. Generic description with page number as last resort
        """
        if vlm_caption:
            truncated = vlm_caption.strip()[:100]
            if len(vlm_caption) > 100:
                truncated = truncated[:97] + "..."
            return truncated

        if caption:
            truncated = caption.strip()[:100]
            if len(caption) > 100:
                truncated = truncated[:97] + "..."
            return truncated

        return f"Medical image from page {page_number}"

    async def _summarize_chunks(
        self,
        chunks: List[SemanticChunk],
        tracker: ProgressTracker
    ) -> None:
        """
        Stage 4.5: Generate brief human-readable summaries for chunks.

        Uses efficient batching with concurrent API calls for performance.
        Summaries identify: subject + distinguishing aspect.

        Format: "[Subject] — [distinguishing aspect]"
        Example: "Pterional approach — scalp incision technique"
        """
        logger.info(f"[SUMMARIZATION] _summarize_chunks called. enable_summaries={self.config.enable_summaries}, chunk_count={len(chunks)}")

        if not self.config.enable_summaries or not chunks:
            tracker.update(
                ProcessingStage.CHUNK_SUMMARIZATION,
                1.0,
                "Chunk summarization disabled"
            )
            return

        tracker.update(
            ProcessingStage.CHUNK_SUMMARIZATION,
            0.0,
            f"Summarizing {len(chunks)} chunks"
        )

        try:
            from src.ingest.chunk_summarizer import ContentSummarizer
            import anthropic

            # Create summarizer with efficient concurrency
            client = anthropic.AsyncAnthropic()
            summarizer = ContentSummarizer(
                client=client,
                model=self.config.summary_model,
                max_concurrent=self.config.summary_max_concurrent
            )

            total_chunks = len(chunks)
            summarized = 0

            # Process in batches for progress updates
            batch_size = 20
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i + batch_size]

                # Concurrent summarization within batch
                tasks = [
                    summarizer.summarize_chunk(chunk.content)
                    for chunk in batch
                ]

                summaries = await asyncio.gather(*tasks, return_exceptions=True)

                # Store summaries on chunks
                for chunk, summary in zip(batch, summaries):
                    if isinstance(summary, Exception):
                        logger.warning(f"Chunk summary failed: {summary}")
                        # Fallback: first sentence or title
                        fallback = self._generate_fallback_summary(chunk.content, getattr(chunk, 'title', None))
                        chunk.summary = fallback
                        summarized += 1  # Count fallbacks as success
                    else:
                        chunk.summary = summary
                        summarized += 1

                # Progress update
                progress = min((i + batch_size) / total_chunks, 1.0)
                tracker.update(
                    ProcessingStage.CHUNK_SUMMARIZATION,
                    progress,
                    f"Summarized {summarized}/{total_chunks} chunks"
                )

            logger.info(f"Chunk summarization complete: {summarized}/{total_chunks} chunks")

        except ImportError as e:
            logger.warning(f"Summarizer not available: {e}, using fallbacks")
            # Generate fallback summaries for all chunks
            for chunk in chunks:
                if not chunk.summary:
                    chunk.summary = self._generate_fallback_summary(
                        chunk.content, getattr(chunk, 'title', None)
                    )
            tracker.update(
                ProcessingStage.CHUNK_SUMMARIZATION,
                1.0,
                "Used fallback summaries (summarizer unavailable)"
            )
        except Exception as e:
            logger.error(f"Chunk summarization failed: {e}, using fallbacks")
            # Generate fallback summaries for all chunks
            for chunk in chunks:
                if not chunk.summary:
                    chunk.summary = self._generate_fallback_summary(
                        chunk.content, getattr(chunk, 'title', None)
                    )
            tracker.update(
                ProcessingStage.CHUNK_SUMMARIZATION,
                1.0,
                f"Used fallback summaries (error: {e})"
            )

    async def _summarize_captions(
        self,
        images: List[ExtractedImage],
        tracker: ProgressTracker
    ) -> None:
        """
        Stage 8.5: Generate brief summaries for VLM captions.

        Only processes images that have VLM captions, avoiding redundant work.
        Uses efficient concurrent processing.

        Format: "[Image type] — [key detail]"
        Example: "MRI T1 axial — vestibular schwannoma"
        """
        logger.info(f"[SUMMARIZATION] _summarize_captions called. enable_summaries={self.config.enable_summaries}, image_count={len(images)}")

        if not self.config.enable_summaries:
            tracker.update(
                ProcessingStage.CAPTION_SUMMARIZATION,
                1.0,
                "Caption summarization disabled"
            )
            return

        # Split images: those with VLM captions for API summarization,
        # those without get fallback summaries
        images_with_captions = []
        images_without_captions = []

        for img in images:
            if img.is_decorative:
                continue
            if img.vlm_caption:
                images_with_captions.append(img)
            else:
                images_without_captions.append(img)

        # Assign fallback summaries to images without VLM captions
        for img in images_without_captions:
            img.caption_summary = self._generate_fallback_caption_summary(
                img.vlm_caption, img.caption, img.page_number
            )

        if not images_with_captions:
            tracker.update(
                ProcessingStage.CAPTION_SUMMARIZATION,
                1.0,
                f"Generated {len(images_without_captions)} fallback summaries (no VLM captions)"
            )
            return

        tracker.update(
            ProcessingStage.CAPTION_SUMMARIZATION,
            0.0,
            f"Summarizing {len(images_with_captions)} captions"
        )

        try:
            from src.ingest.chunk_summarizer import ContentSummarizer
            import anthropic

            client = anthropic.AsyncAnthropic()
            summarizer = ContentSummarizer(
                client=client,
                model=self.config.summary_model,
                max_concurrent=self.config.summary_max_concurrent
            )

            total = len(images_with_captions)
            summarized = 0

            # Concurrent summarization
            tasks = [
                summarizer.summarize_image_caption(img.vlm_caption)
                for img in images_with_captions
            ]

            summaries = await asyncio.gather(*tasks, return_exceptions=True)

            for img, summary in zip(images_with_captions, summaries):
                if isinstance(summary, Exception):
                    logger.warning(f"Caption summary failed: {summary}")
                    # Fallback: truncated caption or generic
                    fallback = self._generate_fallback_caption_summary(
                        img.vlm_caption, img.caption, img.page_number
                    )
                    img.caption_summary = fallback
                    summarized += 1  # Count fallbacks as success
                else:
                    img.caption_summary = summary
                    summarized += 1

            total_with_fallbacks = summarized + len(images_without_captions)
            tracker.update(
                ProcessingStage.CAPTION_SUMMARIZATION,
                1.0,
                f"Summarized {total_with_fallbacks} captions ({len(images_without_captions)} fallbacks)"
            )

            logger.info(f"Caption summarization complete: {summarized}/{total} API, {len(images_without_captions)} fallbacks")

        except ImportError as e:
            logger.warning(f"Summarizer not available: {e}, using fallbacks")
            # Generate fallback summaries for all non-decorative images
            for img in images:
                if not img.is_decorative and not img.caption_summary:
                    img.caption_summary = self._generate_fallback_caption_summary(
                        img.vlm_caption, img.caption, img.page_number
                    )
            tracker.update(
                ProcessingStage.CAPTION_SUMMARIZATION,
                1.0,
                "Used fallback summaries (summarizer unavailable)"
            )
        except Exception as e:
            logger.error(f"Caption summarization failed: {e}, using fallbacks")
            # Generate fallback summaries for all non-decorative images
            for img in images:
                if not img.is_decorative and not img.caption_summary:
                    img.caption_summary = self._generate_fallback_caption_summary(
                        img.vlm_caption, img.caption, img.page_number
                    )
            tracker.update(
                ProcessingStage.CAPTION_SUMMARIZATION,
                1.0,
                f"Used fallback summaries (error: {e})"
            )

    def export_for_phase2(
        self,
        result: PipelineResult,
        output_dir: Optional[Path] = None,
        include_embeddings: bool = True
    ) -> Path:
        """
        Export processed document in Phase 2 compatible format.

        Creates a structured export with:
        - chunks.pkl: List[SemanticChunk] with embeddings and CUIs
        - images.pkl: List[ExtractedImage] with dual embeddings
        - links.json: List[LinkResult] as JSON
        - manifest.json: Processing metadata for Phase 2 consumption

        Args:
            result: PipelineResult from process_document()
            output_dir: Output directory (defaults to config.output_dir/exports/{doc_id})
            include_embeddings: Whether to include embeddings in pickle files

        Returns:
            Path to the export directory
        """
        doc_id = result.document.id if result.document else "unknown"

        if output_dir is None:
            output_dir = self.config.output_dir / "exports" / doc_id

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting Phase 2 data to: {output_dir}")

        # Prepare chunks for export
        chunks_data = result.chunks
        if not include_embeddings:
            # Strip embeddings for smaller export
            chunks_data = []
            for chunk in result.chunks:
                chunk_copy = SemanticChunk(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_path=chunk.section_path,
                    chunk_type=chunk.chunk_type,
                    entities=chunk.entities,
                    entity_names=chunk.entity_names,
                    specialty_tags=chunk.specialty_tags,
                    cuis=chunk.cuis,
                    # Omit embeddings
                )
                chunks_data.append(chunk_copy)

        # Export chunks
        chunks_file = output_dir / "chunks.pkl"
        with open(chunks_file, "wb") as f:
            pickle.dump(chunks_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Export images
        images_file = output_dir / "images.pkl"
        with open(images_file, "wb") as f:
            pickle.dump(result.images, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Export links as JSON
        links_file = output_dir / "links.json"
        links_data = [
            link.to_dict() if hasattr(link, 'to_dict') else {
                "chunk_id": link.chunk_id,
                "image_id": link.image_id,
                "strength": link.strength,
                "match_type": link.match_type,
                "details": link.details
            }
            for link in result.links
        ]
        with open(links_file, "w") as f:
            json.dump(links_data, f, indent=2)

        # Count statistics
        chunks_with_cuis = sum(1 for c in result.chunks if c.cuis)
        images_with_cuis = sum(1 for i in result.images if i.cuis)
        images_with_caption_emb = sum(
            1 for i in result.images
            if i.caption_embedding is not None
        )

        # Create manifest
        manifest = ProcessingManifest(
            document_id=doc_id,
            source_path=str(result.document.file_path) if result.document else "",
            text_embedding_dim=self.text_embedder.dimension if self.text_embedder else 0,
            image_embedding_dim=self.image_embedder.dimension if self.image_embedder else 0,
            text_embedding_provider=getattr(self.text_embedder, 'provider', 'unknown') if self.text_embedder else "none",
            chunk_count=len(result.chunks),
            image_count=len(result.images),
            link_count=len(result.links),
            chunks_with_cuis=chunks_with_cuis,
            images_with_cuis=images_with_cuis,
            files={
                "chunks": "chunks.pkl",
                "images": "images.pkl",
                "links": "links.json",
            },
            processing_time_seconds=result.duration_seconds,
        )

        # Export manifest
        manifest_file = output_dir / "manifest.json"
        manifest_dict = {
            "document_id": manifest.document_id,
            "source_path": manifest.source_path,
            "text_embedding_dim": manifest.text_embedding_dim,
            "image_embedding_dim": manifest.image_embedding_dim,
            "text_embedding_provider": manifest.text_embedding_provider,
            "chunk_count": manifest.chunk_count,
            "image_count": manifest.image_count,
            "link_count": manifest.link_count,
            "chunks_with_cuis": manifest.chunks_with_cuis,
            "images_with_cuis": manifest.images_with_cuis,
            "images_with_caption_embedding": images_with_caption_emb,
            "files": manifest.files,
            "processing_time_seconds": manifest.processing_time_seconds,
            "created_at": manifest.created_at,
        }
        with open(manifest_file, "w") as f:
            json.dump(manifest_dict, f, indent=2)

        logger.info(
            f"Phase 2 export complete: {len(result.chunks)} chunks, "
            f"{len(result.images)} images, {len(result.links)} links"
        )

        return output_dir


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def ingest_document(
    pdf_path: Path,
    output_dir: Path,
    database_dsn: Optional[str] = None,
    text_embedder: Optional[TextEmbedder] = None,
    image_embedder: Optional[ImageEmbedder] = None,
    enable_ocr: bool = True
) -> PipelineResult:
    """
    Convenience function for single document ingestion.
    """
    config = PipelineConfig(output_dir=output_dir, enable_ocr=enable_ocr)
    
    database = None
    if database_dsn:
        database = await NeuroDatabase.connect(database_dsn)
    
    try:
        pipeline = NeuroIngestPipeline(
            config=config,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            database=database
        )
        
        return await pipeline.process_document(pdf_path)
        
    finally:
        if database:
            await database.close()


async def ingest_directory(
    directory: Path,
    output_dir: Path,
    database_dsn: str,
    text_embedder: Optional[TextEmbedder] = None,
    image_embedder: Optional[ImageEmbedder] = None,
    recursive: bool = True,
    enable_ocr: bool = True
) -> List[PipelineResult]:
    """
    Convenience function for directory ingestion.
    """
    config = PipelineConfig(output_dir=output_dir, enable_ocr=enable_ocr)
    database = await NeuroDatabase.connect(database_dsn)
    
    try:
        pipeline = NeuroIngestPipeline(
            config=config,
            text_embedder=text_embedder,
            image_embedder=image_embedder,
            database=database
        )
        
        # Find all PDFs
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory.glob(pattern))
        
        results = []
        for pdf_path in pdf_files:
            result = await pipeline.process_document(pdf_path)
            results.append(result)
        
        return results
        
    finally:
        await database.close()
