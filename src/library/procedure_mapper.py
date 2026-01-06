"""
NeuroSynth - Procedure-Library Mapper
=====================================

Service for mapping library chunks to surgical procedures.

This service populates the chunk_procedure_relevance table by:
1. Loading procedure taxonomy with semantic tags
2. Analyzing each chunk for procedure relevance
3. Classifying content type (anatomy/technique/complication/evidence)
4. Detecting pearls (teaching wisdom) and pitfalls (danger warnings)
5. Inferring surgical phase from content
6. Computing confidence scores

Usage:
    from src.library.procedure_mapper import ProcedureMapper

    mapper = ProcedureMapper(db_connection)
    stats = await mapper.map_all_chunks()
    print(f"Mapped {stats['mappings_created']} chunk-procedure links")
"""

import logging
import re
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from uuid import UUID
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MapperConfig:
    """Configuration for procedure mapping."""
    min_relevance_score: float = 0.3     # Minimum score to create mapping
    min_confidence: float = 0.4          # Minimum confidence threshold
    batch_size: int = 100                # Chunks per batch
    use_embeddings: bool = True          # Use vector similarity
    embedding_weight: float = 0.6        # Weight for embedding similarity
    keyword_weight: float = 0.4          # Weight for keyword matching
    max_procedures_per_chunk: int = 5    # Limit mappings per chunk


# =============================================================================
# Content Type Patterns
# =============================================================================

CONTENT_TYPE_PATTERNS = {
    "anatomy": [
        r"\b(anatom|structur|nerve|artery|vein|vessel|brain|skull|bone|muscle|ligament)\b",
        r"\b(location|course|origin|insertion|branch|relationship|adjacent)\b",
        r"\b(anterior|posterior|superior|inferior|medial|lateral|proximal|distal)\b",
        r"\b(fissure|sulcus|gyrus|foramen|fossa|canal|sinus)\b",
    ],
    "technique": [
        r"\b(technique|method|approach|step|procedure|dissect|incision|retract)\b",
        r"\b(craniotomy|exposure|drilling|remove|clip|coagulat|cut|suture)\b",
        r"\b(positioning|fixation|navigation|endoscop|microscop)\b",
        r"\b(instrument|equipment|tool|device|forcep|scissor|drill|burr)\b",
    ],
    "complication": [
        r"\b(complication|risk|adverse|injury|damage|deficit|hemorrhage|bleeding)\b",
        r"\b(avoid|prevent|careful|caution|danger|warning|pitfall)\b",
        r"\b(failure|infection|leak|stroke|death|mortalit|morbidit)\b",
        r"\b(iatrogenic|postoperative|intraoperative|sequela)\b",
    ],
    "evidence": [
        r"\b(study|trial|research|evidence|literature|published|journal)\b",
        r"\b(outcome|result|rate|percent|statistic|significant|cohort)\b",
        r"\b(meta.?analysis|systematic|review|retrospective|prospective)\b",
        r"\b(p.?value|confidence interval|odds ratio|hazard ratio)\b",
    ],
}

# Compile patterns for efficiency
COMPILED_CONTENT_PATTERNS = {
    ctype: [re.compile(p, re.IGNORECASE) for p in patterns]
    for ctype, patterns in CONTENT_TYPE_PATTERNS.items()
}


# =============================================================================
# Pearl & Pitfall Detection
# =============================================================================

PEARL_PATTERNS = [
    r"\b(key|essential|critical|important|crucial|pearl|tip|trick)\b",
    r"\b(always|never|must|should|ensure|remember|note)\b",
    r"\b(secret|technique|maneuver|principle|rule)\b",
    r"(?i)(the\s+key\s+is|important\s+to|essential\s+for|crucial\s+step)",
    r"(?i)(teaching\s+point|clinical\s+pearl|surgical\s+tip)",
]

PITFALL_PATTERNS = [
    r"\b(pitfall|danger|hazard|risk|avoid|caution|warning|beware)\b",
    r"\b(complication|injury|damage|error|mistake|failure)\b",
    r"(?i)(can\s+lead\s+to|may\s+cause|risk\s+of|danger\s+of)",
    r"(?i)(avoid|prevent|careful\s+not\s+to|do\s+not)",
    r"(?i)(common\s+mistake|frequent\s+error|easily\s+missed)",
]

COMPILED_PEARL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PEARL_PATTERNS]
COMPILED_PITFALL_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PITFALL_PATTERNS]


# =============================================================================
# Surgical Phase Detection
# =============================================================================

PHASE_KEYWORDS = {
    "PLANNING": [
        "indication", "contraindication", "imaging", "mri", "ct", "angiograph",
        "selection", "workup", "evaluation", "assessment", "plan", "preoperative",
        "consent", "discussion", "differential", "diagnosis"
    ],
    "POSITIONING": [
        "position", "positioning", "supine", "prone", "lateral", "park bench",
        "pin", "mayfield", "horseshoe", "head holder", "fixation", "drape",
        "preparation", "sterile", "setup"
    ],
    "EXPOSURE": [
        "incision", "skin", "soft tissue", "muscle", "fascia", "periosteum",
        "craniotomy", "drill", "burr hole", "bone flap", "rongeur", "dura",
        "exposure", "approach", "opening", "hemostasis"
    ],
    "INTRADURAL": [
        "arachnoid", "dissect", "sylvian", "fissure", "cistern", "csf",
        "retract", "brain", "vessel", "clip", "aneurysm", "tumor", "resect",
        "coagulat", "bipolar", "microscop", "target"
    ],
    "CLOSURE": [
        "closure", "close", "dura", "duraplasty", "cranioplasty", "bone flap",
        "replace", "plate", "screw", "suture", "staple", "drain", "wound"
    ],
    "POSTOPERATIVE": [
        "postoperative", "postop", "recovery", "icu", "extubation", "imaging",
        "monitoring", "follow-up", "rehabilitation", "outcome"
    ],
}


# =============================================================================
# Procedure Mapper Class
# =============================================================================

@dataclass
class MappingResult:
    """Result of chunk-procedure mapping."""
    chunk_id: UUID
    procedure_id: int
    relevance_score: float
    confidence: float
    content_type: str
    surgical_phase: Optional[str]
    is_pearl: bool
    is_pitfall: bool
    is_critical: bool
    keyword_matches: List[str] = field(default_factory=list)


class ProcedureMapper:
    """
    Maps library chunks to surgical procedures.

    Uses a hybrid approach combining:
    1. Keyword matching (procedure tags vs chunk content)
    2. Semantic similarity (if embeddings available)
    """

    def __init__(self, db, config: MapperConfig = None, embedder=None):
        """
        Initialize the procedure mapper.

        Args:
            db: Database connection (asyncpg pool or DatabaseConnection)
            config: Mapping configuration
            embedder: Optional text embedder for semantic similarity
        """
        self.db = db
        self.config = config or MapperConfig()
        self.embedder = embedder

        # Cache for procedure data
        self._procedures: Dict[int, Dict[str, Any]] = {}
        self._procedure_embeddings: Dict[int, np.ndarray] = {}

    async def load_procedures(self) -> int:
        """
        Load all procedures from taxonomy table.

        Returns:
            Number of procedures loaded
        """
        query = """
            SELECT
                id, slug, name, description, specialty,
                anatomy_tags, pathology_tags, approach_tags, keyword_aliases
            FROM procedure_taxonomy
            ORDER BY id
        """

        rows = await self.db.fetch(query)

        self._procedures = {}
        for row in rows:
            proc_id = row['id']
            self._procedures[proc_id] = {
                'id': proc_id,
                'slug': row['slug'],
                'name': row['name'],
                'description': row['description'],
                'specialty': row['specialty'],
                'all_keywords': set(
                    (row['anatomy_tags'] or []) +
                    (row['pathology_tags'] or []) +
                    (row['approach_tags'] or []) +
                    (row['keyword_aliases'] or [])
                )
            }

        logger.info(f"Loaded {len(self._procedures)} procedures from taxonomy")
        return len(self._procedures)

    async def _generate_procedure_embeddings(self):
        """Generate embeddings for procedure descriptions if embedder available."""
        if not self.embedder or not self._procedures:
            return

        texts = []
        proc_ids = []

        for proc_id, proc in self._procedures.items():
            # Create rich text representation
            text = f"{proc['name']}. {proc['description'] or ''} "
            text += " ".join(proc['all_keywords'])
            texts.append(text)
            proc_ids.append(proc_id)

        try:
            embeddings = await self.embedder.embed_batch(texts)
            for proc_id, embedding in zip(proc_ids, embeddings):
                self._procedure_embeddings[proc_id] = embedding
            logger.info(f"Generated embeddings for {len(embeddings)} procedures")
        except Exception as e:
            logger.warning(f"Failed to generate procedure embeddings: {e}")

    def _compute_keyword_score(
        self,
        chunk_text: str,
        procedure: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        Compute keyword-based relevance score.

        Returns:
            (score, matched_keywords)
        """
        text_lower = chunk_text.lower()
        keywords = procedure['all_keywords']

        if not keywords:
            return 0.0, []

        matches = []
        for kw in keywords:
            if kw.lower() in text_lower:
                matches.append(kw)

        if not matches:
            return 0.0, []

        # Score based on number of matches, diminishing returns
        base_score = len(matches) / len(keywords)

        # Boost for name match
        if procedure['name'].lower() in text_lower:
            base_score = min(1.0, base_score + 0.3)

        # Boost for slug match (e.g., "pterional" in text)
        slug_words = procedure['slug'].replace('-', ' ').split()
        for word in slug_words:
            if len(word) > 3 and word in text_lower:
                base_score = min(1.0, base_score + 0.1)

        return min(1.0, base_score), matches

    def _compute_embedding_score(
        self,
        chunk_embedding: np.ndarray,
        procedure_id: int
    ) -> float:
        """Compute embedding-based similarity score."""
        if procedure_id not in self._procedure_embeddings:
            return 0.0

        proc_embedding = self._procedure_embeddings[procedure_id]

        # Cosine similarity
        dot_product = np.dot(chunk_embedding, proc_embedding)
        norm_a = np.linalg.norm(chunk_embedding)
        norm_b = np.linalg.norm(proc_embedding)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)

        # Convert to 0-1 range (cosine can be negative)
        return max(0.0, (similarity + 1) / 2)

    def _classify_content_type(self, text: str) -> str:
        """
        Classify chunk content into category.

        Returns:
            One of: anatomy, technique, complication, evidence
        """
        scores = {}

        for ctype, patterns in COMPILED_CONTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(text)
                score += len(matches)
            scores[ctype] = score

        if not scores or max(scores.values()) == 0:
            return "technique"  # Default

        return max(scores, key=scores.get)

    def _detect_pearl(self, text: str) -> bool:
        """Detect if chunk contains teaching pearl."""
        for pattern in COMPILED_PEARL_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _detect_pitfall(self, text: str) -> bool:
        """Detect if chunk contains pitfall warning."""
        for pattern in COMPILED_PITFALL_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _infer_surgical_phase(self, text: str) -> Optional[str]:
        """
        Infer surgical phase from chunk content.

        Returns:
            Phase enum value or None
        """
        text_lower = text.lower()
        scores = {}

        for phase, keywords in PHASE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[phase] = score

        if not scores:
            return None

        return max(scores, key=scores.get)

    def _is_critical_content(
        self,
        text: str,
        is_pitfall: bool,
        content_type: str
    ) -> bool:
        """Determine if content is critical must-read."""
        # Pitfalls are always critical
        if is_pitfall:
            return True

        # Complications are critical
        if content_type == "complication":
            return True

        # Check for explicit critical markers
        critical_patterns = [
            r"\b(critical|essential|mandatory|must)\b",
            r"\b(life.?threatening|fatal|devastating)\b",
            r"\b(never|always|absolute)\b",
        ]

        text_lower = text.lower()
        for pattern in critical_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    async def map_chunk_to_procedures(
        self,
        chunk: Dict[str, Any]
    ) -> List[MappingResult]:
        """
        Map a single chunk to relevant procedures.

        Args:
            chunk: Chunk dict with id, content, embedding

        Returns:
            List of MappingResult objects
        """
        if not self._procedures:
            await self.load_procedures()

        chunk_id = chunk['id']
        text = chunk.get('content', '')
        embedding = chunk.get('embedding')

        if not text:
            return []

        # Classify content characteristics (once per chunk)
        content_type = self._classify_content_type(text)
        is_pearl = self._detect_pearl(text)
        is_pitfall = self._detect_pitfall(text)
        is_critical = self._is_critical_content(text, is_pitfall, content_type)
        surgical_phase = self._infer_surgical_phase(text)

        results = []

        for proc_id, procedure in self._procedures.items():
            # Keyword matching
            kw_score, matches = self._compute_keyword_score(text, procedure)

            # Embedding similarity (if available)
            emb_score = 0.0
            if embedding is not None and self.config.use_embeddings:
                emb_score = self._compute_embedding_score(embedding, proc_id)

            # Combined score
            if self.config.use_embeddings and embedding is not None:
                relevance_score = (
                    self.config.keyword_weight * kw_score +
                    self.config.embedding_weight * emb_score
                )
            else:
                relevance_score = kw_score

            # Skip if below threshold
            if relevance_score < self.config.min_relevance_score:
                continue

            # Compute confidence based on evidence quality
            confidence = 0.5
            if matches:
                confidence += 0.1 * min(len(matches), 3)  # Max +0.3
            if emb_score > 0.6:
                confidence += 0.2
            confidence = min(1.0, confidence)

            if confidence < self.config.min_confidence:
                continue

            results.append(MappingResult(
                chunk_id=chunk_id,
                procedure_id=proc_id,
                relevance_score=relevance_score,
                confidence=confidence,
                content_type=content_type,
                surgical_phase=surgical_phase,
                is_pearl=is_pearl,
                is_pitfall=is_pitfall,
                is_critical=is_critical,
                keyword_matches=matches,
            ))

        # Sort by relevance and limit
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:self.config.max_procedures_per_chunk]

    async def save_mappings(self, mappings: List[MappingResult]) -> int:
        """
        Batch insert mappings to database.

        Returns:
            Number of mappings saved
        """
        if not mappings:
            return 0

        records = []
        for m in mappings:
            records.append((
                m.chunk_id,
                m.procedure_id,
                m.relevance_score,
                m.confidence,
                m.content_type,
                m.surgical_phase,
                m.is_pearl,
                m.is_pitfall,
                m.is_critical,
            ))

        # Use INSERT ... ON CONFLICT to handle duplicates
        query = """
            INSERT INTO chunk_procedure_relevance (
                chunk_id, procedure_id, relevance_score, confidence,
                content_type, surgical_phase, is_pearl, is_pitfall, is_critical
            ) VALUES ($1, $2, $3, $4, $5, $6::surgical_phase_enum, $7, $8, $9)
            ON CONFLICT (chunk_id, procedure_id) DO UPDATE SET
                relevance_score = EXCLUDED.relevance_score,
                confidence = EXCLUDED.confidence,
                content_type = EXCLUDED.content_type,
                surgical_phase = EXCLUDED.surgical_phase,
                is_pearl = EXCLUDED.is_pearl,
                is_pitfall = EXCLUDED.is_pitfall,
                is_critical = EXCLUDED.is_critical
        """

        async with self.db.transaction() as conn:
            await conn.executemany(query, records)

        return len(records)

    async def map_all_chunks(
        self,
        document_ids: List[UUID] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Map all chunks to procedures.

        Args:
            document_ids: Optional filter by documents
            progress_callback: Optional (processed, total) -> None

        Returns:
            Statistics dict
        """
        # Load procedures
        await self.load_procedures()

        if not self._procedures:
            logger.warning("No procedures in taxonomy - nothing to map")
            return {"error": "No procedures loaded"}

        # Generate procedure embeddings if embedder available
        if self.config.use_embeddings and self.embedder:
            await self._generate_procedure_embeddings()

        # Query chunks
        if document_ids:
            chunk_query = """
                SELECT id, content, embedding
                FROM chunks
                WHERE document_id = ANY($1::uuid[])
                AND content IS NOT NULL
                AND LENGTH(content) > 50
            """
            chunks = await self.db.fetch(chunk_query, document_ids)
        else:
            chunk_query = """
                SELECT id, content, embedding
                FROM chunks
                WHERE content IS NOT NULL
                AND LENGTH(content) > 50
            """
            chunks = await self.db.fetch(chunk_query)

        total_chunks = len(chunks)
        logger.info(f"Processing {total_chunks} chunks for procedure mapping")

        # Process in batches
        stats = {
            "chunks_processed": 0,
            "mappings_created": 0,
            "pearls_found": 0,
            "pitfalls_found": 0,
            "by_content_type": {},
            "by_phase": {},
        }

        all_mappings = []

        for i in range(0, total_chunks, self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]

            for row in batch:
                chunk = {
                    'id': row['id'],
                    'content': row['content'],
                    'embedding': row.get('embedding'),
                }

                mappings = await self.map_chunk_to_procedures(chunk)
                all_mappings.extend(mappings)

                for m in mappings:
                    if m.is_pearl:
                        stats["pearls_found"] += 1
                    if m.is_pitfall:
                        stats["pitfalls_found"] += 1

                    stats["by_content_type"][m.content_type] = \
                        stats["by_content_type"].get(m.content_type, 0) + 1

                    if m.surgical_phase:
                        stats["by_phase"][m.surgical_phase] = \
                            stats["by_phase"].get(m.surgical_phase, 0) + 1

            stats["chunks_processed"] = i + len(batch)

            if progress_callback:
                progress_callback(stats["chunks_processed"], total_chunks)

            # Save batch
            if len(all_mappings) >= 500:
                saved = await self.save_mappings(all_mappings)
                stats["mappings_created"] += saved
                all_mappings = []
                logger.info(f"Progress: {stats['chunks_processed']}/{total_chunks} chunks")

        # Save remaining
        if all_mappings:
            saved = await self.save_mappings(all_mappings)
            stats["mappings_created"] += saved

        logger.info(
            f"Mapping complete: {stats['mappings_created']} mappings "
            f"from {stats['chunks_processed']} chunks"
        )

        return stats

    async def get_procedure_content(
        self,
        procedure_slug: str,
        phase: str = None,
        content_type: str = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get organized content for a procedure.

        Args:
            procedure_slug: Procedure slug (e.g., "pterional-craniotomy")
            phase: Optional filter by surgical phase
            content_type: Optional filter by content type
            limit: Max results

        Returns:
            List of chunks with relevance metadata
        """
        params = [procedure_slug, limit]
        where_parts = ["pt.slug = $1"]
        param_idx = 2

        if phase:
            param_idx += 1
            params.insert(-1, phase)
            where_parts.append(f"cpr.surgical_phase = ${param_idx}::surgical_phase_enum")

        if content_type:
            param_idx += 1
            params.insert(-1, content_type)
            where_parts.append(f"cpr.content_type = ${param_idx}")

        query = f"""
            SELECT
                c.id,
                c.content,
                c.page_number,
                d.title as document_title,
                d.authority_score,
                cpr.relevance_score,
                cpr.confidence,
                cpr.content_type,
                cpr.surgical_phase,
                cpr.is_pearl,
                cpr.is_pitfall,
                cpr.is_critical
            FROM chunk_procedure_relevance cpr
            JOIN procedure_taxonomy pt ON pt.id = cpr.procedure_id
            JOIN chunks c ON c.id = cpr.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE {' AND '.join(where_parts)}
            ORDER BY
                cpr.is_critical DESC,
                cpr.is_pitfall DESC,
                cpr.relevance_score DESC,
                d.authority_score DESC
            LIMIT ${param_idx + 1}
        """

        rows = await self.db.fetch(query, *params)
        return [dict(row) for row in rows]


# =============================================================================
# Standalone Mapping Script
# =============================================================================

async def run_full_mapping(database_url: str = None):
    """
    Run full chunk-to-procedure mapping.

    Usage:
        python -m src.library.procedure_mapper
    """
    import os
    from src.database.connection import DatabaseConnection

    db_url = database_url or os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL not set")

    db = DatabaseConnection(db_url)
    await db.connect()

    try:
        mapper = ProcedureMapper(db)

        def progress(done, total):
            pct = (done / total * 100) if total else 0
            print(f"\rProgress: {done}/{total} ({pct:.1f}%)", end="", flush=True)

        stats = await mapper.map_all_chunks(progress_callback=progress)

        print("\n\nMapping Statistics:")
        print(f"  Chunks processed: {stats.get('chunks_processed', 0)}")
        print(f"  Mappings created: {stats.get('mappings_created', 0)}")
        print(f"  Pearls found: {stats.get('pearls_found', 0)}")
        print(f"  Pitfalls found: {stats.get('pitfalls_found', 0)}")

        print("\nBy Content Type:")
        for ctype, count in stats.get('by_content_type', {}).items():
            print(f"  {ctype}: {count}")

        print("\nBy Surgical Phase:")
        for phase, count in stats.get('by_phase', {}).items():
            print(f"  {phase}: {count}")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(run_full_mapping())
