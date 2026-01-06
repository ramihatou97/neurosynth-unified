"""
NeuroSynth 2.0 - Physics Bootstrapper
======================================

Automated bootstrapping of physics-aware anatomical entities.

Problem: Manually populating physics properties for 500+ entities is infeasible.
Solution: Use existing chunks + LLM extraction to auto-populate.

Strategy:
1. Query existing entities table
2. Find chunks mentioning each entity
3. Use LLM to extract physics properties from chunk context
4. Insert as "draft" records with lower confidence for human verification

Usage:
    bootstrapper = PhysicsBootstrapper(database, llm_client)
    
    # Bootstrap vascular entities first (highest impact)
    results = await bootstrapper.bootstrap_vascular_entities(batch_size=50)
    
    # Or bootstrap all entities
    results = await bootstrapper.bootstrap_all(batch_size=100)
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# KNOWN ENTITY DATA
# =============================================================================

# Known end arteries - high-confidence data that doesn't need LLM extraction
KNOWN_END_ARTERIES = {
    "lenticulostriate": {
        "is_end_artery": True,
        "collateral_capacity": "none",
        "territory_supplied": ["basal ganglia", "internal capsule"],
        "confidence": 0.98
    },
    "anterior choroidal": {
        "is_end_artery": True,
        "collateral_capacity": "poor",
        "territory_supplied": ["internal capsule", "optic tract", "hippocampus"],
        "confidence": 0.98
    },
    "heubner": {
        "is_end_artery": True,
        "collateral_capacity": "none",
        "territory_supplied": ["caudate head", "anterior internal capsule"],
        "confidence": 0.95
    },
    "thalamoperforator": {
        "is_end_artery": True,
        "collateral_capacity": "none",
        "territory_supplied": ["thalamus"],
        "confidence": 0.95
    },
    "pontine perforator": {
        "is_end_artery": True,
        "collateral_capacity": "none",
        "territory_supplied": ["pons"],
        "confidence": 0.95
    },
    "labyrinthine": {
        "is_end_artery": True,
        "collateral_capacity": "none",
        "territory_supplied": ["inner ear"],
        "confidence": 0.98
    },
    "central retinal": {
        "is_end_artery": True,
        "collateral_capacity": "none",
        "territory_supplied": ["retina"],
        "confidence": 0.98
    },
}

# Known tethered structures
KNOWN_TETHERED_STRUCTURES = {
    "m1 segment": "tethered_by_perforators",
    "basilar artery": "tethered_by_perforators",
    "p1 segment": "tethered_by_perforators",
    "a1 segment": "tethered_by_perforators",
    "superior cerebellar artery": "tethered_by_perforators",
}

# Known eloquent structures
KNOWN_ELOQUENT_STRUCTURES = {
    "motor cortex": {"eloquence_grade": "eloquent", "functional_role": "motor"},
    "sensory cortex": {"eloquence_grade": "eloquent", "functional_role": "sensory"},
    "broca area": {"eloquence_grade": "eloquent", "functional_role": "language_production"},
    "wernicke area": {"eloquence_grade": "eloquent", "functional_role": "language_comprehension"},
    "visual cortex": {"eloquence_grade": "eloquent", "functional_role": "vision"},
    "primary auditory cortex": {"eloquence_grade": "eloquent", "functional_role": "hearing"},
}


# =============================================================================
# LLM EXTRACTION PROMPTS
# =============================================================================

PHYSICS_EXTRACTION_SYSTEM_PROMPT = """You are a neurosurgical anatomy expert. Your task is to extract 
physical and surgical properties of anatomical structures from medical text.

For each structure, you must determine these properties:

1. mobility: How the structure responds to surgical manipulation
   - fixed: Cannot be moved (bone, some dural attachments)
   - tethered_by_nerve: Limited by neural attachments
   - tethered_by_vessel: Limited by vascular attachments
   - tethered_by_perforators: Limited by perforating arteries
   - freely_mobile: Can be safely displaced
   - semi_mobile: Limited mobility with technique
   - elastic: Returns to position (brain parenchyma)

2. consistency: Physical properties
   - solid_bone, firm_dura, soft_brain, vascular, neural, membranous, fibrous

3. is_end_artery: true/false (for vessels only)

4. collateral_capacity: none/poor/variable/moderate/rich (for vessels)

5. retraction_tolerance: none/minimal/moderate/good/excellent

6. sacrifice_safety: never/conditional/usually_safe/safe

7. eloquence_grade: non_eloquent/near_eloquent/eloquent

Be conservative - only extract properties you are confident about from the text.
Return ONLY properties that are clearly supported by the provided context."""

PHYSICS_EXTRACTION_USER_PROMPT = """Analyze these neurosurgical text chunks and extract physical properties 
for the anatomical structure: {entity_name}

Text context:
---
{chunk_text}
---

Return a JSON object with the properties you can confidently extract:
{{
    "name": "{entity_name}",
    "mobility": "...",
    "consistency": "...",
    "is_end_artery": true/false,
    "collateral_capacity": "...",
    "retraction_tolerance": "...",
    "sacrifice_safety": "...",
    "eloquence_grade": "...",
    "territory_supplied": ["..."],
    "spatial_context": {{
        "region": "...",
        "laterality": "..."
    }},
    "confidence": 0.0-1.0,
    "extraction_notes": "brief notes on what informed your extraction"
}}

Only include properties you can determine from the text. Omit properties you're uncertain about.
Set confidence based on how explicit the text evidence is (0.5 for inferred, 0.8 for explicit)."""


# =============================================================================
# BOOTSTRAPPER
# =============================================================================

@dataclass
class ExtractionResult:
    """Result of physics extraction for an entity."""
    entity_name: str
    properties: Dict[str, Any]
    confidence: float
    source_chunks: List[str]
    extraction_method: str  # 'known', 'llm', 'rule_based'
    success: bool
    error: Optional[str] = None


class PhysicsBootstrapper:
    """
    Bootstraps anatomical_entities from existing chunk data.
    
    Strategy:
    1. Use known data for well-established entities
    2. Query existing entities table for candidates
    3. Find chunks mentioning each entity
    4. LLM extraction of physics properties
    5. Insert as "draft" records for human verification
    """
    
    def __init__(
        self,
        database,
        llm_client = None,
        llm_model: str = "claude-sonnet-4-20250514"
    ):
        """
        Initialize the bootstrapper.
        
        Args:
            database: Async database connection pool
            llm_client: Anthropic client (or compatible)
            llm_model: Model to use for extraction
        """
        self.db = database
        self.llm = llm_client
        self.llm_model = llm_model
        
        # Statistics
        self.stats = {
            "processed": 0,
            "known_data": 0,
            "llm_extracted": 0,
            "failed": 0
        }
    
    async def bootstrap_entity(
        self,
        entity_name: str,
        max_chunks: int = 10
    ) -> ExtractionResult:
        """
        Extract physics properties for a single entity.
        
        Args:
            entity_name: Name of the anatomical entity
            max_chunks: Maximum chunks to use for context
            
        Returns:
            ExtractionResult with extracted properties
        """
        entity_lower = entity_name.lower()
        
        # =================================================================
        # 1. Check known data first
        # =================================================================
        
        # Check end arteries
        for known_name, known_props in KNOWN_END_ARTERIES.items():
            if known_name in entity_lower:
                self.stats["known_data"] += 1
                return ExtractionResult(
                    entity_name=entity_name,
                    properties={
                        "name": entity_name,
                        "canonical_name": entity_name,
                        "consistency": "vascular",
                        **known_props
                    },
                    confidence=known_props.get("confidence", 0.95),
                    source_chunks=[],
                    extraction_method="known",
                    success=True
                )
        
        # Check tethered structures
        for known_name, mobility in KNOWN_TETHERED_STRUCTURES.items():
            if known_name in entity_lower:
                self.stats["known_data"] += 1
                return ExtractionResult(
                    entity_name=entity_name,
                    properties={
                        "name": entity_name,
                        "canonical_name": entity_name,
                        "mobility": mobility,
                        "consistency": "vascular",
                        "confidence": 0.9
                    },
                    confidence=0.9,
                    source_chunks=[],
                    extraction_method="known",
                    success=True
                )
        
        # Check eloquent structures
        for known_name, known_props in KNOWN_ELOQUENT_STRUCTURES.items():
            if known_name in entity_lower:
                self.stats["known_data"] += 1
                return ExtractionResult(
                    entity_name=entity_name,
                    properties={
                        "name": entity_name,
                        "canonical_name": entity_name,
                        "consistency": "soft_brain",
                        "sacrifice_safety": "never",
                        **known_props,
                        "confidence": 0.95
                    },
                    confidence=0.95,
                    source_chunks=[],
                    extraction_method="known",
                    success=True
                )
        
        # =================================================================
        # 2. Query chunks for context
        # =================================================================
        
        try:
            chunks = await self.db.fetch("""
                SELECT id, content, document_id, chunk_type
                FROM chunks
                WHERE content ILIKE $1
                ORDER BY 
                    CASE WHEN chunk_type = 'ANATOMY' THEN 0
                         WHEN chunk_type = 'PROCEDURE' THEN 1
                         ELSE 2 END,
                    LENGTH(content) DESC
                LIMIT $2
            """, f"%{entity_name}%", max_chunks)
            
            if not chunks:
                return ExtractionResult(
                    entity_name=entity_name,
                    properties={
                        "name": entity_name,
                        "canonical_name": entity_name
                    },
                    confidence=0.0,
                    source_chunks=[],
                    extraction_method="none",
                    success=False,
                    error="No chunks found mentioning this entity"
                )
                
        except Exception as e:
            return ExtractionResult(
                entity_name=entity_name,
                properties={},
                confidence=0.0,
                source_chunks=[],
                extraction_method="none",
                success=False,
                error=f"Database query failed: {str(e)}"
            )
        
        # =================================================================
        # 3. Rule-based extraction for simple cases
        # =================================================================
        
        chunk_text = "\n---\n".join([c["content"][:800] for c in chunks])
        chunk_ids = [str(c["id"]) for c in chunks]
        
        # Simple pattern-based extraction
        rule_based_props = self._extract_by_rules(entity_name, chunk_text)
        
        if rule_based_props.get("confidence", 0) >= 0.7:
            self.stats["processed"] += 1
            return ExtractionResult(
                entity_name=entity_name,
                properties={
                    "name": entity_name,
                    "canonical_name": entity_name,
                    **rule_based_props
                },
                confidence=rule_based_props.get("confidence", 0.7),
                source_chunks=chunk_ids,
                extraction_method="rule_based",
                success=True
            )
        
        # =================================================================
        # 4. LLM extraction
        # =================================================================
        
        if self.llm is None:
            return ExtractionResult(
                entity_name=entity_name,
                properties=rule_based_props,
                confidence=rule_based_props.get("confidence", 0.3),
                source_chunks=chunk_ids,
                extraction_method="rule_based",
                success=True
            )
        
        try:
            response = await self._llm_extract(entity_name, chunk_text)
            
            if response:
                self.stats["llm_extracted"] += 1
                return ExtractionResult(
                    entity_name=entity_name,
                    properties=response,
                    confidence=response.get("confidence", 0.6),
                    source_chunks=chunk_ids,
                    extraction_method="llm",
                    success=True
                )
                
        except Exception as e:
            logger.warning(f"LLM extraction failed for {entity_name}: {e}")
            self.stats["failed"] += 1
        
        # Fall back to rule-based results
        self.stats["processed"] += 1
        return ExtractionResult(
            entity_name=entity_name,
            properties={
                "name": entity_name,
                "canonical_name": entity_name,
                **rule_based_props
            },
            confidence=rule_based_props.get("confidence", 0.3),
            source_chunks=chunk_ids,
            extraction_method="rule_based",
            success=True
        )
    
    def _extract_by_rules(self, entity_name: str, chunk_text: str) -> Dict[str, Any]:
        """
        Rule-based extraction using pattern matching.
        
        Faster than LLM, good for obvious cases.
        """
        props = {}
        text_lower = chunk_text.lower()
        entity_lower = entity_name.lower()
        
        # Detect vascular structures
        vascular_keywords = ["artery", "arteria", "vein", "venous", "sinus", "vessel"]
        if any(kw in entity_lower for kw in vascular_keywords):
            props["consistency"] = "vascular"
            
            # Check for end artery mentions
            end_artery_patterns = [
                r"end.?arter",
                r"no.{0,20}collateral",
                r"terminal.{0,10}branch",
                r"perforat"
            ]
            for pattern in end_artery_patterns:
                if re.search(pattern, text_lower):
                    props["is_end_artery"] = True
                    props["collateral_capacity"] = "none"
                    props["sacrifice_safety"] = "never"
                    break
        
        # Detect neural structures
        neural_keywords = ["nerve", "nucleus", "tract", "ganglion", "plexus"]
        if any(kw in entity_lower for kw in neural_keywords):
            props["consistency"] = "neural"
            props["coagulation_tolerance"] = "minimal"
            
            # Cranial nerves
            if "cranial nerve" in entity_lower or re.search(r"cn\s*[ivx]+", entity_lower):
                props["retraction_tolerance"] = "minimal"
                props["sacrifice_safety"] = "never"
        
        # Detect brain regions
        brain_keywords = ["cortex", "lobe", "gyrus", "sulcus", "white matter", "parenchyma"]
        if any(kw in entity_lower for kw in brain_keywords):
            props["consistency"] = "soft_brain"
            props["mobility"] = "elastic"
            
            # Check for eloquence
            eloquent_mentions = [
                r"eloquent",
                r"motor\s+cortex",
                r"speech\s+area",
                r"language\s+area",
                r"primary\s+sensory"
            ]
            for pattern in eloquent_mentions:
                if re.search(pattern, text_lower):
                    props["eloquence_grade"] = "eloquent"
                    props["sacrifice_safety"] = "never"
                    break
        
        # Detect bone structures
        bone_keywords = ["bone", "skull", "vertebra", "spine", "petrous", "mastoid"]
        if any(kw in entity_lower for kw in bone_keywords):
            props["consistency"] = "solid_bone"
            props["mobility"] = "fixed"
            props["retraction_tolerance"] = "excellent"
        
        # Detect dural structures
        if "dura" in entity_lower:
            props["consistency"] = "firm_dura"
            props["mobility"] = "semi_mobile"
        
        # Set confidence based on how many properties we extracted
        if len(props) >= 3:
            props["confidence"] = 0.7
        elif len(props) >= 1:
            props["confidence"] = 0.5
        else:
            props["confidence"] = 0.3
        
        return props
    
    async def _llm_extract(
        self,
        entity_name: str,
        chunk_text: str
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to extract physics properties.
        """
        if self.llm is None:
            return None
        
        prompt = PHYSICS_EXTRACTION_USER_PROMPT.format(
            entity_name=entity_name,
            chunk_text=chunk_text[:4000]  # Limit context
        )
        
        try:
            response = await asyncio.to_thread(
                self.llm.messages.create,
                model=self.llm_model,
                max_tokens=1000,
                system=PHYSICS_EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            response_text = response.content[0].text
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                return json.loads(json_match.group())
            
            return None
            
        except Exception as e:
            logger.warning(f"LLM extraction error: {e}")
            return None
    
    async def bootstrap_vascular_entities(
        self,
        batch_size: int = 50
    ) -> List[ExtractionResult]:
        """
        Priority bootstrap: Vascular structures first.
        
        These are most critical for surgical safety reasoning.
        """
        results = []
        
        try:
            # Get vascular entities from existing entities table
            vascular_types = ["T023", "T024"]  # Blood Vessel, Artery
            
            entities = await self.db.fetch("""
                SELECT DISTINCT name 
                FROM entities
                WHERE tui IN ($1, $2)
                   OR semantic_type ILIKE '%vessel%'
                   OR semantic_type ILIKE '%artery%'
                   OR semantic_type ILIKE '%vein%'
                ORDER BY chunk_count DESC
                LIMIT $3
            """, vascular_types[0], vascular_types[1], batch_size)
            
            logger.info(f"Bootstrapping {len(entities)} vascular entities")
            
            for entity in entities:
                result = await self.bootstrap_entity(entity["name"])
                results.append(result)
                
                # Small delay to avoid rate limits
                await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Vascular bootstrap failed: {e}")
        
        return results
    
    async def bootstrap_neural_entities(
        self,
        batch_size: int = 50
    ) -> List[ExtractionResult]:
        """
        Bootstrap neural structures (cranial nerves, nuclei, tracts).
        """
        results = []
        
        try:
            entities = await self.db.fetch("""
                SELECT DISTINCT name 
                FROM entities
                WHERE semantic_type ILIKE '%nerve%'
                   OR semantic_type ILIKE '%nucleus%'
                   OR name ILIKE '%nerve%'
                   OR name ILIKE '%CN%'
                ORDER BY chunk_count DESC
                LIMIT $1
            """, batch_size)
            
            logger.info(f"Bootstrapping {len(entities)} neural entities")
            
            for entity in entities:
                result = await self.bootstrap_entity(entity["name"])
                results.append(result)
                await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Neural bootstrap failed: {e}")
        
        return results
    
    async def bootstrap_all(
        self,
        batch_size: int = 100,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> List[ExtractionResult]:
        """
        Bootstrap all entities from the entities table.
        
        Args:
            batch_size: Number of entities to process
            on_progress: Callback for progress updates (current, total)
        """
        results = []
        
        try:
            # Get all unique entities
            entities = await self.db.fetch("""
                SELECT DISTINCT name, chunk_count
                FROM entities
                WHERE name IS NOT NULL AND name != ''
                ORDER BY chunk_count DESC
                LIMIT $1
            """, batch_size)
            
            total = len(entities)
            logger.info(f"Bootstrapping {total} entities")
            
            for i, entity in enumerate(entities):
                result = await self.bootstrap_entity(entity["name"])
                results.append(result)
                
                if on_progress:
                    on_progress(i + 1, total)
                
                await asyncio.sleep(0.05)
            
        except Exception as e:
            logger.error(f"Full bootstrap failed: {e}")
        
        return results
    
    async def save_results(
        self,
        results: List[ExtractionResult],
        verify_before_insert: bool = True
    ) -> Dict[str, int]:
        """
        Save extraction results to anatomical_entities table.
        
        Args:
            results: List of extraction results
            verify_before_insert: If True, check for duplicates
            
        Returns:
            Statistics dict {inserted, updated, skipped, failed}
        """
        stats = {
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "failed": 0
        }
        
        for result in results:
            if not result.success:
                stats["skipped"] += 1
                continue
            
            props = result.properties
            
            try:
                # Check if entity already exists
                if verify_before_insert:
                    existing = await self.db.fetchval("""
                        SELECT id FROM anatomical_entities
                        WHERE LOWER(name) = LOWER($1)
                    """, props.get("name", result.entity_name))
                    
                    if existing:
                        # Update only if new confidence is higher
                        existing_conf = await self.db.fetchval("""
                            SELECT confidence FROM anatomical_entities
                            WHERE id = $1
                        """, existing)
                        
                        if result.confidence <= (existing_conf or 0):
                            stats["skipped"] += 1
                            continue
                        
                        # Update existing
                        await self.db.execute("""
                            UPDATE anatomical_entities SET
                                mobility = COALESCE($2, mobility),
                                consistency = COALESCE($3, consistency),
                                is_end_artery = COALESCE($4, is_end_artery),
                                collateral_capacity = COALESCE($5, collateral_capacity),
                                retraction_tolerance = COALESCE($6, retraction_tolerance),
                                sacrifice_safety = COALESCE($7, sacrifice_safety),
                                eloquence_grade = COALESCE($8, eloquence_grade),
                                spatial_context = COALESCE($9, spatial_context),
                                confidence = $10,
                                source_chunk_ids = $11,
                                extraction_method = $12,
                                updated_at = NOW()
                            WHERE id = $1
                        """,
                            existing,
                            props.get("mobility"),
                            props.get("consistency"),
                            props.get("is_end_artery"),
                            props.get("collateral_capacity"),
                            props.get("retraction_tolerance"),
                            props.get("sacrifice_safety"),
                            props.get("eloquence_grade"),
                            json.dumps(props.get("spatial_context", {})),
                            result.confidence,
                            result.source_chunks,
                            result.extraction_method
                        )
                        
                        stats["updated"] += 1
                        continue
                
                # Insert new entity
                await self.db.execute("""
                    INSERT INTO anatomical_entities (
                        name, canonical_name, mobility, consistency,
                        is_end_artery, collateral_capacity, territory_supplied,
                        retraction_tolerance, sacrifice_safety, eloquence_grade,
                        spatial_context, confidence, source_chunk_ids,
                        extraction_method
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                    )
                    ON CONFLICT (name) DO UPDATE SET
                        confidence = GREATEST(anatomical_entities.confidence, EXCLUDED.confidence),
                        updated_at = NOW()
                """,
                    props.get("name", result.entity_name),
                    props.get("canonical_name", props.get("name", result.entity_name)),
                    props.get("mobility", "fixed"),
                    props.get("consistency", "soft_brain"),
                    props.get("is_end_artery", False),
                    props.get("collateral_capacity"),
                    props.get("territory_supplied", []),
                    props.get("retraction_tolerance", "minimal"),
                    props.get("sacrifice_safety", "conditional"),
                    props.get("eloquence_grade", "non_eloquent"),
                    json.dumps(props.get("spatial_context", {})),
                    result.confidence,
                    result.source_chunks,
                    result.extraction_method
                )
                
                stats["inserted"] += 1
                
            except Exception as e:
                logger.error(f"Failed to save entity {result.entity_name}: {e}")
                stats["failed"] += 1
        
        return stats
