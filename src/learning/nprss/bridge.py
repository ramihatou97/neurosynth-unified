# src/learning/nprss/bridge.py
"""
Procedure Bridge: NeuroSynth Synthesis -> NPRSS Procedure

Converts NeuroSynth synthesis output into NPRSS-compatible procedure format,
enabling the learning system to work with both:
1. Manually created Impeccable Format procedures
2. Automatically extracted procedures from synthesis

This is the integration point between NeuroSynth's RAG capabilities
and NPRSS's procedural learning system.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from uuid import uuid4, UUID

from .models import (
    Procedure, ProcedureElement, PhaseType, ElementType,
    ACGMESubspecialty, Complexity, SafeEntryZone, DangerZone,
    VisualDescription
)

logger = logging.getLogger(__name__)


@dataclass
class BridgeSettings:
    """Configuration for procedure bridge"""
    use_llm_extraction: bool = True
    max_substeps_per_step: int = 5
    extract_measurements: bool = True
    extract_instruments: bool = True
    min_substep_confidence: float = 0.6


class ProcedureBridge:
    """
    Converts NeuroSynth synthesis output to NPRSS procedure format.

    This enables the learning system to work with both:
    1. Manually created Impeccable Format procedures
    2. Automatically extracted procedures from synthesis

    Usage:
        bridge = ProcedureBridge(llm_client=anthropic_client)

        # From synthesis result
        procedure = await bridge.synthesis_to_procedure(
            synthesis=synthesis_result,
            procedure_name="Pterional Craniotomy",
            subspecialty="cerebrovascular"
        )

        # From RAG response
        procedure = await bridge.rag_response_to_procedure(
            question="Describe the pterional approach",
            response=rag_response,
            context_chunks=chunks
        )
    """

    # Section titles that map to phases
    PHASE_SECTION_MAPPING = {
        # Architecture
        'positioning': PhaseType.ARCHITECTURE,
        'patient positioning': PhaseType.ARCHITECTURE,
        'setup': PhaseType.ARCHITECTURE,
        'preparation': PhaseType.ARCHITECTURE,
        'head fixation': PhaseType.ARCHITECTURE,
        'navigation': PhaseType.ARCHITECTURE,
        'registration': PhaseType.ARCHITECTURE,

        # Approach
        'approach': PhaseType.APPROACH,
        'surgical approach': PhaseType.APPROACH,
        'exposure': PhaseType.APPROACH,
        'craniotomy': PhaseType.APPROACH,
        'incision': PhaseType.APPROACH,
        'bone work': PhaseType.APPROACH,
        'dural opening': PhaseType.APPROACH,
        'durotomy': PhaseType.APPROACH,

        # Target
        'technique': PhaseType.TARGET,
        'procedure': PhaseType.TARGET,
        'resection': PhaseType.TARGET,
        'clipping': PhaseType.TARGET,
        'dissection': PhaseType.TARGET,
        'tumor removal': PhaseType.TARGET,
        'decompression': PhaseType.TARGET,

        # Closure
        'closure': PhaseType.CLOSURE,
        'reconstruction': PhaseType.CLOSURE,
        'dural closure': PhaseType.CLOSURE,
        'wound closure': PhaseType.CLOSURE,
    }

    # LLM prompt for procedure extraction
    EXTRACTION_PROMPT = """
Extract the surgical procedure steps from this content into a structured format.

Content:
{content}

Return a JSON object with:
{{
    "procedure_name": "Name of the procedure",
    "approach": "Surgical approach used",
    "phases": [
        {{
            "phase_type": "architecture|approach|target|closure",
            "steps": [
                {{
                    "name": "Step name",
                    "description": "Brief description",
                    "the_maneuver": "Specific surgical action",
                    "instrument": "Primary instrument used",
                    "critical": true/false,
                    "substeps": [
                        {{
                            "name": "Substep name",
                            "the_maneuver": "Action",
                            "visual_cue": "What surgeon sees"
                        }}
                    ]
                }}
            ]
        }}
    ],
    "danger_zones": [
        {{
            "name": "Zone name",
            "structure_at_risk": "Structure",
            "mechanism": "How injury occurs"
        }}
    ],
    "safe_zones": [
        {{
            "name": "Zone name",
            "boundaries": "Boundary description",
            "safe_depth_mm": number or null
        }}
    ]
}}

Extract only what is explicitly mentioned. Return valid JSON only.
"""

    def __init__(
        self,
        settings: Optional[BridgeSettings] = None,
        llm_client: Any = None
    ):
        self.settings = settings or BridgeSettings()
        self.llm = llm_client

    async def synthesis_to_procedure(
        self,
        synthesis: Dict[str, Any],
        procedure_name: Optional[str] = None,
        subspecialty: str = "brain_tumor",
        complexity: str = "routine"
    ) -> Procedure:
        """
        Convert SynthesisResult to Procedure model.

        Args:
            synthesis: NeuroSynth synthesis output dict with 'sections', 'topic', etc.
            procedure_name: Override name (uses synthesis topic if None)
            subspecialty: ACGME subspecialty domain
            complexity: Procedure complexity level

        Returns:
            Procedure ready for NPRSS enrichment
        """
        # Create base procedure
        procedure = Procedure(
            name=procedure_name or synthesis.get('topic', 'Unknown Procedure'),
            description=synthesis.get('abstract', ''),
            subspecialty_domain=ACGMESubspecialty(subspecialty) if subspecialty in [e.value for e in ACGMESubspecialty] else ACGMESubspecialty.BRAIN_TUMOR,
            complexity=Complexity(complexity) if complexity in [e.value for e in Complexity] else Complexity.ROUTINE,
            source_synthesis_id=synthesis.get('id'),
            surgical_approach=self._detect_approach(synthesis)
        )

        # Extract elements from sections
        elements = []
        sections = synthesis.get('sections', [])

        for idx, section in enumerate(sections):
            title = section.get('title', '') or section.get('heading', '')
            content = section.get('content', '') or section.get('text', '')

            if not title and not content:
                continue

            phase_type = self._detect_phase(title)

            # Create step from section
            step = ProcedureElement(
                procedure_id=procedure.id,
                element_type=ElementType.STEP,
                granularity_level=3,
                sequence_order=idx + 1,
                name=title or f"Step {idx + 1}",
                description=content[:500] if content else None,
                phase_type=phase_type
            )
            elements.append(step)

            # Extract substeps from content
            if self.settings.use_llm_extraction and self.llm and content:
                substeps = await self._extract_substeps(
                    content,
                    step.id,
                    phase_type
                )
                elements.extend(substeps)
            else:
                # Simple extraction without LLM
                substeps = self._extract_substeps_simple(content, step.id, phase_type)
                elements.extend(substeps)

        procedure.elements = elements
        return procedure

    async def rag_response_to_procedure(
        self,
        question: str,
        response: Dict[str, Any],
        context_chunks: Optional[List[Dict[str, Any]]] = None,
        procedure_name: Optional[str] = None
    ) -> Procedure:
        """
        Convert RAG response to Procedure.

        Args:
            question: Original question asked
            response: RAG engine response with 'answer', 'citations', etc.
            context_chunks: Source chunks used for answer
            procedure_name: Override name

        Returns:
            Procedure model
        """
        answer = response.get('answer', '')

        # Try to detect procedure name from question
        if not procedure_name:
            procedure_name = self._extract_procedure_name(question, answer)

        # Use LLM to extract structured procedure
        if self.settings.use_llm_extraction and self.llm:
            procedure = await self._llm_extract_procedure(
                content=answer,
                procedure_name=procedure_name,
                context_chunks=context_chunks
            )
        else:
            # Simple extraction
            procedure = self._simple_extract_procedure(
                content=answer,
                procedure_name=procedure_name
            )

        # Link to source chunks
        if context_chunks:
            for chunk in context_chunks:
                doc_id = chunk.get('document_id')
                if doc_id:
                    procedure.source_document_id = UUID(doc_id) if isinstance(doc_id, str) else doc_id
                    break

        return procedure

    def chunks_to_procedure(
        self,
        chunks: List[Dict[str, Any]],
        procedure_name: str,
        subspecialty: str = "brain_tumor"
    ) -> Procedure:
        """
        Create procedure directly from NeuroSynth chunks.

        Useful when chunks are already well-organized procedural content
        (e.g., from Rhoton or procedural textbooks).

        Args:
            chunks: List of NeuroSynth chunks with 'content', 'chunk_type', etc.
            procedure_name: Name of the procedure
            subspecialty: ACGME subspecialty

        Returns:
            Procedure model
        """
        procedure = Procedure(
            name=procedure_name,
            subspecialty_domain=ACGMESubspecialty(subspecialty) if subspecialty in [e.value for e in ACGMESubspecialty] else ACGMESubspecialty.BRAIN_TUMOR,
            complexity=Complexity.ROUTINE
        )

        elements = []

        # Group chunks by type or position
        procedural_chunks = [c for c in chunks if c.get('chunk_type') == 'PROCEDURE']
        anatomical_chunks = [c for c in chunks if c.get('chunk_type') == 'ANATOMY']

        # Create elements from procedural chunks
        for idx, chunk in enumerate(procedural_chunks):
            content = chunk.get('content', '')

            step = ProcedureElement(
                procedure_id=procedure.id,
                source_chunk_id=chunk.get('id'),
                element_type=ElementType.STEP,
                granularity_level=3,
                sequence_order=idx + 1,
                name=self._extract_step_name(content),
                description=content[:500],
                phase_type=self._detect_phase_from_content(content)
            )
            elements.append(step)

        # If no procedural chunks, use all chunks
        if not elements:
            for idx, chunk in enumerate(chunks):
                content = chunk.get('content', '')

                step = ProcedureElement(
                    procedure_id=procedure.id,
                    source_chunk_id=chunk.get('id'),
                    element_type=ElementType.STEP,
                    granularity_level=3,
                    sequence_order=idx + 1,
                    name=self._extract_step_name(content),
                    description=content[:500],
                    phase_type=self._detect_phase_from_content(content)
                )
                elements.append(step)

        # Extract danger zones from anatomical chunks
        danger_zones = []
        for chunk in anatomical_chunks:
            content = chunk.get('content', '')
            zones = self._extract_danger_zones_simple(content)
            danger_zones.extend(zones)

        procedure.elements = elements
        procedure.danger_zones = danger_zones

        return procedure

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _detect_phase(self, section_title: str) -> Optional[PhaseType]:
        """Detect phase from section title."""
        title_lower = section_title.lower()

        for keyword, phase in self.PHASE_SECTION_MAPPING.items():
            if keyword in title_lower:
                return phase

        return None

    def _detect_phase_from_content(self, content: str) -> Optional[PhaseType]:
        """Detect phase from content keywords."""
        content_lower = content.lower()

        # Count keyword matches for each phase
        scores = {phase: 0 for phase in PhaseType}

        for keyword, phase in self.PHASE_SECTION_MAPPING.items():
            if keyword in content_lower:
                scores[phase] += 1

        # Return phase with highest score
        best = max(scores.items(), key=lambda x: x[1])
        if best[1] > 0:
            return best[0]

        return None

    def _detect_approach(self, synthesis: Dict[str, Any]) -> str:
        """Detect surgical approach from synthesis."""
        # Check topic
        topic = synthesis.get('topic', '').lower()

        approaches = [
            'pterional', 'subfrontal', 'transcallosal', 'interhemispheric',
            'retrosigmoid', 'translabyrinthine', 'transpetrosal',
            'transsphenoidal', 'endoscopic', 'supracerebellar',
            'far lateral', 'suboccipital', 'transtentorial'
        ]

        for approach in approaches:
            if approach in topic:
                return approach.title()

        # Check sections
        for section in synthesis.get('sections', []):
            content = (section.get('content', '') + section.get('title', '')).lower()
            for approach in approaches:
                if approach in content:
                    return approach.title()

        return ""

    def _extract_procedure_name(self, question: str, answer: str) -> str:
        """Extract procedure name from question/answer."""
        # Common procedure patterns
        patterns = [
            r'(pterional\s+craniotomy)',
            r'(suboccipital\s+craniectomy)',
            r'(transsphenoidal\s+approach)',
            r'(clipping\s+of\s+\w+\s+aneurysm)',
            r'(resection\s+of\s+\w+)',
            r'(decompression\s+of\s+\w+)',
        ]

        combined = f"{question} {answer}".lower()

        for pattern in patterns:
            match = re.search(pattern, combined, re.IGNORECASE)
            if match:
                return match.group(1).title()

        # Fall back to question
        return question[:50] if len(question) > 50 else question

    def _extract_step_name(self, content: str) -> str:
        """Extract step name from content."""
        # Take first sentence or first 50 chars
        sentences = content.split('.')
        if sentences:
            first = sentences[0].strip()
            if len(first) <= 80:
                return first
            return first[:80] + "..."
        return content[:50] + "..."

    def _extract_substeps_simple(
        self,
        content: str,
        parent_id: UUID,
        phase_type: Optional[PhaseType]
    ) -> List[ProcedureElement]:
        """Extract substeps without LLM."""
        substeps = []

        # Split by sentences and filter for action-oriented ones
        sentences = content.split('.')
        action_verbs = [
            'incise', 'dissect', 'retract', 'coagulate', 'clip', 'remove',
            'drill', 'elevate', 'open', 'close', 'suture', 'place', 'insert',
            'identify', 'expose', 'mobilize', 'divide', 'ligate'
        ]

        substep_count = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if sentence contains action verbs
            sentence_lower = sentence.lower()
            if any(verb in sentence_lower for verb in action_verbs):
                substep_count += 1
                if substep_count > self.settings.max_substeps_per_step:
                    break

                substeps.append(ProcedureElement(
                    procedure_id=None,  # Set later
                    parent_id=parent_id,
                    element_type=ElementType.SUBSTEP,
                    granularity_level=2,
                    sequence_order=substep_count,
                    name=sentence[:80],
                    the_maneuver=sentence,
                    phase_type=phase_type
                ))

        return substeps

    async def _extract_substeps(
        self,
        content: str,
        parent_id: UUID,
        phase_type: Optional[PhaseType]
    ) -> List[ProcedureElement]:
        """Extract substeps using LLM."""
        if not self.llm:
            return self._extract_substeps_simple(content, parent_id, phase_type)

        prompt = f"""Extract the key surgical substeps from this content.

Content:
{content[:2000]}

Return a JSON array of substeps, each with:
- name: Brief substep name (max 80 chars)
- the_maneuver: The specific action
- instrument_specification: Instruments used (if mentioned)
- visual_description: What the surgeon sees (if mentioned)

Return only the JSON array, no explanation."""

        try:
            response = await self.llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text.strip()

            # Extract JSON array
            if '[' in result_text:
                json_str = result_text[result_text.index('['):result_text.rindex(']')+1]
                substeps_data = json.loads(json_str)
            else:
                return self._extract_substeps_simple(content, parent_id, phase_type)

            substeps = []
            for idx, data in enumerate(substeps_data[:self.settings.max_substeps_per_step]):
                visual_desc = None
                if data.get('visual_description'):
                    visual_desc = VisualDescription(
                        expected_view=data['visual_description']
                    )

                substep = ProcedureElement(
                    procedure_id=None,
                    parent_id=parent_id,
                    element_type=ElementType.SUBSTEP,
                    granularity_level=2,
                    sequence_order=idx + 1,
                    name=data.get('name', '')[:80],
                    the_maneuver=data.get('the_maneuver'),
                    instrument_specification=data.get('instrument_specification'),
                    visual_description=visual_desc,
                    phase_type=phase_type
                )
                substeps.append(substep)

            return substeps

        except Exception as e:
            logger.warning(f"Substep extraction failed: {e}")
            return self._extract_substeps_simple(content, parent_id, phase_type)

    async def _llm_extract_procedure(
        self,
        content: str,
        procedure_name: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> Procedure:
        """Extract full procedure using LLM."""
        if not self.llm:
            return self._simple_extract_procedure(content, procedure_name)

        # Include context if available
        full_content = content
        if context_chunks:
            chunk_content = "\n\n".join([c.get('content', '') for c in context_chunks[:3]])
            full_content = f"{content}\n\nContext:\n{chunk_content}"

        prompt = self.EXTRACTION_PROMPT.format(content=full_content[:4000])

        try:
            response = await self.llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text.strip()

            # Extract JSON
            if '{' in result_text:
                json_str = result_text[result_text.index('{'):result_text.rindex('}')+1]
                data = json.loads(json_str)
            else:
                return self._simple_extract_procedure(content, procedure_name)

            # Build procedure
            procedure = Procedure(
                name=data.get('procedure_name', procedure_name),
                surgical_approach=data.get('approach', ''),
                subspecialty_domain=ACGMESubspecialty.BRAIN_TUMOR,
                complexity=Complexity.ROUTINE
            )

            # Build elements
            elements = []
            element_order = 0

            for phase_data in data.get('phases', []):
                phase_type_str = phase_data.get('phase_type', 'target')
                try:
                    phase_type = PhaseType(phase_type_str)
                except ValueError:
                    phase_type = PhaseType.TARGET

                for step_data in phase_data.get('steps', []):
                    element_order += 1

                    step = ProcedureElement(
                        procedure_id=procedure.id,
                        element_type=ElementType.STEP,
                        granularity_level=3,
                        sequence_order=element_order,
                        name=step_data.get('name', ''),
                        description=step_data.get('description'),
                        the_maneuver=step_data.get('the_maneuver'),
                        instrument_specification=step_data.get('instrument'),
                        critical_step=step_data.get('critical', False),
                        phase_type=phase_type
                    )
                    elements.append(step)

                    # Add substeps
                    for substep_idx, substep_data in enumerate(step_data.get('substeps', [])):
                        visual_desc = None
                        if substep_data.get('visual_cue'):
                            visual_desc = VisualDescription(
                                expected_view=substep_data['visual_cue']
                            )

                        substep = ProcedureElement(
                            procedure_id=procedure.id,
                            parent_id=step.id,
                            element_type=ElementType.SUBSTEP,
                            granularity_level=2,
                            sequence_order=substep_idx + 1,
                            name=substep_data.get('name', ''),
                            the_maneuver=substep_data.get('the_maneuver'),
                            visual_description=visual_desc,
                            phase_type=phase_type
                        )
                        elements.append(substep)

            procedure.elements = elements

            # Build danger zones
            danger_zones = []
            for dz_data in data.get('danger_zones', []):
                danger_zones.append(DangerZone(
                    name=dz_data.get('name', ''),
                    structures_at_risk=[dz_data.get('structure_at_risk', '')],
                    mechanism_of_injury=dz_data.get('mechanism')
                ))
            procedure.danger_zones = danger_zones

            # Build safe zones
            safe_zones = []
            for sz_data in data.get('safe_zones', []):
                safe_zones.append(SafeEntryZone(
                    name=sz_data.get('name', ''),
                    mean_safe_depth_mm=sz_data.get('safe_depth_mm')
                ))
            procedure.safe_zones = safe_zones

            return procedure

        except Exception as e:
            logger.warning(f"LLM procedure extraction failed: {e}")
            return self._simple_extract_procedure(content, procedure_name)

    def _simple_extract_procedure(
        self,
        content: str,
        procedure_name: str
    ) -> Procedure:
        """Simple procedure extraction without LLM."""
        procedure = Procedure(
            name=procedure_name,
            description=content[:500],
            subspecialty_domain=ACGMESubspecialty.BRAIN_TUMOR,
            complexity=Complexity.ROUTINE
        )

        # Split content into paragraphs and create steps
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        elements = []
        for idx, para in enumerate(paragraphs[:10]):  # Max 10 steps
            phase = self._detect_phase_from_content(para)

            step = ProcedureElement(
                procedure_id=procedure.id,
                element_type=ElementType.STEP,
                granularity_level=3,
                sequence_order=idx + 1,
                name=self._extract_step_name(para),
                description=para[:500],
                phase_type=phase
            )
            elements.append(step)

        procedure.elements = elements
        procedure.danger_zones = self._extract_danger_zones_simple(content)

        return procedure

    def _extract_danger_zones_simple(self, content: str) -> List[DangerZone]:
        """Extract danger zones using pattern matching."""
        danger_zones = []

        # Common danger zone patterns
        patterns = [
            (r'(facial\s+nerve)', 'Facial nerve', 'Direct injury during dissection'),
            (r'(optic\s+nerve)', 'Optic nerve', 'Compression or direct injury'),
            (r'(carotid\s+artery)', 'Internal carotid artery', 'Vascular injury'),
            (r'(vertebral\s+artery)', 'Vertebral artery', 'Vascular injury'),
            (r'(spinal\s+cord)', 'Spinal cord', 'Direct compression or injury'),
            (r'(brainstem)', 'Brainstem', 'Direct injury'),
        ]

        content_lower = content.lower()

        for pattern, structure, mechanism in patterns:
            if re.search(pattern, content_lower):
                # Avoid duplicates
                if not any(dz.structures_at_risk[0] == structure for dz in danger_zones if dz.structures_at_risk):
                    danger_zones.append(DangerZone(
                        name=f"{structure} danger zone",
                        structures_at_risk=[structure],
                        mechanism_of_injury=mechanism
                    ))

        return danger_zones
