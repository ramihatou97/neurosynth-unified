# src/learning/nprss/card_generators.py
"""
Learning Content Generators for NPRSS

Generates learning materials from enriched procedures:
- CardGenerator: FSRS flashcards (sequence, CSP, dictation, MCQ)
- SurgicalCardGenerator: One-page rapid reference summaries
"""

import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .models import (
    PhaseType,
    CardType,
    SurgicalCard,
    SurgicalCardRow,
    LearningCard,
)


# =============================================================================
# CARD GENERATOR
# =============================================================================

@dataclass
class CardGeneratorSettings:
    """Configuration for card generation"""
    generate_sequence_cards: bool = True
    generate_csp_cards: bool = True
    generate_dictation_cards: bool = True
    generate_mcq_cards: bool = True
    generate_safe_zone_cards: bool = True
    max_cards_per_type: int = 20


class CardGenerator:
    """
    Generates FSRS learning cards from procedure content

    Card Types:
    - SEQUENCE: Order the phases/steps
    - CSP_TRIGGER: Rapid-fire CSP trigger-action pairs
    - DICTATION: Operative note dictation prompts
    - MCQ: Multiple choice questions
    - SAFE_ZONE: Safe zone identification
    - SCENARIO: Clinical scenario decisions
    """

    def __init__(self, settings: Optional[CardGeneratorSettings] = None):
        self.settings = settings or CardGeneratorSettings()

        # Difficulty presets by card type
        self.difficulty_presets = {
            CardType.SEQUENCE: 0.3,
            CardType.CSP_TRIGGER: 0.4,
            CardType.DICTATION: 0.5,
            CardType.MCQ: 0.3,
            CardType.SAFE_ZONE: 0.35,
            CardType.SCENARIO: 0.5,
            CardType.IMAGE: 0.4,
        }

    def generate_all_cards(
        self,
        procedure_id: str,
        procedure_name: str,
        elements: List[Dict[str, Any]],
        csps: List[Dict[str, Any]],
        safe_zones: List[Dict[str, Any]],
        phase_mapping: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Generate all card types for a procedure

        Args:
            procedure_id: UUID of procedure
            procedure_name: Human-readable name
            elements: Procedure elements with phase assignments
            csps: Critical Safety Points
            safe_zones: Safe Entry Zones
            phase_mapping: Element ID -> Phase type mapping

        Returns:
            List of card dicts ready for database insertion
        """
        cards = []

        # Enrich elements with phase info
        for element in elements:
            element_id = str(element.get('id'))
            if element_id in phase_mapping:
                element['phase_type'] = phase_mapping[element_id]

        if self.settings.generate_sequence_cards:
            cards.extend(self._generate_sequence_cards(procedure_id, procedure_name, elements))

        if self.settings.generate_csp_cards:
            cards.extend(self._generate_csp_cards(procedure_id, csps))

        if self.settings.generate_dictation_cards:
            cards.extend(self._generate_dictation_cards(procedure_id, procedure_name, elements))

        if self.settings.generate_safe_zone_cards:
            cards.extend(self._generate_safe_zone_cards(procedure_id, safe_zones))

        if self.settings.generate_mcq_cards:
            cards.extend(self._generate_instrument_cards(procedure_id, elements))

        return cards

    def _generate_sequence_cards(
        self,
        procedure_id: str,
        procedure_name: str,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate phase/step sequencing cards"""
        cards = []

        # Phase sequence card
        phases = []
        for phase in [PhaseType.ARCHITECTURE, PhaseType.APPROACH, PhaseType.TARGET, PhaseType.CLOSURE]:
            phase_elements = [e for e in elements if e.get('phase_type') == phase.value]
            if phase_elements:
                phases.append(phase.value.upper())

        if len(phases) >= 2:
            cards.append({
                "id": str(uuid.uuid4()),
                "procedure_id": procedure_id,
                "element_id": None,
                "csp_id": None,
                "card_type": CardType.SEQUENCE.value,
                "prompt": f"What is the phase sequence for {procedure_name}?",
                "answer": " → ".join(phases),
                "options": None,
                "difficulty_preset": self.difficulty_presets[CardType.SEQUENCE],
                "tags": ["sequence", "phases", "overview"]
            })

        # Per-phase step sequence cards
        for phase in [PhaseType.ARCHITECTURE, PhaseType.APPROACH, PhaseType.TARGET, PhaseType.CLOSURE]:
            phase_steps = sorted(
                [e for e in elements if e.get('phase_type') == phase.value and e.get('element_type') == 'step'],
                key=lambda x: x.get('sequence_order', 0)
            )

            if len(phase_steps) >= 2:
                step_names = [s.get('name', '')[:50] for s in phase_steps[:5]]
                cards.append({
                    "id": str(uuid.uuid4()),
                    "procedure_id": procedure_id,
                    "element_id": None,
                    "csp_id": None,
                    "card_type": CardType.SEQUENCE.value,
                    "prompt": f"List the key steps in {phase.value.upper()} phase of {procedure_name}",
                    "answer": "\n".join([f"{i+1}. {name}" for i, name in enumerate(step_names)]),
                    "options": None,
                    "difficulty_preset": self.difficulty_presets[CardType.SEQUENCE],
                    "tags": ["sequence", phase.value, "steps"]
                })

        return cards[:self.settings.max_cards_per_type]

    def _generate_csp_cards(
        self,
        procedure_id: str,
        csps: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate CSP trigger-action cards"""
        cards = []

        for csp in csps:
            # Trigger → Structure card
            cards.append({
                "id": str(uuid.uuid4()),
                "procedure_id": procedure_id,
                "element_id": csp.get('element_id'),
                "csp_id": csp.get('id'),
                "card_type": CardType.CSP_TRIGGER.value,
                "prompt": f"CSP-{csp.get('csp_number')}: When {csp.get('when_action', '')}...\nSTOP IF: {csp.get('stop_if_trigger', '')}?\n\nWhat structure is at risk?",
                "answer": csp.get('structure_at_risk', ''),
                "options": None,
                "difficulty_preset": self.difficulty_presets[CardType.CSP_TRIGGER],
                "tags": ["csp", f"csp_{csp.get('csp_number')}", "anatomy"]
            })

            # Mechanism card
            if csp.get('mechanism_of_injury'):
                cards.append({
                    "id": str(uuid.uuid4()),
                    "procedure_id": procedure_id,
                    "element_id": csp.get('element_id'),
                    "csp_id": csp.get('id'),
                    "card_type": CardType.CSP_TRIGGER.value,
                    "prompt": f"How would {csp.get('structure_at_risk')} be injured at this step?",
                    "answer": csp.get('mechanism_of_injury', ''),
                    "options": None,
                    "difficulty_preset": self.difficulty_presets[CardType.CSP_TRIGGER] + 0.1,
                    "tags": ["csp", "mechanism", "safety"]
                })

            # Recovery card
            if csp.get('if_violated_action'):
                cards.append({
                    "id": str(uuid.uuid4()),
                    "procedure_id": procedure_id,
                    "element_id": csp.get('element_id'),
                    "csp_id": csp.get('id'),
                    "card_type": CardType.CSP_TRIGGER.value,
                    "prompt": f"What is the bailout if CSP-{csp.get('csp_number')} is violated?",
                    "answer": csp.get('if_violated_action', ''),
                    "options": None,
                    "difficulty_preset": self.difficulty_presets[CardType.CSP_TRIGGER] + 0.15,
                    "tags": ["csp", "bailout", "safety"]
                })

        return cards[:self.settings.max_cards_per_type]

    def _generate_safe_zone_cards(
        self,
        procedure_id: str,
        safe_zones: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate safe entry zone identification cards"""
        cards = []

        for sz in safe_zones:
            # Depth limit card
            if sz.get('mean_safe_depth_mm'):
                cards.append({
                    "id": str(uuid.uuid4()),
                    "procedure_id": procedure_id,
                    "element_id": None,
                    "csp_id": None,
                    "card_type": CardType.SAFE_ZONE.value,
                    "prompt": f"What is the safe depth for the {sz.get('name', 'safe zone')}?",
                    "answer": f"{sz.get('mean_safe_depth_mm')}mm (range: {sz.get('min_safe_depth_mm', '?')}-{sz.get('max_safe_depth_mm', '?')}mm)",
                    "options": None,
                    "difficulty_preset": self.difficulty_presets[CardType.SAFE_ZONE],
                    "tags": ["safe_zone", "measurements"]
                })

            # Boundary card
            boundaries = []
            if sz.get('superior_boundary'):
                boundaries.append(f"Superior: {sz.get('superior_boundary')}")
            if sz.get('inferior_boundary'):
                boundaries.append(f"Inferior: {sz.get('inferior_boundary')}")
            if sz.get('lateral_boundary'):
                boundaries.append(f"Lateral: {sz.get('lateral_boundary')}")
            if sz.get('medial_boundary'):
                boundaries.append(f"Medial: {sz.get('medial_boundary')}")

            if boundaries:
                cards.append({
                    "id": str(uuid.uuid4()),
                    "procedure_id": procedure_id,
                    "element_id": None,
                    "csp_id": None,
                    "card_type": CardType.SAFE_ZONE.value,
                    "prompt": f"What are the boundaries of the {sz.get('name', 'safe zone')}?",
                    "answer": "\n".join(boundaries),
                    "options": None,
                    "difficulty_preset": self.difficulty_presets[CardType.SAFE_ZONE],
                    "tags": ["safe_zone", "anatomy", "boundaries"]
                })

        return cards[:self.settings.max_cards_per_type]

    def _generate_instrument_cards(
        self,
        procedure_id: str,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate instrument selection cards for critical steps"""
        cards = []

        for e in elements:
            if not e.get('critical_step'):
                continue

            instrument_seq = e.get('instrument_sequence', [])
            instrument_spec = e.get('instrument_specification', '')

            if instrument_seq and len(instrument_seq) > 0:
                primary_instrument = instrument_seq[0] if isinstance(instrument_seq[0], str) else instrument_seq[0].get('name', '')

                cards.append({
                    "id": str(uuid.uuid4()),
                    "procedure_id": procedure_id,
                    "element_id": e.get('id'),
                    "csp_id": None,
                    "card_type": CardType.MCQ.value,
                    "prompt": f"Primary instrument for: {e.get('name', '')}",
                    "answer": primary_instrument,
                    "options": None,
                    "difficulty_preset": self.difficulty_presets[CardType.MCQ],
                    "tags": ["instruments", "equipment"]
                })
            elif instrument_spec:
                cards.append({
                    "id": str(uuid.uuid4()),
                    "procedure_id": procedure_id,
                    "element_id": e.get('id'),
                    "csp_id": None,
                    "card_type": CardType.MCQ.value,
                    "prompt": f"What instrument is used for: {e.get('name', '')}",
                    "answer": instrument_spec,
                    "options": None,
                    "difficulty_preset": self.difficulty_presets[CardType.MCQ],
                    "tags": ["instruments", "equipment"]
                })

        return cards[:self.settings.max_cards_per_type]

    def _generate_dictation_cards(
        self,
        procedure_id: str,
        procedure_name: str,
        elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate dictation prompt cards"""
        cards = []

        # Overall procedure dictation
        cards.append({
            "id": str(uuid.uuid4()),
            "procedure_id": procedure_id,
            "element_id": None,
            "csp_id": None,
            "card_type": CardType.DICTATION.value,
            "prompt": f"Dictate the opening of an operative note for {procedure_name}",
            "answer": "Position, head fixation, prep, drape, time-out, incision description",
            "options": None,
            "difficulty_preset": self.difficulty_presets[CardType.DICTATION],
            "tags": ["dictation", "operative_note"]
        })

        # Phase-specific dictation
        for phase in ['architecture', 'approach', 'target', 'closure']:
            phase_elements = [e for e in elements if e.get('phase_type') == phase]
            if phase_elements:
                key_elements = [e.get('name', '')[:30] for e in phase_elements[:3]]
                cards.append({
                    "id": str(uuid.uuid4()),
                    "procedure_id": procedure_id,
                    "element_id": None,
                    "csp_id": None,
                    "card_type": CardType.DICTATION.value,
                    "prompt": f"Dictate the {phase.upper()} phase of {procedure_name}",
                    "answer": f"Key elements: {', '.join(key_elements)}",
                    "options": None,
                    "difficulty_preset": self.difficulty_presets[CardType.DICTATION],
                    "tags": ["dictation", phase]
                })

        return cards[:self.settings.max_cards_per_type]


# =============================================================================
# SURGICAL CARD GENERATOR
# =============================================================================

@dataclass
class SurgicalCardGeneratorSettings:
    """Configuration for surgical card generation"""
    max_actions_per_phase: int = 5
    include_csp_summary: bool = True
    include_dictation_template: bool = True


class SurgicalCardGenerator:
    """
    Generates one-page surgical cards for rapid reference

    A Surgical Card contains:
    - Header: Title, approach, corridor, exam relevance
    - 4 Phase rows: Key actions + anchors/CSPs
    - CSP Summary: Quick reference triggers
    - Dictation Template: Operative note structure
    - Mantra: Learning mnemonic
    """

    DICTATION_TEMPLATE = """
OPERATIVE NOTE TEMPLATE
=======================

PROCEDURE: {procedure_name}
DATE: [Date]
SURGEON: [Name]
ASSISTANT: [Name]
ANESTHESIA: General endotracheal

PREOPERATIVE DIAGNOSIS: [Diagnosis]
POSTOPERATIVE DIAGNOSIS: Same

INDICATION: [Brief indication]

DESCRIPTION OF PROCEDURE:
The patient was brought to the operating room and placed in the {position} position.
After induction of general anesthesia, the head was secured in a {fixation}.

[ARCHITECTURE]
{architecture_summary}

[APPROACH]
{approach_summary}

[TARGET]
{target_summary}

[CLOSURE]
{closure_summary}

ESTIMATED BLOOD LOSS: [X] mL
DRAINS: [Type/Location]
SPECIMENS: [Description]
COMPLICATIONS: None

The patient tolerated the procedure well and was transferred to [PACU/ICU] in stable condition.
"""

    def __init__(self, settings: Optional[SurgicalCardGeneratorSettings] = None):
        self.settings = settings or SurgicalCardGeneratorSettings()

    def generate(
        self,
        procedure_id: str,
        procedure_name: str,
        surgical_approach: str,
        elements: List[Dict[str, Any]],
        csps: List[Dict[str, Any]],
        anchors: List[Dict[str, Any]],
        phase_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Generate a surgical card for a procedure

        Args:
            procedure_id: UUID of procedure
            procedure_name: Human-readable name
            surgical_approach: Approach description
            elements: Procedure elements
            csps: Critical Safety Points
            anchors: Visuospatial anchors
            phase_mapping: Element ID -> Phase mapping

        Returns:
            SurgicalCard dict ready for database insertion
        """
        # Enrich elements with phase info
        for element in elements:
            element_id = str(element.get('id'))
            if element_id in phase_mapping:
                element['phase_type'] = phase_mapping[element_id]

        # Build anchor lookup
        anchor_lookup = {}
        for anchor in anchors:
            element_id = str(anchor.get('element_id'))
            if element_id not in anchor_lookup:
                anchor_lookup[element_id] = []
            anchor_lookup[element_id].append(anchor)

        # Build CSP lookup by phase
        csp_by_phase = {}
        for csp in csps:
            phase = csp.get('phase_type', 'target')
            if phase not in csp_by_phase:
                csp_by_phase[phase] = []
            csp_by_phase[phase].append(csp)

        # Generate card rows
        card_rows = []
        phase_labels = {
            'architecture': 'I. ARCHITECTURE',
            'approach': 'II. APPROACH',
            'target': 'III. TARGET',
            'closure': 'IV. CLOSURE'
        }

        for phase_value, phase_label in phase_labels.items():
            phase_elements = sorted(
                [e for e in elements if e.get('phase_type') == phase_value],
                key=lambda x: x.get('sequence_order', 0)
            )

            # Get key actions
            key_actions = []
            for elem in phase_elements[:self.settings.max_actions_per_phase]:
                action = elem.get('name', '')
                if elem.get('critical_step'):
                    action = f"⚠️ {action}"
                key_actions.append(action)

            # Get anchor or CSP for this phase
            anchor_or_csp = ""
            anchor_type = "none"

            # Prioritize CSPs
            phase_csps = csp_by_phase.get(phase_value, [])
            if phase_csps:
                csp = phase_csps[0]
                anchor_or_csp = f"CSP-{csp.get('csp_number')}: {csp.get('stop_if_trigger', '')[:40]}"
                anchor_type = "csp"
            else:
                # Fall back to anchor
                for elem in phase_elements:
                    elem_anchors = anchor_lookup.get(str(elem.get('id')), [])
                    if elem_anchors:
                        anchor = elem_anchors[0]
                        anchor_or_csp = f"ANCHOR: {anchor.get('expected_view', '')[:40]}"
                        anchor_type = "anchor"
                        break

            card_rows.append({
                "phase": phase_value,
                "phase_label": phase_label,
                "key_actions": key_actions,
                "anchor_or_csp": anchor_or_csp,
                "anchor_type": anchor_type
            })

        # Generate CSP summary
        csp_summary = []
        if self.settings.include_csp_summary:
            for csp in csps[:7]:  # Max 7 CSPs in summary
                csp_summary.append({
                    "id": str(csp.get('id')),
                    "short_name": f"CSP-{csp.get('csp_number')}",
                    "trigger": csp.get('stop_if_trigger', '')[:50],
                    "structure": csp.get('structure_at_risk', '')[:30]
                })

        # Generate dictation template
        dictation_template = ""
        if self.settings.include_dictation_template:
            dictation_template = self._generate_dictation_template(
                procedure_name=procedure_name,
                elements=elements
            )

        # Determine corridor from approach
        corridor = self._infer_corridor(surgical_approach, elements)

        surgical_card = {
            "id": str(uuid.uuid4()),
            "procedure_id": procedure_id,
            "title": procedure_name,
            "subtitle": surgical_approach or "",
            "approach": surgical_approach or "",
            "corridor": corridor,
            "exam_relevance": "Royal College Core",
            "card_rows": card_rows,
            "csp_summary": csp_summary,
            "dictation_template": dictation_template,
            "mantra": "4 folders → 3-5 substeps → visual anchors → CSP triggers",
            "generated_at": datetime.now().isoformat(),
            "version": 1
        }

        return surgical_card

    def _generate_dictation_template(
        self,
        procedure_name: str,
        elements: List[Dict[str, Any]]
    ) -> str:
        """Generate operative note dictation template"""
        # Extract phase summaries
        phase_summaries = {}

        for phase in ['architecture', 'approach', 'target', 'closure']:
            phase_elements = [e for e in elements if e.get('phase_type') == phase]
            if phase_elements:
                names = [e.get('name', '') for e in phase_elements[:3]]
                phase_summaries[phase] = ', '.join(names)
            else:
                phase_summaries[phase] = f"[Describe {phase}]"

        # Infer position and fixation
        position = "supine"
        fixation = "Mayfield 3-point fixation"

        for elem in elements:
            name_lower = elem.get('name', '').lower()
            if 'prone' in name_lower:
                position = "prone"
            elif 'lateral' in name_lower:
                position = "lateral decubitus"
            elif 'park bench' in name_lower:
                position = "park bench"

        return self.DICTATION_TEMPLATE.format(
            procedure_name=procedure_name,
            position=position,
            fixation=fixation,
            architecture_summary=phase_summaries.get('architecture', ''),
            approach_summary=phase_summaries.get('approach', ''),
            target_summary=phase_summaries.get('target', ''),
            closure_summary=phase_summaries.get('closure', '')
        )

    def _infer_corridor(
        self,
        surgical_approach: str,
        elements: List[Dict[str, Any]]
    ) -> str:
        """Infer surgical corridor from approach and elements"""
        if not surgical_approach:
            return ""

        approach_lower = surgical_approach.lower()

        corridor_patterns = {
            'pterional': 'Sylvian fissure',
            'subfrontal': 'Frontobasal',
            'interhemispheric': 'Interhemispheric',
            'retrosigmoid': 'Cerebellopontine angle',
            'translabyrinthine': 'Through labyrinth',
            'transcallosal': 'Corpus callosum',
            'endonasal': 'Nasal corridor',
            'transsphenoidal': 'Sphenoid sinus',
            'suboccipital': 'Posterior fossa',
            'far lateral': 'Transcondylar',
            'anterior': 'Anterior cervical',
            'posterior': 'Posterior',
            'lateral': 'Lateral',
        }

        for pattern, corridor in corridor_patterns.items():
            if pattern in approach_lower:
                return corridor

        return surgical_approach

    def export_to_markdown(self, card: Dict[str, Any]) -> str:
        """Export surgical card to markdown format"""
        lines = []

        # Header
        lines.append(f"# {card.get('title', 'Surgical Card')}")
        lines.append(f"*{card.get('subtitle', '')}*")
        lines.append("")
        lines.append(f"**Approach:** {card.get('approach', '')}")
        lines.append(f"**Corridor:** {card.get('corridor', '')}")
        lines.append(f"**Exam Relevance:** {card.get('exam_relevance', '')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Phase rows
        for row in card.get('card_rows', []):
            lines.append(f"## {row.get('phase_label', '')}")
            lines.append("")
            for action in row.get('key_actions', []):
                lines.append(f"- {action}")
            lines.append("")
            if row.get('anchor_or_csp'):
                lines.append(f"**{row.get('anchor_or_csp')}**")
            lines.append("")

        # CSP Summary
        if card.get('csp_summary'):
            lines.append("---")
            lines.append("")
            lines.append("## CSP Summary")
            lines.append("")
            lines.append("| CSP | Trigger | Structure at Risk |")
            lines.append("|-----|---------|-------------------|")
            for csp in card.get('csp_summary', []):
                lines.append(f"| {csp.get('short_name', '')} | {csp.get('trigger', '')} | {csp.get('structure', '')} |")
            lines.append("")

        # Mantra
        lines.append("---")
        lines.append("")
        lines.append(f"**Mantra:** *{card.get('mantra', '')}*")

        return "\n".join(lines)

    def export_to_html(self, card: Dict[str, Any]) -> str:
        """Export surgical card to HTML format for printing"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{card.get('title', 'Surgical Card')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 20px; }}
        .meta {{ color: #7f8c8d; margin-bottom: 20px; }}
        .phase {{ background: #ecf0f1; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }}
        .phase h3 {{ margin-top: 0; color: #2c3e50; }}
        .actions {{ list-style-type: none; padding-left: 0; }}
        .actions li {{ padding: 5px 0; }}
        .anchor {{ font-weight: bold; color: #e74c3c; margin-top: 10px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        .mantra {{ font-style: italic; color: #7f8c8d; text-align: center; margin-top: 30px; }}
        @media print {{ body {{ max-width: 100%; }} }}
    </style>
</head>
<body>
    <h1>{card.get('title', 'Surgical Card')}</h1>
    <div class="meta">
        <p><strong>Approach:</strong> {card.get('approach', '')} | <strong>Corridor:</strong> {card.get('corridor', '')}</p>
        <p><strong>Exam Relevance:</strong> {card.get('exam_relevance', '')}</p>
    </div>
"""

        # Phase rows
        for row in card.get('card_rows', []):
            actions_html = "".join([f"<li>{action}</li>" for action in row.get('key_actions', [])])
            html += f"""
    <div class="phase">
        <h3>{row.get('phase_label', '')}</h3>
        <ul class="actions">{actions_html}</ul>
        <div class="anchor">{row.get('anchor_or_csp', '')}</div>
    </div>
"""

        # CSP Summary
        if card.get('csp_summary'):
            html += """
    <h2>CSP Summary</h2>
    <table>
        <tr><th>CSP</th><th>Trigger</th><th>Structure at Risk</th></tr>
"""
            for csp in card.get('csp_summary', []):
                html += f"        <tr><td>{csp.get('short_name', '')}</td><td>{csp.get('trigger', '')}</td><td>{csp.get('structure', '')}</td></tr>\n"
            html += "    </table>\n"

        # Mantra
        html += f"""
    <div class="mantra">{card.get('mantra', '')}</div>
</body>
</html>
"""

        return html
