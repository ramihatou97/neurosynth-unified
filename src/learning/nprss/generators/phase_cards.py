# src/learning/nprss/generators/phase_cards.py
"""
Phase-Based Flashcard Generator

Generates detailed cards for individual surgical phases.

Card Format:
- Front: "Describe the [Phase Name] phase of [Procedure]"
- Back: "[Description] + Key Steps + Critical Structures"

Example:
    Front: "Describe the tumor resection phase of retrosigmoid acoustic neuroma surgery"
    Back: "DESCRIPTION: Internal debulking followed by capsular dissection
           KEY STEPS:
           - Ultrasonic aspiration of tumor core
           - Preserve arachnoid plane
           - Identify facial nerve on anterior capsule

           CRITICAL STRUCTURES: CN VII, CN VIII, AICA

           PEARLS: Debulk centrally first, preserve capsule for traction"
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from uuid import uuid4

from ..models import LearningCard, CardType

logger = logging.getLogger(__name__)


# =============================================================================
# PHASE TYPE TEMPLATES
# =============================================================================

PHASE_CARD_TEMPLATES = {
    'positioning': {
        'main': "Describe the positioning for {procedure_name}",
        'specifics': [
            "What patient position is used for {procedure_name}?",
            "What are the key positioning considerations for {procedure_name}?",
            "What positioning-related complications should you avoid in {procedure_name}?"
        ],
        'tags': ['positioning', 'setup']
    },
    'approach': {
        'main': "Describe the surgical approach for {procedure_name}",
        'specifics': [
            "What incision is used for {procedure_name}?",
            "What are the anatomical landmarks for the {procedure_name} approach?",
            "What muscles/structures must be divided in the approach?"
        ],
        'tags': ['approach', 'exposure']
    },
    'exposure': {
        'main': "Describe the exposure phase of {procedure_name}",
        'specifics': [
            "What is the bony exposure for {procedure_name}?",
            "What dural opening is performed?",
            "What structures are visualized after exposure?"
        ],
        'tags': ['exposure', 'craniotomy']
    },
    'resection': {
        'main': "Describe the resection phase of {procedure_name}",
        'specifics': [
            "What is the resection strategy for {procedure_name}?",
            "What planes of dissection are used?",
            "How do you handle tumor adherent to critical structures?"
        ],
        'tags': ['resection', 'technique']
    },
    'closure': {
        'main': "Describe the closure for {procedure_name}",
        'specifics': [
            "What dural closure technique is used?",
            "What are the layers of closure?",
            "What CSF management is performed?"
        ],
        'tags': ['closure', 'reconstruction']
    },
    'hemostasis': {
        'main': "Describe hemostasis techniques for {procedure_name}",
        'specifics': [
            "What hemostatic agents are commonly used?",
            "How do you manage venous bleeding?",
            "What are the critical bleeding points?"
        ],
        'tags': ['hemostasis', 'bleeding']
    },
    'nerve_preservation': {
        'main': "Describe nerve preservation in {procedure_name}",
        'specifics': [
            "Which nerves are at risk?",
            "What monitoring is used?",
            "What techniques preserve nerve function?"
        ],
        'tags': ['nerve', 'monitoring', 'preservation']
    },
    'vascular': {
        'main': "Describe vascular management in {procedure_name}",
        'specifics': [
            "Which vessels are encountered?",
            "How do you preserve perforators?",
            "What is your strategy for temporary occlusion?"
        ],
        'tags': ['vascular', 'vessels']
    },
    'default': {
        'main': "Describe the {phase_name} phase of {procedure_name}",
        'specifics': [
            "What are the key steps of the {phase_name} phase?",
            "What structures are encountered in this phase?",
            "What are the critical considerations?"
        ],
        'tags': ['phase']
    }
}


# =============================================================================
# PHASE ELEMENT MODEL
# =============================================================================

@dataclass
class PhaseElement:
    """
    Represents a surgical phase/step.
    Mirrors procedural_elements from NPRSS.
    """
    id: str = None
    procedure_id: str = None
    phase_name: str = ""
    phase_type: str = ""  # positioning, approach, exposure, etc.
    phase_number: int = 0
    description: str = ""
    key_steps: List[str] = None
    critical_structures: List[str] = None
    pitfalls: List[str] = None
    pearls: List[str] = None

    def __post_init__(self):
        if self.key_steps is None:
            self.key_steps = []
        if self.critical_structures is None:
            self.critical_structures = []
        if self.pitfalls is None:
            self.pitfalls = []
        if self.pearls is None:
            self.pearls = []


# =============================================================================
# PHASE CARD GENERATOR
# =============================================================================

class PhaseCardGenerator:
    """
    Generate flashcards from surgical phases.

    Creates cards for:
    - Phase overview (description + steps)
    - Critical structures in phase
    - Phase-specific pitfalls
    - Surgical pearls
    - Step sequences

    Usage:
        generator = PhaseCardGenerator()

        phase = PhaseElement(
            procedure_id="uuid",
            phase_name="Tumor Resection",
            phase_type="resection",
            description="Internal debulking followed by capsular dissection",
            key_steps=["Ultrasonic aspiration", "Preserve arachnoid", "Identify CN VII"],
            critical_structures=["CN VII", "CN VIII", "AICA"],
            pitfalls=["Traction injury to facial nerve"],
            pearls=["Debulk centrally first"]
        )

        cards = generator.generate_from_phase(phase, procedure_name="Retrosigmoid Acoustic")
    """

    def __init__(
        self,
        generate_specifics: bool = True,
        generate_step_cards: bool = True,
        max_cards_per_phase: int = 5
    ):
        """
        Initialize generator.

        Args:
            generate_specifics: Generate specific aspect cards
            generate_step_cards: Generate individual step sequence cards
            max_cards_per_phase: Maximum cards to generate per phase
        """
        self.generate_specifics = generate_specifics
        self.generate_step_cards = generate_step_cards
        self.max_cards_per_phase = max_cards_per_phase

    def generate_from_phase(
        self,
        phase: PhaseElement,
        procedure_name: str
    ) -> List[LearningCard]:
        """
        Generate flashcards from a surgical phase.

        Args:
            phase: PhaseElement to generate from
            procedure_name: Name of the procedure

        Returns:
            List of LearningCard objects
        """
        cards = []

        # Get template for phase type
        template = PHASE_CARD_TEMPLATES.get(
            phase.phase_type.lower() if phase.phase_type else 'default',
            PHASE_CARD_TEMPLATES['default']
        )

        # Card 1: Main phase overview
        overview_card = self._create_overview_card(phase, procedure_name, template)
        cards.append(overview_card)

        # Card 2: Key steps card (if steps exist)
        if phase.key_steps:
            steps_card = self._create_steps_card(phase, procedure_name)
            cards.append(steps_card)

        # Card 3: Critical structures card
        if phase.critical_structures:
            structures_card = self._create_structures_card(phase, procedure_name)
            cards.append(structures_card)

        # Card 4: Pitfalls card
        if phase.pitfalls:
            pitfalls_card = self._create_pitfalls_card(phase, procedure_name)
            cards.append(pitfalls_card)

        # Card 5: Pearls card
        if phase.pearls:
            pearls_card = self._create_pearls_card(phase, procedure_name)
            cards.append(pearls_card)

        # Generate specific question cards
        if self.generate_specifics:
            specific_cards = self._create_specific_cards(
                phase, procedure_name, template
            )
            cards.extend(specific_cards)

        # Generate step sequence cards
        if self.generate_step_cards and len(phase.key_steps) >= 3:
            sequence_cards = self._create_step_sequence_cards(
                phase, procedure_name
            )
            cards.extend(sequence_cards)

        # Limit total cards
        return cards[:self.max_cards_per_phase]

    def _create_overview_card(
        self,
        phase: PhaseElement,
        procedure_name: str,
        template: Dict[str, Any]
    ) -> LearningCard:
        """Create main phase overview card."""
        question = template['main'].format(
            procedure_name=procedure_name,
            phase_name=phase.phase_name
        )

        # Build comprehensive answer
        answer_parts = []

        # Description
        answer_parts.append(f"**DESCRIPTION:**\n{phase.description}")

        # Key steps
        if phase.key_steps:
            steps_text = "\n".join([f"- {step}" for step in phase.key_steps])
            answer_parts.append(f"\n\n**KEY STEPS:**\n{steps_text}")

        # Critical structures
        if phase.critical_structures:
            structures = ", ".join(phase.critical_structures)
            answer_parts.append(f"\n\n**CRITICAL STRUCTURES:** {structures}")

        # Pearls
        if phase.pearls:
            pearls_text = "\n".join([f"- {pearl}" for pearl in phase.pearls[:2]])
            answer_parts.append(f"\n\n**PEARLS:**\n{pearls_text}")

        answer = "".join(answer_parts)

        # Build tags
        tags = list(template.get('tags', []))
        tags.extend(['phase_overview', phase.phase_type or 'general'])

        return LearningCard(
            procedure_id=phase.procedure_id,
            element_id=phase.id,
            card_type=CardType.DICTATION,
            prompt=question,
            answer=answer,
            explanation=f"Phase {phase.phase_number}: {phase.phase_name}",
            difficulty_preset=0.5,
            tags=tags,
            generation_method='phase_based',
            quality_score=0.8
        )

    def _create_steps_card(
        self,
        phase: PhaseElement,
        procedure_name: str
    ) -> LearningCard:
        """Create key steps recall card."""
        question = f"What are the key steps of {phase.phase_name} in {procedure_name}?"

        steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(phase.key_steps)])
        answer = f"**KEY STEPS:**\n{steps_text}"

        return LearningCard(
            procedure_id=phase.procedure_id,
            element_id=phase.id,
            card_type=CardType.SEQUENCE,
            prompt=question,
            answer=answer,
            difficulty_preset=0.6,
            tags=['steps', 'sequence', phase.phase_type or 'general'],
            generation_method='phase_based',
            quality_score=0.85
        )

    def _create_structures_card(
        self,
        phase: PhaseElement,
        procedure_name: str
    ) -> LearningCard:
        """Create critical structures card."""
        question = f"What are the critical structures encountered during {phase.phase_name} of {procedure_name}?"

        structures_text = "\n".join([f"- {s}" for s in phase.critical_structures])
        answer = f"**CRITICAL STRUCTURES:**\n{structures_text}"

        if phase.pitfalls:
            # Add relevant pitfall as warning
            answer += f"\n\n**WATCH FOR:** {phase.pitfalls[0]}"

        return LearningCard(
            procedure_id=phase.procedure_id,
            element_id=phase.id,
            card_type=CardType.MCQ,
            prompt=question,
            answer=answer,
            difficulty_preset=0.55,
            tags=['anatomy', 'critical_structures', phase.phase_type or 'general'],
            generation_method='phase_based',
            quality_score=0.8
        )

    def _create_pitfalls_card(
        self,
        phase: PhaseElement,
        procedure_name: str
    ) -> LearningCard:
        """Create pitfalls/complications card."""
        question = f"What are the pitfalls to avoid during {phase.phase_name} of {procedure_name}?"

        pitfalls_text = "\n".join([f"- {p}" for p in phase.pitfalls])
        answer = f"**PITFALLS:**\n{pitfalls_text}"

        if phase.pearls:
            # Add prevention strategy
            answer += f"\n\n**PREVENTION:** {phase.pearls[0]}"

        return LearningCard(
            procedure_id=phase.procedure_id,
            element_id=phase.id,
            card_type=CardType.CSP_TRIGGER,
            prompt=question,
            answer=answer,
            difficulty_preset=0.7,
            tags=['pitfalls', 'safety', 'complications'],
            generation_method='phase_based',
            quality_score=0.9  # High priority safety content
        )

    def _create_pearls_card(
        self,
        phase: PhaseElement,
        procedure_name: str
    ) -> LearningCard:
        """Create surgical pearls card."""
        question = f"What are the surgical pearls for {phase.phase_name} in {procedure_name}?"

        pearls_text = "\n".join([f"- {p}" for p in phase.pearls])
        answer = f"**SURGICAL PEARLS:**\n{pearls_text}"

        return LearningCard(
            procedure_id=phase.procedure_id,
            element_id=phase.id,
            card_type=CardType.DICTATION,
            prompt=question,
            answer=answer,
            difficulty_preset=0.5,
            tags=['pearls', 'tips', 'technique'],
            generation_method='phase_based',
            quality_score=0.75
        )

    def _create_specific_cards(
        self,
        phase: PhaseElement,
        procedure_name: str,
        template: Dict[str, Any]
    ) -> List[LearningCard]:
        """Create cards for specific aspects."""
        cards = []

        specific_questions = template.get('specifics', [])

        for question_template in specific_questions[:2]:  # Limit to 2 specifics
            question = question_template.format(
                procedure_name=procedure_name,
                phase_name=phase.phase_name
            )

            # Generate contextual answer from phase data
            answer = self._generate_specific_answer(phase, question_template)

            if answer:
                cards.append(LearningCard(
                    procedure_id=phase.procedure_id,
                    element_id=phase.id,
                    card_type=CardType.MCQ,
                    prompt=question,
                    answer=answer,
                    difficulty_preset=0.6,
                    tags=['specific', phase.phase_type or 'general'],
                    generation_method='phase_based',
                    quality_score=0.7
                ))

        return cards

    def _create_step_sequence_cards(
        self,
        phase: PhaseElement,
        procedure_name: str
    ) -> List[LearningCard]:
        """Create step sequence cards for ordering practice."""
        cards = []

        if len(phase.key_steps) < 3:
            return cards

        # Card: What comes after step X?
        for i in range(min(2, len(phase.key_steps) - 1)):
            question = f"In the {phase.phase_name} phase of {procedure_name}, what step comes after: '{phase.key_steps[i]}'?"
            answer = phase.key_steps[i + 1]

            cards.append(LearningCard(
                procedure_id=phase.procedure_id,
                element_id=phase.id,
                card_type=CardType.SEQUENCE,
                prompt=question,
                answer=answer,
                explanation=f"Step {i+2} of {len(phase.key_steps)}",
                difficulty_preset=0.65,
                tags=['sequence', 'ordering'],
                generation_method='phase_based',
                quality_score=0.75
            ))

        return cards

    def _generate_specific_answer(
        self,
        phase: PhaseElement,
        question_template: str
    ) -> Optional[str]:
        """Generate answer for specific question from phase data."""
        question_lower = question_template.lower()

        # Match question to relevant data
        if 'position' in question_lower:
            if phase.description and 'position' in phase.description.lower():
                return phase.description

        if 'incision' in question_lower or 'landmark' in question_lower:
            if phase.key_steps:
                return phase.key_steps[0]

        if 'structure' in question_lower:
            if phase.critical_structures:
                return ", ".join(phase.critical_structures)

        if 'complication' in question_lower or 'avoid' in question_lower:
            if phase.pitfalls:
                return phase.pitfalls[0]

        # Default: use description
        return phase.description if phase.description else None

    def generate_batch(
        self,
        phases: List[PhaseElement],
        procedure_name: str
    ) -> List[LearningCard]:
        """
        Generate cards for multiple phases.

        Args:
            phases: List of phase elements
            procedure_name: Procedure name

        Returns:
            All generated cards
        """
        all_cards = []

        for phase in phases:
            cards = self.generate_from_phase(phase, procedure_name)
            all_cards.extend(cards)

        return all_cards


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def generate_phase_cards(
    phase_name: str,
    phase_type: str,
    procedure_name: str,
    description: str,
    key_steps: List[str] = None,
    critical_structures: List[str] = None,
    pitfalls: List[str] = None,
    pearls: List[str] = None
) -> List[LearningCard]:
    """
    Convenience function to generate phase cards.

    Args:
        phase_name: Name of the phase
        phase_type: Type (positioning, approach, etc.)
        procedure_name: Name of procedure
        description: Phase description
        key_steps: List of key steps
        critical_structures: Critical anatomical structures
        pitfalls: Potential pitfalls
        pearls: Surgical pearls

    Returns:
        List of LearningCard objects
    """
    generator = PhaseCardGenerator()

    phase = PhaseElement(
        id=str(uuid4()),
        phase_name=phase_name,
        phase_type=phase_type,
        description=description,
        key_steps=key_steps or [],
        critical_structures=critical_structures or [],
        pitfalls=pitfalls or [],
        pearls=pearls or []
    )

    return generator.generate_from_phase(phase, procedure_name)
