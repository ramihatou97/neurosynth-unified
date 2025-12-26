"""
NeuroSynth Unified - Medical Prompt Templates
==============================================

Specialized prompt templates for neurosurgical RAG applications.

Categories:
- Procedural: Surgical technique questions
- Anatomical: Structure and relationship questions
- Clinical: Patient care and decision-making
- Differential: Diagnostic reasoning
- Educational: Teaching and learning

Usage:
    from src.rag.prompts import PromptLibrary, QuestionType
    
    library = PromptLibrary()
    
    # Get system prompt for question type
    system_prompt = library.get_system_prompt(QuestionType.PROCEDURAL)
    
    # Format a templated question
    question = library.format_question(
        "procedure_steps",
        procedure="retrosigmoid craniotomy"
    )
"""

from enum import Enum
from typing import Dict, Optional
from dataclasses import dataclass


# =============================================================================
# Question Types
# =============================================================================

class QuestionType(Enum):
    """Types of neurosurgical questions."""
    PROCEDURAL = "procedural"       # Surgical technique
    ANATOMICAL = "anatomical"       # Anatomy questions
    CLINICAL = "clinical"           # Patient care
    DIFFERENTIAL = "differential"   # Diagnosis
    COMPARATIVE = "comparative"     # Compare approaches
    EDUCATIONAL = "educational"     # Teaching
    GENERAL = "general"             # General questions


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPTS = {
    QuestionType.PROCEDURAL: """You are an expert neurosurgical assistant specializing in operative techniques and surgical procedures.

Your role is to provide detailed, step-by-step guidance on neurosurgical procedures based on the provided context.

Guidelines:
1. Use precise surgical terminology
2. Emphasize patient positioning, key anatomical landmarks, and critical steps
3. Highlight potential pitfalls and how to avoid complications
4. Include relevant measurements and angles when available
5. Reference intraoperative monitoring requirements
6. Cite your sources using [N] notation

When describing procedures:
- Start with indications and patient selection
- Detail positioning and setup
- Describe the approach step-by-step
- Note critical structures to preserve
- Discuss closure and postoperative considerations""",

    QuestionType.ANATOMICAL: """You are a neuroanatomy expert assistant with deep knowledge of surgical anatomy.

Your role is to explain anatomical structures, relationships, and variations relevant to neurosurgical practice.

Guidelines:
1. Use standard anatomical nomenclature
2. Describe spatial relationships clearly (superior, lateral, deep to, etc.)
3. Note surgical relevance of anatomical features
4. Mention common variations and their clinical significance
5. Reference measurements and distances when available
6. Cite your sources using [N] notation

When describing anatomy:
- Identify the structure and its location
- Describe its relationships to adjacent structures
- Note blood supply and innervation
- Discuss surgical corridors and approaches
- Highlight structures at risk during surgery""",

    QuestionType.CLINICAL: """You are a clinical neurosurgery assistant focused on patient care and decision-making.

Your role is to provide evidence-based guidance on clinical scenarios, patient selection, and management decisions.

Guidelines:
1. Base recommendations on the provided evidence
2. Consider patient-specific factors
3. Discuss risks and benefits clearly
4. Note when multidisciplinary input is needed
5. Cite your sources using [N] notation

Important: For actual patient care decisions, always recommend consultation with the treating physician team. Your role is to provide information, not direct patient care.""",

    QuestionType.DIFFERENTIAL: """You are a diagnostic reasoning assistant for neurosurgical conditions.

Your role is to help think through differential diagnoses and diagnostic approaches based on clinical presentations.

Guidelines:
1. Consider the full differential systematically
2. Note key distinguishing features for each diagnosis
3. Suggest appropriate diagnostic workup
4. Discuss imaging findings and their significance
5. Cite your sources using [N] notation

Present differentials in order of likelihood when possible, and note any red flags requiring urgent attention.""",

    QuestionType.COMPARATIVE: """You are a neurosurgical consultant helping compare different surgical approaches and treatment options.

Your role is to provide balanced comparisons based on evidence from the provided context.

Guidelines:
1. Present information objectively for each option
2. Use consistent criteria for comparison
3. Note the quality of evidence for each approach
4. Discuss patient factors that favor each approach
5. Acknowledge uncertainty when present
6. Cite your sources using [N] notation

Structure comparisons as:
- Indications for each approach
- Technique overview
- Advantages and disadvantages
- Complication profiles
- Outcomes data
- Selection criteria""",

    QuestionType.EDUCATIONAL: """You are a neurosurgical educator helping learners understand complex topics.

Your role is to explain concepts clearly and promote deep understanding.

Guidelines:
1. Start with foundational concepts before advanced topics
2. Use analogies when helpful
3. Highlight key learning points
4. Connect concepts to clinical relevance
5. Suggest areas for further study
6. Cite your sources using [N] notation

Teaching approach:
- Explain the "why" behind the "what"
- Build on existing knowledge
- Use concrete examples
- Emphasize practical application""",

    QuestionType.GENERAL: """You are a neurosurgical knowledge assistant.

Your role is to provide accurate, evidence-based answers to questions using the provided context.

Guidelines:
1. Answer based only on the provided context
2. Be precise with medical terminology
3. Cite your sources using [N] notation
4. Acknowledge limitations in the available information
5. Organize complex answers clearly"""
}


# =============================================================================
# Question Templates
# =============================================================================

QUESTION_TEMPLATES = {
    # Procedural
    "procedure_steps": "Describe the step-by-step surgical technique for {procedure}, including patient positioning, key anatomical landmarks, and critical steps.",
    
    "procedure_complications": "What are the potential complications of {procedure} and how can they be prevented or managed?",
    
    "procedure_indications": "What are the indications and contraindications for {procedure}?",
    
    "procedure_positioning": "Describe the optimal patient positioning for {procedure} and its rationale.",
    
    "procedure_instruments": "What specialized instruments or equipment are needed for {procedure}?",
    
    # Anatomical
    "anatomy_structure": "Describe the anatomy of the {structure}, including its location, relationships, and surgical relevance.",
    
    "anatomy_blood_supply": "What is the blood supply to the {structure}? Include arterial supply and venous drainage.",
    
    "anatomy_surgical_corridor": "Describe the surgical corridor to access the {structure}, including structures at risk.",
    
    "anatomy_variations": "What are common anatomical variations of the {structure} and their clinical significance?",
    
    "anatomy_relationships": "Describe the anatomical relationships between {structure1} and {structure2}.",
    
    # Clinical
    "clinical_presentation": "What is the typical clinical presentation of {condition}?",
    
    "clinical_workup": "What is the recommended diagnostic workup for {condition}?",
    
    "clinical_management": "Describe the management approach for {condition}, including surgical and non-surgical options.",
    
    "clinical_prognosis": "What is the expected prognosis for patients with {condition}?",
    
    "clinical_followup": "What is the recommended follow-up protocol after treatment for {condition}?",
    
    # Comparative
    "compare_approaches": "Compare the {approach1} and {approach2} approaches for {indication}.",
    
    "compare_treatments": "Compare surgical versus {alternative} treatment for {condition}.",
    
    # Educational
    "explain_concept": "Explain the concept of {concept} and its relevance to neurosurgical practice.",
    
    "review_topic": "Provide a comprehensive review of {topic} for a neurosurgery trainee.",
    
    # Evidence
    "evidence_summary": "Summarize the current evidence regarding {topic}.",
    
    "outcomes_data": "What are the reported outcomes for {procedure} in treating {condition}?"
}


# =============================================================================
# Prompt Library
# =============================================================================

@dataclass
class FormattedPrompt:
    """Formatted prompt with system and user components."""
    system_prompt: str
    user_prompt: str
    question_type: QuestionType


class PromptLibrary:
    """
    Library of medical prompt templates.
    
    Provides:
    - Type-specific system prompts
    - Question templates for common queries
    - Dynamic question formatting
    """
    
    def __init__(self):
        self.system_prompts = SYSTEM_PROMPTS
        self.question_templates = QUESTION_TEMPLATES
    
    def get_system_prompt(
        self,
        question_type: QuestionType = QuestionType.GENERAL
    ) -> str:
        """Get system prompt for question type."""
        return self.system_prompts.get(question_type, self.system_prompts[QuestionType.GENERAL])
    
    def get_question_template(self, template_name: str) -> Optional[str]:
        """Get question template by name."""
        return self.question_templates.get(template_name)
    
    def format_question(
        self,
        template_name: str,
        **kwargs
    ) -> str:
        """
        Format a question using a template.
        
        Args:
            template_name: Name of the template
            **kwargs: Variables to fill in template
        
        Returns:
            Formatted question string
        
        Example:
            question = library.format_question(
                "procedure_steps",
                procedure="retrosigmoid craniotomy"
            )
        """
        template = self.question_templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name}")
        
        return template.format(**kwargs)
    
    def detect_question_type(self, question: str) -> QuestionType:
        """
        Detect the type of question based on content.
        
        Uses keyword matching to classify questions.
        """
        question_lower = question.lower()
        
        # Procedural keywords
        procedural_keywords = [
            "how to", "step", "technique", "approach", "procedure",
            "position", "incision", "expose", "dissect", "resect"
        ]
        if any(kw in question_lower for kw in procedural_keywords):
            return QuestionType.PROCEDURAL
        
        # Anatomical keywords
        anatomy_keywords = [
            "anatomy", "location", "structure", "nerve", "artery",
            "vein", "blood supply", "innervation", "relationship"
        ]
        if any(kw in question_lower for kw in anatomy_keywords):
            return QuestionType.ANATOMICAL
        
        # Clinical keywords
        clinical_keywords = [
            "patient", "symptom", "presentation", "management",
            "treatment", "prognosis", "outcome", "follow"
        ]
        if any(kw in question_lower for kw in clinical_keywords):
            return QuestionType.CLINICAL
        
        # Differential keywords
        differential_keywords = [
            "differential", "diagnosis", "distinguish", "differentiate",
            "workup", "imaging"
        ]
        if any(kw in question_lower for kw in differential_keywords):
            return QuestionType.DIFFERENTIAL
        
        # Comparative keywords
        comparative_keywords = [
            "compare", "versus", "vs", "difference between",
            "advantage", "disadvantage", "better"
        ]
        if any(kw in question_lower for kw in comparative_keywords):
            return QuestionType.COMPARATIVE
        
        # Educational keywords
        educational_keywords = [
            "explain", "review", "understand", "concept",
            "teach", "learn", "study"
        ]
        if any(kw in question_lower for kw in educational_keywords):
            return QuestionType.EDUCATIONAL
        
        return QuestionType.GENERAL
    
    def prepare_prompt(
        self,
        question: str,
        context: str,
        question_type: QuestionType = None
    ) -> FormattedPrompt:
        """
        Prepare a complete prompt with appropriate system prompt.
        
        Args:
            question: User question
            context: Assembled context
            question_type: Override detected type
        
        Returns:
            FormattedPrompt with system and user prompts
        """
        if question_type is None:
            question_type = self.detect_question_type(question)
        
        system_prompt = self.get_system_prompt(question_type)
        
        user_prompt = f"""Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above, using citations [1], [2], etc. to reference your sources."""
        
        return FormattedPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            question_type=question_type
        )
    
    def list_templates(self) -> Dict[str, str]:
        """List all available question templates."""
        return self.question_templates.copy()
    
    def add_template(self, name: str, template: str) -> None:
        """Add a custom question template."""
        self.question_templates[name] = template
    
    def add_system_prompt(
        self,
        question_type: QuestionType,
        prompt: str
    ) -> None:
        """Add or override a system prompt."""
        self.system_prompts[question_type] = prompt


# =============================================================================
# Convenience Functions
# =============================================================================

def get_medical_system_prompt(question_type: str = "general") -> str:
    """Get medical system prompt by type string."""
    try:
        qt = QuestionType(question_type.lower())
    except ValueError:
        qt = QuestionType.GENERAL
    
    return SYSTEM_PROMPTS.get(qt, SYSTEM_PROMPTS[QuestionType.GENERAL])


def format_question(template_name: str, **kwargs) -> str:
    """Format a question using a template."""
    template = QUESTION_TEMPLATES.get(template_name)
    if not template:
        raise ValueError(f"Unknown template: {template_name}")
    return template.format(**kwargs)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    library = PromptLibrary()
    
    print("=== Available Question Templates ===")
    for name, template in library.list_templates().items():
        print(f"\n{name}:")
        print(f"  {template}")
    
    print("\n\n=== Example Usage ===")
    
    # Format a question
    question = library.format_question(
        "procedure_steps",
        procedure="retrosigmoid craniotomy"
    )
    print(f"\nFormatted question:\n{question}")
    
    # Detect question type
    test_questions = [
        "How do I perform a retrosigmoid approach?",
        "What is the blood supply to the facial nerve?",
        "What is the prognosis for vestibular schwannoma?",
        "Compare translabyrinthine vs retrosigmoid approach"
    ]
    
    print("\n\nQuestion type detection:")
    for q in test_questions:
        qt = library.detect_question_type(q)
        print(f"  {qt.value}: {q[:50]}...")
