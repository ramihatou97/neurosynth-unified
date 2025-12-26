"""
NeuroSynth - Prompt Library Unit Tests
=======================================

Tests for medical prompt templates.
"""

import pytest


# =============================================================================
# PromptLibrary Tests
# =============================================================================

class TestPromptLibrary:
    """Tests for PromptLibrary."""
    
    def test_get_system_prompt_procedural(self):
        """Get procedural system prompt."""
        from src.rag.prompts import PromptLibrary, QuestionType
        
        library = PromptLibrary()
        prompt = library.get_system_prompt(QuestionType.PROCEDURAL)
        
        assert "neurosurgical" in prompt.lower() or "surgical" in prompt.lower()
        assert "step" in prompt.lower()
    
    def test_get_system_prompt_anatomical(self):
        """Get anatomical system prompt."""
        from src.rag.prompts import PromptLibrary, QuestionType
        
        library = PromptLibrary()
        prompt = library.get_system_prompt(QuestionType.ANATOMICAL)
        
        assert "anatomy" in prompt.lower() or "anatomical" in prompt.lower()
    
    def test_get_system_prompt_general(self):
        """Get general system prompt as default."""
        from src.rag.prompts import PromptLibrary, QuestionType
        
        library = PromptLibrary()
        prompt = library.get_system_prompt(QuestionType.GENERAL)
        
        assert len(prompt) > 0
    
    def test_format_question_procedure_steps(self):
        """Format procedure steps template."""
        from src.rag.prompts import PromptLibrary
        
        library = PromptLibrary()
        question = library.format_question(
            "procedure_steps",
            procedure="retrosigmoid craniotomy"
        )
        
        assert "retrosigmoid craniotomy" in question
        assert "step" in question.lower()
    
    def test_format_question_anatomy_structure(self):
        """Format anatomy structure template."""
        from src.rag.prompts import PromptLibrary
        
        library = PromptLibrary()
        question = library.format_question(
            "anatomy_structure",
            structure="facial nerve"
        )
        
        assert "facial nerve" in question
        assert "anatomy" in question.lower()
    
    def test_format_question_compare_approaches(self):
        """Format compare approaches template."""
        from src.rag.prompts import PromptLibrary
        
        library = PromptLibrary()
        question = library.format_question(
            "compare_approaches",
            approach1="retrosigmoid",
            approach2="translabyrinthine",
            indication="acoustic neuroma"
        )
        
        assert "retrosigmoid" in question
        assert "translabyrinthine" in question
    
    def test_format_question_unknown_template(self):
        """Unknown template raises error."""
        from src.rag.prompts import PromptLibrary
        
        library = PromptLibrary()
        
        with pytest.raises(ValueError):
            library.format_question("unknown_template", foo="bar")
    
    def test_detect_question_type_procedural(self):
        """Detect procedural question type."""
        from src.rag.prompts import PromptLibrary, QuestionType
        
        library = PromptLibrary()
        
        questions = [
            "How do I perform a retrosigmoid approach?",
            "What are the steps for craniotomy?",
            "Describe the surgical technique for tumor resection"
        ]
        
        for q in questions:
            qtype = library.detect_question_type(q)
            assert qtype == QuestionType.PROCEDURAL
    
    def test_detect_question_type_anatomical(self):
        """Detect anatomical question type."""
        from src.rag.prompts import PromptLibrary, QuestionType
        
        library = PromptLibrary()
        
        questions = [
            "What is the anatomy of the facial nerve?",
            "Describe the blood supply to the brainstem",
            "What structures are near the cavernous sinus?"
        ]
        
        for q in questions:
            qtype = library.detect_question_type(q)
            assert qtype == QuestionType.ANATOMICAL
    
    def test_detect_question_type_clinical(self):
        """Detect clinical question type."""
        from src.rag.prompts import PromptLibrary, QuestionType
        
        library = PromptLibrary()
        
        questions = [
            "What is the prognosis for this patient?",
            "How should we manage this condition?",
            "What are the treatment options?"
        ]
        
        for q in questions:
            qtype = library.detect_question_type(q)
            assert qtype == QuestionType.CLINICAL
    
    def test_detect_question_type_comparative(self):
        """Detect comparative question type."""
        from src.rag.prompts import PromptLibrary, QuestionType
        
        library = PromptLibrary()
        
        questions = [
            "Compare retrosigmoid versus translabyrinthine",
            "What is the difference between these approaches?",
            "Which approach is better for large tumors?"
        ]
        
        for q in questions:
            qtype = library.detect_question_type(q)
            assert qtype == QuestionType.COMPARATIVE
    
    def test_detect_question_type_general(self):
        """Detect general question type for ambiguous queries."""
        from src.rag.prompts import PromptLibrary, QuestionType
        
        library = PromptLibrary()
        
        questions = [
            "Tell me about vestibular schwannoma",
            "What is this?"
        ]
        
        for q in questions:
            qtype = library.detect_question_type(q)
            assert qtype == QuestionType.GENERAL
    
    def test_prepare_prompt(self):
        """Prepare complete prompt."""
        from src.rag.prompts import PromptLibrary
        
        library = PromptLibrary()
        
        prompt = library.prepare_prompt(
            question="How do I perform this surgery?",
            context="[1] Step one is positioning..."
        )
        
        assert prompt.question_type.value == "procedural"
        assert "context" in prompt.user_prompt.lower()
        assert "[1]" in prompt.user_prompt
    
    def test_list_templates(self):
        """List all templates."""
        from src.rag.prompts import PromptLibrary
        
        library = PromptLibrary()
        templates = library.list_templates()
        
        assert "procedure_steps" in templates
        assert "anatomy_structure" in templates
        assert len(templates) > 10
    
    def test_add_custom_template(self):
        """Add custom template."""
        from src.rag.prompts import PromptLibrary
        
        library = PromptLibrary()
        library.add_template(
            "custom_template",
            "Custom question about {topic}"
        )
        
        question = library.format_question(
            "custom_template",
            topic="neurosurgery"
        )
        
        assert "neurosurgery" in question


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_get_medical_system_prompt(self):
        """Get medical system prompt by string."""
        from src.rag.prompts import get_medical_system_prompt
        
        prompt = get_medical_system_prompt("procedural")
        assert len(prompt) > 0
        
        prompt = get_medical_system_prompt("unknown")
        assert len(prompt) > 0  # Falls back to general
    
    def test_format_question_function(self):
        """Format question convenience function."""
        from src.rag.prompts import format_question
        
        question = format_question(
            "procedure_steps",
            procedure="test procedure"
        )
        
        assert "test procedure" in question


# =============================================================================
# QuestionType Enum Tests
# =============================================================================

class TestQuestionType:
    """Tests for QuestionType enum."""
    
    def test_all_types_have_prompts(self):
        """All question types have system prompts."""
        from src.rag.prompts import PromptLibrary, QuestionType
        
        library = PromptLibrary()
        
        for qtype in QuestionType:
            prompt = library.get_system_prompt(qtype)
            assert len(prompt) > 100, f"Prompt too short for {qtype}"
    
    def test_question_type_values(self):
        """QuestionType enum values."""
        from src.rag.prompts import QuestionType
        
        assert QuestionType.PROCEDURAL.value == "procedural"
        assert QuestionType.ANATOMICAL.value == "anatomical"
        assert QuestionType.CLINICAL.value == "clinical"
        assert QuestionType.DIFFERENTIAL.value == "differential"
        assert QuestionType.COMPARATIVE.value == "comparative"
        assert QuestionType.EDUCATIONAL.value == "educational"
        assert QuestionType.GENERAL.value == "general"
