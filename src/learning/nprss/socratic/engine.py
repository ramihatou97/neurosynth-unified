# src/learning/nprss/socratic/engine.py
"""
Socratic Learning Engine

Implements guided learning through questioning.
Integrates with RAG retrieval to provide contextually-aware guidance.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from uuid import uuid4
from datetime import datetime
from enum import Enum

from .prompts import (
    SOCRATIC_SYSTEM_PROMPT,
    SOCRATIC_TEMPLATES,
    HINT_PROMPT,
    REVEAL_ANSWER_PROMPT,
    CORRECT_RESPONSE_PROMPT,
    PARTIAL_CORRECT_PROMPT,
    BEGINNER_ADAPTATION,
    ADVANCED_ADAPTATION
)

logger = logging.getLogger(__name__)


# =============================================================================
# MODELS
# =============================================================================

class ConversationState(str, Enum):
    """State of Socratic conversation"""
    QUESTIONING = "questioning"      # Still guiding with questions
    HINT_PROVIDED = "hint_provided"  # Gave a hint
    REVEALED = "revealed"            # Answered revealed
    COMPLETED = "completed"          # Student understood


@dataclass
class SocraticTurn:
    """Single turn in Socratic conversation"""
    role: str  # 'student' or 'guide'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocraticConversation:
    """Tracks a Socratic learning conversation"""
    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    original_question: str = ""
    topic: str = ""
    content_type: str = "default"
    turns: List[SocraticTurn] = field(default_factory=list)
    state: ConversationState = ConversationState.QUESTIONING
    hints_given: int = 0
    max_hints: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    context_chunks: List[Dict] = field(default_factory=list)

    def add_turn(self, role: str, content: str, **metadata):
        """Add a conversation turn"""
        self.turns.append(SocraticTurn(
            role=role,
            content=content,
            metadata=metadata
        ))

    def get_history(self) -> str:
        """Format conversation history for prompts"""
        if not self.turns:
            return "No previous conversation."

        history = []
        for turn in self.turns[-6:]:  # Last 6 turns
            prefix = "Student" if turn.role == "student" else "Guide"
            history.append(f"{prefix}: {turn.content}")

        return "\n".join(history)

    def get_student_attempts(self) -> List[str]:
        """Get all student responses"""
        return [
            turn.content
            for turn in self.turns
            if turn.role == "student"
        ]


# =============================================================================
# SOCRATIC ENGINE
# =============================================================================

class SocraticEngine:
    """
    Main engine for Socratic learning interactions.

    Usage:
        # With RAG integration
        from src.rag.engine import RAGEngine

        rag = RAGEngine(...)
        socratic = SocraticEngine(rag_engine=rag, llm_client=client)

        # Start conversation
        response = await socratic.ask(
            question="What supplies blood to the optic nerve?",
            user_id="student-123"
        )

        # Continue conversation
        response = await socratic.respond(
            conversation_id=response['conversation_id'],
            student_response="Is it the carotid artery?"
        )

        # Get hint
        hint = await socratic.get_hint(conversation_id=...)

        # Reveal answer
        answer = await socratic.reveal(conversation_id=...)
    """

    def __init__(
        self,
        rag_engine=None,
        llm_client=None,
        default_top_k: int = 5,
        max_hints: int = 3
    ):
        """
        Initialize Socratic engine.

        Args:
            rag_engine: RAG engine for context retrieval
            llm_client: LLM client (Anthropic, etc.)
            default_top_k: Default chunks to retrieve
            max_hints: Maximum hints before revealing answer
        """
        self.rag = rag_engine
        self.llm = llm_client
        self.default_top_k = default_top_k
        self.max_hints = max_hints

        # Store active conversations
        self._conversations: Dict[str, SocraticConversation] = {}

    async def ask(
        self,
        question: str,
        user_id: str = "default",
        content_type: str = None,
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Start a new Socratic conversation.

        Args:
            question: Student's question
            user_id: User identifier
            content_type: Type of content (anatomy, procedure, etc.)
            top_k: Number of context chunks to retrieve

        Returns:
            Dict with guiding questions and conversation_id
        """
        # Detect content type if not provided
        if content_type is None:
            content_type = self._detect_content_type(question)

        # Create conversation
        conversation = SocraticConversation(
            user_id=user_id,
            original_question=question,
            content_type=content_type,
            max_hints=self.max_hints
        )

        # Add student's question
        conversation.add_turn("student", question)

        # Retrieve context
        context = ""
        if self.rag:
            try:
                retrieval = await self.rag.retrieve(
                    query=question,
                    top_k=top_k or self.default_top_k,
                    filters={'chunk_type': self._map_content_type(content_type)}
                )
                chunks = retrieval.get('chunks', [])
                conversation.context_chunks = chunks
                context = self._format_context(chunks)
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                context = "No specific context available."

        # Generate Socratic response
        response = await self._generate_response(
            conversation=conversation,
            context=context,
            template_type=content_type
        )

        # Add guide response
        conversation.add_turn("guide", response)

        # Store conversation
        self._conversations[conversation.id] = conversation

        return {
            'response': response,
            'conversation_id': conversation.id,
            'mode': 'socratic',
            'content_type': content_type,
            'state': conversation.state.value,
            'sources': [self._format_source(c) for c in conversation.context_chunks[:3]]
        }

    async def respond(
        self,
        conversation_id: str,
        student_response: str
    ) -> Dict[str, Any]:
        """
        Continue Socratic conversation with student's response.

        Args:
            conversation_id: Active conversation ID
            student_response: Student's response/attempt

        Returns:
            Dict with next guiding questions or feedback
        """
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Add student response
        conversation.add_turn("student", student_response)

        # Check if answer is correct (simple heuristic)
        is_correct = await self._check_correctness(
            conversation, student_response
        )

        if is_correct:
            # Generate positive feedback
            response = await self._generate_correct_feedback(
                conversation, student_response
            )
            conversation.state = ConversationState.COMPLETED
        else:
            # Generate follow-up questions
            context = self._format_context(conversation.context_chunks)
            response = await self._generate_response(
                conversation=conversation,
                context=context,
                template_type=conversation.content_type
            )

        # Add guide response
        conversation.add_turn("guide", response)

        return {
            'response': response,
            'conversation_id': conversation_id,
            'state': conversation.state.value,
            'is_correct': is_correct,
            'turns': len(conversation.turns)
        }

    async def get_hint(
        self,
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Provide a hint for stuck student.

        Args:
            conversation_id: Active conversation ID

        Returns:
            Dict with hint
        """
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Check hint limit
        if conversation.hints_given >= conversation.max_hints:
            return await self.reveal(conversation_id)

        conversation.hints_given += 1
        conversation.state = ConversationState.HINT_PROVIDED

        # Get student's last response
        student_attempts = conversation.get_student_attempts()
        last_response = student_attempts[-1] if student_attempts else ""

        # Generate hint
        context = self._format_context(conversation.context_chunks)

        prompt = HINT_PROMPT.format(
            context=context,
            question=conversation.original_question,
            attempts="\n".join(student_attempts[:-1]) if len(student_attempts) > 1 else "None yet",
            student_response=last_response
        )

        hint = await self._call_llm(
            system_prompt=SOCRATIC_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.6
        )

        conversation.add_turn("guide", f"[HINT] {hint}", hint_number=conversation.hints_given)

        return {
            'hint': hint,
            'conversation_id': conversation_id,
            'hints_remaining': conversation.max_hints - conversation.hints_given,
            'state': conversation.state.value
        }

    async def reveal(
        self,
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Reveal the answer after attempts.

        Args:
            conversation_id: Active conversation ID

        Returns:
            Dict with full answer and explanation
        """
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")

        conversation.state = ConversationState.REVEALED

        # Format student attempts
        attempts = conversation.get_student_attempts()

        # Generate full answer
        context = self._format_context(conversation.context_chunks)

        prompt = REVEAL_ANSWER_PROMPT.format(
            context=context,
            question=conversation.original_question,
            attempts="\n".join([f"- {a}" for a in attempts]) if attempts else "No attempts made"
        )

        answer = await self._call_llm(
            system_prompt="You are a neurosurgery educator providing clear, accurate answers with explanations.",
            user_prompt=prompt,
            temperature=0.3
        )

        conversation.add_turn("guide", f"[ANSWER] {answer}")

        return {
            'answer': answer,
            'conversation_id': conversation_id,
            'state': conversation.state.value,
            'sources': [self._format_source(c) for c in conversation.context_chunks[:3]],
            'total_attempts': len(attempts),
            'hints_used': conversation.hints_given
        }

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation details"""
        conversation = self._conversations.get(conversation_id)
        if not conversation:
            return None

        return {
            'id': conversation.id,
            'user_id': conversation.user_id,
            'original_question': conversation.original_question,
            'content_type': conversation.content_type,
            'state': conversation.state.value,
            'turns': [
                {
                    'role': t.role,
                    'content': t.content,
                    'timestamp': t.timestamp.isoformat()
                }
                for t in conversation.turns
            ],
            'hints_given': conversation.hints_given,
            'created_at': conversation.created_at.isoformat()
        }

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    async def _generate_response(
        self,
        conversation: SocraticConversation,
        context: str,
        template_type: str
    ) -> str:
        """Generate Socratic response using template"""
        template = SOCRATIC_TEMPLATES.get(
            template_type,
            SOCRATIC_TEMPLATES['default']
        )

        prompt = template.format(
            context=context,
            question=conversation.original_question,
            history=conversation.get_history()
        )

        return await self._call_llm(
            system_prompt=SOCRATIC_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.7
        )

    async def _generate_correct_feedback(
        self,
        conversation: SocraticConversation,
        student_answer: str
    ) -> str:
        """Generate positive feedback for correct answer"""
        context = self._format_context(conversation.context_chunks)

        prompt = CORRECT_RESPONSE_PROMPT.format(
            question=conversation.original_question,
            student_answer=student_answer,
            correct_answer=context[:500]  # Use context as reference
        )

        return await self._call_llm(
            system_prompt=SOCRATIC_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.5
        )

    async def _check_correctness(
        self,
        conversation: SocraticConversation,
        student_response: str
    ) -> bool:
        """Check if student response is correct (heuristic)"""
        # Simple heuristic - could be enhanced with LLM
        context_text = " ".join([
            c.get('content', '')
            for c in conversation.context_chunks
        ]).lower()

        response_lower = student_response.lower()

        # Check for key terms from context in response
        # This is a simple heuristic - production would use LLM
        context_words = set(context_text.split())
        response_words = set(response_lower.split())

        overlap = len(context_words.intersection(response_words))

        # Very rough heuristic
        return overlap > 10 and len(response_lower) > 50

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7
    ) -> str:
        """Call LLM with prompts"""
        if not self.llm:
            return "[LLM not configured - would generate Socratic response here]"

        try:
            response = await self.llm.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature
            )

            return response.content[0].text
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return "I'd like to guide you through this. Can you tell me what you already know about this topic?"

    def _detect_content_type(self, question: str) -> str:
        """Detect content type from question"""
        question_lower = question.lower()

        # Anatomy indicators
        if any(term in question_lower for term in [
            'supplies', 'innervates', 'drains', 'passes through',
            'adjacent', 'artery', 'nerve', 'vein', 'muscle', 'bone',
            'anatomy', 'location', 'course', 'origin', 'insertion'
        ]):
            return 'anatomy'

        # Procedure indicators
        if any(term in question_lower for term in [
            'approach', 'craniotomy', 'procedure', 'technique',
            'how to', 'steps', 'incision', 'position', 'exposure'
        ]):
            return 'procedure'

        # Pathology indicators
        if any(term in question_lower for term in [
            'tumor', 'lesion', 'disease', 'syndrome', 'pathology',
            'causes', 'etiology', 'mechanism', 'hemorrhage', 'infarct'
        ]):
            return 'pathology'

        # Clinical indicators
        if any(term in question_lower for term in [
            'diagnosis', 'treatment', 'management', 'prognosis',
            'symptoms', 'signs', 'presentation', 'workup', 'exam'
        ]):
            return 'clinical'

        # Imaging indicators
        if any(term in question_lower for term in [
            'mri', 'ct', 'imaging', 'scan', 'radiograph', 'angiogram',
            'sequence', 't1', 't2', 'flair', 'contrast'
        ]):
            return 'imaging'

        return 'default'

    def _map_content_type(self, content_type: str) -> str:
        """Map content type to chunk_type filter"""
        mapping = {
            'anatomy': 'ANATOMY',
            'procedure': 'PROCEDURE',
            'pathology': 'PATHOLOGY',
            'clinical': 'CLINICAL',
            'imaging': 'ANATOMY',  # Imaging often relates to anatomy
            'pharmacology': 'CLINICAL',
            'default': None
        }
        return mapping.get(content_type)

    def _format_context(self, chunks: List[Dict]) -> str:
        """Format chunks as context string"""
        if not chunks:
            return "No specific context available."

        contexts = []
        for chunk in chunks[:5]:
            content = chunk.get('content', '')[:500]
            source = chunk.get('source_document', 'Unknown')
            page = chunk.get('page_number', '?')
            contexts.append(f"[Source: {source}, p.{page}]\n{content}")

        return "\n\n---\n\n".join(contexts)

    def _format_source(self, chunk: Dict) -> Dict[str, Any]:
        """Format chunk as source reference"""
        return {
            'document': chunk.get('source_document', 'Unknown'),
            'page': chunk.get('page_number'),
            'chunk_type': chunk.get('chunk_type'),
            'score': chunk.get('similarity_score')
        }
