"""
Q&A Interaction Repository
==========================

Repository for tracking Q&A interactions used in gap detection Stage 5 (User Demand).
Tracks questions asked and how well they were answered to identify recurring gaps.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QAInteraction:
    """Represents a Q&A interaction record."""

    id: str
    question: str
    answer: Optional[str] = None
    was_answered: bool = False
    answer_quality_score: Optional[float] = None
    chapter_topic: Optional[str] = None
    subspecialty: Optional[str] = None
    template_type: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None


class QARepository:
    """
    Repository for Q&A interactions.

    Used by GapDetectionService Stage 5 to identify frequently asked
    but poorly answered questions.

    Usage:
        repo = QARepository(db_pool, embedding_service)

        # Record a new question
        await repo.record_question(
            question="What is the ICP threshold?",
            chapter_topic="TBI management",
            subspecialty="trauma",
        )

        # Update with answer quality
        await repo.update_answer(
            qa_id=id,
            answer="22 mmHg per BTF guidelines",
            quality_score=0.9,
        )

        # Get related questions for gap analysis
        questions = await repo.get_related_questions("ICP management")
    """

    def __init__(
        self,
        db_pool: Any,
        embedding_service: Optional[Any] = None,
    ):
        self.pool = db_pool
        self.embedder = embedding_service

    async def record_question(
        self,
        question: str,
        chapter_topic: Optional[str] = None,
        subspecialty: Optional[str] = None,
        template_type: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Record a new question.

        Returns the generated Q&A interaction ID.
        """
        qa_id = str(uuid.uuid4())

        # Generate embedding if service available
        embedding = None
        if self.embedder:
            try:
                embedding = await self.embedder.embed_text(question)
            except Exception as e:
                logger.warning(f"Failed to generate embedding for question: {e}")

        query = """
            INSERT INTO qa_interactions (
                id, question, question_embedding,
                chapter_topic, subspecialty, template_type,
                session_id, user_id, was_answered
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    uuid.UUID(qa_id),
                    question,
                    embedding,
                    chapter_topic,
                    subspecialty,
                    template_type,
                    session_id,
                    user_id,
                    False,
                )
            return qa_id
        except Exception as e:
            logger.error(f"Failed to record question: {e}")
            raise

    async def update_answer(
        self,
        qa_id: str,
        answer: str,
        quality_score: float,
        answer_source: Optional[str] = None,
    ) -> None:
        """
        Update a Q&A record with the answer and quality score.

        Args:
            qa_id: The Q&A interaction ID
            answer: The answer provided
            quality_score: Quality score (0-1)
            answer_source: Source of answer (internal, external, both)
        """
        query = """
            UPDATE qa_interactions
            SET answer = $2,
                was_answered = $3,
                answer_quality_score = $4,
                answer_source = $5,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
        """

        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    uuid.UUID(qa_id),
                    answer,
                    quality_score >= 0.5,  # Considered "answered" if quality >= 0.5
                    quality_score,
                    answer_source,
                )
        except Exception as e:
            logger.error(f"Failed to update answer: {e}")
            raise

    async def get_related_questions(
        self,
        topic: str,
        limit: int = 20,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Get questions related to a topic for gap analysis.

        Uses semantic similarity if embeddings available,
        otherwise falls back to text matching.

        Returns questions that were not answered or poorly answered.
        """
        # Try semantic search first
        if self.embedder:
            try:
                return await self._get_related_semantic(topic, limit, min_similarity)
            except Exception as e:
                logger.warning(f"Semantic search failed, falling back to text: {e}")

        # Fall back to text matching
        return await self._get_related_text(topic, limit)

    async def _get_related_semantic(
        self,
        topic: str,
        limit: int,
        min_similarity: float,
    ) -> List[Dict[str, Any]]:
        """Get related questions using semantic similarity."""
        topic_embedding = await self.embedder.embed_text(topic)

        query = """
            SELECT
                id, question, answer, was_answered, answer_quality_score,
                chapter_topic, subspecialty,
                1 - (question_embedding <=> $1::vector) AS similarity
            FROM qa_interactions
            WHERE question_embedding IS NOT NULL
              AND (NOT was_answered OR answer_quality_score < 0.5)
              AND created_at > NOW() - INTERVAL '90 days'
            ORDER BY question_embedding <=> $1::vector
            LIMIT $2
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, topic_embedding, limit)

        return [
            {
                "id": str(row["id"]),
                "question": row["question"],
                "answer": row["answer"],
                "was_answered": row["was_answered"],
                "answer_quality_score": row["answer_quality_score"],
                "chapter_topic": row["chapter_topic"],
                "subspecialty": row["subspecialty"],
                "similarity": row["similarity"],
            }
            for row in rows
            if row["similarity"] >= min_similarity
        ]

    async def _get_related_text(
        self,
        topic: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Get related questions using text matching."""
        # Extract key terms from topic
        topic_terms = [t.lower() for t in topic.split() if len(t) > 3]

        if not topic_terms:
            return []

        # Build LIKE conditions
        like_conditions = " OR ".join(
            f"LOWER(question) LIKE '%{term}%'"
            for term in topic_terms[:5]
        )

        query = f"""
            SELECT
                id, question, answer, was_answered, answer_quality_score,
                chapter_topic, subspecialty
            FROM qa_interactions
            WHERE ({like_conditions})
              AND (NOT was_answered OR answer_quality_score < 0.5)
              AND created_at > NOW() - INTERVAL '90 days'
            ORDER BY created_at DESC
            LIMIT $1
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, limit)

        return [
            {
                "id": str(row["id"]),
                "question": row["question"],
                "answer": row["answer"],
                "was_answered": row["was_answered"],
                "answer_quality_score": row["answer_quality_score"],
                "chapter_topic": row["chapter_topic"],
                "subspecialty": row["subspecialty"],
            }
            for row in rows
        ]

    async def get_unanswered_by_topic(
        self,
        topic: Optional[str] = None,
        subspecialty: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get aggregated unanswered questions by topic.

        Useful for identifying systematic gaps in coverage.
        """
        conditions = ["(NOT was_answered OR answer_quality_score < 0.5)"]
        params = []
        param_idx = 1

        if topic:
            conditions.append(f"chapter_topic ILIKE ${param_idx}")
            params.append(f"%{topic}%")
            param_idx += 1

        if subspecialty:
            conditions.append(f"subspecialty = ${param_idx}")
            params.append(subspecialty)
            param_idx += 1

        where_clause = " AND ".join(conditions)
        params.append(limit)

        query = f"""
            SELECT
                chapter_topic,
                subspecialty,
                COUNT(*) as question_count,
                AVG(answer_quality_score) as avg_quality,
                array_agg(question ORDER BY created_at DESC) FILTER (WHERE question IS NOT NULL) as questions
            FROM qa_interactions
            WHERE {where_clause}
              AND created_at > NOW() - INTERVAL '90 days'
            GROUP BY chapter_topic, subspecialty
            HAVING COUNT(*) >= 2
            ORDER BY COUNT(*) DESC
            LIMIT ${param_idx}
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        return [
            {
                "chapter_topic": row["chapter_topic"],
                "subspecialty": row["subspecialty"],
                "question_count": row["question_count"],
                "avg_quality": float(row["avg_quality"]) if row["avg_quality"] else None,
                "sample_questions": row["questions"][:5] if row["questions"] else [],
            }
            for row in rows
        ]

    async def get_stats(
        self,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get Q&A statistics for monitoring."""
        query = """
            SELECT
                COUNT(*) as total_questions,
                COUNT(*) FILTER (WHERE was_answered) as answered_count,
                COUNT(*) FILTER (WHERE NOT was_answered) as unanswered_count,
                AVG(answer_quality_score) FILTER (WHERE answer_quality_score IS NOT NULL) as avg_quality,
                COUNT(DISTINCT chapter_topic) as unique_topics,
                COUNT(DISTINCT subspecialty) as unique_subspecialties
            FROM qa_interactions
            WHERE created_at > NOW() - INTERVAL '%s days'
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query % days)

        return {
            "total_questions": row["total_questions"],
            "answered_count": row["answered_count"],
            "unanswered_count": row["unanswered_count"],
            "answer_rate": (
                row["answered_count"] / row["total_questions"]
                if row["total_questions"] > 0
                else 0
            ),
            "avg_quality": float(row["avg_quality"]) if row["avg_quality"] else None,
            "unique_topics": row["unique_topics"],
            "unique_subspecialties": row["unique_subspecialties"],
            "period_days": days,
        }
