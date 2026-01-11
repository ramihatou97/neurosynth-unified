# src/learning/nprss/dependencies.py
"""
NPRSS Dependency Injection

FastAPI dependency factories for learning system components.
Follows NeuroSynth's existing dependency patterns.

Usage in routes:
    from src.learning.nprss.dependencies import (
        get_review_service,
        get_flashcard_generator,
        get_socratic_engine
    )

    @router.post("/review")
    async def submit_review(
        review_service = Depends(get_review_service)
    ):
        ...
"""

from typing import Optional
from functools import lru_cache

# Database
from asyncpg import Pool

# Repositories
from .repositories.learning_item import LearningItemRepository
from .repositories.user_state import UserLearningStateRepository
from .repositories.review_history import ReviewHistoryRepository
from .repositories.study_session import StudySessionRepository
from .repositories.milestone import MilestoneRepository
from .repositories.socratic import SocraticPromptRepository, SocraticResponseRepository

# Services
from .service import (
    LearningEnrichmentService,
    RetrievalScheduleService,
    MasteryService
)

# Generators
from .generators.hybrid_generator import HybridFlashcardGenerator
from .generators.phase_cards import PhaseCardGenerator

# Socratic
from .socratic.engine import SocraticEngine

# FSRS
from .fsrs import FSRS

# FRCSC Enhancement Services
from .services.fsrs_enhanced import FSRSEnhanced, FSRSConfig
from .services.question_bank import QuestionBankService
from .services.exam_simulation import ExamSimulationService


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

_db_pool: Optional[Pool] = None


def set_db_pool(pool: Pool):
    """Set the database connection pool (called at app startup)."""
    global _db_pool
    _db_pool = pool


def get_db_pool() -> Pool:
    """Get database connection pool."""
    if _db_pool is None:
        raise RuntimeError(
            "Database pool not initialized. "
            "Call set_db_pool() at application startup."
        )
    return _db_pool


# =============================================================================
# REPOSITORY DEPENDENCIES
# =============================================================================

async def get_learning_item_repository() -> LearningItemRepository:
    """Get LearningItemRepository instance."""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        yield LearningItemRepository(conn)


async def get_user_state_repository() -> UserLearningStateRepository:
    """Get UserLearningStateRepository instance."""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        yield UserLearningStateRepository(conn)


async def get_review_history_repository() -> ReviewHistoryRepository:
    """Get ReviewHistoryRepository instance."""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        yield ReviewHistoryRepository(conn)


async def get_study_session_repository() -> StudySessionRepository:
    """Get StudySessionRepository instance."""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        yield StudySessionRepository(conn)


async def get_milestone_repository() -> MilestoneRepository:
    """Get MilestoneRepository instance."""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        yield MilestoneRepository(conn)


async def get_socratic_prompt_repository() -> SocraticPromptRepository:
    """Get SocraticPromptRepository instance."""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        yield SocraticPromptRepository(conn)


async def get_socratic_response_repository() -> SocraticResponseRepository:
    """Get SocraticResponseRepository instance."""
    pool = get_db_pool()
    async with pool.acquire() as conn:
        yield SocraticResponseRepository(conn)


# =============================================================================
# SERVICE DEPENDENCIES
# =============================================================================

@lru_cache()
def get_fsrs_algorithm() -> FSRS:
    """Get cached FSRS algorithm instance."""
    from .config import get_nprss_settings
    from .fsrs import FSRSParameters
    settings = get_nprss_settings()

    params = FSRSParameters(
        request_retention=settings.fsrs_request_retention,
        maximum_interval=settings.fsrs_maximum_interval,
        w=settings.fsrs_weights
    )
    return FSRS(params=params)


async def get_review_service():
    """
    Get ReviewService for handling FSRS reviews.

    Combines FSRS algorithm with database operations.
    """
    pool = get_db_pool()
    fsrs = get_fsrs_algorithm()

    async with pool.acquire() as conn:
        state_repo = UserLearningStateRepository(conn)
        history_repo = ReviewHistoryRepository(conn)

        # Create a simple review service wrapper
        class ReviewService:
            def __init__(self, state_repo, history_repo, fsrs):
                self.state_repo = state_repo
                self.history_repo = history_repo
                self.fsrs = fsrs

            async def submit_review(
                self,
                user_id: str,
                card_id: str,
                rating: int,
                response_time_ms: int = None,
                session_id: str = None
            ):
                from uuid import UUID
                from datetime import datetime
                from .fsrs import Rating, MemoryState, State

                card_uuid = UUID(card_id) if isinstance(card_id, str) else card_id

                # Get current state from database
                db_state = await self.state_repo.get_or_create(user_id, card_uuid)

                # Convert to MemoryState for FSRS
                state_map = {
                    'new': State.NEW,
                    'learning': State.LEARNING,
                    'review': State.REVIEW,
                    'relearning': State.RELEARNING
                }
                memory = MemoryState(
                    difficulty=db_state.difficulty,
                    stability=db_state.stability,
                    state=state_map.get(db_state.state, State.NEW),
                    step=0,
                    due=db_state.due_date,
                    last_review=db_state.last_review,
                    reps=db_state.reps,
                    lapses=db_state.lapses
                )

                # Record state before review
                state_before = {
                    'difficulty': db_state.difficulty,
                    'stability': db_state.stability,
                    'retrievability': db_state.retrievability,
                    'scheduled_days': db_state.scheduled_days
                }

                # Get review outcome from FSRS
                new_memory, review_log = self.fsrs.review(
                    memory=memory,
                    rating=Rating(rating),
                    review_time=datetime.now()
                )

                # Calculate scheduled days
                scheduled_days = 0.0
                if new_memory.due:
                    scheduled_days = (new_memory.due - datetime.now()).total_seconds() / 86400

                # Map state back to string
                state_str_map = {
                    State.NEW: 'new',
                    State.LEARNING: 'learning',
                    State.REVIEW: 'review',
                    State.RELEARNING: 'relearning'
                }
                new_state_str = state_str_map.get(new_memory.state, 'new')

                # Record in history
                state_after = {
                    'difficulty': new_memory.difficulty,
                    'stability': new_memory.stability,
                    'retrievability': 1.0,  # Will be recalculated
                    'scheduled_days': scheduled_days
                }

                await self.history_repo.record_review(
                    user_id=user_id,
                    card_id=card_uuid,
                    rating=rating,
                    state_before=state_before,
                    state_after=state_after,
                    response_time_ms=response_time_ms,
                    session_id=UUID(session_id) if session_id else None
                )

                # Update state in database
                updated = await self.state_repo.update_after_review(
                    user_id=user_id,
                    card_id=card_uuid,
                    new_difficulty=new_memory.difficulty,
                    new_stability=new_memory.stability,
                    new_state=new_state_str,
                    next_review=new_memory.due,
                    scheduled_days=scheduled_days,
                    r_level=None  # R-level tracking separate from FSRS
                )

                return {
                    'next_review': new_memory.due,
                    'interval_days': scheduled_days,
                    'new_difficulty': new_memory.difficulty,
                    'new_stability': new_memory.stability,
                    'new_state': new_state_str,
                    'r_level_achieved': None
                }

            async def get_due_cards(self, user_id: str, limit: int = 20):
                return await self.state_repo.get_due_cards(user_id, limit)

            async def get_forecast(self, user_id: str, days: int = 7):
                return await self.state_repo.get_forecast(user_id, days)

        yield ReviewService(state_repo, history_repo, fsrs)


async def get_mastery_service():
    """Get MasteryService for procedure mastery tracking."""
    pool = get_db_pool()

    async with pool.acquire() as conn:
        # Would instantiate actual MasteryService here
        # For now, return placeholder
        yield MasteryService(db=conn)


# =============================================================================
# GENERATOR DEPENDENCIES
# =============================================================================

def get_flashcard_generator() -> HybridFlashcardGenerator:
    """
    Get HybridFlashcardGenerator instance.

    Orchestrates all flashcard generation strategies.
    """
    return HybridFlashcardGenerator()


def get_phase_card_generator() -> PhaseCardGenerator:
    """Get PhaseCardGenerator instance."""
    return PhaseCardGenerator()


async def get_hybrid_generator():
    """
    Get HybridFlashcardGenerator with database access.

    For use in routes that need to persist generated cards.
    """
    generator = HybridFlashcardGenerator()
    yield generator


# =============================================================================
# SOCRATIC DEPENDENCIES
# =============================================================================

_rag_engine = None
_llm_client = None


def set_rag_engine(engine):
    """Set RAG engine for Socratic mode (called at app startup)."""
    global _rag_engine
    _rag_engine = engine


def set_llm_client(client):
    """Set LLM client for Socratic mode (called at app startup)."""
    global _llm_client
    _llm_client = client


async def get_socratic_engine() -> SocraticEngine:
    """
    Get SocraticEngine instance.

    Requires RAG engine and LLM client to be configured.
    """
    engine = SocraticEngine(
        rag_engine=_rag_engine,
        llm_client=_llm_client
    )
    yield engine


# =============================================================================
# COMBINED SERVICE DEPENDENCIES
# =============================================================================

async def get_learning_services():
    """
    Get all learning services in a single dependency.

    Useful for routes that need multiple services.
    """
    pool = get_db_pool()
    fsrs = get_fsrs_algorithm()

    async with pool.acquire() as conn:
        services = {
            'learning_items': LearningItemRepository(conn),
            'user_state': UserLearningStateRepository(conn),
            'review_history': ReviewHistoryRepository(conn),
            'study_sessions': StudySessionRepository(conn),
            'milestones': MilestoneRepository(conn),
            'fsrs': fsrs,
            'generator': HybridFlashcardGenerator()
        }
        yield services


# =============================================================================
# CHUNK REPOSITORY (for compatibility with extended routes)
# =============================================================================

async def get_chunk_repository():
    """
    Get ChunkRepository from main NeuroSynth.

    Falls back to placeholder if not available.
    """
    try:
        # Try to import from main NeuroSynth
        from src.database.repositories.chunk import ChunkRepository
        pool = get_db_pool()
        async with pool.acquire() as conn:
            yield ChunkRepository(conn)
    except ImportError:
        # Placeholder for standalone testing
        class PlaceholderChunkRepo:
            async def get_by_id(self, chunk_id):
                return None
            async def find_by(self, filters, limit=100):
                return []

        yield PlaceholderChunkRepo()


# =============================================================================
# FRCSC ENHANCEMENT DEPENDENCIES
# =============================================================================

async def get_question_bank_service():
    """
    Get QuestionBankService for IRT-based adaptive questioning.

    Provides:
    - Adaptive question selection based on user ability
    - IRT parameter tracking
    - Category performance analytics
    """
    pool = get_db_pool()

    async with pool.acquire() as conn:
        # Create a simple repository wrapper for the service
        class FRCSCRepository:
            def __init__(self, conn):
                self.conn = conn

            async def get_questions(self, category=None, exclude_ids=None):
                """Get available questions from database."""
                query = """
                    SELECT
                        id, question_code, category_id, stem, options, answer,
                        explanation, format, yield_rating, cognitive_level,
                        difficulty_param, discrimination_param, guessing_param,
                        year_asked, key_points
                    FROM frcsc_questions
                    WHERE ($1::UUID IS NULL OR category_id = $1)
                    AND ($2::UUID[] IS NULL OR id != ALL($2))
                    ORDER BY yield_rating DESC, difficulty_param
                """
                rows = await self.conn.fetch(query, category, exclude_ids or [])
                # Convert to Question objects (simplified)
                return rows

            async def get_question(self, question_id):
                """Get single question by ID."""
                query = """
                    SELECT
                        id, question_code, category_id, stem, options, answer,
                        explanation, format, yield_rating, cognitive_level,
                        difficulty_param, discrimination_param, guessing_param,
                        year_asked, key_points
                    FROM frcsc_questions
                    WHERE id = $1
                """
                return await self.conn.fetchrow(query, question_id)

            async def record_attempt(self, user_id, question_id, user_answer,
                                   is_correct, time_spent, ability_estimate):
                """Record question attempt."""
                query = """
                    INSERT INTO frcsc_question_attempts
                    (user_id, question_id, user_answer, is_correct,
                     time_spent_seconds, ability_estimate)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                """
                return await self.conn.fetchrow(
                    query, user_id, question_id, user_answer,
                    is_correct, time_spent, ability_estimate
                )

            async def get_user_attempts(self, user_id, limit=100):
                """Get user's attempt history."""
                query = """
                    SELECT
                        question_id, user_answer, is_correct,
                        time_spent_seconds, ability_estimate, attempted_at
                    FROM frcsc_question_attempts
                    WHERE user_id = $1
                    ORDER BY attempted_at DESC
                    LIMIT $2
                """
                return await self.conn.fetch(query, user_id, limit)

            async def get_all_questions(self):
                """Get all questions for exam simulation."""
                query = """
                    SELECT
                        id, question_code, category_id, stem, options, answer,
                        explanation, format, yield_rating, cognitive_level,
                        difficulty_param, discrimination_param, guessing_param,
                        year_asked, key_points
                    FROM frcsc_questions
                    ORDER BY RANDOM()
                """
                return await self.conn.fetch(query)

        repo = FRCSCRepository(conn)
        service = QuestionBankService(repository=repo)
        yield service


async def get_exam_service():
    """
    Get ExamSimulationService for FRCSC exam practice.

    Provides:
    - Timed exam simulations (written, oral, mock)
    - Performance analytics
    - Pass probability prediction
    """
    pool = get_db_pool()
    question_bank = None

    # Get question bank service for exam question selection
    async for qb in get_question_bank_service():
        question_bank = qb
        break

    async with pool.acquire() as conn:
        # Create repository wrapper
        class ExamRepository:
            def __init__(self, conn):
                self.conn = conn

            async def create_exam_session(self, session):
                """Create new exam session."""
                query = """
                    INSERT INTO exam_simulations
                    (id, user_id, exam_type, status, started_at,
                     time_limit_minutes, total_questions, question_ids)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """
                import json
                await self.conn.execute(
                    query,
                    session.id, session.user_id, session.exam_type.value,
                    session.status.value, session.started_at,
                    session.time_limit_minutes, len(session.questions),
                    json.dumps([str(q.question_id) for q in session.questions])
                )

            async def save_exam_result(self, result):
                """Save exam results."""
                query = """
                    UPDATE exam_simulations
                    SET status = 'completed',
                        completed_at = NOW(),
                        correct_answers = $1,
                        score_percentage = $2,
                        passed = $3,
                        category_scores = $4
                    WHERE id = $5
                """
                import json
                await self.conn.execute(
                    query,
                    result.correct_answers, result.score_percentage,
                    result.passed, json.dumps(result.category_scores),
                    result.exam_id
                )

            async def get_exam_history(self, user_id, limit=10):
                """Get user's exam history."""
                query = """
                    SELECT
                        id, exam_type, score_percentage, passed,
                        started_at, completed_at, category_scores
                    FROM exam_simulations
                    WHERE user_id = $1 AND status = 'completed'
                    ORDER BY completed_at DESC
                    LIMIT $2
                """
                rows = await self.conn.fetch(query, user_id, limit)
                return [dict(row) for row in rows]

            async def get_category_performance(self, user_id):
                """Get performance by category."""
                query = """
                    SELECT
                        q.category_id::text as category,
                        COUNT(*) as total,
                        SUM(CASE WHEN a.is_correct THEN 1 ELSE 0 END) as correct
                    FROM frcsc_question_attempts a
                    JOIN frcsc_questions q ON a.question_id = q.id
                    WHERE a.user_id = $1
                    GROUP BY q.category_id
                """
                rows = await self.conn.fetch(query, user_id)
                result = {}
                for row in rows:
                    result[row['category']] = {
                        'total': row['total'],
                        'correct': row['correct'],
                        'accuracy': row['correct'] / row['total'] if row['total'] > 0 else 0
                    }
                return result

            async def update_exam_status(self, exam_id, status):
                """Update exam status."""
                from .services.exam_simulation import ExamStatus
                query = "UPDATE exam_simulations SET status = $1 WHERE id = $2"
                await self.conn.execute(query, status.value, exam_id)

        repo = ExamRepository(conn)
        service = ExamSimulationService(repository=repo, question_bank=question_bank)
        yield service


@lru_cache()
def get_fsrs_enhanced() -> FSRSEnhanced:
    """
    Get cached FSRSEnhanced algorithm instance.

    Provides dual-track FSRS with exam-aware scheduling.
    """
    from .config import get_nprss_settings
    settings = get_nprss_settings()

    config = FSRSConfig(
        exam_date=None,  # Will be set per-user
        retention_factual=0.90,
        retention_procedural=0.92,
        retention_csp=0.95
    )
    return FSRSEnhanced(config=config)


# =============================================================================
# INITIALIZATION HELPER
# =============================================================================

def initialize_nprss_dependencies(
    db_pool: Pool,
    rag_engine=None,
    llm_client=None
):
    """
    Initialize all NPRSS dependencies.

    Call this at application startup.

    Args:
        db_pool: AsyncPG connection pool
        rag_engine: Optional RAG engine for Socratic mode
        llm_client: Optional LLM client (Anthropic) for Socratic mode

    Example:
        @app.on_event("startup")
        async def startup():
            pool = await asyncpg.create_pool(DATABASE_URL)
            initialize_nprss_dependencies(
                db_pool=pool,
                rag_engine=rag_engine,
                llm_client=anthropic_client
            )
    """
    set_db_pool(db_pool)

    if rag_engine:
        set_rag_engine(rag_engine)

    if llm_client:
        set_llm_client(llm_client)

    # Warm up cached instances
    get_fsrs_algorithm()
    get_fsrs_enhanced()

    print("NPRSS dependencies initialized successfully")
