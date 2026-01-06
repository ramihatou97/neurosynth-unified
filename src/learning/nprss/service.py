# src/learning/nprss/service.py
"""
NPRSS Learning Services

Core services for procedural learning:
- LearningEnrichmentService: Enriches procedures with learning features
- RetrievalScheduleService: Manages R1-R7 schedules
- MasteryService: Tracks and updates mastery state
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4, UUID

from .models import (
    PhaseType, MasteryLevel, Procedure, ProcedureElement,
    CriticalSafetyPoint, VisuospatialAnchor, SurgicalCard,
    LearningCard, CardMemoryState, RetrievalSchedule, RetrievalSession,
    ProcedureMastery, RetrievalScheduleFactory
)
from .transformers import PhaseMapper, CSPExtractor, AnchorGenerator
from .fsrs import FSRS, MemoryState, Rating, HybridScheduler

logger = logging.getLogger(__name__)


# =============================================================================
# LEARNING ENRICHMENT SERVICE
# =============================================================================

class LearningEnrichmentService:
    """
    Main service for enriching procedures with learning features

    Pipeline:
    1. Load procedure and elements
    2. Map elements to 4-phase framework
    3. Extract CSPs from danger zones and maneuvers
    4. Generate visuospatial anchors
    5. Generate surgical card
    6. Generate FSRS learning cards
    """

    def __init__(
        self,
        db,
        llm_client=None
    ):
        """
        Initialize service

        Args:
            db: Database connection (asyncpg pool or similar)
            llm_client: Anthropic client for LLM operations
        """
        self.db = db
        self.llm = llm_client

        # Initialize components
        self.phase_mapper = PhaseMapper(llm_client=llm_client)
        self.csp_extractor = CSPExtractor(llm_client=llm_client)
        self.anchor_generator = AnchorGenerator()

    async def enrich_procedure(
        self,
        procedure_id: str,
        generate_surgical_card: bool = True,
        generate_fsrs_cards: bool = True,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrich a procedure with learning features

        Args:
            procedure_id: UUID of the procedure to enrich
            generate_surgical_card: Whether to generate surgical card
            generate_fsrs_cards: Whether to generate FSRS cards
            user_id: User ID for schedule creation

        Returns:
            Dict with enrichment results
        """
        # Load procedure data
        procedure = await self._get_procedure(procedure_id)
        if not procedure:
            raise ValueError(f"Procedure {procedure_id} not found")

        elements = await self._get_elements(procedure_id)
        danger_zones = await self._get_danger_zones(procedure_id)
        safe_zones = await self._get_safe_zones(procedure_id)

        # Step 1: Map phases
        phase_mapping = self.phase_mapper.map_elements(elements)

        # Update elements with phase types
        for element in elements:
            element_id = str(element.get('id'))
            if element_id in phase_mapping:
                await self._update_element_phase(element_id, phase_mapping[element_id])

        # Generate phase gates
        phase_gates = self.phase_mapper.generate_phase_gates(procedure_id)
        for gate in phase_gates:
            await self._save_phase_gate(gate)

        # Step 2: Extract CSPs
        csps = self.csp_extractor.extract_csps(
            procedure_id, elements, danger_zones, safe_zones
        )

        for csp in csps:
            csp_dict = csp if isinstance(csp, dict) else (csp.to_dict() if hasattr(csp, 'to_dict') else asdict(csp))
            await self._save_csp(csp_dict)

        # Step 3: Generate anchors
        anchors = self.anchor_generator.generate_anchors(elements)

        for anchor in anchors:
            anchor_dict = anchor if isinstance(anchor, dict) else (anchor.to_dict() if hasattr(anchor, 'to_dict') else asdict(anchor))
            await self._save_anchor(anchor_dict)

        result = {
            "procedure_id": procedure_id,
            "phases_mapped": len(phase_mapping),
            "csps_extracted": len(csps),
            "anchors_generated": len(anchors),
            "phase_gates_created": len(phase_gates),
            "surgical_card_id": None,
            "cards_generated": 0,
        }

        logger.info(f"Enriched procedure {procedure_id}: {result}")
        return result

    # =========================================================================
    # Database Operations
    # =========================================================================

    async def _get_procedure(self, procedure_id: str) -> Optional[Dict[str, Any]]:
        """Get procedure by ID"""
        query = "SELECT * FROM procedures WHERE id = $1"
        row = await self.db.fetchrow(query, procedure_id)
        return dict(row) if row else None

    async def _get_elements(self, procedure_id: str) -> List[Dict[str, Any]]:
        """Get procedure elements"""
        query = """
            SELECT * FROM procedure_elements
            WHERE procedure_id = $1
            ORDER BY sequence_order
        """
        rows = await self.db.fetch(query, procedure_id)
        return [dict(r) for r in rows]

    async def _get_danger_zones(self, procedure_id: str) -> List[Dict[str, Any]]:
        """Get danger zones for procedure"""
        query = """
            SELECT dz.* FROM danger_zones dz
            JOIN procedure_danger_zones pdz ON pdz.danger_zone_id = dz.id
            WHERE pdz.procedure_id = $1
        """
        rows = await self.db.fetch(query, procedure_id)
        return [dict(r) for r in rows]

    async def _get_safe_zones(self, procedure_id: str) -> List[Dict[str, Any]]:
        """Get safe entry zones for procedure"""
        query = """
            SELECT sz.* FROM safe_entry_zones sz
            JOIN procedure_safe_zones psz ON psz.safe_zone_id = sz.id
            WHERE psz.procedure_id = $1
        """
        rows = await self.db.fetch(query, procedure_id)
        return [dict(r) for r in rows]

    async def _update_element_phase(self, element_id: str, phase_type: str):
        """Update element with phase type"""
        phase_value = phase_type.value if hasattr(phase_type, 'value') else phase_type
        query = "UPDATE procedure_elements SET phase_type = $2 WHERE id = $1"
        await self.db.execute(query, element_id, phase_value)

    async def _save_phase_gate(self, gate: Dict):
        """Save phase gate"""
        query = """
            INSERT INTO phase_gates (id, procedure_id, from_phase, to_phase, verification_questions, prerequisites)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (procedure_id, from_phase, to_phase) DO UPDATE
            SET verification_questions = EXCLUDED.verification_questions,
                prerequisites = EXCLUDED.prerequisites
        """
        await self.db.execute(
            query, gate['id'], gate['procedure_id'],
            gate['from_phase'], gate['to_phase'],
            gate['verification_questions'], gate.get('prerequisites', [])
        )

    async def _save_csp(self, csp: Dict):
        """Save critical safety point"""
        query = """
            INSERT INTO critical_safety_points (
                id, procedure_id, element_id, csp_number, phase_type,
                when_action, stop_if_trigger, visual_cue,
                structure_at_risk, mechanism_of_injury, if_violated_action,
                derived_from_danger_zone_id, derived_from_safe_zone_id,
                retrieval_cue, common_errors
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            ON CONFLICT (procedure_id, csp_number) DO UPDATE
            SET when_action = EXCLUDED.when_action,
                stop_if_trigger = EXCLUDED.stop_if_trigger,
                structure_at_risk = EXCLUDED.structure_at_risk
        """
        phase_type = csp.get('phase_type')
        if hasattr(phase_type, 'value'):
            phase_type = phase_type.value

        await self.db.execute(
            query,
            str(csp.get('id', uuid4())),
            str(csp['procedure_id']) if csp.get('procedure_id') else None,
            str(csp['element_id']) if csp.get('element_id') else None,
            csp['csp_number'],
            phase_type,
            csp['when_action'],
            csp['stop_if_trigger'],
            csp.get('visual_cue'),
            csp['structure_at_risk'],
            csp.get('mechanism_of_injury'),
            csp.get('if_violated_action'),
            str(csp['derived_from_danger_zone_id']) if csp.get('derived_from_danger_zone_id') else None,
            str(csp['derived_from_safe_zone_id']) if csp.get('derived_from_safe_zone_id') else None,
            csp.get('retrieval_cue'),
            csp.get('common_errors', [])
        )

    async def _save_anchor(self, anchor: Dict):
        """Save visuospatial anchor"""
        query = """
            INSERT INTO visuospatial_anchors (
                id, element_id, expected_view, landmarks, color_cues,
                mental_rotation_prompt, spatial_relationship, depth_reference, viewing_angle
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT DO NOTHING
        """
        await self.db.execute(
            query,
            str(anchor.get('id', uuid4())),
            str(anchor['element_id']) if anchor.get('element_id') else None,
            anchor.get('expected_view'),
            anchor.get('landmarks', []),
            anchor.get('color_cues'),
            anchor.get('mental_rotation_prompt'),
            anchor.get('spatial_relationship'),
            anchor.get('depth_reference'),
            anchor.get('viewing_angle')
        )

    async def _save_learning_card(self, card: Dict):
        """Save FSRS learning card"""
        card_type = card.get('card_type')
        if hasattr(card_type, 'value'):
            card_type = card_type.value

        query = """
            INSERT INTO nprss_learning_cards (
                id, procedure_id, element_id, csp_id, card_type,
                prompt, answer, options, difficulty_preset, tags
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """
        await self.db.execute(
            query,
            str(card.get('id', uuid4())),
            str(card['procedure_id']) if card.get('procedure_id') else None,
            str(card['element_id']) if card.get('element_id') else None,
            str(card['csp_id']) if card.get('csp_id') else None,
            card_type,
            card['prompt'],
            card['answer'],
            json.dumps(card.get('options')) if card.get('options') else None,
            card.get('difficulty_preset', 0.3),
            card.get('tags', [])
        )


# =============================================================================
# RETRIEVAL SCHEDULE SERVICE
# =============================================================================

class RetrievalScheduleService:
    """
    Service for managing R1-R7 retrieval schedules
    """

    R1_R7_TEMPLATE = [
        (1, "R1", "Dictate operative note from memory", "free_recall", 15),
        (3, "R2", "Write Surgical Card from memory -> compare", "free_recall", 20),
        (7, "R3", "Mental rehearsal (full procedure)", "rehearsal", 20),
        (14, "R4", "Verbal teach-back (explain to peer/recorder)", "elaboration", 25),
        (30, "R5", "CSP rapid-fire quiz + case variation", "cued_recall", 15),
        (60, "R6", "Interleaved review (mix with similar procedures)", "interleaved", 30),
        (120, "R7", "Full simulation or cadaver lab if available", "application", 60),
    ]

    def __init__(self, db):
        self.db = db
        self.hybrid_scheduler = HybridScheduler()

    async def create_schedule(
        self,
        user_id: str,
        procedure_id: str,
        target_retention_days: int = 180,
        encoding_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Create R1-R7 schedule for a user-procedure pair
        """
        encoding_date = encoding_date or datetime.now()
        schedule_id = str(uuid4())

        # Insert schedule
        schedule_query = """
            INSERT INTO retrieval_schedules (id, user_id, procedure_id, encoding_date, target_retention_days)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (user_id, procedure_id) DO UPDATE
            SET encoding_date = EXCLUDED.encoding_date,
                target_retention_days = EXCLUDED.target_retention_days
            RETURNING id
        """
        result = await self.db.fetchrow(
            schedule_query, schedule_id, user_id, procedure_id, encoding_date, target_retention_days
        )
        schedule_id = str(result['id'])

        # Delete existing sessions
        await self.db.execute(
            "DELETE FROM retrieval_sessions WHERE schedule_id = $1",
            schedule_id
        )

        # Create sessions
        sessions = []
        for day, label, task, task_type, duration in self.R1_R7_TEMPLATE:
            # Adjust days based on retention target
            adjusted_day = int(day * (target_retention_days / 180))
            scheduled_date = encoding_date + timedelta(days=adjusted_day)

            session_id = str(uuid4())
            session_query = """
                INSERT INTO retrieval_sessions (
                    id, schedule_id, session_number, scheduled_date, days_from_encoding,
                    retrieval_task, task_type, estimated_duration_min
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """
            await self.db.execute(
                session_query,
                session_id, schedule_id, int(label[1]), scheduled_date, adjusted_day,
                task, task_type, duration
            )

            sessions.append({
                "id": session_id,
                "session_number": int(label[1]),
                "label": label,
                "scheduled_date": scheduled_date.isoformat(),
                "days_from_encoding": adjusted_day,
                "retrieval_task": task,
                "task_type": task_type,
                "estimated_duration_min": duration,
                "completed": False
            })

        return {
            "schedule_id": schedule_id,
            "user_id": user_id,
            "procedure_id": procedure_id,
            "encoding_date": encoding_date.isoformat(),
            "target_retention_days": target_retention_days,
            "sessions": sessions
        }

    async def get_upcoming_sessions(
        self,
        user_id: str,
        days_ahead: int = 14
    ) -> List[Dict[str, Any]]:
        """Get upcoming retrieval sessions for a user"""
        query = f"""
            SELECT
                rs.id, rs.session_number, rs.scheduled_date, rs.days_from_encoding,
                rs.retrieval_task, rs.task_type, rs.estimated_duration_min,
                rs.completed, rs.completed_at, rs.self_assessment_score,
                p.id as procedure_id, p.name as procedure_name,
                rsch.encoding_date
            FROM retrieval_sessions rs
            JOIN retrieval_schedules rsch ON rs.schedule_id = rsch.id
            JOIN procedures p ON rsch.procedure_id = p.id
            WHERE rsch.user_id = $1
            AND rs.scheduled_date <= NOW() + INTERVAL '{days_ahead} days'
            AND rs.completed = FALSE
            ORDER BY rs.scheduled_date
        """

        rows = await self.db.fetch(query, user_id)

        sessions = []
        for row in rows:
            sessions.append({
                "id": str(row['id']),
                "session_number": row['session_number'],
                "scheduled_date": row['scheduled_date'].isoformat() if row['scheduled_date'] else None,
                "days_from_encoding": row['days_from_encoding'],
                "retrieval_task": row['retrieval_task'],
                "task_type": row['task_type'],
                "estimated_duration_min": row['estimated_duration_min'],
                "procedure_id": str(row['procedure_id']),
                "procedure_name": row['procedure_name'],
                "is_overdue": row['scheduled_date'] < datetime.now() if row['scheduled_date'] else False
            })

        return sessions

    async def complete_session(
        self,
        session_id: str,
        self_assessment_score: int,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mark a session as complete"""
        query = """
            UPDATE retrieval_sessions
            SET completed = TRUE,
                completed_at = NOW(),
                self_assessment_score = $2,
                notes = $3
            WHERE id = $1
            RETURNING *
        """
        row = await self.db.fetchrow(query, session_id, self_assessment_score, notes)

        if not row:
            raise ValueError(f"Session {session_id} not found")

        # Update mastery
        schedule_query = """
            SELECT user_id, procedure_id FROM retrieval_schedules
            WHERE id = $1
        """
        schedule = await self.db.fetchrow(schedule_query, row['schedule_id'])

        if schedule:
            mastery_service = MasteryService(self.db)
            await mastery_service.update_from_session(
                user_id=schedule['user_id'],
                procedure_id=str(schedule['procedure_id']),
                session_number=row['session_number'],
                self_assessment_score=self_assessment_score
            )

        return dict(row)


# =============================================================================
# MASTERY SERVICE
# =============================================================================

class MasteryService:
    """
    Service for tracking and updating procedure mastery
    """

    # Mastery level thresholds based on session performance
    LEVEL_THRESHOLDS = {
        MasteryLevel.NOT_YET: 0.0,
        MasteryLevel.DEVELOPING: 0.4,
        MasteryLevel.COMPETENT: 0.7,
        MasteryLevel.MASTERY: 0.9,
    }

    def __init__(self, db):
        self.db = db
        self.fsrs = FSRS()

    async def get_mastery(
        self,
        user_id: str,
        procedure_id: str
    ) -> Dict[str, Any]:
        """Get mastery state for a user-procedure pair"""
        query = """
            SELECT * FROM procedure_mastery
            WHERE user_id = $1 AND procedure_id = $2
        """
        row = await self.db.fetchrow(query, user_id, procedure_id)

        if not row:
            # Return default mastery state
            return {
                "user_id": user_id,
                "procedure_id": procedure_id,
                "current_level": 1,
                "level_name": "NOT_YET",
                "phase_scores": {},
                "weak_csps": [],
                "weak_phases": [],
                "predicted_retention_score": 0.5,
                "next_optimal_review": None,
                "total_retrieval_sessions": 0
            }

        level_names = {1: "NOT_YET", 2: "DEVELOPING", 3: "COMPETENT", 4: "MASTERY"}

        return {
            "id": str(row['id']),
            "user_id": row['user_id'],
            "procedure_id": str(row['procedure_id']),
            "current_level": row['current_level'],
            "level_name": level_names.get(row['current_level'], "UNKNOWN"),
            "phase_scores": row['phase_scores'] or {},
            "weak_csps": row['weak_csps'] or [],
            "weak_phases": row['weak_phases'] or [],
            "predicted_retention_score": float(row['predicted_retention_score']) if row['predicted_retention_score'] else 0.5,
            "next_optimal_review": row['next_optimal_review'].isoformat() if row['next_optimal_review'] else None,
            "total_retrieval_sessions": row['total_retrieval_sessions'] or 0,
            "last_session_date": row['last_session_date'].isoformat() if row['last_session_date'] else None,
            "entrustment_level": row['entrustment_level']
        }

    async def get_user_mastery_overview(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Get mastery overview for all procedures for a user"""
        query = """
            SELECT
                pm.*,
                p.name as procedure_name,
                p.subspecialty_domain,
                p.complexity
            FROM procedure_mastery pm
            JOIN procedures p ON pm.procedure_id = p.id
            WHERE pm.user_id = $1
            ORDER BY pm.next_optimal_review NULLS LAST
        """
        rows = await self.db.fetch(query, user_id)

        level_names = {1: "NOT_YET", 2: "DEVELOPING", 3: "COMPETENT", 4: "MASTERY"}

        return [
            {
                "procedure_id": str(row['procedure_id']),
                "procedure_name": row['procedure_name'],
                "subspecialty_domain": row['subspecialty_domain'],
                "complexity": row['complexity'],
                "current_level": row['current_level'],
                "level_name": level_names.get(row['current_level'], "UNKNOWN"),
                "predicted_retention_score": float(row['predicted_retention_score']) if row['predicted_retention_score'] else 0.5,
                "next_optimal_review": row['next_optimal_review'].isoformat() if row['next_optimal_review'] else None,
                "total_retrieval_sessions": row['total_retrieval_sessions'] or 0
            }
            for row in rows
        ]

    async def update_from_session(
        self,
        user_id: str,
        procedure_id: str,
        session_number: int,
        self_assessment_score: int
    ) -> Dict[str, Any]:
        """Update mastery based on completed session"""
        # Get or create mastery record
        mastery = await self.get_mastery(user_id, procedure_id)

        # Calculate new level based on self-assessment history
        current_level = mastery['current_level']
        total_sessions = mastery['total_retrieval_sessions'] + 1

        # Simple level update logic
        # Score 4 on R5+ can upgrade to COMPETENT
        # Score 4 on R7 can upgrade to MASTERY
        if self_assessment_score >= 4 and session_number >= 7:
            new_level = min(4, current_level + 1)
        elif self_assessment_score >= 3 and session_number >= 5:
            new_level = max(3, current_level)
        elif self_assessment_score >= 3:
            new_level = max(2, current_level)
        elif self_assessment_score <= 1:
            new_level = max(1, current_level - 1)
        else:
            new_level = current_level

        # Calculate predicted retention
        # Higher scores and more sessions = higher retention
        base_retention = 0.5
        session_bonus = min(0.3, total_sessions * 0.05)
        score_bonus = (self_assessment_score - 2) * 0.1
        predicted_retention = min(0.95, base_retention + session_bonus + score_bonus)

        # Calculate next optimal review
        if new_level >= 3:
            # Competent/Mastery: longer intervals
            next_review = datetime.now() + timedelta(days=30 * new_level)
        else:
            # Earlier levels: shorter intervals
            next_review = datetime.now() + timedelta(days=7 * (session_number + 1))

        # Upsert mastery record
        query = """
            INSERT INTO procedure_mastery (
                id, user_id, procedure_id, current_level,
                predicted_retention_score, next_optimal_review,
                total_retrieval_sessions, last_session_date
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            ON CONFLICT (user_id, procedure_id) DO UPDATE
            SET current_level = $4,
                predicted_retention_score = $5,
                next_optimal_review = $6,
                total_retrieval_sessions = procedure_mastery.total_retrieval_sessions + 1,
                last_session_date = NOW()
            RETURNING *
        """

        mastery_id = mastery.get('id') or str(uuid4())
        await self.db.fetchrow(
            query,
            mastery_id, user_id, procedure_id, new_level,
            predicted_retention, next_review, total_sessions
        )

        return await self.get_mastery(user_id, procedure_id)

    async def update_from_cards(
        self,
        user_id: str,
        procedure_id: str,
        card_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Update mastery based on FSRS card reviews

        Args:
            user_id: User identifier
            procedure_id: Procedure UUID
            card_results: List of {card_id, rating, tags} dicts
        """
        # Calculate phase-level scores
        phase_scores: Dict[str, List[int]] = {
            "architecture": [],
            "approach": [],
            "target": [],
            "closure": []
        }

        csp_scores: Dict[int, List[int]] = {}

        for result in card_results:
            rating = result.get('rating', 3)
            tags = result.get('tags', [])

            # Update phase scores
            for tag in tags:
                if tag in phase_scores:
                    phase_scores[tag].append(rating)

                # Track CSP scores
                if tag.startswith('csp_'):
                    try:
                        csp_num = int(tag.split('_')[1])
                        if csp_num not in csp_scores:
                            csp_scores[csp_num] = []
                        csp_scores[csp_num].append(rating)
                    except (ValueError, IndexError):
                        pass

        # Calculate averages
        phase_averages = {}
        weak_phases = []
        for phase, scores in phase_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                phase_averages[phase] = avg
                if avg < 2.5:  # Below GOOD rating
                    weak_phases.append(phase)

        # Find weak CSPs
        weak_csps = []
        for csp_num, scores in csp_scores.items():
            if scores and sum(scores) / len(scores) < 2.5:
                weak_csps.append(csp_num)

        # Update mastery record
        mastery = await self.get_mastery(user_id, procedure_id)
        mastery_id = mastery.get('id') or str(uuid4())

        query = """
            INSERT INTO procedure_mastery (
                id, user_id, procedure_id, phase_scores, weak_phases, weak_csps
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (user_id, procedure_id) DO UPDATE
            SET phase_scores = $4,
                weak_phases = $5,
                weak_csps = $6
        """

        await self.db.execute(
            query,
            mastery_id, user_id, procedure_id,
            json.dumps(phase_averages), weak_phases, weak_csps
        )

        return await self.get_mastery(user_id, procedure_id)
