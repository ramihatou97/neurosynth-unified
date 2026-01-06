-- ============================================================================
-- Migration: 002_frcsc_learning_enhancements.sql
-- Strategy: EXTEND existing tables, ADD only truly new ones
-- Safe: All changes are additive, no data loss
-- ============================================================================

-- ============================================================================
-- PHASE 1: EXTEND EXISTING TABLES
-- These ALTER statements are safe - they add columns with defaults
-- ============================================================================

-- Extend existing learning cards table (if it exists)
-- CORRECTED: Uses actual existing table names from neurosynth-unified
DO $$
BEGIN
    -- Add card_track for dual-track FSRS to nprss_learning_cards
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'nprss_learning_cards') THEN
        ALTER TABLE nprss_learning_cards
            ADD COLUMN IF NOT EXISTS card_track VARCHAR(20) DEFAULT 'factual',
            ADD COLUMN IF NOT EXISTS yield_rating INTEGER DEFAULT 1,
            ADD COLUMN IF NOT EXISTS source_question_id UUID,
            ADD COLUMN IF NOT EXISTS quality_score FLOAT;
    END IF;

    -- Add enhanced fields to memory states (nprss_card_memory_state)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'nprss_card_memory_state') THEN
        ALTER TABLE nprss_card_memory_state
            ADD COLUMN IF NOT EXISTS card_track VARCHAR(20) DEFAULT 'factual';
    END IF;

    -- Extend existing procedure_mastery tracking
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'procedure_mastery') THEN
        ALTER TABLE procedure_mastery
            ADD COLUMN IF NOT EXISTS entrustment_level_enhanced VARCHAR(50) DEFAULT 'observation',
            ADD COLUMN IF NOT EXISTS acgme_milestone VARCHAR(50);
    END IF;

END $$;

-- ============================================================================
-- PHASE 2: NEW TABLES - Question Bank System
-- ============================================================================

-- Question categories (FRCSC exam structure)
CREATE TABLE IF NOT EXISTS frcsc_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL UNIQUE,
    display_name VARCHAR(200) NOT NULL,
    exam_weight FLOAT DEFAULT 0.0,
    description TEXT,
    parent_id UUID REFERENCES frcsc_categories(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Question bank
CREATE TABLE IF NOT EXISTS frcsc_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_code VARCHAR(50) UNIQUE,
    category_id UUID REFERENCES frcsc_categories(id),

    stem TEXT NOT NULL,
    options JSONB,  -- MCQ options
    answer TEXT,
    explanation TEXT,

    format VARCHAR(20) DEFAULT 'short_answer' CHECK (format IN ('mcq', 'short_answer', 'essay', 'oral')),
    yield_rating INTEGER DEFAULT 1 CHECK (yield_rating BETWEEN 1 AND 3),
    cognitive_level INTEGER DEFAULT 1 CHECK (cognitive_level BETWEEN 1 AND 6),

    -- IRT parameters for adaptive testing
    difficulty_param FLOAT DEFAULT 0.0,  -- IRT b parameter
    discrimination_param FLOAT DEFAULT 1.0,  -- IRT a parameter
    guessing_param FLOAT DEFAULT 0.25,  -- IRT c parameter

    year_asked INTEGER,
    source VARCHAR(200),
    source_document_id UUID,

    key_points JSONB DEFAULT '[]'::jsonb,
    numerical_facts JSONB DEFAULT '[]'::jsonb,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Question attempts
CREATE TABLE IF NOT EXISTS frcsc_question_attempts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    question_id UUID NOT NULL REFERENCES frcsc_questions(id) ON DELETE CASCADE,

    user_answer TEXT,
    is_correct BOOLEAN,
    score FLOAT,
    time_spent_seconds INTEGER,

    -- IRT ability estimate after this attempt
    ability_estimate FLOAT,

    feedback TEXT,
    attempted_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- PHASE 3: NEW TABLES - Enhanced CSP Quiz System (extends existing CSPs)
-- ============================================================================

-- CSP quiz results (complements existing critical_safety_points table)
CREATE TABLE IF NOT EXISTS csp_quiz_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    csp_id UUID NOT NULL REFERENCES critical_safety_points(id) ON DELETE CASCADE,

    quiz_type VARCHAR(30) NOT NULL CHECK (quiz_type IN ('trigger_recognition', 'action_selection', 'consequence_awareness', 'rapid_fire')),
    is_correct BOOLEAN NOT NULL,
    response_time_ms INTEGER,
    achieved_automaticity BOOLEAN DEFAULT FALSE,

    attempted_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- PHASE 4: NEW TABLES - Gamification System
-- ============================================================================

-- User streaks
CREATE TABLE IF NOT EXISTS user_streaks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE,

    current_streak INTEGER NOT NULL DEFAULT 0,
    longest_streak INTEGER NOT NULL DEFAULT 0,
    streak_freezes_available INTEGER NOT NULL DEFAULT 1,

    last_study_date DATE,
    freeze_used_dates JSONB DEFAULT '[]'::jsonb,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Badge definitions
CREATE TABLE IF NOT EXISTS badges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    badge_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    category VARCHAR(30) NOT NULL CHECK (category IN ('streak', 'mastery', 'achievement', 'special')),
    icon VARCHAR(10),
    points INTEGER DEFAULT 0,
    requirement JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Earned badges
CREATE TABLE IF NOT EXISTS user_badges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    badge_id UUID NOT NULL REFERENCES badges(id) ON DELETE CASCADE,
    earned_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, badge_id)
);

-- User points
CREATE TABLE IF NOT EXISTS user_points (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE,

    total_points INTEGER NOT NULL DEFAULT 0,
    weekly_points INTEGER NOT NULL DEFAULT 0,
    monthly_points INTEGER NOT NULL DEFAULT 0,

    review_points INTEGER DEFAULT 0,
    quiz_points INTEGER DEFAULT 0,
    streak_points INTEGER DEFAULT 0,
    badge_points INTEGER DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- PHASE 5: NEW TABLES - Exam Simulation
-- ============================================================================

CREATE TABLE IF NOT EXISTS exam_simulations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,

    exam_type VARCHAR(20) NOT NULL CHECK (exam_type IN ('written', 'oral', 'mock_full')),
    status VARCHAR(20) DEFAULT 'in_progress' CHECK (status IN ('in_progress', 'completed', 'abandoned')),

    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    time_limit_minutes INTEGER,
    actual_duration_minutes INTEGER,

    -- Results
    total_questions INTEGER,
    correct_answers INTEGER,
    score_percentage FLOAT,
    pass_threshold FLOAT DEFAULT 0.70,
    passed BOOLEAN,

    -- Category breakdown
    category_scores JSONB DEFAULT '{}'::jsonb,

    -- Questions used
    question_ids JSONB DEFAULT '[]'::jsonb,
    responses JSONB DEFAULT '[]'::jsonb
);

-- ============================================================================
-- PHASE 6: INDEXES
-- ============================================================================

-- Question bank indexes
CREATE INDEX IF NOT EXISTS idx_frcsc_questions_category ON frcsc_questions(category_id);
CREATE INDEX IF NOT EXISTS idx_frcsc_questions_yield ON frcsc_questions(yield_rating DESC);
CREATE INDEX IF NOT EXISTS idx_frcsc_questions_year ON frcsc_questions(year_asked);
CREATE INDEX IF NOT EXISTS idx_frcsc_attempts_user ON frcsc_question_attempts(user_id);
CREATE INDEX IF NOT EXISTS idx_frcsc_attempts_question ON frcsc_question_attempts(question_id);

-- CSP indexes
CREATE INDEX IF NOT EXISTS idx_csp_quiz_user ON csp_quiz_results(user_id);
CREATE INDEX IF NOT EXISTS idx_csp_quiz_csp ON csp_quiz_results(csp_id);

-- Gamification indexes
CREATE INDEX IF NOT EXISTS idx_user_badges_user ON user_badges(user_id);
CREATE INDEX IF NOT EXISTS idx_exam_sim_user ON exam_simulations(user_id);
CREATE INDEX IF NOT EXISTS idx_exam_sim_status ON exam_simulations(user_id, status);

-- ============================================================================
-- PHASE 7: SEED DATA
-- ============================================================================

-- FRCSC Categories
INSERT INTO frcsc_categories (name, display_name, exam_weight) VALUES
    ('neuro_oncology', 'Neuro-Oncology', 0.20),
    ('vascular', 'Cerebrovascular', 0.17),
    ('trauma', 'Trauma', 0.13),
    ('spine', 'Spine', 0.13),
    ('pediatrics', 'Pediatrics', 0.10),
    ('functional', 'Functional', 0.08),
    ('peripheral_nerve', 'Peripheral Nerve', 0.05),
    ('anatomy', 'Anatomy', 0.04),
    ('neurology', 'Neurology/Neuro-critical Care', 0.04),
    ('radiology', 'Radiology', 0.03),
    ('pathology', 'Pathology', 0.02),
    ('research', 'Research/Statistics', 0.01)
ON CONFLICT (name) DO NOTHING;

-- Default badges
INSERT INTO badges (badge_id, name, description, category, icon, points, requirement) VALUES
    ('week_warrior', 'Week Warrior', '7-day study streak', 'streak', '🔥', 50, '{"streak_days": 7}'::jsonb),
    ('monthly_master', 'Monthly Master', '30-day study streak', 'streak', '🏆', 200, '{"streak_days": 30}'::jsonb),
    ('century_club', 'Century Club', '100-day study streak', 'streak', '💯', 500, '{"streak_days": 100}'::jsonb),
    ('yearly_legend', 'Yearly Legend', '365-day study streak', 'streak', '👑', 2000, '{"streak_days": 365}'::jsonb),
    ('first_blood', 'First Blood', 'Complete first review session', 'achievement', '🎯', 10, '{"reviews": 1}'::jsonb),
    ('safety_first', 'Safety First', '95%+ CSP accuracy', 'achievement', '🛡️', 300, '{"csp_accuracy": 0.95}'::jsonb),
    ('high_yield_hunter', 'High Yield Hunter', 'Master all 3-star topics', 'achievement', '⭐', 500, '{"high_yield_mastery": 1.0}'::jsonb),
    ('exam_ready', 'Exam Ready', 'Pass mock FRCSC exam', 'achievement', '📝', 400, '{"exam_passed": true}'::jsonb),
    ('speed_demon', 'Speed Demon', '100 CSP automaticity responses', 'achievement', '⚡', 250, '{"automaticity_count": 100}'::jsonb),
    ('vascular_expert', 'Vascular Expert', '90% vascular mastery', 'mastery', '🧠', 200, '{"category_mastery": {"vascular": 0.9}}'::jsonb),
    ('tumor_master', 'Tumor Master', '90% neuro-oncology mastery', 'mastery', '🎗️', 200, '{"category_mastery": {"neuro_oncology": 0.9}}'::jsonb),
    ('spine_specialist', 'Spine Specialist', '90% spine mastery', 'mastery', '🦴', 200, '{"category_mastery": {"spine": 0.9}}'::jsonb),
    ('perfect_week', 'Perfect Week', '7 days 100% accuracy', 'special', '⭐', 300, '{"perfect_days": 7}'::jsonb),
    ('comeback_kid', 'Comeback Kid', 'Rebuild 7-day streak after break', 'special', '💪', 100, '{"rebuilt_streak": 7}'::jsonb)
ON CONFLICT (badge_id) DO NOTHING;

-- ============================================================================
-- MIGRATION COMPLETE
-- ============================================================================
