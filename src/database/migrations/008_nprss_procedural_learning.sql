-- migrations/007_nprss_procedural_learning.sql
-- NPRSS Procedural Learning System Integration
-- Extends NeuroSynth with procedural hierarchy, CSPs, and spaced repetition

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- PART 1: CORE PROCEDURAL HIERARCHY
-- =============================================================================

-- Procedures (Level 5) - Links to NeuroSynth documents
CREATE TABLE IF NOT EXISTS procedures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to NeuroSynth content
    source_document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    source_synthesis_id UUID,  -- Reference to synthesis that generated this

    -- Identification
    snomed_ct_code VARCHAR(50),
    icd10_pcs_code VARCHAR(10),
    cpt_code VARCHAR(10),
    name VARCHAR(500) NOT NULL,
    description TEXT,

    -- ACGME Classification
    subspecialty_domain VARCHAR(50) NOT NULL CHECK (subspecialty_domain IN (
        'brain_tumor', 'spine', 'cerebrovascular', 'functional',
        'pediatric', 'pain_peripheral', 'trauma', 'critical_care'
    )),
    complexity VARCHAR(20) NOT NULL CHECK (complexity IN ('routine', 'complex', 'advanced')),

    -- Milestone Targets
    milestone_pc_target INTEGER CHECK (milestone_pc_target BETWEEN 1 AND 5),
    milestone_mk_target INTEGER CHECK (milestone_mk_target BETWEEN 1 AND 5),

    -- Anatomical Context
    primary_target_structure_fma INTEGER,
    surgical_approach VARCHAR(100),

    -- Versioning
    version INTEGER DEFAULT 1,
    status VARCHAR(20) DEFAULT 'draft' CHECK (status IN ('draft', 'review', 'published', 'archived')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by UUID,

    -- Search
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(name, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(surgical_approach, '')), 'C')
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_procedures_search ON procedures USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_procedures_subspecialty ON procedures(subspecialty_domain);
CREATE INDEX IF NOT EXISTS idx_procedures_complexity ON procedures(complexity);
CREATE INDEX IF NOT EXISTS idx_procedures_status ON procedures(status);
CREATE INDEX IF NOT EXISTS idx_procedures_source_doc ON procedures(source_document_id);

-- Safe Entry Zones
CREATE TABLE IF NOT EXISTS safe_entry_zones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    anatomical_region VARCHAR(100),

    -- Quantitative Measurements
    mean_safe_depth_mm DECIMAL(6,2),
    min_safe_depth_mm DECIMAL(6,2),
    max_safe_depth_mm DECIMAL(6,2),
    length_mm DECIMAL(6,2),
    width_mm DECIMAL(6,2),

    -- Boundaries
    superior_boundary TEXT,
    inferior_boundary TEXT,
    lateral_boundary TEXT,
    medial_boundary TEXT,

    -- Critical Distances: [{structure: "pyramidal tract", distance_mm: 4.64}]
    distance_to_critical_structures JSONB DEFAULT '[]',

    -- Visualization
    diagram_asset_id UUID,
    coordinates_3d JSONB,

    -- Source
    source_reference TEXT,
    evidence_grade VARCHAR(10) CHECK (evidence_grade IN ('A', 'B', 'C', 'D', 'E')),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_safe_zones_region ON safe_entry_zones(anatomical_region);

-- Danger Zones
CREATE TABLE IF NOT EXISTS danger_zones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(200) NOT NULL,
    structures_at_risk TEXT[] NOT NULL,
    mechanism_of_injury TEXT,
    prevention_strategy TEXT,
    management_if_violated TEXT,

    -- Anatomical Reference
    anatomical_region VARCHAR(100),
    fma_ids INTEGER[],

    -- Visualization
    diagram_asset_id UUID,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_danger_zones_structures ON danger_zones USING GIN(structures_at_risk);

-- Procedure-Zone Associations
CREATE TABLE IF NOT EXISTS procedure_safe_zones (
    procedure_id UUID NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,
    safe_zone_id UUID NOT NULL REFERENCES safe_entry_zones(id) ON DELETE CASCADE,
    PRIMARY KEY (procedure_id, safe_zone_id)
);

CREATE TABLE IF NOT EXISTS procedure_danger_zones (
    procedure_id UUID NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,
    danger_zone_id UUID NOT NULL REFERENCES danger_zones(id) ON DELETE CASCADE,
    PRIMARY KEY (procedure_id, danger_zone_id)
);

-- Procedure Elements (Levels 4-0: phase, step, substep, task, motion)
CREATE TABLE IF NOT EXISTS procedure_elements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    procedure_id UUID NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,
    parent_id UUID REFERENCES procedure_elements(id) ON DELETE CASCADE,

    -- Link to NeuroSynth chunk (provenance)
    source_chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,

    -- Hierarchy
    element_type VARCHAR(20) NOT NULL CHECK (element_type IN ('phase', 'step', 'substep', 'task', 'motion')),
    granularity_level INTEGER NOT NULL CHECK (granularity_level BETWEEN 0 AND 4),
    sequence_order INTEGER NOT NULL,

    -- Content
    name VARCHAR(500) NOT NULL,
    description TEXT,
    critical_step BOOLEAN DEFAULT FALSE,

    -- Phase Classification (Learning System)
    phase_type VARCHAR(20) CHECK (phase_type IN ('architecture', 'approach', 'target', 'closure')),

    -- Anatomical References
    anatomical_structure_fma INTEGER,
    action_verb VARCHAR(100),
    safe_zone_refs UUID[],
    danger_zone_refs UUID[],

    -- Technical Details
    instrument_sequence JSONB DEFAULT '[]',
    ionm_requirements VARCHAR(100)[],

    -- Impeccable 7-Element Fields (for substeps)
    standard_measurements TEXT,
    trajectory_specification TEXT,
    instrument_specification TEXT,
    the_maneuver TEXT,
    bailout_protocol TEXT,
    visual_description JSONB,  -- {expected_view, landmarks, color_cues}

    -- Versioning
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_hierarchy CHECK (
        (element_type = 'phase' AND granularity_level = 4) OR
        (element_type = 'step' AND granularity_level = 3) OR
        (element_type = 'substep' AND granularity_level = 2) OR
        (element_type = 'task' AND granularity_level = 1) OR
        (element_type = 'motion' AND granularity_level = 0)
    )
);

CREATE INDEX IF NOT EXISTS idx_elements_procedure ON procedure_elements(procedure_id);
CREATE INDEX IF NOT EXISTS idx_elements_parent ON procedure_elements(parent_id);
CREATE INDEX IF NOT EXISTS idx_elements_type ON procedure_elements(element_type);
CREATE INDEX IF NOT EXISTS idx_elements_phase ON procedure_elements(phase_type);
CREATE INDEX IF NOT EXISTS idx_elements_critical ON procedure_elements(critical_step) WHERE critical_step = TRUE;
CREATE INDEX IF NOT EXISTS idx_elements_source_chunk ON procedure_elements(source_chunk_id);

-- Decision Branches
CREATE TABLE IF NOT EXISTS decision_branches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_element_id UUID NOT NULL REFERENCES procedure_elements(id) ON DELETE CASCADE,

    condition_type VARCHAR(50) CHECK (condition_type IN ('anatomical_variant', 'complication', 'finding', 'intraoperative')),
    condition_criteria JSONB NOT NULL,

    alternative_path_id UUID REFERENCES procedure_elements(id),

    evidence_grade VARCHAR(10),
    evidence_reference TEXT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_branches_source ON decision_branches(source_element_id);

-- =============================================================================
-- PART 2: LEARNING ENRICHMENT
-- =============================================================================

-- Phase Gates
CREATE TABLE IF NOT EXISTS phase_gates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    procedure_id UUID NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,

    from_phase VARCHAR(20) NOT NULL CHECK (from_phase IN ('architecture', 'approach', 'target')),
    to_phase VARCHAR(20) NOT NULL CHECK (to_phase IN ('approach', 'target', 'closure')),

    verification_questions TEXT[] NOT NULL,
    prerequisites TEXT[],

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_phase_transition CHECK (
        (from_phase = 'architecture' AND to_phase = 'approach') OR
        (from_phase = 'approach' AND to_phase = 'target') OR
        (from_phase = 'target' AND to_phase = 'closure')
    ),
    UNIQUE(procedure_id, from_phase, to_phase)
);

-- Critical Safety Points (CSPs)
CREATE TABLE IF NOT EXISTS critical_safety_points (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    procedure_id UUID NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,
    element_id UUID REFERENCES procedure_elements(id) ON DELETE SET NULL,

    csp_number INTEGER NOT NULL,
    phase_type VARCHAR(20) CHECK (phase_type IN ('architecture', 'approach', 'target', 'closure')),

    -- Trigger-Action Circuit
    when_action TEXT NOT NULL,
    stop_if_trigger TEXT NOT NULL,
    visual_cue TEXT,

    -- Consequence
    structure_at_risk TEXT NOT NULL,
    mechanism_of_injury TEXT,

    -- Recovery
    if_violated_action TEXT,

    -- Source Tracing
    derived_from_danger_zone_id UUID REFERENCES danger_zones(id),
    derived_from_safe_zone_id UUID REFERENCES safe_entry_zones(id),

    -- Learning Metadata
    retrieval_cue TEXT,
    common_errors TEXT[],

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(procedure_id, csp_number)
);

CREATE INDEX IF NOT EXISTS idx_csps_procedure ON critical_safety_points(procedure_id);
CREATE INDEX IF NOT EXISTS idx_csps_element ON critical_safety_points(element_id);

-- Visuospatial Anchors
CREATE TABLE IF NOT EXISTS visuospatial_anchors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    element_id UUID NOT NULL REFERENCES procedure_elements(id) ON DELETE CASCADE,

    -- From Visual Description
    expected_view TEXT,
    landmarks TEXT[],
    color_cues TEXT,

    -- Learning Enhancements
    mental_rotation_prompt TEXT,
    spatial_relationship TEXT,
    depth_reference TEXT,
    viewing_angle TEXT,

    -- 3D Coordinates (for AR/VR)
    coordinates_3d JSONB,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_anchors_element ON visuospatial_anchors(element_id);

-- Surgical Cards
CREATE TABLE IF NOT EXISTS surgical_cards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    procedure_id UUID NOT NULL REFERENCES procedures(id) ON DELETE CASCADE UNIQUE,

    -- Header
    title VARCHAR(255) NOT NULL,
    subtitle VARCHAR(255),
    approach VARCHAR(100),
    corridor VARCHAR(100),
    exam_relevance VARCHAR(100),

    -- Content (JSON for flexibility)
    card_rows JSONB NOT NULL,  -- [{phase, phase_label, key_actions[], anchor_or_csp, anchor_type}]
    csp_summary JSONB,         -- [{id, short_name, trigger}]

    -- Dictation Template
    dictation_template TEXT,

    -- Mantra
    mantra TEXT DEFAULT '4 folders -> 3-5 substeps -> visual anchors -> CSP triggers',

    -- Generation Metadata
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER DEFAULT 1
);

-- =============================================================================
-- PART 3: LEARNING SYSTEM (FSRS + Spaced Repetition)
-- =============================================================================

-- Learning Cards (NPRSS card types)
CREATE TABLE IF NOT EXISTS nprss_learning_cards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    procedure_id UUID NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,
    element_id UUID REFERENCES procedure_elements(id) ON DELETE SET NULL,
    csp_id UUID REFERENCES critical_safety_points(id) ON DELETE SET NULL,

    card_type VARCHAR(50) NOT NULL CHECK (card_type IN (
        'sequence', 'image', 'mcq', 'scenario', 'csp_trigger', 'dictation', 'safe_zone'
    )),

    -- Content
    prompt TEXT NOT NULL,
    answer TEXT NOT NULL,
    options JSONB,  -- For MCQ
    image_asset_id UUID,

    -- Metadata
    difficulty_preset DECIMAL(5,4) DEFAULT 0.3,
    tags TEXT[],

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nprss_cards_procedure ON nprss_learning_cards(procedure_id);
CREATE INDEX IF NOT EXISTS idx_nprss_cards_type ON nprss_learning_cards(card_type);
CREATE INDEX IF NOT EXISTS idx_nprss_cards_tags ON nprss_learning_cards USING GIN(tags);

-- FSRS Memory State (Per user per card)
CREATE TABLE IF NOT EXISTS nprss_card_memory_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    card_id UUID NOT NULL REFERENCES nprss_learning_cards(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,

    -- FSRS Parameters
    difficulty DECIMAL(5,4) DEFAULT 0.3,
    stability DECIMAL(10,4) DEFAULT 1.0,
    retrievability DECIMAL(5,4) DEFAULT 1.0,

    -- State
    state VARCHAR(20) DEFAULT 'new' CHECK (state IN ('new', 'learning', 'review', 'relearning')),
    step INTEGER DEFAULT 0,
    due_date TIMESTAMP WITH TIME ZONE,

    -- History
    review_count INTEGER DEFAULT 0,
    lapses INTEGER DEFAULT 0,
    last_review TIMESTAMP WITH TIME ZONE,

    UNIQUE(card_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_nprss_memory_user ON nprss_card_memory_state(user_id);
CREATE INDEX IF NOT EXISTS idx_nprss_memory_due ON nprss_card_memory_state(due_date) WHERE due_date IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_nprss_memory_state ON nprss_card_memory_state(state);

-- Review Logs
CREATE TABLE IF NOT EXISTS nprss_card_review_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    card_id UUID NOT NULL REFERENCES nprss_learning_cards(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,

    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 4),
    state_before VARCHAR(20),
    stability_after DECIMAL(10,4),
    difficulty_after DECIMAL(5,4),
    elapsed_days INTEGER,
    scheduled_days INTEGER,

    review_time TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_nprss_review_logs_card ON nprss_card_review_logs(card_id);
CREATE INDEX IF NOT EXISTS idx_nprss_review_logs_user ON nprss_card_review_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_nprss_review_logs_time ON nprss_card_review_logs(review_time);

-- Retrieval Schedules (R1-R7 Expansion)
CREATE TABLE IF NOT EXISTS retrieval_schedules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    procedure_id UUID NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,

    encoding_date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    target_retention_days INTEGER DEFAULT 180,

    -- Adaptive Parameters
    interval_multiplier DECIMAL(4,2) DEFAULT 1.0,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(user_id, procedure_id)
);

CREATE INDEX IF NOT EXISTS idx_schedules_user ON retrieval_schedules(user_id);

-- Retrieval Sessions
CREATE TABLE IF NOT EXISTS retrieval_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    schedule_id UUID NOT NULL REFERENCES retrieval_schedules(id) ON DELETE CASCADE,

    session_number INTEGER NOT NULL,
    scheduled_date TIMESTAMP WITH TIME ZONE NOT NULL,
    days_from_encoding INTEGER NOT NULL,

    -- Task
    retrieval_task VARCHAR(500),
    task_type VARCHAR(50) CHECK (task_type IN (
        'free_recall', 'cued_recall', 'recognition', 'rehearsal', 'elaboration', 'interleaved', 'application'
    )),
    estimated_duration_min INTEGER,

    -- Focus Areas
    focus_phases TEXT[],
    focus_csps INTEGER[],

    -- Completion
    completed BOOLEAN DEFAULT FALSE,
    completed_at TIMESTAMP WITH TIME ZONE,
    self_assessment_score INTEGER CHECK (self_assessment_score BETWEEN 1 AND 4),
    notes TEXT,

    UNIQUE(schedule_id, session_number)
);

CREATE INDEX IF NOT EXISTS idx_sessions_schedule ON retrieval_sessions(schedule_id);
CREATE INDEX IF NOT EXISTS idx_sessions_date ON retrieval_sessions(scheduled_date);
CREATE INDEX IF NOT EXISTS idx_sessions_pending ON retrieval_sessions(completed, scheduled_date) WHERE completed = FALSE;

-- Procedure Mastery
CREATE TABLE IF NOT EXISTS procedure_mastery (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    procedure_id UUID NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,

    -- Current State (1=NOT_YET, 2=DEVELOPING, 3=COMPETENT, 4=MASTERY)
    current_level INTEGER DEFAULT 1 CHECK (current_level BETWEEN 1 AND 4),

    -- Phase-Level Granularity
    phase_scores JSONB DEFAULT '{}',

    -- Weak Points
    weak_csps INTEGER[],
    weak_phases TEXT[],

    -- History
    assessment_history JSONB DEFAULT '[]',
    total_retrieval_sessions INTEGER DEFAULT 0,
    last_session_date TIMESTAMP WITH TIME ZONE,

    -- Predictions
    predicted_retention_score DECIMAL(4,3) DEFAULT 0.5,
    next_optimal_review TIMESTAMP WITH TIME ZONE,

    -- Entrustment (EPA) - Zwisch Scale
    entrustment_level INTEGER CHECK (entrustment_level BETWEEN 1 AND 4),

    UNIQUE(user_id, procedure_id)
);

CREATE INDEX IF NOT EXISTS idx_mastery_user ON procedure_mastery(user_id);
CREATE INDEX IF NOT EXISTS idx_mastery_level ON procedure_mastery(current_level);
CREATE INDEX IF NOT EXISTS idx_mastery_review ON procedure_mastery(next_optimal_review);

-- =============================================================================
-- PART 4: ASSESSMENT (Miller's Pyramid)
-- =============================================================================

-- Assessment Items
CREATE TABLE IF NOT EXISTS nprss_assessment_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    procedure_id UUID REFERENCES procedures(id) ON DELETE SET NULL,
    element_id UUID REFERENCES procedure_elements(id) ON DELETE SET NULL,

    miller_level VARCHAR(20) NOT NULL CHECK (miller_level IN ('knows', 'knows_how', 'shows_how', 'does')),
    item_type VARCHAR(50) NOT NULL CHECK (item_type IN ('mcq', 'sequence', 'scenario', 'labeled_anatomy', 'video_assessment')),

    prompt TEXT NOT NULL,
    correct_answer TEXT,
    options JSONB,
    scoring_rubric JSONB,

    difficulty_level VARCHAR(20) CHECK (difficulty_level IN ('easy', 'medium', 'hard')),
    tags TEXT[],

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_assessment_procedure ON nprss_assessment_items(procedure_id);
CREATE INDEX IF NOT EXISTS idx_assessment_miller ON nprss_assessment_items(miller_level);
CREATE INDEX IF NOT EXISTS idx_assessment_type ON nprss_assessment_items(item_type);

-- Assessment Responses
CREATE TABLE IF NOT EXISTS nprss_assessment_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id UUID NOT NULL REFERENCES nprss_assessment_items(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,

    response TEXT,
    score DECIMAL(5,2),
    time_taken_seconds INTEGER,

    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_responses_user ON nprss_assessment_responses(user_id);
CREATE INDEX IF NOT EXISTS idx_responses_item ON nprss_assessment_responses(item_id);

-- Entrustment Assessments (Workplace)
CREATE TABLE IF NOT EXISTS entrustment_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    procedure_id UUID NOT NULL REFERENCES procedures(id) ON DELETE CASCADE,
    assessor_id TEXT NOT NULL,

    assessment_date TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Zwisch Scale (1-4)
    entrustment_level INTEGER NOT NULL CHECK (entrustment_level BETWEEN 1 AND 4),

    -- Milestone Mapping
    milestone_pc_level INTEGER CHECK (milestone_pc_level BETWEEN 1 AND 5),
    milestone_mk_level INTEGER CHECK (milestone_mk_level BETWEEN 1 AND 5),

    narrative_feedback TEXT,
    verified BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entrustment_user ON entrustment_assessments(user_id);
CREATE INDEX IF NOT EXISTS idx_entrustment_procedure ON entrustment_assessments(procedure_id);

-- =============================================================================
-- PART 5: FUNCTIONS & TRIGGERS
-- =============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_procedures_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS procedures_updated_at ON procedures;
CREATE TRIGGER procedures_updated_at
    BEFORE UPDATE ON procedures
    FOR EACH ROW
    EXECUTE FUNCTION update_procedures_updated_at();

-- =============================================================================
-- PART 6: VIEWS
-- =============================================================================

-- Procedure summary view
CREATE OR REPLACE VIEW procedure_summary AS
SELECT
    p.id,
    p.name,
    p.subspecialty_domain,
    p.complexity,
    p.status,
    COUNT(DISTINCT pe.id) FILTER (WHERE pe.element_type = 'phase') AS phase_count,
    COUNT(DISTINCT pe.id) FILTER (WHERE pe.element_type = 'step') AS step_count,
    COUNT(DISTINCT pe.id) FILTER (WHERE pe.element_type = 'substep') AS substep_count,
    COUNT(DISTINCT csp.id) AS csp_count,
    EXISTS(SELECT 1 FROM surgical_cards sc WHERE sc.procedure_id = p.id) AS has_surgical_card,
    p.created_at
FROM procedures p
LEFT JOIN procedure_elements pe ON pe.procedure_id = p.id
LEFT JOIN critical_safety_points csp ON csp.procedure_id = p.id
GROUP BY p.id;

-- User learning progress view
CREATE OR REPLACE VIEW user_learning_progress AS
SELECT
    pm.user_id,
    pm.procedure_id,
    p.name AS procedure_name,
    pm.current_level,
    CASE pm.current_level
        WHEN 1 THEN 'NOT_YET'
        WHEN 2 THEN 'DEVELOPING'
        WHEN 3 THEN 'COMPETENT'
        WHEN 4 THEN 'MASTERY'
    END AS level_name,
    pm.phase_scores,
    pm.predicted_retention_score,
    pm.next_optimal_review,
    COUNT(DISTINCT rs.id) FILTER (WHERE rs.completed = TRUE) AS sessions_completed,
    COUNT(DISTINCT rs.id) FILTER (WHERE rs.completed = FALSE AND rs.scheduled_date <= NOW()) AS sessions_overdue,
    COUNT(DISTINCT cms.card_id) AS cards_total,
    COUNT(DISTINCT cms.card_id) FILTER (WHERE cms.state = 'review') AS cards_graduated
FROM procedure_mastery pm
JOIN procedures p ON p.id = pm.procedure_id
LEFT JOIN retrieval_schedules rsc ON rsc.user_id = pm.user_id AND rsc.procedure_id = pm.procedure_id
LEFT JOIN retrieval_sessions rs ON rs.schedule_id = rsc.id
LEFT JOIN nprss_learning_cards lc ON lc.procedure_id = pm.procedure_id
LEFT JOIN nprss_card_memory_state cms ON cms.card_id = lc.id AND cms.user_id = pm.user_id
GROUP BY pm.id, p.name;

-- Due cards view
CREATE OR REPLACE VIEW nprss_due_cards AS
SELECT
    cms.user_id,
    cms.card_id,
    lc.procedure_id,
    p.name AS procedure_name,
    lc.card_type,
    lc.prompt,
    cms.due_date,
    cms.state,
    cms.difficulty,
    cms.stability,
    EXTRACT(EPOCH FROM (NOW() - cms.due_date))/86400 AS days_overdue
FROM nprss_card_memory_state cms
JOIN nprss_learning_cards lc ON lc.id = cms.card_id
JOIN procedures p ON p.id = lc.procedure_id
WHERE cms.due_date <= NOW()
ORDER BY cms.due_date;

-- =============================================================================
-- PART 7: ADDITIONAL TABLES FOR ANALYTICS
-- =============================================================================

-- Review History (Detailed audit trail of all reviews)
CREATE TABLE IF NOT EXISTS review_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    card_id UUID NOT NULL REFERENCES nprss_learning_cards(id) ON DELETE CASCADE,

    -- Review outcome
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 4),

    -- Time tracking
    response_time_ms INTEGER,
    reviewed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- FSRS state snapshot (before update)
    difficulty_before FLOAT,
    stability_before FLOAT,
    retrievability_before FLOAT,

    -- FSRS state snapshot (after update)
    difficulty_after FLOAT,
    stability_after FLOAT,
    retrievability_after FLOAT,

    -- Scheduling
    interval_before_days FLOAT,
    interval_after_days FLOAT,

    -- Context
    review_mode TEXT DEFAULT 'standard',
    session_id UUID,
    device_type TEXT,

    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_review_history_user ON review_history(user_id);
CREATE INDEX IF NOT EXISTS idx_review_history_card ON review_history(card_id);
CREATE INDEX IF NOT EXISTS idx_review_history_time ON review_history(reviewed_at DESC);
CREATE INDEX IF NOT EXISTS idx_review_history_user_time ON review_history(user_id, reviewed_at DESC);

-- Study Sessions (Aggregated session analytics)
CREATE TABLE IF NOT EXISTS study_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,

    -- Session timing
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_seconds INTEGER,

    -- Session stats
    cards_reviewed INTEGER DEFAULT 0,
    cards_correct INTEGER DEFAULT 0,
    cards_incorrect INTEGER DEFAULT 0,
    cards_skipped INTEGER DEFAULT 0,

    -- Average metrics
    avg_response_time_ms FLOAT,
    avg_difficulty FLOAT,

    -- Session type and focus
    session_type TEXT NOT NULL DEFAULT 'daily_review',
    focus_procedure_id UUID REFERENCES procedures(id) ON DELETE SET NULL,
    focus_specialty TEXT,

    -- R-level progress
    r_levels_achieved TEXT[] DEFAULT '{}',

    device_type TEXT,
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_study_sessions_user ON study_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_study_sessions_start ON study_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_study_sessions_type ON study_sessions(session_type);

-- Add session_id FK to review_history after study_sessions exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
        WHERE constraint_name = 'fk_review_history_session'
    ) THEN
        ALTER TABLE review_history
        ADD CONSTRAINT fk_review_history_session
        FOREIGN KEY (session_id) REFERENCES study_sessions(id) ON DELETE SET NULL;
    END IF;
END $$;

-- Mastery Milestones (Achievement tracking)
CREATE TABLE IF NOT EXISTS mastery_milestones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,

    -- Milestone identification
    milestone_type TEXT NOT NULL,
    milestone_name TEXT NOT NULL,
    milestone_level INTEGER DEFAULT 1,

    -- Achievement details
    achieved_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Context
    procedure_id UUID REFERENCES procedures(id) ON DELETE SET NULL,
    specialty TEXT,

    -- Metrics at achievement
    metrics JSONB DEFAULT '{}'::jsonb,

    notified BOOLEAN DEFAULT FALSE,
    notified_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_milestones_user ON mastery_milestones(user_id);
CREATE INDEX IF NOT EXISTS idx_milestones_type ON mastery_milestones(milestone_type);
CREATE INDEX IF NOT EXISTS idx_milestones_achieved ON mastery_milestones(achieved_at DESC);

-- Unique constraint for milestone deduplication
CREATE UNIQUE INDEX IF NOT EXISTS idx_milestones_unique
ON mastery_milestones(user_id, milestone_type, milestone_name, COALESCE(procedure_id, '00000000-0000-0000-0000-000000000000'::uuid));

-- Socratic Prompts (Pre-generated Socratic questions)
CREATE TABLE IF NOT EXISTS socratic_prompts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    card_id UUID NOT NULL REFERENCES nprss_learning_cards(id) ON DELETE CASCADE,

    -- Socratic level
    level TEXT NOT NULL CHECK (level IN ('guided', 'reflective', 'challenging')),

    -- Prompt content
    prompt_text TEXT NOT NULL,
    hint_text TEXT,
    followup_prompts TEXT[] DEFAULT '{}',

    -- Expected response criteria
    expected_concepts TEXT[] DEFAULT '{}',
    key_terms TEXT[] DEFAULT '{}',
    min_response_length INTEGER DEFAULT 20,

    -- Quality metrics
    quality_score FLOAT DEFAULT 0.5,
    usage_count INTEGER DEFAULT 0,
    avg_user_score FLOAT,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_socratic_prompts_card ON socratic_prompts(card_id);
CREATE INDEX IF NOT EXISTS idx_socratic_prompts_level ON socratic_prompts(level);

-- Socratic Responses (User responses to Socratic prompts)
CREATE TABLE IF NOT EXISTS socratic_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    prompt_id UUID NOT NULL REFERENCES socratic_prompts(id) ON DELETE CASCADE,
    session_id UUID REFERENCES study_sessions(id) ON DELETE SET NULL,

    -- User's response
    response_text TEXT NOT NULL,
    response_time_ms INTEGER,

    -- Evaluation
    concepts_covered TEXT[] DEFAULT '{}',
    key_terms_used TEXT[] DEFAULT '{}',
    completeness_score FLOAT CHECK (completeness_score BETWEEN 0 AND 1),
    accuracy_score FLOAT CHECK (accuracy_score BETWEEN 0 AND 1),

    -- AI feedback
    ai_feedback TEXT,
    ai_score FLOAT,

    -- User self-assessment
    user_confidence INTEGER CHECK (user_confidence BETWEEN 1 AND 5),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_socratic_responses_user ON socratic_responses(user_id);
CREATE INDEX IF NOT EXISTS idx_socratic_responses_prompt ON socratic_responses(prompt_id);
CREATE INDEX IF NOT EXISTS idx_socratic_responses_time ON socratic_responses(created_at DESC);

-- =============================================================================
-- PART 8: HELPER FUNCTIONS
-- =============================================================================

-- Function: Get user study streak
CREATE OR REPLACE FUNCTION get_user_streak(p_user_id TEXT)
RETURNS INTEGER AS $$
DECLARE
    v_streak INTEGER := 0;
    v_current_date DATE := CURRENT_DATE;
    v_study_date DATE;
BEGIN
    FOR v_study_date IN
        SELECT DISTINCT DATE(started_at)
        FROM study_sessions
        WHERE user_id = p_user_id
        ORDER BY DATE(started_at) DESC
    LOOP
        IF v_study_date = v_current_date - v_streak THEN
            v_streak := v_streak + 1;
        ELSE
            EXIT;
        END IF;
    END LOOP;

    RETURN v_streak;
END;
$$ LANGUAGE plpgsql STABLE;

-- Function: Get procedure accuracy
CREATE OR REPLACE FUNCTION get_procedure_accuracy(p_user_id TEXT, p_procedure_id UUID)
RETURNS FLOAT AS $$
DECLARE
    v_accuracy FLOAT;
BEGIN
    SELECT
        CASE WHEN COUNT(*) > 0
             THEN SUM(CASE WHEN rating >= 3 THEN 1 ELSE 0 END)::FLOAT / COUNT(*)
             ELSE 0.0
        END INTO v_accuracy
    FROM review_history rh
    JOIN nprss_learning_cards lc ON rh.card_id = lc.id
    WHERE rh.user_id = p_user_id
      AND lc.procedure_id = p_procedure_id;

    RETURN COALESCE(v_accuracy, 0.0);
END;
$$ LANGUAGE plpgsql STABLE;

-- Function: Check and award milestones
CREATE OR REPLACE FUNCTION check_milestones(p_user_id TEXT)
RETURNS TABLE (
    milestone_type TEXT,
    milestone_name TEXT,
    newly_achieved BOOLEAN
) AS $$
DECLARE
    v_total_reviews INTEGER;
    v_streak INTEGER;
BEGIN
    SELECT COUNT(*) INTO v_total_reviews FROM review_history WHERE user_id = p_user_id;
    SELECT get_user_streak(p_user_id) INTO v_streak;

    -- Card count milestones
    IF v_total_reviews >= 100 THEN
        INSERT INTO mastery_milestones (user_id, milestone_type, milestone_name, metrics)
        VALUES (p_user_id, 'card_count', 'Century Reviewer', jsonb_build_object('cards', v_total_reviews))
        ON CONFLICT (user_id, milestone_type, milestone_name, COALESCE(procedure_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING;
    END IF;

    IF v_total_reviews >= 1000 THEN
        INSERT INTO mastery_milestones (user_id, milestone_type, milestone_name, metrics)
        VALUES (p_user_id, 'card_count', 'Millennium Scholar', jsonb_build_object('cards', v_total_reviews))
        ON CONFLICT (user_id, milestone_type, milestone_name, COALESCE(procedure_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING;
    END IF;

    -- Streak milestones
    IF v_streak >= 7 THEN
        INSERT INTO mastery_milestones (user_id, milestone_type, milestone_name, metrics)
        VALUES (p_user_id, 'streak_achievement', 'Week Warrior', jsonb_build_object('streak_days', v_streak))
        ON CONFLICT (user_id, milestone_type, milestone_name, COALESCE(procedure_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING;
    END IF;

    IF v_streak >= 30 THEN
        INSERT INTO mastery_milestones (user_id, milestone_type, milestone_name, metrics)
        VALUES (p_user_id, 'streak_achievement', 'Monthly Master', jsonb_build_object('streak_days', v_streak))
        ON CONFLICT (user_id, milestone_type, milestone_name, COALESCE(procedure_id, '00000000-0000-0000-0000-000000000000'::uuid)) DO NOTHING;
    END IF;

    -- Return milestones
    RETURN QUERY
    SELECT mm.milestone_type, mm.milestone_name,
           mm.achieved_at > NOW() - INTERVAL '24 hours' as newly_achieved
    FROM mastery_milestones mm
    WHERE mm.user_id = p_user_id
    ORDER BY mm.achieved_at DESC;
END;
$$ LANGUAGE plpgsql;
