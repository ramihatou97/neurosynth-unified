-- Migration: 009_neurosynth2_physics_schema
-- Description: Adds physics-aware tables for NeuroSynth 2.0 surgical reasoning module
-- Depends on: 008_nprss_procedural_learning.sql
-- Author: NeuroSynth 2.0 Integration
-- Date: 2026-01-05

BEGIN;

-- =============================================================================
-- 1. ANATOMICAL ENTITIES (The "Body")
-- =============================================================================
-- Core entity table storing physical properties of anatomical structures.
-- Used by the ClinicalReasoner to assess surgical risks.

CREATE TABLE IF NOT EXISTS anatomical_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,
    canonical_name TEXT NOT NULL,
    entity_cui TEXT,                            -- Link to UMLS if available
    aliases TEXT[] DEFAULT '{}',

    -- Physical Properties
    mobility TEXT DEFAULT 'fixed',              -- fixed, tethered_by_*, elastic, freely_mobile
    consistency TEXT DEFAULT 'soft_brain',      -- vascular, neural, soft_brain, firm_tumor, bone
    is_end_artery BOOLEAN DEFAULT FALSE,
    has_collaterals BOOLEAN DEFAULT TRUE,
    collateral_capacity TEXT DEFAULT 'moderate', -- none, poor, variable, moderate, rich
    vessel_diameter_mm REAL,
    territory_supplied TEXT[] DEFAULT '{}',     -- Regions supplied by this entity

    -- Surgical Properties
    eloquence_grade TEXT DEFAULT 'non_eloquent', -- non_eloquent, near_eloquent, eloquent
    retraction_tolerance TEXT DEFAULT 'moderate', -- none, minimal, moderate, significant
    sacrifice_safety TEXT DEFAULT 'variable',    -- never, with_collaterals, variable, acceptable
    coagulation_tolerance TEXT DEFAULT 'moderate',

    -- Spatial Context (MNI coordinates, region info)
    spatial_context JSONB DEFAULT '{}'::jsonb,

    -- Confidence & Provenance
    confidence FLOAT DEFAULT 0.5,
    source_chunk_ids TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 2. SURGICAL CORRIDORS (The "Path")
-- =============================================================================
-- Defines surgical approaches with structure sequences and risk profiles.

CREATE TABLE IF NOT EXISTS surgical_corridors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL UNIQUE,                  -- e.g., 'pterional', 'retrosigmoid'
    display_name TEXT NOT NULL,
    approach_type TEXT,                         -- craniotomy, endoscopic, minimally_invasive
    category TEXT,                              -- cranial_base, supratentorial, infratentorial
    subspecialty TEXT,                          -- vascular, tumor, functional

    -- Sequence & Risk
    structure_sequence TEXT[] NOT NULL,         -- Ordered list of structures encountered
    structures_at_risk TEXT[] DEFAULT '{}',     -- Vulnerable structures
    critical_steps JSONB DEFAULT '[]'::jsonb,   -- Step number -> risk info

    -- Positioning
    patient_position TEXT,                      -- supine, lateral, prone, park_bench
    head_position TEXT,                         -- neutral, rotated_30_contralateral, extended

    -- Requirements
    required_monitoring TEXT[] DEFAULT '{}',    -- SSEP, MEP, EEG, EMG, etc.
    required_equipment TEXT[] DEFAULT '{}',     -- microscope, endoscope, high_speed_drill

    -- Indications
    primary_indications TEXT[] DEFAULT '{}',
    contraindications TEXT[] DEFAULT '{}',

    -- Evidence
    evidence_level TEXT DEFAULT 'expert_opinion', -- I, II, III, IV, expert_opinion

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 3. CLINICAL PRINCIPLES (The "Logic")
-- =============================================================================
-- IF-THEN rules encoding surgical safety knowledge.

CREATE TABLE IF NOT EXISTS clinical_principles (
    id TEXT PRIMARY KEY,                        -- e.g., 'VASC_001', 'NEURO_003'
    name TEXT NOT NULL UNIQUE,
    statement TEXT NOT NULL,                    -- Human-readable summary

    -- Rule Logic
    antecedent TEXT NOT NULL,                   -- IF condition (parsed expression)
    consequent TEXT NOT NULL,                   -- THEN result
    mechanism TEXT,                             -- WHY (physiological explanation)

    -- Classification
    domain TEXT NOT NULL,                       -- vascular, neural, surgical_technique
    category TEXT,                              -- perforator, eloquent, retraction, thermal
    severity TEXT DEFAULT 'warning',            -- critical, high, moderate, low, warning

    -- Exceptions & Examples
    exceptions JSONB DEFAULT '[]'::jsonb,       -- Conditions that override this rule
    examples JSONB DEFAULT '[]'::jsonb,         -- Case examples

    -- Triggers
    trigger_entities TEXT[] DEFAULT '{}',       -- What entities can trigger this
    trigger_actions TEXT[] DEFAULT '{}',        -- What actions can trigger this (retract, sacrifice, coagulate)

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    evidence_level TEXT DEFAULT 'expert_opinion', -- Ib, II, III, IV

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 4. CAUSAL EDGES (The "Reasoning")
-- =============================================================================
-- Relationships between anatomical entities for graph-based reasoning.

CREATE TABLE IF NOT EXISTS causal_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_entity TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    relation_type TEXT NOT NULL,                -- supplies, drains, tethers, innervates, adjacent_to

    -- Causal Chain
    mechanism_chain JSONB DEFAULT '[]'::jsonb,  -- Step-by-step causal mechanism

    -- Quantification
    probability FLOAT,                          -- 0.0-1.0, likelihood of relationship
    effect_magnitude TEXT,                      -- minimal, moderate, severe, catastrophic
    latency TEXT,                               -- immediate, minutes, hours, days
    reversibility TEXT,                         -- reversible, partially_reversible, irreversible

    -- Confidence
    confidence FLOAT DEFAULT 0.5,
    evidence_sources TEXT[] DEFAULT '{}',

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(source_entity, target_entity, relation_type)
);

-- =============================================================================
-- 5. SIMULATION SESSIONS (The "Memory")
-- =============================================================================
-- Logs of surgical simulations for analysis and learning.

CREATE TABLE IF NOT EXISTS simulation_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Session Info
    patient_id TEXT,                            -- Optional patient identifier
    approach TEXT NOT NULL,
    target_pathology TEXT NOT NULL,

    -- Results
    verdict TEXT,                               -- SAFE, CAUTION, HIGH_RISK, CONTRAINDICATED
    confidence FLOAT,

    -- Detailed Logs
    steps JSONB DEFAULT '[]'::jsonb,            -- Array of SimulationStep objects
    final_state JSONB DEFAULT '{}'::jsonb,      -- Final PatientState

    -- Metrics
    total_steps INTEGER,
    highest_risk_level TEXT,
    execution_time_ms INTEGER,
    warnings TEXT[] DEFAULT '{}',
    recommendations TEXT[] DEFAULT '{}',
    data_gaps TEXT[] DEFAULT '{}',

    -- Metadata
    patient_factors JSONB DEFAULT '{}'::jsonb,  -- age, comorbidities, etc.
    notes TEXT,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Anatomical entities
CREATE INDEX IF NOT EXISTS idx_ns2_entities_name ON anatomical_entities(name);
CREATE INDEX IF NOT EXISTS idx_ns2_entities_cui ON anatomical_entities(entity_cui) WHERE entity_cui IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ns2_entities_end_artery ON anatomical_entities(is_end_artery) WHERE is_end_artery = TRUE;
CREATE INDEX IF NOT EXISTS idx_ns2_entities_eloquence ON anatomical_entities(eloquence_grade);

-- Surgical corridors
CREATE INDEX IF NOT EXISTS idx_ns2_corridors_name ON surgical_corridors(name);
CREATE INDEX IF NOT EXISTS idx_ns2_corridors_category ON surgical_corridors(category);

-- Clinical principles
CREATE INDEX IF NOT EXISTS idx_ns2_principles_domain ON clinical_principles(domain);
CREATE INDEX IF NOT EXISTS idx_ns2_principles_severity ON clinical_principles(severity);
CREATE INDEX IF NOT EXISTS idx_ns2_principles_triggers ON clinical_principles USING GIN(trigger_entities);
CREATE INDEX IF NOT EXISTS idx_ns2_principles_actions ON clinical_principles USING GIN(trigger_actions);
CREATE INDEX IF NOT EXISTS idx_ns2_principles_active ON clinical_principles(is_active) WHERE is_active = TRUE;

-- Causal edges
CREATE INDEX IF NOT EXISTS idx_ns2_edges_source ON causal_edges(source_entity);
CREATE INDEX IF NOT EXISTS idx_ns2_edges_target ON causal_edges(target_entity);
CREATE INDEX IF NOT EXISTS idx_ns2_edges_relation ON causal_edges(relation_type);

-- Simulation sessions
CREATE INDEX IF NOT EXISTS idx_ns2_simulations_approach ON simulation_sessions(approach);
CREATE INDEX IF NOT EXISTS idx_ns2_simulations_verdict ON simulation_sessions(verdict);
CREATE INDEX IF NOT EXISTS idx_ns2_simulations_created ON simulation_sessions(created_at DESC);

-- =============================================================================
-- TRIGGERS FOR updated_at
-- =============================================================================

CREATE OR REPLACE FUNCTION update_ns2_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_anatomical_entities_updated_at ON anatomical_entities;
CREATE TRIGGER update_anatomical_entities_updated_at
    BEFORE UPDATE ON anatomical_entities
    FOR EACH ROW EXECUTE FUNCTION update_ns2_updated_at();

DROP TRIGGER IF EXISTS update_surgical_corridors_updated_at ON surgical_corridors;
CREATE TRIGGER update_surgical_corridors_updated_at
    BEFORE UPDATE ON surgical_corridors
    FOR EACH ROW EXECUTE FUNCTION update_ns2_updated_at();

DROP TRIGGER IF EXISTS update_clinical_principles_updated_at ON clinical_principles;
CREATE TRIGGER update_clinical_principles_updated_at
    BEFORE UPDATE ON clinical_principles
    FOR EACH ROW EXECUTE FUNCTION update_ns2_updated_at();

COMMIT;
