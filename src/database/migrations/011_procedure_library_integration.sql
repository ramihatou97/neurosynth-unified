-- ============================================================================
-- Migration 011: Procedure-Library Integration
-- ============================================================================
-- This migration creates the foundation for procedure-centric library browsing.
-- It links library content (chunks/images) to surgical procedures with temporal
-- phase awareness and clinical intelligence extraction.
-- ============================================================================

-- ============================================================================
-- 1. PROCEDURE TAXONOMY (extends existing procedures table)
-- ============================================================================

-- Master procedure taxonomy for library organization
CREATE TABLE IF NOT EXISTS procedure_taxonomy (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(100) UNIQUE NOT NULL,           -- "pterional-craniotomy"
    name VARCHAR(200) NOT NULL,                   -- "Pterional Craniotomy"
    description TEXT,

    -- Hierarchy
    parent_id INTEGER REFERENCES procedure_taxonomy(id) ON DELETE SET NULL,
    level INTEGER NOT NULL DEFAULT 1,             -- 1=category, 2=procedure, 3=variant

    -- Classification
    specialty VARCHAR(50) NOT NULL,               -- "vascular", "skull_base", "spine"
    acgme_category VARCHAR(100),                  -- "Cranial - Vascular"
    complexity_level INTEGER CHECK (complexity_level BETWEEN 1 AND 5),

    -- Semantic tags for matching
    anatomy_tags TEXT[] NOT NULL DEFAULT '{}',    -- ["sylvian_fissure", "mca", "sphenoid"]
    pathology_tags TEXT[] NOT NULL DEFAULT '{}',  -- ["aneurysm", "avm"]
    approach_tags TEXT[] NOT NULL DEFAULT '{}',   -- ["pterional", "frontotemporal"]
    keyword_aliases TEXT[] NOT NULL DEFAULT '{}', -- ["pterional", "frontotemporal craniotomy"]

    -- Link to NPRSS (if procedure exists there)
    nprss_procedure_id UUID REFERENCES procedures(id),

    -- Metadata
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_proc_taxonomy_slug ON procedure_taxonomy(slug);
CREATE INDEX idx_proc_taxonomy_specialty ON procedure_taxonomy(specialty);
CREATE INDEX idx_proc_taxonomy_parent ON procedure_taxonomy(parent_id);
CREATE INDEX idx_proc_taxonomy_anatomy ON procedure_taxonomy USING GIN(anatomy_tags);
CREATE INDEX idx_proc_taxonomy_pathology ON procedure_taxonomy USING GIN(pathology_tags);

-- ============================================================================
-- 2. SURGICAL PHASES (Temporal Logic)
-- ============================================================================

-- Standard surgical phases (aligned with NPRSS 4-phase framework)
CREATE TYPE surgical_phase_enum AS ENUM (
    'PLANNING',           -- Phase 0: Indications, imaging review, approach selection
    'POSITIONING',        -- Phase 1: Patient positioning, head fixation
    'EXPOSURE',           -- Phase 2: Incision, soft tissue, bone work (APPROACH in NPRSS)
    'INTRADURAL',         -- Phase 3: Target handling, pathology management (TARGET in NPRSS)
    'CLOSURE',            -- Phase 4: Reconstruction, wound closure
    'POSTOPERATIVE'       -- Phase 5: Immediate postop concerns
);

-- Procedure-specific step definitions
CREATE TABLE IF NOT EXISTS procedure_steps (
    id SERIAL PRIMARY KEY,
    procedure_id INTEGER NOT NULL REFERENCES procedure_taxonomy(id) ON DELETE CASCADE,

    -- Step identity
    step_order INTEGER NOT NULL,                  -- 1, 2, 3...
    phase surgical_phase_enum NOT NULL,           -- Which phase this step belongs to
    name VARCHAR(200) NOT NULL,                   -- "Pterion drilling"
    description TEXT,

    -- Risk assessment
    criticality_score INTEGER CHECK (criticality_score BETWEEN 1 AND 10),
    danger_structures TEXT[],                     -- ["middle meningeal artery", "orbit"]

    -- Content hints for matching
    keywords TEXT[] NOT NULL DEFAULT '{}',

    UNIQUE(procedure_id, step_order)
);

CREATE INDEX idx_proc_steps_procedure ON procedure_steps(procedure_id);
CREATE INDEX idx_proc_steps_phase ON procedure_steps(phase);

-- ============================================================================
-- 3. CHUNK-PROCEDURE RELEVANCE (The Core Link)
-- ============================================================================

-- Links chunks to procedures with rich metadata
CREATE TABLE IF NOT EXISTS chunk_procedure_relevance (
    id SERIAL PRIMARY KEY,
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    procedure_id INTEGER NOT NULL REFERENCES procedure_taxonomy(id) ON DELETE CASCADE,

    -- Relevance scoring
    relevance_score FLOAT NOT NULL CHECK (relevance_score BETWEEN 0 AND 1),
    confidence FLOAT DEFAULT 0.5,                 -- How confident is this mapping?

    -- Content classification
    content_type VARCHAR(50) NOT NULL,            -- "anatomy", "technique", "complication", "evidence", "pearl", "pitfall"
    surgical_phase surgical_phase_enum,           -- Which phase is this content about?
    step_id INTEGER REFERENCES procedure_steps(id),

    -- Clinical intelligence flags
    is_pearl BOOLEAN DEFAULT FALSE,               -- Teaching wisdom
    is_pitfall BOOLEAN DEFAULT FALSE,             -- Danger warning
    is_critical BOOLEAN DEFAULT FALSE,            -- Must-read content

    -- Visual assessment
    has_key_image BOOLEAN DEFAULT FALSE,          -- Linked to important image
    visual_priority INTEGER DEFAULT 0,            -- Higher = show image more prominently

    -- Source tracking
    extraction_method VARCHAR(50),                -- "semantic", "keyword", "manual", "llm"
    extracted_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(chunk_id, procedure_id)
);

CREATE INDEX idx_chunk_proc_chunk ON chunk_procedure_relevance(chunk_id);
CREATE INDEX idx_chunk_proc_procedure ON chunk_procedure_relevance(procedure_id);
CREATE INDEX idx_chunk_proc_type ON chunk_procedure_relevance(content_type);
CREATE INDEX idx_chunk_proc_phase ON chunk_procedure_relevance(surgical_phase);
CREATE INDEX idx_chunk_proc_relevance ON chunk_procedure_relevance(relevance_score DESC);
CREATE INDEX idx_chunk_proc_pearls ON chunk_procedure_relevance(procedure_id) WHERE is_pearl = TRUE;
CREATE INDEX idx_chunk_proc_pitfalls ON chunk_procedure_relevance(procedure_id) WHERE is_pitfall = TRUE;

-- ============================================================================
-- 4. IMAGE-PROCEDURE RELEVANCE
-- ============================================================================

-- Links images to procedures (separate from chunks for visual-first browsing)
CREATE TABLE IF NOT EXISTS image_procedure_relevance (
    id SERIAL PRIMARY KEY,
    image_id UUID NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    procedure_id INTEGER NOT NULL REFERENCES procedure_taxonomy(id) ON DELETE CASCADE,

    -- Relevance
    relevance_score FLOAT NOT NULL CHECK (relevance_score BETWEEN 0 AND 1),

    -- Classification
    image_role VARCHAR(50),                       -- "anatomy_reference", "surgical_photo", "positioning", "complication"
    surgical_phase surgical_phase_enum,
    step_id INTEGER REFERENCES procedure_steps(id),

    -- Display priority
    display_priority INTEGER DEFAULT 0,           -- Higher = show first in carousel
    is_hero_image BOOLEAN DEFAULT FALSE,          -- The "poster" image for this procedure

    UNIQUE(image_id, procedure_id)
);

CREATE INDEX idx_img_proc_image ON image_procedure_relevance(image_id);
CREATE INDEX idx_img_proc_procedure ON image_procedure_relevance(procedure_id);
CREATE INDEX idx_img_proc_priority ON image_procedure_relevance(procedure_id, display_priority DESC);

-- ============================================================================
-- 5. CLINICAL ENTITIES (For Case Prep Mode)
-- ============================================================================

-- Clinical scenarios that map to procedures
CREATE TABLE IF NOT EXISTS clinical_entities (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(50) NOT NULL,             -- "pathology", "anatomy_variant", "complication"
    name VARCHAR(200) NOT NULL,                   -- "MCA Aneurysm", "Prefixed Chiasm"
    slug VARCHAR(100) UNIQUE NOT NULL,

    -- Semantic matching
    synonyms TEXT[] NOT NULL DEFAULT '{}',        -- ["middle cerebral artery aneurysm", "MCA aneurysm"]
    icd10_codes TEXT[],                           -- ["I67.1"]
    umls_cuis TEXT[],                             -- ["C0917996"]

    -- Clinical metadata
    typical_location VARCHAR(100),                -- "M1 bifurcation"
    typical_presentation TEXT,

    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_clinical_entity_type ON clinical_entities(entity_type);
CREATE INDEX idx_clinical_entity_slug ON clinical_entities(slug);
CREATE INDEX idx_clinical_entity_synonyms ON clinical_entities USING GIN(synonyms);

-- Link clinical entities to procedures
CREATE TABLE IF NOT EXISTS clinical_entity_procedures (
    clinical_entity_id INTEGER NOT NULL REFERENCES clinical_entities(id) ON DELETE CASCADE,
    procedure_id INTEGER NOT NULL REFERENCES procedure_taxonomy(id) ON DELETE CASCADE,

    relevance_score FLOAT DEFAULT 1.0,            -- How often is this procedure used for this entity?
    is_primary BOOLEAN DEFAULT FALSE,             -- Is this THE procedure for this entity?

    PRIMARY KEY (clinical_entity_id, procedure_id)
);

-- ============================================================================
-- 6. SEED DATA: Core Neurosurgical Procedures
-- ============================================================================

-- Level 1: Specialties (categories)
INSERT INTO procedure_taxonomy (slug, name, level, specialty, description) VALUES
('vascular', 'Vascular Neurosurgery', 1, 'vascular', 'Cerebrovascular procedures'),
('tumor', 'Neuro-oncology', 1, 'tumor', 'Brain and spine tumor surgery'),
('skull-base', 'Skull Base Surgery', 1, 'skull_base', 'Skull base approaches'),
('spine', 'Spine Surgery', 1, 'spine', 'Spinal procedures'),
('functional', 'Functional Neurosurgery', 1, 'functional', 'DBS, epilepsy, pain'),
('pediatric', 'Pediatric Neurosurgery', 1, 'pediatric', 'Pediatric procedures'),
('trauma', 'Neurotrauma', 1, 'trauma', 'Traumatic brain and spine injury')
ON CONFLICT (slug) DO NOTHING;

-- Level 2: Core Procedures (vascular)
INSERT INTO procedure_taxonomy (slug, name, parent_id, level, specialty, acgme_category, complexity_level, anatomy_tags, pathology_tags, approach_tags, keyword_aliases) VALUES
('pterional-craniotomy', 'Pterional Craniotomy',
    (SELECT id FROM procedure_taxonomy WHERE slug = 'vascular'), 2, 'vascular',
    'Cranial - Vascular', 3,
    ARRAY['sylvian_fissure', 'mca', 'sphenoid_wing', 'temporal_lobe', 'frontal_lobe'],
    ARRAY['aneurysm', 'meningioma', 'tumor'],
    ARRAY['pterional', 'frontotemporal', 'transsylvian'],
    ARRAY['pterional', 'pterional approach', 'frontotemporal craniotomy', 'sylvian fissure approach']),

('mca-aneurysm-clipping', 'MCA Aneurysm Clipping',
    (SELECT id FROM procedure_taxonomy WHERE slug = 'vascular'), 2, 'vascular',
    'Cranial - Vascular', 4,
    ARRAY['mca', 'm1', 'm2', 'sylvian_fissure', 'lenticulostriate'],
    ARRAY['aneurysm', 'sah', 'vasospasm'],
    ARRAY['pterional', 'transsylvian'],
    ARRAY['mca aneurysm', 'middle cerebral artery aneurysm', 'mca clipping']),

('acom-aneurysm-clipping', 'ACoM Aneurysm Clipping',
    (SELECT id FROM procedure_taxonomy WHERE slug = 'vascular'), 2, 'vascular',
    'Cranial - Vascular', 4,
    ARRAY['acom', 'a1', 'a2', 'recurrent_artery_heubner', 'gyrus_rectus'],
    ARRAY['aneurysm', 'sah'],
    ARRAY['pterional', 'interhemispheric'],
    ARRAY['acom aneurysm', 'anterior communicating artery aneurysm']),

('ec-ic-bypass', 'EC-IC Bypass',
    (SELECT id FROM procedure_taxonomy WHERE slug = 'vascular'), 2, 'vascular',
    'Cranial - Vascular', 5,
    ARRAY['sta', 'mca', 'temporal_artery', 'm2', 'm3'],
    ARRAY['moyamoya', 'occlusion', 'flow_augmentation'],
    ARRAY['pterional', 'temporal'],
    ARRAY['bypass', 'sta-mca bypass', 'ec-ic bypass', 'revascularization'])
ON CONFLICT (slug) DO NOTHING;

-- Level 2: Core Procedures (skull base)
INSERT INTO procedure_taxonomy (slug, name, parent_id, level, specialty, acgme_category, complexity_level, anatomy_tags, pathology_tags, approach_tags, keyword_aliases) VALUES
('retrosigmoid-approach', 'Retrosigmoid Approach',
    (SELECT id FROM procedure_taxonomy WHERE slug = 'skull-base'), 2, 'skull_base',
    'Cranial - Skull Base', 4,
    ARRAY['sigmoid_sinus', 'cerebellopontine_angle', 'cn_vii', 'cn_viii', 'porus_acusticus'],
    ARRAY['vestibular_schwannoma', 'acoustic_neuroma', 'meningioma'],
    ARRAY['retrosigmoid', 'suboccipital'],
    ARRAY['retrosigmoid', 'retrosigmoid approach', 'cpa approach']),

('translabyrinthine-approach', 'Translabyrinthine Approach',
    (SELECT id FROM procedure_taxonomy WHERE slug = 'skull-base'), 2, 'skull_base',
    'Cranial - Skull Base', 4,
    ARRAY['labyrinth', 'iac', 'facial_nerve', 'sigmoid_sinus'],
    ARRAY['vestibular_schwannoma', 'acoustic_neuroma'],
    ARRAY['translabyrinthine'],
    ARRAY['translabyrinthine', 'translab', 'translabyrinthine approach']),

('transsphenoidal-approach', 'Transsphenoidal Approach',
    (SELECT id FROM procedure_taxonomy WHERE slug = 'skull-base'), 2, 'skull_base',
    'Cranial - Skull Base', 3,
    ARRAY['sella', 'sphenoid_sinus', 'pituitary', 'carotid', 'optic_chiasm'],
    ARRAY['pituitary_adenoma', 'craniopharyngioma', 'rathke_cyst'],
    ARRAY['transsphenoidal', 'endonasal', 'endoscopic'],
    ARRAY['transsphenoidal', 'endoscopic endonasal', 'pituitary surgery'])
ON CONFLICT (slug) DO NOTHING;

-- ============================================================================
-- 7. SEED DATA: Procedure Steps (Pterional Craniotomy Example)
-- ============================================================================

INSERT INTO procedure_steps (procedure_id, step_order, phase, name, criticality_score, danger_structures, keywords) VALUES
((SELECT id FROM procedure_taxonomy WHERE slug = 'pterional-craniotomy'), 1, 'PLANNING',
    'Approach Selection & Planning', 2,
    ARRAY[]::TEXT[],
    ARRAY['indication', 'imaging', 'planning', 'approach selection']),

((SELECT id FROM procedure_taxonomy WHERE slug = 'pterional-craniotomy'), 2, 'POSITIONING',
    'Patient Positioning & Head Fixation', 3,
    ARRAY['cervical spine', 'eyes', 'pressure points'],
    ARRAY['positioning', 'mayfield', 'head holder', 'pin placement']),

((SELECT id FROM procedure_taxonomy WHERE slug = 'pterional-craniotomy'), 3, 'EXPOSURE',
    'Skin Incision & Soft Tissue', 4,
    ARRAY['frontal branch facial nerve', 'superficial temporal artery'],
    ARRAY['incision', 'skin flap', 'temporalis', 'interfascial dissection']),

((SELECT id FROM procedure_taxonomy WHERE slug = 'pterional-craniotomy'), 4, 'EXPOSURE',
    'Craniotomy & Bone Work', 6,
    ARRAY['middle meningeal artery', 'orbit', 'frontal sinus'],
    ARRAY['craniotomy', 'burr hole', 'sphenoid wing', 'pterion', 'drilling']),

((SELECT id FROM procedure_taxonomy WHERE slug = 'pterional-craniotomy'), 5, 'EXPOSURE',
    'Dural Opening', 5,
    ARRAY['cortical veins', 'sylvian veins'],
    ARRAY['dural opening', 'dural flap', 'tack-up sutures']),

((SELECT id FROM procedure_taxonomy WHERE slug = 'pterional-craniotomy'), 6, 'INTRADURAL',
    'Sylvian Fissure Dissection', 7,
    ARRAY['mca branches', 'sylvian veins', 'lenticulostriate arteries'],
    ARRAY['sylvian fissure', 'arachnoid dissection', 'inside-out', 'outside-in']),

((SELECT id FROM procedure_taxonomy WHERE slug = 'pterional-craniotomy'), 7, 'INTRADURAL',
    'Target Exposure & Management', 8,
    ARRAY['perforators', 'cranial nerves', 'eloquent cortex'],
    ARRAY['aneurysm', 'clipping', 'tumor', 'resection']),

((SELECT id FROM procedure_taxonomy WHERE slug = 'pterional-craniotomy'), 8, 'CLOSURE',
    'Dural Closure', 4,
    ARRAY['csf leak'],
    ARRAY['dural closure', 'duraplasty', 'watertight']),

((SELECT id FROM procedure_taxonomy WHERE slug = 'pterional-craniotomy'), 9, 'CLOSURE',
    'Bone Flap & Wound Closure', 3,
    ARRAY['epidural hematoma'],
    ARRAY['bone flap', 'cranioplasty', 'wound closure', 'scalp'])
ON CONFLICT DO NOTHING;

-- ============================================================================
-- 8. SEED DATA: Clinical Entities
-- ============================================================================

INSERT INTO clinical_entities (entity_type, name, slug, synonyms, typical_location) VALUES
('pathology', 'MCA Aneurysm', 'mca-aneurysm',
    ARRAY['middle cerebral artery aneurysm', 'MCA bifurcation aneurysm', 'M1 aneurysm'],
    'M1 bifurcation'),
('pathology', 'ACoM Aneurysm', 'acom-aneurysm',
    ARRAY['anterior communicating artery aneurysm', 'A1-A2 junction aneurysm'],
    'Anterior communicating artery'),
('pathology', 'Vestibular Schwannoma', 'vestibular-schwannoma',
    ARRAY['acoustic neuroma', 'CN VIII schwannoma', 'acoustic schwannoma'],
    'Cerebellopontine angle'),
('pathology', 'Pituitary Adenoma', 'pituitary-adenoma',
    ARRAY['pituitary tumor', 'sellar mass', 'pituitary macroadenoma'],
    'Sella turcica'),
('anatomy_variant', 'Prefixed Chiasm', 'prefixed-chiasm',
    ARRAY['anterior chiasm', 'prefixed optic chiasm'],
    'Suprasellar'),
('anatomy_variant', 'Dominant A1', 'dominant-a1',
    ARRAY['hypoplastic contralateral A1', 'A1 dominance'],
    'Anterior circulation')
ON CONFLICT (slug) DO NOTHING;

-- Link clinical entities to procedures
INSERT INTO clinical_entity_procedures (clinical_entity_id, procedure_id, relevance_score, is_primary) VALUES
((SELECT id FROM clinical_entities WHERE slug = 'mca-aneurysm'),
 (SELECT id FROM procedure_taxonomy WHERE slug = 'mca-aneurysm-clipping'), 1.0, TRUE),
((SELECT id FROM clinical_entities WHERE slug = 'mca-aneurysm'),
 (SELECT id FROM procedure_taxonomy WHERE slug = 'pterional-craniotomy'), 0.9, FALSE),
((SELECT id FROM clinical_entities WHERE slug = 'acom-aneurysm'),
 (SELECT id FROM procedure_taxonomy WHERE slug = 'acom-aneurysm-clipping'), 1.0, TRUE),
((SELECT id FROM clinical_entities WHERE slug = 'vestibular-schwannoma'),
 (SELECT id FROM procedure_taxonomy WHERE slug = 'retrosigmoid-approach'), 0.9, TRUE),
((SELECT id FROM clinical_entities WHERE slug = 'vestibular-schwannoma'),
 (SELECT id FROM procedure_taxonomy WHERE slug = 'translabyrinthine-approach'), 0.9, FALSE),
((SELECT id FROM clinical_entities WHERE slug = 'pituitary-adenoma'),
 (SELECT id FROM procedure_taxonomy WHERE slug = 'transsphenoidal-approach'), 1.0, TRUE)
ON CONFLICT DO NOTHING;

-- ============================================================================
-- 9. HELPER FUNCTIONS
-- ============================================================================

-- Get all procedures for a clinical scenario
CREATE OR REPLACE FUNCTION get_procedures_for_pathology(pathology_slug TEXT)
RETURNS TABLE (
    procedure_id INTEGER,
    procedure_name TEXT,
    relevance_score FLOAT,
    is_primary BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        pt.id,
        pt.name,
        cep.relevance_score,
        cep.is_primary
    FROM clinical_entities ce
    JOIN clinical_entity_procedures cep ON ce.id = cep.clinical_entity_id
    JOIN procedure_taxonomy pt ON pt.id = cep.procedure_id
    WHERE ce.slug = pathology_slug
    ORDER BY cep.is_primary DESC, cep.relevance_score DESC;
END;
$$ LANGUAGE plpgsql;

-- Get procedure content organized by phase
CREATE OR REPLACE FUNCTION get_procedure_content_by_phase(proc_slug TEXT)
RETURNS TABLE (
    phase surgical_phase_enum,
    step_name TEXT,
    chunk_id UUID,
    content TEXT,
    relevance_score FLOAT,
    is_pearl BOOLEAN,
    is_pitfall BOOLEAN,
    source_title TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        cpr.surgical_phase,
        ps.name,
        c.id,
        c.content,
        cpr.relevance_score,
        cpr.is_pearl,
        cpr.is_pitfall,
        d.title
    FROM procedure_taxonomy pt
    JOIN chunk_procedure_relevance cpr ON pt.id = cpr.procedure_id
    JOIN chunks c ON c.id = cpr.chunk_id
    JOIN documents d ON d.id = c.document_id
    LEFT JOIN procedure_steps ps ON ps.id = cpr.step_id
    WHERE pt.slug = proc_slug
    ORDER BY
        cpr.surgical_phase,
        COALESCE(ps.step_order, 999),
        cpr.relevance_score DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 10. COMMENTS
-- ============================================================================

COMMENT ON TABLE procedure_taxonomy IS 'Master taxonomy of surgical procedures for library organization';
COMMENT ON TABLE procedure_steps IS 'Sequential steps within each procedure, organized by surgical phase';
COMMENT ON TABLE chunk_procedure_relevance IS 'Links text chunks to procedures with relevance scoring and clinical intelligence flags';
COMMENT ON TABLE image_procedure_relevance IS 'Links images to procedures for visual-first browsing';
COMMENT ON TABLE clinical_entities IS 'Clinical scenarios (pathologies, anatomy variants) that map to procedures';
