-- Migration 010: Document Hierarchy, Materialized Paths, and Specialty Taxonomy
-- Purpose: Enable book→chapter grouping, O(1) tree queries, and hierarchical specialty filtering

-- ============================================================================
-- 1. ADD HIERARCHY COLUMNS TO DOCUMENTS TABLE
-- ============================================================================

ALTER TABLE documents ADD COLUMN IF NOT EXISTS parent_id UUID REFERENCES documents(id) ON DELETE CASCADE;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS sort_order INTEGER DEFAULT 0;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS section_type VARCHAR(20) DEFAULT 'document';

-- Partial index for efficient parent lookups (only indexed when parent exists)
CREATE INDEX IF NOT EXISTS idx_docs_parent ON documents(parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_docs_sort ON documents(parent_id, sort_order);

COMMENT ON COLUMN documents.parent_id IS 'Reference to parent document for book→chapter hierarchy';
COMMENT ON COLUMN documents.sort_order IS 'Order within parent (0=first chapter, 1=second, etc)';
COMMENT ON COLUMN documents.section_type IS 'document|book|chapter|section|appendix';

-- ============================================================================
-- 2. MATERIALIZED PATHS TABLE FOR O(1) TREE QUERIES
-- ============================================================================

CREATE TABLE IF NOT EXISTS document_paths (
    document_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    path VARCHAR(2000) NOT NULL,  -- e.g., '/RHOTON/Microsurgical_Anatomy/Ch_01'
    depth INTEGER NOT NULL,        -- 0=root book, 1=chapter, 2=section
    root_id UUID REFERENCES documents(id) ON DELETE CASCADE,  -- Top-level book
    confidence_score FLOAT DEFAULT 1.0,  -- Grouping algorithm confidence
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_path_lookup ON document_paths(path);
CREATE INDEX IF NOT EXISTS idx_path_prefix ON document_paths(path varchar_pattern_ops);  -- For LIKE 'prefix%' queries
CREATE INDEX IF NOT EXISTS idx_path_depth ON document_paths(depth);
CREATE INDEX IF NOT EXISTS idx_path_root ON document_paths(root_id);

COMMENT ON TABLE document_paths IS 'Materialized paths for O(1) ancestor/descendant queries';
COMMENT ON COLUMN document_paths.path IS 'Hierarchical path like /BookTitle/ChapterName';
COMMENT ON COLUMN document_paths.depth IS '0=book, 1=chapter, 2=section, etc';
COMMENT ON COLUMN document_paths.confidence_score IS 'How confident the grouper was (0.0-1.0)';

-- ============================================================================
-- 3. SPECIALTY HIERARCHY TABLE (REPLACES FLAT ENUM)
-- ============================================================================

CREATE TABLE IF NOT EXISTS specialties (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    parent_id INTEGER REFERENCES specialties(id) ON DELETE CASCADE,
    level INTEGER NOT NULL DEFAULT 1,  -- 1=specialty, 2=subspecialty, 3=sub-subspecialty
    keywords TEXT[] NOT NULL DEFAULT '{}',  -- Keywords for auto-classification
    icon VARCHAR(20),  -- Emoji icon for UI
    description TEXT,
    sort_order INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_specialty_parent ON specialties(parent_id);
CREATE INDEX IF NOT EXISTS idx_specialty_level ON specialties(level);
CREATE INDEX IF NOT EXISTS idx_specialty_keywords ON specialties USING GIN(keywords);

COMMENT ON TABLE specialties IS 'Hierarchical specialty taxonomy (specialty→subspecialty→sub-subspecialty)';
COMMENT ON COLUMN specialties.level IS '1=major specialty, 2=subspecialty, 3=sub-subspecialty';
COMMENT ON COLUMN specialties.keywords IS 'Keywords that trigger auto-classification to this specialty';

-- ============================================================================
-- 4. SEED SPECIALTY HIERARCHY DATA
-- ============================================================================

-- Level 1: Major Specialties
INSERT INTO specialties (name, parent_id, level, keywords, icon, description, sort_order) VALUES
('Vascular', NULL, 1, ARRAY['aneurysm','avm','bypass','carotid','stroke','hemorrhage','ischemia','thrombosis','embolism','vasospasm'], '🩸', 'Cerebrovascular neurosurgery', 1),
('Tumor', NULL, 1, ARRAY['glioma','meningioma','resection','oncology','tumor','mass','lesion','neoplasm','metastasis','radiation'], '🔬', 'Neuro-oncology and tumor surgery', 2),
('Spine', NULL, 1, ARRAY['cervical','lumbar','thoracic','fusion','disc','laminectomy','decompression','spondylosis','stenosis','myelopathy'], '🦴', 'Spinal neurosurgery', 3),
('Skull Base', NULL, 1, ARRAY['pituitary','acoustic','transsphenoidal','petroclival','jugular','cavernous','clivus','foramen magnum'], '💀', 'Skull base surgery', 4),
('Functional', NULL, 1, ARRAY['dbs','epilepsy','parkinson','stimulation','movement disorder','tremor','dystonia','seizure','ablation'], '⚡', 'Functional and stereotactic neurosurgery', 5),
('Pediatric', NULL, 1, ARRAY['pediatric','child','congenital','shunt','hydrocephalus','craniosynostosis','chiari','myelomeningocele'], '👶', 'Pediatric neurosurgery', 6),
('Trauma', NULL, 1, ARRAY['tbi','subdural','epidural','contusion','fracture','concussion','hematoma','decompressive'], '🚑', 'Neurotrauma', 7),
('Peripheral Nerve', NULL, 1, ARRAY['brachial plexus','carpal tunnel','ulnar','peripheral','nerve transfer','neuroma','entrapment'], '🔌', 'Peripheral nerve surgery', 8)
ON CONFLICT (name) DO NOTHING;

-- Level 2: Subspecialties for Vascular
INSERT INTO specialties (name, parent_id, level, keywords, icon, description, sort_order) VALUES
('Aneurysms', (SELECT id FROM specialties WHERE name='Vascular'), 2, ARRAY['aneurysm','clipping','coiling','sah','subarachnoid','rupture','unruptured','giant aneurysm'], '💧', 'Cerebral aneurysm surgery', 1),
('AVMs', (SELECT id FROM specialties WHERE name='Vascular'), 2, ARRAY['avm','arteriovenous','malformation','nidus','draining vein','feeding artery','spetzler-martin'], '🌀', 'Arteriovenous malformation surgery', 2),
('Bypass', (SELECT id FROM specialties WHERE name='Vascular'), 2, ARRAY['bypass','ec-ic','sta-mca','revascularization','moyamoya','radial artery','saphenous'], '🔀', 'Cerebral revascularization', 3),
('Cavernomas', (SELECT id FROM specialties WHERE name='Vascular'), 2, ARRAY['cavernoma','cavernous malformation','ccm','brainstem cavernoma'], '🫐', 'Cavernous malformation surgery', 4),
('Stroke', (SELECT id FROM specialties WHERE name='Vascular'), 2, ARRAY['stroke','thrombectomy','ischemic','hemorrhagic','decompressive craniectomy','malignant mca'], '🧠', 'Stroke intervention', 5)
ON CONFLICT (name) DO NOTHING;

-- Level 2: Subspecialties for Tumor
INSERT INTO specialties (name, parent_id, level, keywords, icon, description, sort_order) VALUES
('Gliomas', (SELECT id FROM specialties WHERE name='Tumor'), 2, ARRAY['glioma','glioblastoma','astrocytoma','oligodendroglioma','gbm','low grade','high grade','awake craniotomy'], '🧠', 'Glioma surgery', 1),
('Meningiomas', (SELECT id FROM specialties WHERE name='Tumor'), 2, ARRAY['meningioma','convexity','parasagittal','falcine','sphenoid wing','olfactory groove','tuberculum'], '🎯', 'Meningioma surgery', 2),
('Metastases', (SELECT id FROM specialties WHERE name='Tumor'), 2, ARRAY['metastasis','metastatic','brain met','radiosurgery','srs','whole brain'], '🎲', 'Brain metastasis treatment', 3),
('Pediatric Tumors', (SELECT id FROM specialties WHERE name='Tumor'), 2, ARRAY['medulloblastoma','ependymoma','pilocytic','craniopharyngioma','posterior fossa'], '🧒', 'Pediatric brain tumors', 4)
ON CONFLICT (name) DO NOTHING;

-- Level 2: Subspecialties for Spine
INSERT INTO specialties (name, parent_id, level, keywords, icon, description, sort_order) VALUES
('Cervical', (SELECT id FROM specialties WHERE name='Spine'), 2, ARRAY['cervical','acdf','corpectomy','laminoplasty','anterior cervical','c1-c2','odontoid'], '1️⃣', 'Cervical spine surgery', 1),
('Lumbar', (SELECT id FROM specialties WHERE name='Spine'), 2, ARRAY['lumbar','discectomy','microdiscectomy','tlif','plif','stenosis','spondylolisthesis','l4-l5','l5-s1'], '2️⃣', 'Lumbar spine surgery', 2),
('Thoracic', (SELECT id FROM specialties WHERE name='Spine'), 2, ARRAY['thoracic','kyphosis','thoracolumbar','t-spine','costotransversectomy'], '3️⃣', 'Thoracic spine surgery', 3),
('Deformity', (SELECT id FROM specialties WHERE name='Spine'), 2, ARRAY['scoliosis','kyphosis','deformity','sagittal balance','pso','vcr','adult deformity'], '📐', 'Spinal deformity correction', 4),
('Spinal Tumors', (SELECT id FROM specialties WHERE name='Spine'), 2, ARRAY['intradural','extramedullary','intramedullary','schwannoma','ependymoma','hemangioblastoma'], '🎯', 'Spinal tumor surgery', 5)
ON CONFLICT (name) DO NOTHING;

-- Level 2: Subspecialties for Skull Base
INSERT INTO specialties (name, parent_id, level, keywords, icon, description, sort_order) VALUES
('Pituitary', (SELECT id FROM specialties WHERE name='Skull Base'), 2, ARRAY['pituitary','adenoma','transsphenoidal','cushing','acromegaly','prolactinoma','endoscopic endonasal'], '🎪', 'Pituitary surgery', 1),
('Acoustic', (SELECT id FROM specialties WHERE name='Skull Base'), 2, ARRAY['acoustic','vestibular','schwannoma','retrosigmoid','translabyrinthine','middle fossa','facial nerve'], '👂', 'Acoustic neuroma surgery', 2),
('Anterior Skull Base', (SELECT id FROM specialties WHERE name='Skull Base'), 2, ARRAY['anterior skull base','olfactory neuroblastoma','cribriform','planum','tuberculum sellae'], '👃', 'Anterior skull base surgery', 3),
('Lateral Skull Base', (SELECT id FROM specialties WHERE name='Skull Base'), 2, ARRAY['jugular foramen','petroclival','petrous apex','infratemporal','far lateral'], '👂', 'Lateral skull base surgery', 4)
ON CONFLICT (name) DO NOTHING;

-- Level 2: Subspecialties for Functional
INSERT INTO specialties (name, parent_id, level, keywords, icon, description, sort_order) VALUES
('DBS', (SELECT id FROM specialties WHERE name='Functional'), 2, ARRAY['dbs','deep brain stimulation','stn','gpi','vim','parkinson','essential tremor'], '⚡', 'Deep brain stimulation', 1),
('Epilepsy', (SELECT id FROM specialties WHERE name='Functional'), 2, ARRAY['epilepsy','seizure','temporal lobectomy','hemispherectomy','vns','rns','laser ablation'], '🌊', 'Epilepsy surgery', 2),
('Pain', (SELECT id FROM specialties WHERE name='Functional'), 2, ARRAY['trigeminal neuralgia','mvd','spinal cord stimulation','intrathecal pump','facial pain'], '😣', 'Pain neurosurgery', 3),
('Spasticity', (SELECT id FROM specialties WHERE name='Functional'), 2, ARRAY['spasticity','baclofen','selective dorsal rhizotomy','sdr','intrathecal baclofen'], '💪', 'Spasticity treatment', 4)
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- 5. DOCUMENT-SPECIALTY BRIDGE TABLE (MANY-TO-MANY)
-- ============================================================================

CREATE TABLE IF NOT EXISTS document_specialties (
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    specialty_id INTEGER REFERENCES specialties(id) ON DELETE CASCADE,
    relevance_score FLOAT DEFAULT 1.0,  -- 0.0-1.0, how relevant the specialty is
    is_primary BOOLEAN DEFAULT false,   -- Is this the main specialty?
    auto_detected BOOLEAN DEFAULT true, -- Was this auto-detected or manually set?
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (document_id, specialty_id)
);

CREATE INDEX IF NOT EXISTS idx_doc_spec_doc ON document_specialties(document_id);
CREATE INDEX IF NOT EXISTS idx_doc_spec_spec ON document_specialties(specialty_id);
CREATE INDEX IF NOT EXISTS idx_doc_spec_primary ON document_specialties(document_id) WHERE is_primary = true;

COMMENT ON TABLE document_specialties IS 'Many-to-many relationship between documents and specialties';
COMMENT ON COLUMN document_specialties.relevance_score IS 'How strongly this document relates to this specialty (0.0-1.0)';
COMMENT ON COLUMN document_specialties.is_primary IS 'True if this is the primary/main specialty for the document';

-- ============================================================================
-- 6. HELPER FUNCTIONS FOR TREE QUERIES
-- ============================================================================

-- Get all descendants of a document (book → chapters → sections)
CREATE OR REPLACE FUNCTION get_document_descendants(root_doc_id UUID)
RETURNS TABLE(document_id UUID, depth INTEGER) AS $$
BEGIN
    RETURN QUERY
    SELECT dp.document_id, dp.depth
    FROM document_paths dp
    WHERE dp.path LIKE (
        SELECT path || '/%' FROM document_paths WHERE document_id = root_doc_id
    )
    ORDER BY dp.path;
END;
$$ LANGUAGE plpgsql;

-- Get all ancestors of a document (section → chapter → book)
CREATE OR REPLACE FUNCTION get_document_ancestors(child_doc_id UUID)
RETURNS TABLE(document_id UUID, depth INTEGER) AS $$
DECLARE
    child_path VARCHAR;
BEGIN
    SELECT path INTO child_path FROM document_paths WHERE document_id = child_doc_id;

    RETURN QUERY
    SELECT dp.document_id, dp.depth
    FROM document_paths dp
    WHERE child_path LIKE dp.path || '%'
      AND dp.document_id != child_doc_id
    ORDER BY dp.depth;
END;
$$ LANGUAGE plpgsql;

-- Get specialty with all descendants (specialty → subspecialties)
CREATE OR REPLACE FUNCTION get_specialty_with_descendants(spec_id INTEGER)
RETURNS TABLE(specialty_id INTEGER, name VARCHAR, level INTEGER) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE spec_tree AS (
        -- Base: the starting specialty
        SELECT s.id, s.name, s.level
        FROM specialties s
        WHERE s.id = spec_id

        UNION ALL

        -- Recursive: children of current level
        SELECT s.id, s.name, s.level
        FROM specialties s
        JOIN spec_tree st ON s.parent_id = st.id
    )
    SELECT st.id, st.name, st.level FROM spec_tree st;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- 7. TRIGGER TO UPDATE document_paths.updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION update_document_paths_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_document_paths_timestamp ON document_paths;
CREATE TRIGGER trigger_update_document_paths_timestamp
    BEFORE UPDATE ON document_paths
    FOR EACH ROW
    EXECUTE FUNCTION update_document_paths_timestamp();

-- ============================================================================
-- 8. MIGRATION COMPLETE
-- ============================================================================

COMMENT ON SCHEMA public IS 'Migration 010 applied: Document hierarchy, materialized paths, specialty taxonomy';
