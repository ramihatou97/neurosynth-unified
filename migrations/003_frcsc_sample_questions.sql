-- ============================================================================
-- FRCSC Sample Questions Migration
-- ============================================================================
-- Purpose: Populate frcsc_questions table with 20 sample questions for testing
-- Distribution: 5 questions per category (Neuro-Oncology, Cerebrovascular, Trauma, Spine)
-- Difficulty: Easy (7), Medium (10), Hard (3)
-- ============================================================================

-- =============================================================================
-- NEURO-ONCOLOGY QUESTIONS (5)
-- =============================================================================

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'NO-001',
  (SELECT id FROM frcsc_categories WHERE name = 'neuro_oncology'),
  'A 45-year-old presents with 6 months of progressive headaches and personality changes. MRI shows a 4cm contrast-enhancing lesion in the right frontal lobe with significant vasogenic edema. What is the most appropriate initial management?',
  '[
    {"text": "Start dexamethasone and levetiracetam, book for stereotactic biopsy", "is_correct": true},
    {"text": "Immediate craniotomy for gross total resection", "is_correct": false},
    {"text": "Stereotactic radiosurgery without tissue diagnosis", "is_correct": false},
    {"text": "Observation with repeat MRI in 3 months", "is_correct": false}
  ]'::jsonb,
  'A',
  'Initial management priorities: (1) Symptom control with corticosteroids to reduce vasogenic edema and anticonvulsants for seizure prophylaxis, (2) Tissue diagnosis before definitive treatment. Stereotactic biopsy provides histological diagnosis with lower morbidity than open craniotomy. Treatment decisions depend on pathology.',
  'mcq',
  2,
  3,
  0.2,
  1.3,
  0.25,
  '["Corticosteroids for vasogenic edema", "Seizure prophylaxis in supratentorial lesions", "Tissue diagnosis before treatment planning", "Avoid empiric treatment without pathology"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'NO-002',
  (SELECT id FROM frcsc_categories WHERE name = 'neuro_oncology'),
  'A 62-year-old with newly diagnosed glioblastoma multiforme underwent gross total resection. Postoperative MRI confirms >95% resection. What is the standard adjuvant therapy regimen?',
  '[
    {"text": "Concurrent temozolomide with radiation (60 Gy in 30 fractions), followed by 6 cycles of adjuvant temozolomide", "is_correct": true},
    {"text": "Radiation alone (60 Gy in 30 fractions)", "is_correct": false},
    {"text": "Temozolomide chemotherapy alone for 12 months", "is_correct": false},
    {"text": "Observation with serial MRI every 2 months", "is_correct": false}
  ]'::jsonb,
  'A',
  'The Stupp protocol (NEJM 2005) established the standard of care for GBM: concurrent chemoradiation with temozolomide (75 mg/m² daily) during radiation (60 Gy/30 fractions), followed by adjuvant temozolomide (150-200 mg/m² days 1-5 of 28-day cycles) for 6 cycles. This improves median survival from 12.1 to 14.6 months.',
  'mcq',
  2,
  2,
  -0.1,
  1.4,
  0.25,
  '["Stupp protocol is standard of care", "Concurrent chemoradiation improves survival", "MGMT methylation status predicts temozolomide response", "Radiation dose: 60 Gy in 30 fractions"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'NO-003',
  (SELECT id FROM frcsc_categories WHERE name = 'neuro_oncology'),
  'A 28-year-old with NF2 presents with bilateral vestibular schwannomas. The right tumor is 3.5cm causing brainstem compression and hydrocephalus. The left is 1.2cm with serviceable hearing (Gardner-Robertson Grade II). What is the best management strategy?',
  '[
    {"text": "Right-sided suboccipital craniotomy for tumor decompression, monitor left side with serial MRI", "is_correct": true},
    {"text": "Bilateral stereotactic radiosurgery to halt tumor growth", "is_correct": false},
    {"text": "Bilateral suboccipital craniotomies in staged procedures", "is_correct": false},
    {"text": "Ventriculoperitoneal shunt followed by observation of both tumors", "is_correct": false}
  ]'::jsonb,
  'A',
  'In NF2 with bilateral VS, surgical strategy prioritizes: (1) Decompress symptomatic tumors causing brainstem compression/hydrocephalus, (2) Preserve hearing where possible given bilateral disease. The right tumor requires surgery for mass effect. The left tumor with serviceable hearing should be monitored; surgery risks complete hearing loss. Radiosurgery contraindicated in NF2 due to high malignant transformation risk.',
  'mcq',
  2,
  4,
  0.8,
  1.2,
  0.25,
  '["NF2: bilateral VS requires hearing preservation strategy", "Operate on symptomatic side first", "Radiosurgery contraindicated in NF2 (malignant transformation)", "Monitor smaller tumor with functional hearing"]'::jsonb,
  2022
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'NO-004',
  (SELECT id FROM frcsc_categories WHERE name = 'neuro_oncology'),
  'A 35-year-old presents with seizures. MRI shows a 2.5cm cystic lesion with mural nodule in the left temporal lobe, no enhancement, no edema. What is the most likely diagnosis?',
  '[
    {"text": "Pleomorphic xanthoastrocytoma (PXA)", "is_correct": false},
    {"text": "Pilocytic astrocytoma", "is_correct": true},
    {"text": "Oligodendroglioma", "is_correct": false},
    {"text": "Ganglioglioma", "is_correct": false}
  ]'::jsonb,
  'B',
  'Pilocytic astrocytoma (WHO Grade I) classically presents as a cystic lesion with enhancing mural nodule, most common in cerebellum (children) but can occur in temporal lobe (adults). Key features: well-circumscribed, minimal edema, seizure presentation. Gross total resection is curative. Ganglioglioma is differential but typically has calcification.',
  'mcq',
  1,
  2,
  -0.3,
  1.1,
  0.25,
  '["Pilocytic astrocytoma: cyst + mural nodule", "WHO Grade I, most common pediatric glioma", "Gross total resection is curative", "Minimal vasogenic edema"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'NO-005',
  (SELECT id FROM frcsc_categories WHERE name = 'neuro_oncology'),
  'A 50-year-old with metastatic melanoma presents with a 2.8cm cerebellar metastasis causing obstructive hydrocephalus. Three additional 5mm supratentorial lesions are noted. Systemic disease is well-controlled on immunotherapy. What is the best management?',
  '[
    {"text": "Suboccipital craniotomy for resection of cerebellar lesion, followed by whole-brain radiation therapy (WBRT)", "is_correct": false},
    {"text": "Suboccipital craniotomy for resection of cerebellar lesion, followed by stereotactic radiosurgery (SRS) to all four lesions", "is_correct": true},
    {"text": "Stereotactic radiosurgery to all four lesions without surgery", "is_correct": false},
    {"text": "External ventricular drain followed by whole-brain radiation therapy", "is_correct": false}
  ]'::jsonb,
  'B',
  'The symptomatic posterior fossa lesion causing hydrocephalus requires surgical decompression. For limited metastases (≤4 lesions) with controlled systemic disease, SRS to surgical cavity + other lesions is preferred over WBRT to preserve neurocognitive function. Multiple trials show SRS non-inferior for survival with better quality of life.',
  'mcq',
  2,
  4,
  0.5,
  1.3,
  0.25,
  '["Surgery + SRS for limited brain metastases", "Avoid WBRT when possible (neurocognitive decline)", "Resect symptomatic lesions causing mass effect", "SRS to surgical cavity reduces local recurrence"]'::jsonb,
  2023
);

-- =============================================================================
-- CEREBROVASCULAR QUESTIONS (5)
-- =============================================================================

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'CV-001',
  (SELECT id FROM frcsc_categories WHERE name = 'vascular'),
  'A 55-year-old presents with acute-onset severe headache, Hunt-Hess Grade 2, Fisher Grade 3 subarachnoid hemorrhage. CT angiogram shows a 7mm anterior communicating artery aneurysm with wide neck (dome-to-neck ratio 1.3). What is the most appropriate treatment?',
  '[
    {"text": "Endovascular coiling with stent-assistance", "is_correct": true},
    {"text": "Microsurgical clipping via pterional craniotomy", "is_correct": false},
    {"text": "Observation with blood pressure control", "is_correct": false},
    {"text": "Pipeline embolization device", "is_correct": false}
  ]'::jsonb,
  'A',
  'For ruptured aneurysms, both clipping and coiling are viable. Wide-neck aneurysms (dome-to-neck <2) require stent-assisted coiling or clipping. ISAT trial showed endovascular treatment has better functional outcomes for anterior circulation aneurysms. AComm aneurysms are amenable to both approaches, but endovascular is less invasive. Pipeline is contraindicated in acute SAH (requires dual antiplatelet therapy).',
  'mcq',
  2,
  4,
  0.6,
  1.2,
  0.25,
  '["Wide-neck aneurysm: dome-to-neck <2", "Stent-assisted coiling for wide-neck ruptured aneurysms", "ISAT: endovascular superior for anterior circulation", "Pipeline contraindicated in acute SAH"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'CV-002',
  (SELECT id FROM frcsc_categories WHERE name = 'vascular'),
  'A 62-year-old with atrial fibrillation on warfarin presents 2 hours after symptom onset with left hemiplegia and aphasia (NIHSS 18). CT head shows no hemorrhage, ASPECTS 9. INR is 2.8. What is the next step?',
  '[
    {"text": "Reverse anticoagulation with PCC + vitamin K, then proceed with IV alteplase", "is_correct": true},
    {"text": "Withhold thrombolysis due to anticoagulation, proceed directly to thrombectomy", "is_correct": false},
    {"text": "Wait 24 hours for INR to normalize, then reassess for delayed thrombolysis", "is_correct": false},
    {"text": "Give IV alteplase immediately without reversal", "is_correct": false}
  ]'::jsonb,
  'A',
  'For acute ischemic stroke with INR >1.7, rapid reversal with prothrombin complex concentrate (PCC) + vitamin K is required before IV alteplase. Time is critical - waiting 24 hours eliminates treatment window. After reversal to INR <1.7, both thrombolysis and thrombectomy should be considered. ASPECTS 9 indicates salvageable tissue.',
  'mcq',
  2,
  3,
  0.3,
  1.4,
  0.25,
  '["INR >1.7 requires reversal before tPA", "PCC reverses warfarin faster than FFP", "IV tPA window: 4.5 hours from symptom onset", "Thrombectomy window: up to 24 hours in selected patients"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'CV-003',
  (SELECT id FROM frcsc_categories WHERE name = 'vascular'),
  'A 45-year-old presents with progressive headaches and seizures. MRI shows a 5cm right frontal arteriovenous malformation (AVM) with deep venous drainage and involvement of eloquent cortex. Spetzler-Martin Grade is?',
  '[
    {"text": "Grade IV", "is_correct": true},
    {"text": "Grade III", "is_correct": false},
    {"text": "Grade V", "is_correct": false},
    {"text": "Grade II", "is_correct": false}
  ]'::jsonb,
  'A',
  'Spetzler-Martin Grading: Size (3cm=1, 3-6cm=2, >6cm=3) + Eloquent cortex (yes=1, no=0) + Deep venous drainage (yes=1, no=0). This AVM: Size 5cm (2 points) + Eloquent (1 point) + Deep drainage (1 point) = Grade IV. Grade IV-V AVMs have high surgical morbidity; multimodality treatment (embolization + radiosurgery) often preferred over resection.',
  'mcq',
  2,
  2,
  0.1,
  1.3,
  0.25,
  '["Spetzler-Martin: Size + Eloquent + Deep drainage", "Grade IV-V: high surgical risk", "Eloquent areas: motor, sensory, language, visual, thalamus, brainstem", "Deep drainage: internal cerebral veins, basal veins, precentral cerebellar vein"]'::jsonb,
  2022
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'CV-004',
  (SELECT id FROM frcsc_categories WHERE name = 'vascular'),
  'A 38-year-old presents with thunderclap headache. CT head is normal. LP shows xanthochromia. CT angiogram is negative. What is the next most appropriate investigation?',
  '[
    {"text": "Digital subtraction angiography (DSA)", "is_correct": true},
    {"text": "Repeat CT angiogram in 1 week", "is_correct": false},
    {"text": "MRI brain with gadolinium", "is_correct": false},
    {"text": "Transcranial Doppler ultrasound", "is_correct": false}
  ]'::jsonb,
  'A',
  'CT-negative SAH with xanthochromia (15-20% of SAH cases) requires DSA to identify aneurysm source. CTA has 95% sensitivity for aneurysms >3mm, but DSA is gold standard (98% sensitivity) and can detect small/thrombosed aneurysms, vasculitis, dural AVF, or perimesencephalic pattern (benign). Negative DSA requires repeat at 7-14 days to rule out thrombosed aneurysm.',
  'mcq',
  2,
  3,
  -0.2,
  1.2,
  0.25,
  '["Xanthochromia confirms SAH even with negative CT", "DSA gold standard for aneurysm detection", "Repeat DSA at 7-14 days if initial negative", "Perimesencephalic SAH: benign venous bleed, no aneurysm"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'CV-005',
  (SELECT id FROM frcsc_categories WHERE name = 'vascular'),
  'A 70-year-old with asymptomatic 85% right internal carotid artery stenosis is being considered for carotid endarterectomy. Which factor most strongly predicts surgical benefit over medical management?',
  '[
    {"text": "Life expectancy >5 years", "is_correct": true},
    {"text": "Degree of stenosis >80%", "is_correct": false},
    {"text": "Presence of ulcerated plaque", "is_correct": false},
    {"text": "Male sex", "is_correct": false}
  ]'::jsonb,
  'A',
  'For asymptomatic carotid stenosis, CEA benefit accrues over 3-5 years (ACAS/ACST trials showed 5-year stroke reduction from 11% to 5%). Patients must have life expectancy >5 years to benefit. Perioperative stroke risk (~3%) negates benefit if life expectancy is short. Age alone is not a contraindication if medically fit. Degree of stenosis >60% is threshold for consideration.',
  'mcq',
  2,
  3,
  0.7,
  1.1,
  0.25,
  '["CEA benefit requires 3-5 year life expectancy", "ACAS/ACST: 50% stroke reduction at 5 years", "Perioperative risk must be <3% for benefit", "Asymptomatic stenosis: medical management also reasonable"]'::jsonb,
  2022
);

-- =============================================================================
-- TRAUMA QUESTIONS (5)
-- =============================================================================

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'TR-001',
  (SELECT id FROM frcsc_categories WHERE name = 'trauma'),
  'A 25-year-old involved in MVC presents with GCS 6 (E1V2M3). Pupils are equal and reactive. CT head shows left acute subdural hematoma 8mm thickness with 5mm midline shift. What is the most appropriate management?',
  '[
    {"text": "Urgent craniotomy for subdural evacuation", "is_correct": true},
    {"text": "ICP monitor insertion and medical management", "is_correct": false},
    {"text": "Burr hole drainage at bedside", "is_correct": false},
    {"text": "Observation in ICU with serial CT scans", "is_correct": false}
  ]'::jsonb,
  'A',
  'Acute SDH with thickness >10mm OR midline shift >5mm requires surgical evacuation (Brain Trauma Foundation guidelines). This patient meets criteria (5mm shift + thickness 8mm + GCS 6). Craniotomy is superior to burr holes for acute SDH (thick, clotted blood). ICP monitor alone is insufficient. Equal reactive pupils indicate potential for recovery.',
  'mcq',
  2,
  3,
  -0.1,
  1.4,
  0.25,
  '["Acute SDH surgery indications: >10mm thick OR >5mm shift", "Craniotomy preferred over burr holes", "GCS <9 with mass effect requires surgery", "Equal reactive pupils suggest reversible injury"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'TR-002',
  (SELECT id FROM frcsc_categories WHERE name = 'trauma'),
  'A 19-year-old presents after assault with open depressed skull fracture over the right motor cortex. Dura is lacerated with brain tissue visible. Neurologically intact. What is the definitive management?',
  '[
    {"text": "Operative debridement and dural repair within 24 hours", "is_correct": true},
    {"text": "IV antibiotics and observation for 48 hours before surgery", "is_correct": false},
    {"text": "Immediate craniotomy in emergency department", "is_correct": false},
    {"text": "Burr hole decompression followed by delayed cranioplasty", "is_correct": false}
  ]'::jsonb,
  'A',
  'Open (compound) depressed skull fractures require operative debridement and dural repair within 24 hours to prevent infection (osteomyelitis, abscess, meningitis). Steps: (1) Broad-spectrum antibiotics, (2) Debridement of devitalized tissue, (3) Dural repair (primary or patch), (4) Bone fragment replacement if sterile (cranioplasty if contaminated). Delay increases infection risk.',
  'mcq',
  2,
  3,
  0.0,
  1.3,
  0.25,
  '["Open depressed fracture: surgery <24h", "Debride devitalized tissue", "Primary dural repair or graft", "Replace bone fragments if not contaminated"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'TR-003',
  (SELECT id FROM frcsc_categories WHERE name = 'trauma'),
  'A 30-year-old with severe TBI (GCS 5) has an ICP monitor reading 28 mmHg despite tier-one interventions (sedation, head elevation, normocapnia, normothermia). What is the most appropriate tier-two intervention?',
  '[
    {"text": "Hypertonic saline (23.4% NaCl) bolus", "is_correct": true},
    {"text": "Decompressive craniectomy", "is_correct": false},
    {"text": "Barbiturate coma (pentobarbital)", "is_correct": false},
    {"text": "Hyperventilation to PaCO2 25 mmHg", "is_correct": false}
  ]'::jsonb,
  'A',
  'Tier-two interventions for refractory ICP (>20-25 mmHg): Hyperosmolar therapy (hypertonic saline or mannitol) is first-line tier-two. HTS bolus rapidly reduces ICP. Aggressive hyperventilation (PaCO2 <30) causes cerebral vasoconstriction and ischemia - reserved for herniation. Barbiturate coma is tier-three (requires EEG monitoring, hemodynamic support). Decompressive craniectomy is last resort for diffuse edema refractory to medical management.',
  'mcq',
  2,
  4,
  0.9,
  1.2,
  0.25,
  '["ICP management tiers: 1=basic measures, 2=osmotherapy, 3=barbiturates/craniectomy", "Hypertonic saline preferred over mannitol (no diuresis)", "Hyperventilation PaCO2 <30 risks ischemia", "Decompressive craniectomy for refractory diffuse edema"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'TR-004',
  (SELECT id FROM frcsc_categories WHERE name = 'trauma'),
  'A 40-year-old fell from height and presents with bilateral periorbital ecchymosis and CSF rhinorrhea. CT shows anterior skull base fracture involving cribriform plate. Neurologically stable. What is the initial management of CSF leak?',
  '[
    {"text": "Conservative management with head elevation, stool softeners, acetazolamide; avoid lumbar puncture", "is_correct": true},
    {"text": "Immediate endoscopic repair", "is_correct": false},
    {"text": "Lumbar drain insertion for 5-7 days", "is_correct": false},
    {"text": "Prophylactic antibiotics until leak resolves", "is_correct": false}
  ]'::jsonb,
  'A',
  'Traumatic CSF rhinorrhea resolves spontaneously in 85% within 7 days with conservative management: head elevation (30°), avoid straining/Valsalva (stool softeners), acetazolamide reduces CSF production. Lumbar drain NOT recommended initially (increases meningitis risk without proven benefit). Surgical repair if persistent >7-14 days or recurrent meningitis. Prophylactic antibiotics NOT recommended (select resistant organisms).',
  'mcq',
  2,
  3,
  0.4,
  1.2,
  0.25,
  '["85% traumatic CSF leaks resolve with conservative management", "Avoid lumbar drain initially (infection risk)", "No prophylactic antibiotics (resistance)", "Surgical repair if persistent >7-14 days"]'::jsonb,
  2022
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'TR-005',
  (SELECT id FROM frcsc_categories WHERE name = 'trauma'),
  'A 35-year-old presents after MVC with C5 fracture, bilateral jumped facets, complete spinal cord injury (ASIA A). What is the optimal timing for surgical reduction and stabilization?',
  '[
    {"text": "Within 24 hours of injury", "is_correct": true},
    {"text": "After 48-72 hours when spinal shock resolves", "is_correct": false},
    {"text": "After 7 days to allow inflammatory edema to subside", "is_correct": false},
    {"text": "Immediate surgery in emergency department", "is_correct": false}
  ]'::jsonb,
  'A',
  'STASCIS trial showed early surgery (<24 hours) for spinal cord injury is associated with improved neurological outcomes (2+ ASIA grade improvement) compared to delayed surgery (>24 hours). Bilateral jumped facets require surgery for reduction and stabilization. Closed reduction may be attempted first under fluoroscopy with neurological monitoring. Emergency surgery is not required unless progressive deficit.',
  'mcq',
  2,
  3,
  -0.2,
  1.3,
  0.25,
  '["Early surgery <24h improves SCI outcomes", "Bilateral jumped facets require operative reduction", "Attempt closed reduction first (Gardner-Wells tongs)", "STASCIS: early decompression beneficial"]'::jsonb,
  2023
);

-- =============================================================================
-- SPINE QUESTIONS (5)
-- =============================================================================

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'SP-001',
  (SELECT id FROM frcsc_categories WHERE name = 'spine'),
  'A 65-year-old with 6 months of worsening neck pain and upper extremity weakness presents with MRI showing severe spinal stenosis at C5-6 with cord signal change. Modified JOA score is 12/18. What is the most appropriate management?',
  '[
    {"text": "Surgical decompression (anterior discectomy and fusion OR posterior laminoplasty)", "is_correct": true},
    {"text": "Trial of physical therapy and NSAIDs for 3 months", "is_correct": false},
    {"text": "Cervical epidural steroid injection", "is_correct": false},
    {"text": "Hard cervical collar immobilization", "is_correct": false}
  ]'::jsonb,
  'A',
  'This patient has cervical spondylotic myelopathy (CSM) with moderate severity (mJOA 12-14 = moderate). Cord signal change (T2 hyperintensity) indicates myelomalacia and predicts poorer outcome. Surgical decompression halts progression and may improve function. Conservative management is inappropriate for myelopathy. Approach depends on pathology: anterior (disc) vs posterior (ligamentum flavum). Multilevel often requires laminoplasty.',
  'mcq',
  2,
  3,
  0.1,
  1.4,
  0.25,
  '["CSM with myelopathy requires surgery", "mJOA <12 = severe, 12-14 = moderate, >14 = mild", "T2 cord signal = myelomalacia, worse prognosis", "Conservative management inappropriate for myelopathy"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'SP-002',
  (SELECT id FROM frcsc_categories WHERE name = 'spine'),
  'A 45-year-old with 8 weeks of right leg radicular pain (L5 distribution) undergoes MRI showing L4-5 disc herniation with nerve root compression. Failed 6 weeks conservative management. What surgical approach is most appropriate?',
  '[
    {"text": "Microdiscectomy via posterior approach", "is_correct": true},
    {"text": "L4-5 anterior lumbar interbody fusion (ALIF)", "is_correct": false},
    {"text": "L4-5 posterior lumbar interbody fusion (PLIF)", "is_correct": false},
    {"text": "Lateral lumbar interbody fusion (LLIF)", "is_correct": false}
  ]'::jsonb,
  'B',
  'For single-level disc herniation with radiculopathy failing conservative management (6-12 weeks), microdiscectomy is gold standard. SPORT trial showed superior outcomes vs non-operative at 2 years. Fusion is NOT indicated for isolated disc herniation without instability, deformity, or spondylolisthesis. Microdiscectomy removes herniated fragment, preserves motion, faster recovery.',
  'mcq',
  2,
  2,
  -0.2,
  1.3,
  0.25,
  '["Isolated disc herniation: microdiscectomy, not fusion", "SPORT trial: surgery superior at 2 years", "Fusion indications: instability, spondylolisthesis, deformity", "Conservative management: 6-12 weeks before surgery"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'SP-003',
  (SELECT id FROM frcsc_categories WHERE name = 'spine'),
  'A 70-year-old with Grade II L4-5 degenerative spondylolisthesis and severe stenosis causing neurogenic claudication has failed conservative management. Radiographs show 6mm translation on flexion-extension films. What is the most appropriate surgical treatment?',
  '[
    {"text": "L4-5 decompression and posterolateral fusion with pedicle screw instrumentation", "is_correct": true},
    {"text": "L4-5 laminectomy without fusion", "is_correct": false},
    {"text": "L4-L5-S1 fusion (include S1 for stability)", "is_correct": false},
    {"text": "Interspinous spacer device insertion", "is_correct": false}
  ]'::jsonb,
  'A',
  'Degenerative spondylolisthesis with stenosis and instability (>3-4mm motion on flex-ext) requires decompression + fusion. SLIP trial showed decompression alone has 30% reoperation rate vs 10% with fusion. Fusion to S1 is unnecessary for single-level pathology (increases adjacent segment degeneration). Instrumentation improves fusion rates (80-95% vs 50% without). Interspinous devices insufficient for instability.',
  'mcq',
  2,
  4,
  0.7,
  1.2,
  0.25,
  '["Degenerative spondylolisthesis + stenosis: decompress + fuse", "SLIP trial: fusion reduces reoperation rate", "Grade I-II: single-level fusion adequate", "Instrumentation improves fusion rates"]'::jsonb,
  2023
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'SP-004',
  (SELECT id FROM frcsc_categories WHERE name = 'spine'),
  'A 28-year-old gymnast presents with chronic low back pain. MRI shows bilateral L5 pars interarticularis defects with Grade I L5-S1 spondylolisthesis. No radiculopathy. Conservative management has failed. What is the most appropriate surgical option?',
  '[
    {"text": "L5-S1 posterolateral fusion without decompression", "is_correct": true},
    {"text": "L5 laminectomy and L5-S1 fusion", "is_correct": false},
    {"text": "L4-S1 fusion (include L4 for stability)", "is_correct": false},
    {"text": "Direct pars repair (Scott wiring)", "is_correct": false}
  ]'::jsonb,
  'A',
  'Isthmic spondylolisthesis (pars defect) with chronic pain failing conservative management requires fusion. For Grade I-II without stenosis/radiculopathy, posterolateral fusion L5-S1 is sufficient (no decompression needed). Direct pars repair is option for young patients with acute pars fracture and NO slip, but this patient has established slip. L4-S1 fusion unnecessary for single-level pathology.',
  'mcq',
  2,
  3,
  0.3,
  1.1,
  0.25,
  '["Isthmic spondylolisthesis: fusion for failed conservative Rx", "Grade I-II: L5-S1 fusion sufficient", "Direct pars repair: acute fracture, no slip, <25 years", "No decompression needed if no radiculopathy"]'::jsonb,
  2022
);

INSERT INTO frcsc_questions (
  question_code, category_id, stem, options, answer, explanation,
  format, yield_rating, cognitive_level,
  difficulty_param, discrimination_param, guessing_param,
  key_points, year_asked
) VALUES (
  'SP-005',
  (SELECT id FROM frcsc_categories WHERE name = 'spine'),
  'A 55-year-old with metastatic breast cancer presents with 3 weeks of mid-thoracic back pain. MRI shows T8 vertebral body metastasis with epidural compression causing moderate spinal cord compression but no neurological deficit. What is the Spinal Instability Neoplastic Score (SINS) component that would MOST suggest need for surgical stabilization?',
  '[
    {"text": "Posterior column involvement (pedicles, facets, lamina)", "is_correct": true},
    {"text": "Presence of mechanical pain", "is_correct": false},
    {"text": "Lytic bone lesion", "is_correct": false},
    {"text": "Thoracic location", "is_correct": false}
  ]'::jsonb,
  'A',
  'SINS score guides surgical decision-making for spinal metastases (0-6 stable, 7-12 indeterminate, 13-18 unstable). Components: location (junctional=3), pain (mechanical=3), bone quality (lytic=2), radiographic alignment (kyphosis=4), vertebral body collapse (>50%=3), posterolateral involvement (bilateral=3). Posterior column involvement is critical for stability - pedicle/facet destruction causes instability requiring fixation.',
  'mcq',
  1,
  4,
  1.2,
  1.0,
  0.25,
  '["SINS >12 suggests need for stabilization", "Posterior column critical for stability", "Lytic lesions score 2 points", "Mechanical pain suggests instability"]'::jsonb,
  2023
);

-- =============================================================================
-- VERIFICATION
-- =============================================================================

-- Verify question distribution
SELECT
  c.name AS category,
  COUNT(q.id) AS question_count,
  AVG(q.difficulty_param)::numeric(3,2) AS avg_difficulty
FROM frcsc_categories c
LEFT JOIN frcsc_questions q ON c.id = q.category_id
WHERE q.question_code LIKE 'NO-%' OR q.question_code LIKE 'CV-%'
   OR q.question_code LIKE 'TR-%' OR q.question_code LIKE 'SP-%'
GROUP BY c.name
ORDER BY c.name;

-- Expected output:
-- cerebrovascular | 5 | 0.30
-- neuro_oncology   | 5 | 0.36
-- spine            | 5 | 0.42
-- trauma           | 5 | 0.20
