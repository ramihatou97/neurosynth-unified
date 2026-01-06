-- ============================================================================
-- NEUROSYNTH 2.0 - CLINICAL PRINCIPLES SEED DATA
-- ============================================================================
-- Purpose: Populates the reasoning engine with high-value neurosurgical axioms
-- Date: 2026-01-04
-- 
-- These principles encode expert knowledge in machine-readable form:
-- - Antecedent: IF condition (what triggers the rule)
-- - Consequent: THEN result (what happens)
-- - Mechanism: WHY (physiological explanation)
--
-- IMPORTANT: These principles should be validated by domain experts
-- ============================================================================

BEGIN;

-- Clear existing principles (for clean re-seeding)
-- TRUNCATE clinical_principles CASCADE;

INSERT INTO clinical_principles (
    id, name, statement, antecedent, consequent, mechanism,
    domain, category, severity, exceptions, examples,
    evidence_level, trigger_entities, trigger_actions
) VALUES

-- ============================================================================
-- VASCULAR PRINCIPLES
-- ============================================================================

('VASC_001', 
 'End-Artery Vulnerability',
 'End arteries lack collateral supply; their occlusion causes infarction in the supplied territory.',
 'structure.is_end_artery = TRUE AND action IN (''sacrifice'', ''occlude'', ''coagulate'')',
 'Infarction in supplied territory with probability > 0.95',
 'End arteries are the sole blood supply to their territory. Without collaterals, any interruption of flow leads to ischemia and cell death within minutes to hours.',
 'vascular', 'physiology', 'absolute',
 '["established collateral circulation", "chronic occlusion with collateral development"]',
 '[{"scenario": "Lenticulostriate artery sacrifice", "outcome": "Lacunar infarct in basal ganglia"},
   {"scenario": "Anterior choroidal artery injury", "outcome": "Internal capsule infarct with hemiparesis"}]',
 'Ib',
 ARRAY['artery', 'end_artery', 'perforator', 'lenticulostriate', 'anterior_choroidal', 'thalamoperforator'],
 ARRAY['sacrifice', 'coagulate', 'clip', 'occlude', 'ligate']),

('VASC_002',
 'Perforator Tethering Principle',
 'Major vessels tethered by perforating branches cannot be mobilized without risking perforator avulsion.',
 'structure.mobility = ''tethered_by_perforators'' AND action = ''mobilize''',
 'Risk of perforator avulsion causing stroke in perforator territory',
 'Perforating arteries arise at fixed points and have minimal length redundancy. Mobilizing the parent vessel stretches perforators beyond their elastic limit, causing avulsion.',
 'vascular', 'surgical_technique', 'critical',
 '["adequate arachnoid dissection releasing perforators", "intentional perforator sacrifice with confirmed collaterals"]',
 '[{"scenario": "MCA mobilization without perforator release", "outcome": "LSA avulsion with basal ganglia infarct"},
   {"scenario": "Basilar artery manipulation", "outcome": "Pontine perforator injury with quadriparesis"}]',
 'III',
 ARRAY['mca', 'basilar', 'pca', 'perforator', 'm1_segment'],
 ARRAY['mobilize', 'retract', 'displace', 'rotate']),

('VASC_003',
 'Venous Drainage Dominance',
 'Sacrificing the dominant venous drainage pathway causes venous infarction.',
 'structure.type = ''vein'' AND structure.is_dominant_drainage = TRUE AND action = ''sacrifice''',
 'Venous infarction with hemorrhagic transformation probability > 0.7',
 'Venous occlusion increases upstream pressure, causing capillary rupture and parenchymal hemorrhage. Unlike arterial collaterals, venous collaterals develop slowly.',
 'vascular', 'physiology', 'critical',
 '["established venous collaterals", "gradual occlusion allowing collateral development"]',
 '[{"scenario": "Vein of Labbé sacrifice", "outcome": "Temporal lobe venous infarct"},
   {"scenario": "Superior sagittal sinus occlusion", "outcome": "Bilateral parasagittal hemorrhagic infarcts"}]',
 'IIb',
 ARRAY['vein', 'sinus', 'vein_of_labbe', 'superior_sagittal_sinus', 'sigmoid_sinus'],
 ARRAY['sacrifice', 'coagulate', 'ligate', 'occlude']),

('VASC_004',
 'Aneurysm Dome Fragility',
 'The aneurysm dome is the weakest point and ruptures with minimal manipulation.',
 'structure.type = ''aneurysm_dome'' AND action IN (''manipulate'', ''retract'', ''dissect'')',
 'Intraoperative rupture with probability proportional to dome/neck ratio',
 'The aneurysm dome wall is attenuated and lacks the normal arterial wall layers. Mechanical stress concentrates at the dome, making it prone to rupture.',
 'vascular', 'surgical_technique', 'critical',
 '["completely thrombosed aneurysm", "previously coiled aneurysm"]',
 '[{"scenario": "Premature dome dissection", "outcome": "Intraoperative rupture requiring temporary clip"},
   {"scenario": "Excessive brain retraction", "outcome": "Dome avulsion from neck"}]',
 'IIa',
 ARRAY['aneurysm', 'aneurysm_dome', 'aneurysm_neck'],
 ARRAY['dissect', 'manipulate', 'retract', 'touch']),

('VASC_005',
 'Temporary Occlusion Time Limit',
 'Temporary arterial occlusion beyond safe duration causes ischemic injury.',
 'action = ''temporary_occlusion'' AND duration > safe_occlusion_time',
 'Ischemic injury probability increases exponentially after 10-15 minutes',
 'Neurons tolerate ischemia for limited time. ATP depletion leads to calcium influx, excitotoxicity, and cell death. Collateral flow extends but does not eliminate the time limit.',
 'vascular', 'surgical_technique', 'critical',
 '["burst suppression anesthesia", "hypothermia", "pharmacological neuroprotection"]',
 '[{"scenario": "20-minute MCA occlusion", "outcome": "Watershed infarct"},
   {"scenario": "Intermittent 5-minute clips with reperfusion", "outcome": "Usually tolerated"}]',
 'IIa',
 ARRAY['artery', 'temporary_clip'],
 ARRAY['temporary_occlusion', 'temporary_clip']),

-- ============================================================================
-- NEURAL STRUCTURE PRINCIPLES  
-- ============================================================================

('NEUR_001',
 'Cranial Nerve Stretch Injury',
 'Cranial nerves tolerate minimal stretch; excessive traction causes permanent dysfunction.',
 'structure.type = ''cranial_nerve'' AND action = ''retract'' AND force > safe_threshold',
 'Cranial nerve palsy with variable recovery based on stretch magnitude',
 'Cranial nerves have limited epineural support and fixed attachments. Stretch causes axonal injury (neurapraxia to neurotmesis) proportional to force and duration.',
 'neurophysiology', 'surgical_technique', 'critical',
 '["very brief gentle retraction", "intraoperative monitoring showing no change"]',
 '[{"scenario": "CN IV stretch during tentorial tumor", "outcome": "Superior oblique palsy"},
   {"scenario": "CN VII traction during CPA tumor", "outcome": "Facial weakness HB III-IV"}]',
 'III',
 ARRAY['cranial_nerve', 'cn_iii', 'cn_iv', 'cn_vi', 'cn_vii', 'cn_viii'],
 ARRAY['retract', 'stretch', 'mobilize', 'displace']),

('NEUR_002',
 'Eloquent Cortex Preservation',
 'Resection of eloquent cortex causes permanent neurological deficit.',
 'structure.eloquence_grade = ''eloquent'' AND action = ''resect''',
 'Permanent neurological deficit corresponding to cortical function',
 'Eloquent cortex (motor, sensory, language, vision) has localized function that cannot be compensated by other brain regions in adults.',
 'oncology', 'patient_selection', 'absolute',
 '["extensive tumor infiltration making preservation impossible", "patient-accepted trade-off for survival"]',
 '[{"scenario": "Primary motor cortex resection", "outcome": "Contralateral hemiparesis"},
   {"scenario": "Wernicke area resection", "outcome": "Receptive aphasia"}]',
 'Ib',
 ARRAY['motor_cortex', 'sensory_cortex', 'broca_area', 'wernicke_area', 'visual_cortex'],
 ARRAY['resect', 'ablate', 'destroy']),

('NEUR_003',
 'Brainstem Manipulation Sensitivity',
 'The brainstem tolerates minimal manipulation; even gentle pressure causes dysfunction.',
 'structure.name LIKE ''%brainstem%'' OR structure.name LIKE ''%medulla%'' OR structure.name LIKE ''%pons%''',
 'Cardiorespiratory instability, cranial nerve palsies, motor deficits',
 'The brainstem contains vital cardiorespiratory centers, cranial nerve nuclei, and all ascending/descending tracts in a confined space. Minimal edema causes disproportionate dysfunction.',
 'neurophysiology', 'surgical_technique', 'absolute',
 '[]',
 '[{"scenario": "Pons compression during tumor dissection", "outcome": "Hypertension, bradycardia, apnea"},
   {"scenario": "Medulla manipulation", "outcome": "Respiratory arrest"}]',
 'IIb',
 ARRAY['brainstem', 'pons', 'medulla', 'midbrain'],
 ARRAY['manipulate', 'retract', 'compress', 'dissect']),

('NEUR_004',
 'Intraoperative Monitoring Alert Response',
 'Significant IONM changes indicate impending neural injury requiring immediate intervention.',
 'monitoring.change > threshold AND monitoring.type IN (''MEP'', ''SSEP'', ''EMG'')',
 'Permanent deficit if intervention delayed beyond reversal window',
 'IONM detects neural compromise before irreversible injury. MEP loss indicates motor pathway ischemia; SSEP changes indicate sensory pathway compromise. Early intervention can reverse changes.',
 'neurophysiology', 'surgical_technique', 'critical',
 '["known anesthetic effects", "technical artifact"]',
 '[{"scenario": "50% MEP amplitude drop", "outcome": "Release retraction, raise MAP, check temporary clip"},
   {"scenario": "Complete MEP loss", "outcome": "Immediate cessation of causative maneuver"}]',
 'IIa',
 ARRAY['motor_pathway', 'sensory_pathway', 'spinal_cord'],
 ARRAY['monitor_alert', 'mep_change', 'ssep_change']),

-- ============================================================================
-- ONCOLOGICAL PRINCIPLES
-- ============================================================================

('ONCO_001',
 'Tumor Plane Identification',
 'Resecting along the tumor-brain interface preserves function; crossing into normal brain causes deficit.',
 'action = ''resect'' AND tumor.has_defined_plane = TRUE',
 'Safe resection if plane maintained; deficit if plane violated',
 'Well-circumscribed tumors displace rather than infiltrate brain. The gliotic plane between tumor and brain is a safe dissection corridor. Infiltrative tumors lack this plane.',
 'oncology', 'surgical_technique', 'warning',
 '["diffuse infiltrative tumor", "tumor involving eloquent cortex"]',
 '[{"scenario": "Meningioma with clear arachnoid plane", "outcome": "Gross total resection without deficit"},
   {"scenario": "Glioblastoma without plane", "outcome": "Subtotal resection or deficit"}]',
 'III',
 ARRAY['tumor', 'meningioma', 'schwannoma', 'metastasis'],
 ARRAY['resect', 'dissect', 'debulk']),

('ONCO_002',
 'Internal Decompression Before Capsule Dissection',
 'Large tumors require internal debulking before capsule mobilization to prevent brain injury.',
 'tumor.size = ''large'' AND action = ''mobilize_capsule'' AND NOT tumor.internally_decompressed',
 'Brain retraction injury, vessel avulsion, CN injury from excessive force',
 'Large tumors occupy significant intracranial volume. Attempting en-bloc removal requires excessive retraction. Internal decompression creates working space and reduces traction forces.',
 'oncology', 'surgical_technique', 'warning',
 '["small tumor", "cystic tumor easily aspirated"]',
 '[{"scenario": "4cm schwannoma en-bloc attempt", "outcome": "CN VII injury from traction"},
   {"scenario": "Large meningioma without debulking", "outcome": "Venous injury from retraction"}]',
 'III',
 ARRAY['tumor', 'large_tumor', 'schwannoma', 'meningioma'],
 ARRAY['mobilize', 'dissect_capsule', 'remove_en_bloc']),

-- ============================================================================
-- PHYSICAL/MECHANICAL PRINCIPLES
-- ============================================================================

('PHYS_001',
 'Monro-Kellie Doctrine',
 'The cranial vault is fixed volume; increase in one component raises ICP.',
 'intracranial_volume.increase > compensatory_reserve',
 'ICP rises exponentially; herniation if untreated',
 'The skull is a rigid container with fixed volume (~1500mL). Blood, CSF, and brain must sum to this volume. Mass lesions exhaust CSF/venous compliance, then ICP rises exponentially.',
 'neurophysiology', 'physiology', 'critical',
 '["open fontanelles in infants", "craniectomy defect", "skull fracture with dural tear"]',
 '[{"scenario": "Expanding hematoma", "outcome": "Herniation if not evacuated"},
   {"scenario": "Brain swelling after trauma", "outcome": "ICP crisis requiring decompression"}]',
 'Ia',
 ARRAY['intracranial_pressure', 'brain_swelling', 'mass_lesion', 'hematoma'],
 ARRAY['increase_volume', 'cause_swelling']),

('PHYS_002',
 'Brain Retraction Pressure-Time Product',
 'Retraction injury is proportional to pressure × time; minimize both.',
 'action = ''retract'' AND (pressure > 20mmHg OR duration > 15min)',
 'Contusion, ischemia, edema at retraction site',
 'Brain tissue tolerates limited mechanical stress. Sustained pressure compresses microcirculation, causing ischemia. Intermittent release allows reperfusion and extends safe retraction time.',
 'neurophysiology', 'surgical_technique', 'warning',
 '["relaxed brain with mannitol/CSF drainage", "dynamic retraction with intermittent release"]',
 '[{"scenario": "Fixed retractor 30 minutes", "outcome": "Frontal contusion"},
   {"scenario": "Intermittent hand-held retraction", "outcome": "Usually tolerated"}]',
 'IIb',
 ARRAY['brain', 'frontal_lobe', 'temporal_lobe', 'cerebellum'],
 ARRAY['retract', 'compress', 'displace']),

('PHYS_003',
 'Thermal Spread in Bipolar Coagulation',
 'Heat spreads beyond the bipolar tips; nearby neural structures at risk.',
 'action = ''coagulate'' AND distance_to_nerve < 2mm',
 'Thermal injury to adjacent neural structures',
 'Bipolar coagulation generates heat that conducts into surrounding tissue. Neural tissue is particularly susceptible to thermal injury. Low power, irrigation, and brief application minimize spread.',
 'neurophysiology', 'surgical_technique', 'warning',
 '["copious irrigation", "very low power settings", "brief intermittent application"]',
 '[{"scenario": "Bipolar near CN VII", "outcome": "Delayed facial weakness from thermal injury"},
   {"scenario": "Coagulating tumor near optic nerve", "outcome": "Visual loss from thermal damage"}]',
 'III',
 ARRAY['cranial_nerve', 'nerve', 'spinal_cord'],
 ARRAY['coagulate', 'cauterize', 'bipolar']),

('PHYS_004',
 'Drill Heat Generation',
 'High-speed drilling generates significant heat; irrigation essential to prevent bone/neural injury.',
 'action = ''drill'' AND irrigation = FALSE',
 'Thermal necrosis of bone and adjacent neural structures',
 'Friction from drilling converts mechanical energy to heat. Bone conducts heat to nearby structures. Irrigation dissipates heat; without it, temperatures can exceed 50°C causing tissue damage.',
 'neurophysiology', 'surgical_technique', 'warning',
 '["copious continuous irrigation", "intermittent drilling with cooling breaks"]',
 '[{"scenario": "Mastoid drilling without irrigation", "outcome": "Facial nerve thermal injury"},
   {"scenario": "Spinal drilling without irrigation", "outcome": "Spinal cord thermal injury"}]',
 'IIb',
 ARRAY['bone', 'mastoid', 'skull_base', 'spine'],
 ARRAY['drill', 'burr']),

-- ============================================================================
-- APPROACH-SPECIFIC PRINCIPLES
-- ============================================================================

('APPR_001',
 'Sylvian Fissure Opening Technique',
 'Inside-out sylvian dissection preserves bridging veins; outside-in risks venous injury.',
 'approach = ''pterional'' AND action = ''open_sylvian_fissure''',
 'Frontal venous infarct if superficial veins sacrificed',
 'Bridging veins cross the superficial sylvian fissure. Inside-out dissection (starting at the carotid) works in the deep arachnoid plane, avoiding surface veins. Outside-in dissection encounters veins first.',
 'vascular', 'surgical_technique', 'warning',
 '["lateral approach not requiring fissure opening", "aneurysm location allowing limited dissection"]',
 '[{"scenario": "Outside-in aggressive opening", "outcome": "Frontal contusion from vein sacrifice"},
   {"scenario": "Inside-out technique", "outcome": "Preserved venous drainage"}]',
 'III',
 ARRAY['sylvian_fissure', 'superficial_sylvian_vein', 'bridging_vein'],
 ARRAY['dissect', 'open', 'split']),

('APPR_002',
 'Posterior Fossa Venous Anatomy Respect',
 'Superior petrosal and tentorial veins are critical in posterior fossa approaches.',
 'approach IN (''retrosigmoid'', ''subtemporal'') AND structure.type = ''tentorial_vein''',
 'Cerebellar venous infarct or hemorrhagic complications',
 'The posterior fossa has limited venous drainage options. Superior petrosal complex and tentorial veins drain significant territory. Their sacrifice causes venous congestion and hemorrhagic infarction.',
 'vascular', 'surgical_technique', 'critical',
 '["adequate collateral venous drainage confirmed", "very small tributary"]',
 '[{"scenario": "Superior petrosal vein sacrifice", "outcome": "Cerebellar edema and hemorrhagic infarct"},
   {"scenario": "Tentorial sinus division", "outcome": "Usually tolerated if small"}]',
 'III',
 ARRAY['superior_petrosal_vein', 'tentorial_sinus', 'transverse_sinus', 'vein_of_labbe'],
 ARRAY['sacrifice', 'coagulate', 'divide']),

-- ============================================================================
-- HEMOSTASIS PRINCIPLES
-- ============================================================================

('HEMO_001',
 'Sinus Bleeding Management Hierarchy',
 'Venous sinus bleeding requires specific management; blind coagulation worsens injury.',
 'structure.type = ''venous_sinus'' AND complication = ''bleeding''',
 'Air embolism, massive hemorrhage, or sinus thrombosis if mismanaged',
 'Venous sinuses have thin walls and negative pressure gradient. Blind coagulation enlarges defects. Proper management: packing, pressure, gelfoam/surgicel, primary repair, or muscle patch.',
 'vascular', 'hemostasis', 'critical',
 '[]',
 '[{"scenario": "Aggressive bipolar on SSS tear", "outcome": "Larger defect, more bleeding"},
   {"scenario": "Gelfoam with gentle pressure", "outcome": "Hemostasis achieved"}]',
 'III',
 ARRAY['venous_sinus', 'superior_sagittal_sinus', 'transverse_sinus', 'sigmoid_sinus'],
 ARRAY['bleeding', 'injury', 'laceration']),

('HEMO_002',
 'Arterial Bleeding Control Sequence',
 'Arterial bleeding requires proximal control before repair; distal first risks ischemia.',
 'structure.type = ''artery'' AND complication = ''bleeding'' AND severity = ''major''',
 'Ischemic injury if control delayed; repair failure if proximal control inadequate',
 'Proximal control reduces flow, allowing visualization and repair. Distal control first may cause ischemia in the territory between injury and distal clip.',
 'vascular', 'hemostasis', 'critical',
 '["minor bleeding controlled with bipolar", "bleeding at inaccessible location"]',
 '[{"scenario": "MCA injury, distal clip first", "outcome": "MCA territory ischemia"},
   {"scenario": "Proximal temporary clip, then repair", "outcome": "Controlled field, successful repair"}]',
 'III',
 ARRAY['artery', 'mca', 'ica', 'aca', 'pca', 'basilar'],
 ARRAY['bleeding', 'injury', 'rupture']),

-- ============================================================================
-- CSF DYNAMICS PRINCIPLES
-- ============================================================================

('CSF_001',
 'Brain Relaxation for Skull Base Access',
 'Adequate brain relaxation is prerequisite for skull base surgery; tight brain prevents safe access.',
 'approach.type = ''skull_base'' AND brain.relaxation = ''poor''',
 'Inadequate exposure, increased retraction injury, incomplete resection',
 'Skull base approaches traverse narrow corridors. Brain relaxation (mannitol, hyperventilation, CSF drainage, positioning) creates working space and reduces retraction requirements.',
 'general', 'exposure', 'warning',
 '["emergency surgery without time for relaxation"]',
 '[{"scenario": "Pterional without brain relaxation", "outcome": "Excessive retraction injury"},
   {"scenario": "Lumbar drain + mannitol", "outcome": "Slack brain, excellent exposure"}]',
 'III',
 ARRAY['brain', 'frontal_lobe', 'temporal_lobe'],
 ARRAY['expose', 'approach', 'retract']),

('CSF_002',
 'CSF Leak Prevention Hierarchy',
 'Watertight dural closure is essential; repair hierarchy based on defect size.',
 'action = ''close'' AND structure = ''dura'' AND defect.size > ''small''',
 'CSF leak, meningitis, pseudomeningocele if inadequate closure',
 'Small defects: primary closure. Medium: dural substitute. Large: multilayer (dural substitute + muscle + sealant). Skull base: fat graft, tissue sealant, lumbar drain consideration.',
 'general', 'closure', 'warning',
 '["intentional drain left in place"]',
 '[{"scenario": "Large skull base defect, primary closure only", "outcome": "CSF rhinorrhea"},
   {"scenario": "Fat graft + fascia + sealant", "outcome": "Watertight closure"}]',
 'IIb',
 ARRAY['dura', 'skull_base', 'posterior_fossa'],
 ARRAY['close', 'repair']);

-- ============================================================================
-- Add more specialized principles as needed
-- ============================================================================

-- Spine-specific principles
INSERT INTO clinical_principles (
    id, name, statement, antecedent, consequent, mechanism,
    domain, category, severity, evidence_level, trigger_entities, trigger_actions
) VALUES
('SPINE_001',
 'Spinal Cord Tolerance Threshold',
 'The spinal cord tolerates minimal compression or manipulation; even brief insults cause injury.',
 'structure.name = ''spinal_cord'' AND action IN (''compress'', ''manipulate'', ''retract'')',
 'Spinal cord injury with motor/sensory deficit',
 'The spinal cord has minimal redundancy and occupies most of the spinal canal. Even small insults cause edema that further compromises the cord in the confined space.',
 'spine', 'surgical_technique', 'absolute',
 'IIa',
 ARRAY['spinal_cord', 'cervical_cord', 'thoracic_cord'],
 ARRAY['compress', 'manipulate', 'retract', 'instrument']),

('SPINE_002',
 'Vertebral Artery in Cervical Spine',
 'The vertebral artery courses through the transverse foramen; lateral dissection risks injury.',
 'approach.region = ''cervical'' AND action = ''dissect_lateral''',
 'Vertebral artery injury with stroke or exsanguination',
 'The vertebral artery enters the transverse foramen at C6 and courses through C1-C6. Lateral dissection, especially with instruments, risks direct injury.',
 'spine', 'surgical_technique', 'critical',
 'III',
 ARRAY['vertebral_artery', 'transverse_foramen', 'cervical_spine'],
 ARRAY['dissect', 'drill', 'decompress']);

COMMIT;

-- ============================================================================
-- VERIFICATION QUERY
-- ============================================================================
-- Run after seeding to verify:
-- SELECT domain, category, COUNT(*) as count 
-- FROM clinical_principles 
-- GROUP BY domain, category 
-- ORDER BY domain, category;
