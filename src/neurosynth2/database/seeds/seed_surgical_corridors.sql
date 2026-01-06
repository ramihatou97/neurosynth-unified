-- ============================================================================
-- NEUROSYNTH 2.0 - SURGICAL CORRIDORS SEED DATA
-- ============================================================================
-- Purpose: Populates surgical approach definitions with structure sequences
-- Date: 2026-01-04
--
-- Each corridor defines:
-- - The ordered sequence of structures encountered
-- - Structures at risk during the approach
-- - Critical steps with specific risks
-- - Required monitoring and equipment
-- ============================================================================

BEGIN;

INSERT INTO surgical_corridors (
    name, display_name, approach_type, category, subspecialty,
    structure_sequence, structures_at_risk, critical_steps,
    patient_position, head_position, required_monitoring, required_equipment,
    primary_indications, contraindications, evidence_level
) VALUES

-- ============================================================================
-- SKULL BASE APPROACHES
-- ============================================================================

('pterional',
 'Pterional Craniotomy',
 'pterional',
 'cranial_base',
 'skull_base',
 ARRAY[
    'scalp', 'temporalis_muscle', 'periosteum', 'skull_bone',
    'dura', 'sylvian_fissure', 'frontal_lobe', 'temporal_lobe',
    'internal_carotid_artery', 'middle_cerebral_artery', 'optic_nerve',
    'oculomotor_nerve', 'anterior_clinoid'
 ],
 ARRAY[
    'superficial_temporal_artery', 'frontalis_branch_facial_nerve',
    'sylvian_veins', 'middle_cerebral_artery', 'optic_nerve',
    'oculomotor_nerve', 'internal_carotid_artery'
 ],
 '[
    {"step": 4, "action": "drill", "structure": "sphenoid_wing", "risk": "optic_nerve_injury"},
    {"step": 5, "action": "open", "structure": "dura", "risk": "cortical_injury"},
    {"step": 6, "action": "dissect", "structure": "sylvian_fissure", "risk": "vein_sacrifice"},
    {"step": 8, "action": "dissect", "structure": "ica", "risk": "perforator_injury"}
 ]'::jsonb,
 'supine',
 'rotated_30_contralateral, extended',
 ARRAY['SSEP', 'MEP', 'EEG'],
 ARRAY['microscope', 'high_speed_drill', 'bipolar', 'neuronavigation'],
 ARRAY['anterior_circulation_aneurysm', 'sellar_suprasellar_tumor', 'sphenoid_wing_meningioma', 'optic_nerve_decompression'],
 ARRAY['posterior_fossa_pathology', 'far_lateral_pathology'],
 'III'),

('orbitozygomatic',
 'Orbitozygomatic Approach',
 'orbitozygomatic',
 'cranial_base',
 'skull_base',
 ARRAY[
    'scalp', 'temporalis_muscle', 'periosteum', 'skull_bone',
    'orbital_rim', 'zygomatic_arch', 'dura', 'sylvian_fissure',
    'cavernous_sinus', 'basilar_apex', 'posterior_communicating_artery',
    'oculomotor_nerve', 'trochlear_nerve'
 ],
 ARRAY[
    'frontalis_branch_facial_nerve', 'lacrimal_gland', 'periorbita',
    'superior_orbital_fissure', 'optic_nerve', 'oculomotor_nerve',
    'trochlear_nerve', 'abducens_nerve', 'internal_carotid_artery'
 ],
 '[
    {"step": 4, "action": "osteotomy", "structure": "orbital_rim", "risk": "orbital_hematoma"},
    {"step": 5, "action": "osteotomy", "structure": "zygomatic_arch", "risk": "facial_asymmetry"},
    {"step": 8, "action": "dissect", "structure": "cavernous_sinus", "risk": "carotid_injury"}
 ]'::jsonb,
 'supine',
 'rotated_30_contralateral, extended, vertex_down',
 ARRAY['SSEP', 'MEP', 'EEG', 'cranial_nerve_monitoring'],
 ARRAY['microscope', 'high_speed_drill', 'bipolar', 'neuronavigation', 'reciprocating_saw'],
 ARRAY['basilar_apex_aneurysm', 'giant_paraclinoid_aneurysm', 'cavernous_sinus_tumor', 'extensive_sphenoid_wing_meningioma'],
 ARRAY['elderly_patient_with_poor_bone_quality', 'previous_orbital_surgery'],
 'III'),

('retrosigmoid',
 'Retrosigmoid Approach',
 'retrosigmoid',
 'infratentorial',
 'skull_base',
 ARRAY[
    'scalp', 'nuchal_muscles', 'periosteum', 'occipital_bone',
    'dura', 'cerebellum', 'cerebellopontine_angle', 'trigeminal_nerve',
    'facial_nerve', 'vestibulocochlear_nerve', 'aica', 'petrosal_vein'
 ],
 ARRAY[
    'sigmoid_sinus', 'transverse_sinus', 'emissary_veins',
    'facial_nerve', 'vestibulocochlear_nerve', 'cochlear_nerve',
    'aica', 'petrosal_vein', 'trigeminal_nerve'
 ],
 '[
    {"step": 3, "action": "drill", "structure": "occipital_bone", "risk": "sinus_injury"},
    {"step": 4, "action": "open", "structure": "dura", "risk": "cerebellar_injury"},
    {"step": 5, "action": "retract", "structure": "cerebellum", "risk": "contusion"},
    {"step": 7, "action": "dissect", "structure": "cpa_tumor", "risk": "facial_nerve_injury"}
 ]'::jsonb,
 'lateral',
 'flexed, rotated_toward_floor',
 ARRAY['SSEP', 'BAEP', 'facial_EMG', 'direct_CN_stimulation'],
 ARRAY['microscope', 'high_speed_drill', 'bipolar', 'neuronavigation', 'CUSA'],
 ARRAY['vestibular_schwannoma', 'meningioma_cpa', 'trigeminal_neuralgia', 'hemifacial_spasm', 'epidermoid'],
 ARRAY['large_tumor_with_brainstem_compression', 'only_hearing_ear_with_serviceable_hearing'],
 'IIb'),

('translabyrinthine',
 'Translabyrinthine Approach',
 'translabyrinthine',
 'cranial_base',
 'skull_base',
 ARRAY[
    'scalp', 'mastoid_cortex', 'mastoid_air_cells', 'sigmoid_sinus',
    'semicircular_canals', 'vestibule', 'internal_auditory_canal',
    'facial_nerve', 'vestibulocochlear_nerve', 'tumor', 'dura'
 ],
 ARRAY[
    'sigmoid_sinus', 'jugular_bulb', 'facial_nerve', 'dura',
    'tegmen', 'posterior_fossa_dura'
 ],
 '[
    {"step": 2, "action": "drill", "structure": "mastoid", "risk": "facial_nerve_thermal_injury"},
    {"step": 3, "action": "identify", "structure": "sigmoid_sinus", "risk": "sinus_injury"},
    {"step": 4, "action": "drill", "structure": "labyrinth", "risk": "facial_nerve_injury"},
    {"step": 6, "action": "dissect", "structure": "iac_tumor", "risk": "facial_nerve_transection"}
 ]'::jsonb,
 'supine',
 'rotated_away, shoulder_roll',
 ARRAY['facial_EMG', 'direct_facial_nerve_stimulation'],
 ARRAY['microscope', 'high_speed_drill', 'diamond_burr', 'bipolar', 'facial_nerve_monitor'],
 ARRAY['vestibular_schwannoma_nonserviceable_hearing', 'large_vestibular_schwannoma'],
 ARRAY['serviceable_hearing', 'only_hearing_ear'],
 'IIb'),

('middle_fossa',
 'Middle Fossa Approach',
 'middle_fossa',
 'cranial_base',
 'skull_base',
 ARRAY[
    'scalp', 'temporalis_muscle', 'temporal_squama', 'dura',
    'temporal_lobe', 'floor_middle_fossa', 'arcuate_eminence',
    'greater_superficial_petrosal_nerve', 'geniculate_ganglion',
    'internal_auditory_canal', 'facial_nerve', 'superior_vestibular_nerve'
 ],
 ARRAY[
    'middle_meningeal_artery', 'temporal_lobe', 'gspn',
    'geniculate_ganglion', 'facial_nerve', 'cochlea', 'carotid_artery'
 ],
 '[
    {"step": 4, "action": "elevate", "structure": "temporal_lobe", "risk": "temporal_contusion"},
    {"step": 5, "action": "identify", "structure": "arcuate_eminence", "risk": "labyrinth_violation"},
    {"step": 6, "action": "drill", "structure": "petrous_apex", "risk": "carotid_injury"},
    {"step": 7, "action": "dissect", "structure": "iac", "risk": "hearing_loss"}
 ]'::jsonb,
 'supine',
 'rotated_45_contralateral',
 ARRAY['BAEP', 'facial_EMG', 'direct_CN_stimulation'],
 ARRAY['microscope', 'high_speed_drill', 'diamond_burr', 'House_Urban_retractor'],
 ARRAY['small_intracanalicular_vestibular_schwannoma', 'facial_nerve_decompression', 'superior_canal_dehiscence'],
 ARRAY['large_tumor', 'poor_hearing', 'elderly'],
 'III'),

-- ============================================================================
-- VASCULAR APPROACHES
-- ============================================================================

('interhemispheric',
 'Interhemispheric Approach',
 'interhemispheric',
 'supratentorial',
 'vascular',
 ARRAY[
    'scalp', 'periosteum', 'frontal_bone', 'dura',
    'superior_sagittal_sinus', 'falx', 'bridging_veins',
    'cingulate_gyrus', 'pericallosal_artery', 'corpus_callosum',
    'anterior_cerebral_artery', 'a2_segment'
 ],
 ARRAY[
    'superior_sagittal_sinus', 'bridging_veins', 'motor_cortex',
    'pericallosal_artery', 'callosomarginal_artery'
 ],
 '[
    {"step": 3, "action": "open", "structure": "dura", "risk": "sss_injury"},
    {"step": 5, "action": "retract", "structure": "frontal_lobe", "risk": "bridging_vein_avulsion"},
    {"step": 6, "action": "dissect", "structure": "interhemispheric_fissure", "risk": "pericallosal_injury"}
 ]'::jsonb,
 'supine',
 'neutral, slightly_flexed',
 ARRAY['SSEP', 'MEP', 'EEG'],
 ARRAY['microscope', 'bipolar', 'neuronavigation'],
 ARRAY['acom_aneurysm', 'distal_aca_aneurysm', 'falcine_meningioma', 'corpus_callosum_tumor'],
 ARRAY['bilateral_frontal_pathology'],
 'III'),

('far_lateral',
 'Far Lateral Approach',
 'far_lateral',
 'infratentorial',
 'vascular',
 ARRAY[
    'scalp', 'nuchal_muscles', 'c1_posterior_arch', 'foramen_magnum',
    'vertebral_artery', 'dura', 'cerebellum', 'lower_cranial_nerves',
    'hypoglossal_nerve', 'vertebrobasilar_junction', 'pica'
 ],
 ARRAY[
    'vertebral_artery', 'pica', 'hypoglossal_nerve', 'accessory_nerve',
    'vagus_nerve', 'c1_c2_nerve_roots', 'vertebral_venous_plexus'
 ],
 '[
    {"step": 3, "action": "drill", "structure": "occipital_condyle", "risk": "vertebral_artery_injury"},
    {"step": 4, "action": "mobilize", "structure": "vertebral_artery", "risk": "thrombosis"},
    {"step": 6, "action": "dissect", "structure": "lower_cns", "risk": "cn_deficit"}
 ]'::jsonb,
 'lateral',
 'flexed, rotated_toward_floor',
 ARRAY['SSEP', 'MEP', 'BAEP', 'lower_cranial_nerve_monitoring'],
 ARRAY['microscope', 'high_speed_drill', 'bipolar', 'neuronavigation'],
 ARRAY['vertebrobasilar_aneurysm', 'foramen_magnum_meningioma', 'lower_clival_lesion', 'pica_aneurysm'],
 ARRAY['severe_cervical_degenerative_disease', 'craniocervical_instability'],
 'III'),

-- ============================================================================
-- TUMOR APPROACHES
-- ============================================================================

('suboccipital_midline',
 'Suboccipital Midline Approach',
 'suboccipital_midline',
 'infratentorial',
 'tumor',
 ARRAY[
    'scalp', 'nuchal_muscles', 'occipital_bone', 'foramen_magnum',
    'dura', 'cerebellar_vermis', 'fourth_ventricle', 'brainstem',
    'floor_fourth_ventricle', 'choroid_plexus'
 ],
 ARRAY[
    'torcula', 'occipital_sinus', 'pica', 'vermian_veins',
    'floor_fourth_ventricle', 'dentate_nuclei'
 ],
 '[
    {"step": 3, "action": "drill", "structure": "foramen_magnum", "risk": "torcula_injury"},
    {"step": 4, "action": "open", "structure": "dura", "risk": "cerebellar_injury"},
    {"step": 5, "action": "split", "structure": "vermis", "risk": "cerebellar_mutism"},
    {"step": 6, "action": "resect", "structure": "tumor", "risk": "brainstem_injury"}
 ]'::jsonb,
 'prone',
 'flexed',
 ARRAY['SSEP', 'MEP', 'BAEP', 'facial_EMG'],
 ARRAY['microscope', 'high_speed_drill', 'CUSA', 'bipolar'],
 ARRAY['medulloblastoma', 'ependymoma', 'pilocytic_astrocytoma', 'hemangioblastoma'],
 ARRAY['upward_herniation_risk', 'severe_hydrocephalus'],
 'IIb'),

('transsphenoidal',
 'Endoscopic Transsphenoidal Approach',
 'transsphenoidal',
 'transsphenoidal',
 'tumor',
 ARRAY[
    'nasal_mucosa', 'middle_turbinate', 'sphenoid_ostium',
    'sphenoid_sinus', 'sellar_floor', 'dura', 'pituitary_gland',
    'cavernous_sinus', 'internal_carotid_artery', 'optic_chiasm'
 ],
 ARRAY[
    'internal_carotid_artery', 'optic_nerve', 'optic_chiasm',
    'cavernous_sinus', 'sphenopalatine_artery', 'pituitary_stalk'
 ],
 '[
    {"step": 4, "action": "drill", "structure": "sellar_floor", "risk": "carotid_injury"},
    {"step": 5, "action": "open", "structure": "dura", "risk": "csf_leak"},
    {"step": 6, "action": "resect", "structure": "tumor", "risk": "stalk_injury"},
    {"step": 7, "action": "close", "structure": "sella", "risk": "csf_leak"}
 ]'::jsonb,
 'supine',
 'neutral_slightly_extended',
 ARRAY['VEP'],
 ARRAY['endoscope', 'high_speed_drill', 'neuronavigation', 'doppler'],
 ARRAY['pituitary_adenoma', 'rathke_cleft_cyst', 'craniopharyngioma', 'clival_chordoma'],
 ARRAY['active_sinusitis', 'coagulopathy', 'far_lateral_extension'],
 'IIa'),

-- ============================================================================
-- SUPRATENTORIAL APPROACHES  
-- ============================================================================

('frontal_craniotomy',
 'Frontal Craniotomy',
 'frontal',
 'supratentorial',
 'tumor',
 ARRAY[
    'scalp', 'periosteum', 'frontal_bone', 'dura',
    'frontal_lobe', 'frontal_sinus', 'superior_sagittal_sinus',
    'bridging_veins', 'motor_cortex'
 ],
 ARRAY[
    'superior_sagittal_sinus', 'bridging_veins', 'motor_cortex',
    'frontal_sinus_mucosa'
 ],
 '[
    {"step": 3, "action": "craniotomy", "structure": "frontal_bone", "risk": "sinus_entry"},
    {"step": 4, "action": "open", "structure": "dura", "risk": "cortical_injury"},
    {"step": 5, "action": "retract", "structure": "frontal_lobe", "risk": "contusion"}
 ]'::jsonb,
 'supine',
 'neutral',
 ARRAY['SSEP', 'MEP'],
 ARRAY['microscope', 'high_speed_drill', 'bipolar', 'neuronavigation'],
 ARRAY['frontal_glioma', 'frontal_meningioma', 'frontal_metastasis'],
 ARRAY['bilateral_frontal_involvement'],
 'III'),

('temporal_craniotomy',
 'Temporal Craniotomy',
 'temporal',
 'supratentorial',
 'tumor',
 ARRAY[
    'scalp', 'temporalis_muscle', 'temporal_bone', 'dura',
    'temporal_lobe', 'middle_cerebral_artery', 'sylvian_fissure',
    'uncus', 'hippocampus'
 ],
 ARRAY[
    'middle_meningeal_artery', 'sylvian_veins', 'vein_of_labbe',
    'language_cortex', 'optic_radiation'
 ],
 '[
    {"step": 4, "action": "open", "structure": "dura", "risk": "cortical_injury"},
    {"step": 5, "action": "retract", "structure": "temporal_lobe", "risk": "vein_of_labbe_injury"},
    {"step": 6, "action": "resect", "structure": "tumor", "risk": "language_deficit"}
 ]'::jsonb,
 'supine',
 'rotated_contralateral',
 ARRAY['SSEP', 'MEP', 'ECoG', 'language_mapping'],
 ARRAY['microscope', 'bipolar', 'neuronavigation', 'CUSA', 'awake_setup'],
 ARRAY['temporal_glioma', 'temporal_cavernoma', 'mesial_temporal_sclerosis'],
 ARRAY['dominant_hemisphere_eloquent_involvement'],
 'III'),

('parietal_craniotomy',
 'Parietal Craniotomy',
 'parietal',
 'supratentorial',
 'tumor',
 ARRAY[
    'scalp', 'periosteum', 'parietal_bone', 'dura',
    'parietal_lobe', 'sensory_cortex', 'motor_cortex',
    'superior_sagittal_sinus', 'bridging_veins'
 ],
 ARRAY[
    'superior_sagittal_sinus', 'bridging_veins', 'motor_cortex',
    'sensory_cortex', 'angular_gyrus'
 ],
 '[
    {"step": 3, "action": "craniotomy", "structure": "parietal_bone", "risk": "sss_injury"},
    {"step": 4, "action": "open", "structure": "dura", "risk": "cortical_injury"},
    {"step": 5, "action": "map", "structure": "cortex", "risk": "motor_sensory_deficit"}
 ]'::jsonb,
 'supine',
 'lateral_position_or_neutral',
 ARRAY['SSEP', 'MEP', 'phase_reversal_mapping'],
 ARRAY['microscope', 'bipolar', 'neuronavigation', 'cortical_stimulator'],
 ARRAY['parietal_glioma', 'parietal_metastasis', 'parasagittal_meningioma'],
 ARRAY['bilateral_involvement'],
 'III');

COMMIT;

-- ============================================================================
-- VERIFICATION QUERY
-- ============================================================================
-- SELECT approach_type, name, array_length(structure_sequence, 1) as steps,
--        array_length(structures_at_risk, 1) as risk_structures
-- FROM surgical_corridors
-- ORDER BY category, approach_type;
