"""
NeuroSynth 2.0 - Vision Module
===============================

Medical image processing for anatomical extraction.

This module provides:
- AnatomicalExtractor: Extract entity properties from MRI/CTA
- Segmentation pipelines for brain parcellation and vessels
- Vessel enhancement filters

Dependencies (optional):
- torch >= 2.0
- monai >= 1.3
- nibabel
- scipy
- skimage

Usage:
    from src.vision import AnatomicalExtractor
    
    extractor = AnatomicalExtractor(model_path="weights/vessel_seg.pt")
    await extractor.initialize()
    
    result = await extractor.extract_from_nifti("scan.nii.gz")
    entities = extractor.result_to_entity_physics(result)
"""

from src.neurosynth2.vision.anatomical_extractor import (
    AnatomicalExtractor,
    ExtractionResult,
    ExtractedVessel,
    ExtractedRegion,
    VesselSegmentationModel,
    VesselFeatureExtractor,
    RegionFeatureExtractor,
    STANDARD_VESSEL_DIAMETERS,
    MNI_LANDMARKS,
)

from src.neurosynth2.vision.segmentation import (
    SegmentationTask,
    Modality,
    SegmentationConfig,
    SegmentationInference,
    BrainSegmentationPipeline,
    VesselSegmentationPipeline,
    VesselEnhancementFilter,
    SkullStripper,
    BRAIN_LABELS,
    VESSEL_LABELS,
    TUMOR_LABELS,
)

__all__ = [
    # Main extractor
    "AnatomicalExtractor",
    "ExtractionResult",
    "ExtractedVessel",
    "ExtractedRegion",
    
    # Model components
    "VesselSegmentationModel",
    "VesselFeatureExtractor",
    "RegionFeatureExtractor",
    
    # Segmentation
    "SegmentationTask",
    "Modality",
    "SegmentationConfig",
    "SegmentationInference",
    "BrainSegmentationPipeline",
    "VesselSegmentationPipeline",
    
    # Preprocessing
    "VesselEnhancementFilter",
    "SkullStripper",
    
    # Constants
    "STANDARD_VESSEL_DIAMETERS",
    "MNI_LANDMARKS",
    "BRAIN_LABELS",
    "VESSEL_LABELS",
    "TUMOR_LABELS",
]

__version__ = "2.0.0"
