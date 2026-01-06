"""
NeuroSynth 2.0 - Anatomical Extractor
======================================

MONAI/PyTorch-based extraction of anatomical properties from medical images.

This module bridges the gap between raw imaging data and the physics-aware
reasoning system by automatically extracting:
- Vessel diameters and territories
- Structure positions and volumes
- Spatial relationships

Architecture:
    MRI/CTA → Segmentation → Feature Extraction → EntityPhysics

Dependencies:
    - torch >= 2.0
    - monai >= 1.3
    - nibabel
    - numpy
    - scipy

Usage:
    extractor = AnatomicalExtractor(model_path="weights/vessel_seg.pt")
    await extractor.initialize()
    
    entities = await extractor.extract_from_nifti("patient_001.nii.gz")
    # Returns list of EntityPhysics with populated spatial data
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
import json

import numpy as np

logger = logging.getLogger(__name__)

# Conditional imports for optional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    logger.warning("PyTorch not available. Vision features disabled.")

# Dummy decorator for when torch is unavailable
def _no_grad_dummy(func):
    return func

if not TORCH_AVAILABLE:
    class _DummyTorch:
        @staticmethod
        def no_grad():
            return _no_grad_dummy
    torch = _DummyTorch()

try:
    import monai
    from monai.networks.nets import UNet, SwinUNETR, UNETR
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
        Spacingd, ScaleIntensityRanged, CropForegroundd,
        EnsureTyped, Invertd, AsDiscreted, SaveImaged
    )
    from monai.inferers import sliding_window_inference
    from monai.data import decollate_batch
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    logger.warning("MONAI not available. Using fallback extraction.")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractedVessel:
    """Extracted vessel properties."""
    name: str
    centerline: np.ndarray  # Nx3 array of points
    diameter_mm: float
    length_mm: float
    volume_mm3: float
    start_point: Tuple[float, float, float]
    end_point: Tuple[float, float, float]
    branches: List[str] = field(default_factory=list)
    parent_vessel: Optional[str] = None
    confidence: float = 0.8


@dataclass
class ExtractedRegion:
    """Extracted brain region properties."""
    name: str
    centroid: Tuple[float, float, float]
    volume_mm3: float
    bounding_box: Dict[str, float]  # min_x, max_x, etc.
    is_eloquent: bool = False
    hemisphere: str = "bilateral"  # left, right, bilateral
    depth_from_surface_mm: float = 0.0
    adjacent_structures: List[str] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class ExtractionResult:
    """Complete extraction result from a scan."""
    scan_id: str
    modality: str  # MRI, CTA, MRA
    voxel_spacing: Tuple[float, float, float]
    dimensions: Tuple[int, int, int]
    vessels: List[ExtractedVessel] = field(default_factory=list)
    regions: List[ExtractedRegion] = field(default_factory=list)
    landmarks: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_time_ms: int = 0
    model_version: str = "1.0.0"


# =============================================================================
# ATLAS DATA (Fallback when no model available)
# =============================================================================

# Standard vessel diameters from literature (mm)
STANDARD_VESSEL_DIAMETERS = {
    "internal_carotid_artery": 4.5,
    "middle_cerebral_artery_m1": 3.0,
    "middle_cerebral_artery_m2": 2.2,
    "anterior_cerebral_artery_a1": 2.5,
    "anterior_cerebral_artery_a2": 2.0,
    "posterior_cerebral_artery_p1": 2.2,
    "posterior_cerebral_artery_p2": 1.8,
    "basilar_artery": 4.0,
    "vertebral_artery": 3.5,
    "superior_cerebellar_artery": 1.5,
    "aica": 1.2,
    "pica": 1.5,
    "lenticulostriate_artery": 0.3,  # Perforator
    "thalamoperforator": 0.4,
    "anterior_choroidal_artery": 1.0,
    "posterior_communicating_artery": 1.5,
    "anterior_communicating_artery": 1.5,
    "ophthalmic_artery": 1.5,
}

# MNI atlas approximate coordinates (mm from AC-PC origin)
MNI_LANDMARKS = {
    # Vascular landmarks
    "carotid_bifurcation": (12, 0, -5),
    "mca_bifurcation": (35, -5, 10),
    "basilar_tip": (0, -25, -20),
    "acom_complex": (0, 25, -5),
    
    # Anatomical landmarks
    "anterior_commissure": (0, 0, 0),
    "posterior_commissure": (0, -25, 0),
    "pineal_gland": (0, -30, 5),
    "pituitary_gland": (0, 5, -35),
    "foramen_magnum": (0, -35, -45),
    
    # Eloquent regions
    "motor_cortex_hand": (35, -20, 55),
    "brocas_area": (-45, 20, 20),
    "wernickes_area": (-55, -40, 20),
    "primary_visual_cortex": (5, -85, 5),
}


# =============================================================================
# SEGMENTATION MODEL WRAPPER
# =============================================================================

class VesselSegmentationModel:
    """
    Wrapper for vessel segmentation neural network.
    
    Supports multiple architectures:
    - UNet (default, lightweight)
    - UNETR (transformer-based)
    - SwinUNETR (state-of-the-art)
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        architecture: str = "unet",
        device: str = "auto",
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,  # background, artery, vein, other
    ):
        self.model_path = model_path
        self.architecture = architecture
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self._initialized = False
    
    def _build_model(self) -> "nn.Module":
        """Build the segmentation model architecture."""
        if self.architecture == "unet":
            return UNet(
                spatial_dims=self.spatial_dims,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm="batch",
            )
        elif self.architecture == "unetr":
            return UNETR(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                img_size=(96, 96, 96),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="conv",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.0,
            )
        elif self.architecture == "swinunetr":
            return SwinUNETR(
                img_size=(96, 96, 96),
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                feature_size=48,
                use_checkpoint=True,
            )
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
    
    def initialize(self):
        """Initialize the model."""
        if not MONAI_AVAILABLE or not TORCH_AVAILABLE:
            logger.warning("MONAI/PyTorch not available, model not initialized")
            return False
        
        try:
            self.model = self._build_model()
            
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading weights from {self.model_path}")
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                logger.warning("No weights loaded, using random initialization")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            self._initialized = True
            
            logger.info(f"Model initialized on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        roi_size: Tuple[int, int, int] = (96, 96, 96),
        sw_batch_size: int = 4,
        overlap: float = 0.5,
    ) -> np.ndarray:
        """
        Run segmentation inference.
        
        Args:
            image: Input volume (H, W, D) or (C, H, W, D)
            roi_size: Sliding window size
            sw_batch_size: Batch size for sliding window
            overlap: Overlap ratio for sliding window
        
        Returns:
            Segmentation mask (H, W, D) with class labels
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        
        # Ensure correct shape
        if image.ndim == 3:
            image = image[np.newaxis, ...]  # Add channel dim
        if image.ndim == 4 and image.shape[0] != self.in_channels:
            image = image.transpose(3, 0, 1, 2)  # Reorder if needed
        
        # Convert to tensor
        tensor = torch.from_numpy(image).float().unsqueeze(0)  # Add batch dim
        tensor = tensor.to(self.device)
        
        # Sliding window inference
        output = sliding_window_inference(
            tensor,
            roi_size,
            sw_batch_size,
            self.model,
            overlap=overlap,
            mode="gaussian",
        )
        
        # Get class predictions
        pred = torch.argmax(output, dim=1).squeeze(0)
        return pred.cpu().numpy()


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class VesselFeatureExtractor:
    """Extract geometric features from vessel segmentation."""
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.voxel_spacing = voxel_spacing
    
    def extract_centerline(
        self,
        binary_mask: np.ndarray,
        method: str = "skeletonize"
    ) -> np.ndarray:
        """Extract vessel centerline from binary mask."""
        if not SCIPY_AVAILABLE:
            return np.array([])
        
        from skimage.morphology import skeletonize_3d
        
        if method == "skeletonize":
            skeleton = skeletonize_3d(binary_mask.astype(bool))
            points = np.argwhere(skeleton)
            # Convert to physical coordinates
            points = points * np.array(self.voxel_spacing)
            return points
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def measure_diameter(
        self,
        binary_mask: np.ndarray,
        centerline: np.ndarray
    ) -> float:
        """Measure average vessel diameter along centerline."""
        if not SCIPY_AVAILABLE or len(centerline) == 0:
            return 0.0
        
        # Distance transform
        distance = ndimage.distance_transform_edt(
            binary_mask,
            sampling=self.voxel_spacing
        )
        
        # Sample distances at centerline points
        # Convert back to voxel coordinates
        voxel_coords = (centerline / np.array(self.voxel_spacing)).astype(int)
        voxel_coords = np.clip(
            voxel_coords,
            [0, 0, 0],
            [s - 1 for s in binary_mask.shape]
        )
        
        radii = distance[
            voxel_coords[:, 0],
            voxel_coords[:, 1],
            voxel_coords[:, 2]
        ]
        
        # Diameter = 2 * radius
        return float(np.mean(radii) * 2)
    
    def measure_length(self, centerline: np.ndarray) -> float:
        """Measure total vessel length along centerline."""
        if len(centerline) < 2:
            return 0.0
        
        # Sum of segment lengths
        segments = np.diff(centerline, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        return float(np.sum(lengths))
    
    def measure_volume(self, binary_mask: np.ndarray) -> float:
        """Measure vessel volume in mm³."""
        voxel_volume = np.prod(self.voxel_spacing)
        return float(np.sum(binary_mask > 0) * voxel_volume)
    
    def find_endpoints(self, centerline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find vessel endpoints (start and end)."""
        if len(centerline) < 2:
            return np.zeros(3), np.zeros(3)
        return centerline[0], centerline[-1]
    
    def find_bifurcations(
        self,
        skeleton: np.ndarray,
        min_branch_length: float = 5.0
    ) -> List[np.ndarray]:
        """Find bifurcation points in vessel skeleton."""
        if not SCIPY_AVAILABLE:
            return []
        
        # Count neighbors for each skeleton point
        # Bifurcation = point with 3+ neighbors
        kernel = ndimage.generate_binary_structure(3, 3)
        neighbor_count = ndimage.convolve(
            skeleton.astype(int),
            kernel,
            mode='constant'
        )
        
        # Bifurcation points have 4+ neighbors (including self)
        bifurcations = np.argwhere((skeleton > 0) & (neighbor_count >= 4))
        
        return [b * np.array(self.voxel_spacing) for b in bifurcations]


class RegionFeatureExtractor:
    """Extract features from brain region segmentation."""
    
    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.voxel_spacing = voxel_spacing
    
    def extract_centroid(self, binary_mask: np.ndarray) -> Tuple[float, float, float]:
        """Extract region centroid in physical coordinates."""
        if not SCIPY_AVAILABLE:
            return (0.0, 0.0, 0.0)
        
        centroid_voxel = ndimage.center_of_mass(binary_mask)
        centroid_mm = tuple(
            c * s for c, s in zip(centroid_voxel, self.voxel_spacing)
        )
        return centroid_mm
    
    def extract_bounding_box(self, binary_mask: np.ndarray) -> Dict[str, float]:
        """Extract axis-aligned bounding box."""
        points = np.argwhere(binary_mask > 0)
        if len(points) == 0:
            return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0}
        
        mins = points.min(axis=0) * np.array(self.voxel_spacing)
        maxs = points.max(axis=0) * np.array(self.voxel_spacing)
        
        return {
            "min_x": float(mins[0]), "max_x": float(maxs[0]),
            "min_y": float(mins[1]), "max_y": float(maxs[1]),
            "min_z": float(mins[2]), "max_z": float(maxs[2]),
        }
    
    def measure_volume(self, binary_mask: np.ndarray) -> float:
        """Measure region volume in mm³."""
        voxel_volume = np.prod(self.voxel_spacing)
        return float(np.sum(binary_mask > 0) * voxel_volume)
    
    def determine_hemisphere(
        self,
        centroid: Tuple[float, float, float],
        midline_x: float = 0.0,
        threshold: float = 5.0
    ) -> str:
        """Determine which hemisphere the region is in."""
        x = centroid[0] - midline_x
        if abs(x) < threshold:
            return "bilateral"
        return "right" if x > 0 else "left"
    
    def estimate_depth(
        self,
        binary_mask: np.ndarray,
        brain_surface_mask: np.ndarray
    ) -> float:
        """Estimate minimum depth from brain surface."""
        if not SCIPY_AVAILABLE:
            return 0.0
        
        # Distance from surface
        surface_dist = ndimage.distance_transform_edt(
            brain_surface_mask,
            sampling=self.voxel_spacing
        )
        
        # Sample at region points
        region_points = np.argwhere(binary_mask > 0)
        if len(region_points) == 0:
            return 0.0
        
        depths = surface_dist[
            region_points[:, 0],
            region_points[:, 1],
            region_points[:, 2]
        ]
        
        return float(np.min(depths))


# =============================================================================
# MAIN EXTRACTOR
# =============================================================================

class AnatomicalExtractor:
    """
    Main class for extracting anatomical properties from medical images.
    
    Combines segmentation models with feature extraction to populate
    EntityPhysics objects for the reasoning system.
    """
    
    def __init__(
        self,
        vessel_model_path: Optional[str] = None,
        region_model_path: Optional[str] = None,
        device: str = "auto",
        use_atlas_fallback: bool = True,
    ):
        self.vessel_model_path = vessel_model_path
        self.region_model_path = region_model_path
        self.device = device
        self.use_atlas_fallback = use_atlas_fallback
        
        self.vessel_model = None
        self.region_model = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize models asynchronously."""
        if MONAI_AVAILABLE and TORCH_AVAILABLE:
            # Initialize vessel model
            if self.vessel_model_path:
                self.vessel_model = VesselSegmentationModel(
                    model_path=self.vessel_model_path,
                    device=self.device
                )
                self.vessel_model.initialize()
            
            # Initialize region model (could be different architecture)
            if self.region_model_path:
                self.region_model = VesselSegmentationModel(
                    model_path=self.region_model_path,
                    architecture="swinunetr",
                    out_channels=96,  # Many brain regions
                    device=self.device
                )
                self.region_model.initialize()
        
        self._initialized = True
        logger.info("AnatomicalExtractor initialized")
        return True
    
    async def extract_from_nifti(
        self,
        nifti_path: str,
        modality: str = "MRI",
        extract_vessels: bool = True,
        extract_regions: bool = True,
    ) -> ExtractionResult:
        """
        Extract anatomical features from a NIfTI file.
        
        Args:
            nifti_path: Path to NIfTI file
            modality: Image modality (MRI, CTA, MRA)
            extract_vessels: Whether to extract vessel features
            extract_regions: Whether to extract region features
        
        Returns:
            ExtractionResult with vessels, regions, and landmarks
        """
        import time
        start_time = time.time()
        
        # Load image
        if not NIBABEL_AVAILABLE:
            logger.error("nibabel not available")
            return self._fallback_extraction(nifti_path, modality)
        
        try:
            nii = nib.load(nifti_path)
            image = nii.get_fdata()
            spacing = tuple(nii.header.get_zooms()[:3])
            
            result = ExtractionResult(
                scan_id=Path(nifti_path).stem,
                modality=modality,
                voxel_spacing=spacing,
                dimensions=image.shape[:3],
            )
            
            # Vessel extraction
            if extract_vessels:
                vessels = await self._extract_vessels(image, spacing)
                result.vessels = vessels
            
            # Region extraction
            if extract_regions:
                regions = await self._extract_regions(image, spacing)
                result.regions = regions
            
            # Add standard landmarks
            result.landmarks = MNI_LANDMARKS.copy()
            
            result.extraction_time_ms = int((time.time() - start_time) * 1000)
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            if self.use_atlas_fallback:
                return self._fallback_extraction(nifti_path, modality)
            raise
    
    async def _extract_vessels(
        self,
        image: np.ndarray,
        spacing: Tuple[float, float, float]
    ) -> List[ExtractedVessel]:
        """Extract vessel features from image."""
        vessels = []
        feature_extractor = VesselFeatureExtractor(spacing)
        
        if self.vessel_model and self.vessel_model._initialized:
            # Run segmentation
            segmentation = self.vessel_model.predict(image)
            
            # Extract features for each vessel class
            vessel_mask = segmentation == 1  # Assume class 1 = arteries
            
            if np.sum(vessel_mask) > 0:
                centerline = feature_extractor.extract_centerline(vessel_mask)
                diameter = feature_extractor.measure_diameter(vessel_mask, centerline)
                length = feature_extractor.measure_length(centerline)
                volume = feature_extractor.measure_volume(vessel_mask)
                start, end = feature_extractor.find_endpoints(centerline)
                
                vessels.append(ExtractedVessel(
                    name="detected_vessel",
                    centerline=centerline,
                    diameter_mm=diameter,
                    length_mm=length,
                    volume_mm3=volume,
                    start_point=tuple(start),
                    end_point=tuple(end),
                    confidence=0.85,
                ))
        else:
            # Fallback to atlas diameters
            for name, diameter in STANDARD_VESSEL_DIAMETERS.items():
                vessels.append(ExtractedVessel(
                    name=name,
                    centerline=np.array([]),
                    diameter_mm=diameter,
                    length_mm=0.0,
                    volume_mm3=0.0,
                    start_point=(0, 0, 0),
                    end_point=(0, 0, 0),
                    confidence=0.7,  # Lower confidence for atlas values
                ))
        
        return vessels
    
    async def _extract_regions(
        self,
        image: np.ndarray,
        spacing: Tuple[float, float, float]
    ) -> List[ExtractedRegion]:
        """Extract brain region features."""
        regions = []
        feature_extractor = RegionFeatureExtractor(spacing)
        
        # Define eloquent regions
        ELOQUENT_REGIONS = {
            "motor_cortex", "sensory_cortex", "brocas_area", "wernickes_area",
            "primary_visual_cortex", "hippocampus", "thalamus", "brainstem"
        }
        
        if self.region_model and self.region_model._initialized:
            # Run segmentation
            segmentation = self.region_model.predict(image)
            
            # Extract features for each region
            for label_id in np.unique(segmentation):
                if label_id == 0:
                    continue
                
                mask = segmentation == label_id
                centroid = feature_extractor.extract_centroid(mask)
                volume = feature_extractor.measure_volume(mask)
                bbox = feature_extractor.extract_bounding_box(mask)
                hemisphere = feature_extractor.determine_hemisphere(centroid)
                
                regions.append(ExtractedRegion(
                    name=f"region_{label_id}",
                    centroid=centroid,
                    volume_mm3=volume,
                    bounding_box=bbox,
                    hemisphere=hemisphere,
                    confidence=0.8,
                ))
        else:
            # Fallback to standard regions
            for name, coords in MNI_LANDMARKS.items():
                is_eloquent = name in ELOQUENT_REGIONS
                regions.append(ExtractedRegion(
                    name=name,
                    centroid=coords,
                    volume_mm3=0.0,
                    bounding_box={},
                    is_eloquent=is_eloquent,
                    confidence=0.6,
                ))
        
        return regions
    
    def _fallback_extraction(
        self,
        nifti_path: str,
        modality: str
    ) -> ExtractionResult:
        """Return atlas-based fallback when extraction fails."""
        result = ExtractionResult(
            scan_id=Path(nifti_path).stem if nifti_path else "fallback",
            modality=modality,
            voxel_spacing=(1.0, 1.0, 1.0),
            dimensions=(256, 256, 256),
        )
        
        # Add atlas vessels
        for name, diameter in STANDARD_VESSEL_DIAMETERS.items():
            result.vessels.append(ExtractedVessel(
                name=name,
                centerline=np.array([]),
                diameter_mm=diameter,
                length_mm=0.0,
                volume_mm3=0.0,
                start_point=(0, 0, 0),
                end_point=(0, 0, 0),
                confidence=0.5,  # Low confidence for fallback
            ))
        
        # Add atlas landmarks as regions
        for name, coords in MNI_LANDMARKS.items():
            result.regions.append(ExtractedRegion(
                name=name,
                centroid=coords,
                volume_mm3=0.0,
                bounding_box={},
                confidence=0.5,
            ))
        
        result.landmarks = MNI_LANDMARKS.copy()
        result.metadata["fallback"] = True
        
        return result
    
    def result_to_entity_physics(
        self,
        result: ExtractionResult
    ) -> List[Dict[str, Any]]:
        """
        Convert extraction result to EntityPhysics-compatible dicts.
        
        Returns list of dicts ready for database insertion.
        """
        entities = []
        
        # Convert vessels
        for vessel in result.vessels:
            entities.append({
                "name": vessel.name,
                "canonical_name": vessel.name.upper().replace("_", " "),
                "mobility": "tethered_by_perforators" if "cerebral" in vessel.name else "fixed",
                "consistency": "vascular",
                "is_end_artery": "lenticulo" in vessel.name or "perforat" in vessel.name,
                "vessel_diameter_mm": vessel.diameter_mm,
                "spatial_context": {
                    "start_point": vessel.start_point,
                    "end_point": vessel.end_point,
                    "volume_mm3": vessel.volume_mm3,
                    "length_mm": vessel.length_mm,
                },
                "confidence": vessel.confidence,
                "extraction_method": "monai" if vessel.confidence > 0.6 else "atlas",
            })
        
        # Convert regions
        for region in result.regions:
            entities.append({
                "name": region.name,
                "canonical_name": region.name.upper().replace("_", " "),
                "mobility": "fixed",
                "consistency": "soft_brain",
                "eloquence_grade": "eloquent" if region.is_eloquent else "non_eloquent",
                "spatial_context": {
                    "centroid": region.centroid,
                    "hemisphere": region.hemisphere,
                    "depth_mm": region.depth_from_surface_mm,
                    "volume_mm3": region.volume_mm3,
                    "bounding_box": region.bounding_box,
                },
                "confidence": region.confidence,
                "extraction_method": "monai" if region.confidence > 0.6 else "atlas",
            })
        
        return entities


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """CLI for testing extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Anatomical extraction from NIfTI")
    parser.add_argument("input", help="Input NIfTI file")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--model", "-m", help="Model weights path")
    parser.add_argument("--modality", default="MRI", help="Image modality")
    
    args = parser.parse_args()
    
    extractor = AnatomicalExtractor(
        vessel_model_path=args.model,
        use_atlas_fallback=True
    )
    await extractor.initialize()
    
    result = await extractor.extract_from_nifti(
        args.input,
        modality=args.modality
    )
    
    # Convert to JSON-serializable format
    output = {
        "scan_id": result.scan_id,
        "modality": result.modality,
        "voxel_spacing": result.voxel_spacing,
        "dimensions": result.dimensions,
        "vessels": [
            {
                "name": v.name,
                "diameter_mm": v.diameter_mm,
                "length_mm": v.length_mm,
                "confidence": v.confidence
            }
            for v in result.vessels
        ],
        "regions": [
            {
                "name": r.name,
                "centroid": r.centroid,
                "is_eloquent": r.is_eloquent,
                "confidence": r.confidence
            }
            for r in result.regions
        ],
        "extraction_time_ms": result.extraction_time_ms,
    }
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
