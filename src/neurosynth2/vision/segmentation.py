"""
NeuroSynth 2.0 - Segmentation Pipeline
========================================

Medical image segmentation for brain structures and vessels.

This module provides:
- Preprocessing transforms for MRI/CTA normalization
- Multi-class brain segmentation
- Vessel enhancement and segmentation
- Post-processing for clean masks

Model Zoo:
- BrainSegNet: 96-class brain parcellation (based on FreeSurfer labels)
- VesselNet: Arterial/venous segmentation with COW detection
- TumorNet: Glioma segmentation (enhancing, non-enhancing, edema, necrosis)

Dependencies:
    - torch >= 2.0
    - monai >= 1.3
    - numpy
    - scipy (for morphological operations)
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Conditional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

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
    from monai.transforms import (
        Compose,
        LoadImaged,
        EnsureChannelFirstd,
        Orientationd,
        Spacingd,
        ScaleIntensityRanged,
        ScaleIntensityd,
        NormalizeIntensityd,
        CropForegroundd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandRotate90d,
        RandShiftIntensityd,
        RandGaussianNoised,
        EnsureTyped,
        Invertd,
        AsDiscreted,
        KeepLargestConnectedComponentd,
        FillHolesd,
    )
    from monai.data import (
        CacheDataset,
        DataLoader,
        decollate_batch,
    )
    from monai.networks.nets import (
        UNet,
        SwinUNETR,
        UNETR,
        SegResNet,
        DynUNet,
    )
    from monai.losses import (
        DiceLoss,
        DiceCELoss,
        FocalLoss,
        TverskyLoss,
    )
    from monai.metrics import DiceMetric
    from monai.inferers import sliding_window_inference
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# =============================================================================
# ENUMS & CONFIGURATION
# =============================================================================

class SegmentationTask(Enum):
    """Supported segmentation tasks."""
    BRAIN_PARCELLATION = "brain_parcellation"
    VESSEL_SEGMENTATION = "vessel_segmentation"
    TUMOR_SEGMENTATION = "tumor_segmentation"
    LESION_DETECTION = "lesion_detection"


class Modality(Enum):
    """Image modality types."""
    T1 = "T1"
    T1CE = "T1CE"  # T1 contrast-enhanced
    T2 = "T2"
    FLAIR = "FLAIR"
    DWI = "DWI"
    ADC = "ADC"
    SWI = "SWI"
    TOF_MRA = "TOF_MRA"
    CTA = "CTA"
    DSA = "DSA"


@dataclass
class SegmentationConfig:
    """Configuration for segmentation models."""
    task: SegmentationTask
    architecture: str = "swinunetr"
    input_channels: int = 1
    output_channels: int = 2
    spatial_size: Tuple[int, int, int] = (96, 96, 96)
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity_range: Tuple[float, float] = (-1000, 3000)  # HU for CT
    normalize_intensity: bool = True
    use_gpu: bool = True
    batch_size: int = 1
    sw_overlap: float = 0.5
    
    # Post-processing
    apply_crf: bool = False
    keep_largest_component: bool = True
    fill_holes: bool = True
    min_volume_mm3: float = 10.0


# Brain parcellation labels (FreeSurfer-style)
BRAIN_LABELS = {
    0: "background",
    1: "cerebral_white_matter_left",
    2: "cerebral_cortex_left",
    3: "lateral_ventricle_left",
    4: "inferior_lateral_ventricle_left",
    5: "cerebellum_white_matter_left",
    6: "cerebellum_cortex_left",
    7: "thalamus_left",
    8: "caudate_left",
    9: "putamen_left",
    10: "pallidum_left",
    11: "hippocampus_left",
    12: "amygdala_left",
    13: "accumbens_left",
    14: "ventral_dc_left",
    15: "brainstem",
    16: "third_ventricle",
    17: "fourth_ventricle",
    # ... (continue for right hemisphere and more structures)
    41: "cerebral_white_matter_right",
    42: "cerebral_cortex_right",
    # Add more as needed
}

# Vessel segmentation labels
VESSEL_LABELS = {
    0: "background",
    1: "artery",
    2: "vein",
    3: "sinus",
}

# Tumor segmentation labels (BraTS-style)
TUMOR_LABELS = {
    0: "background",
    1: "necrotic_core",
    2: "peritumoral_edema",
    3: "enhancing_tumor",
    4: "non_enhancing_tumor",
}


# =============================================================================
# PREPROCESSING PIPELINES
# =============================================================================

def get_preprocessing_transforms(
    config: SegmentationConfig,
    keys: List[str] = ["image"],
    is_training: bool = False,
) -> "Compose":
    """
    Build preprocessing transform pipeline.
    
    Args:
        config: Segmentation configuration
        keys: Data dictionary keys to transform
        is_training: Whether to include augmentation
    
    Returns:
        MONAI Compose transform
    """
    if not MONAI_AVAILABLE:
        raise RuntimeError("MONAI not available")
    
    transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=config.voxel_spacing,
            mode="bilinear",
        ),
    ]
    
    # Intensity normalization based on modality
    if config.task == SegmentationTask.VESSEL_SEGMENTATION:
        # CTA: Window/level normalization
        transforms.append(
            ScaleIntensityRanged(
                keys=keys,
                a_min=config.intensity_range[0],
                a_max=config.intensity_range[1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        )
    else:
        # MRI: Z-score normalization
        transforms.append(
            NormalizeIntensityd(keys=keys, nonzero=True, channel_wise=True)
        )
    
    # Crop to foreground
    transforms.append(
        CropForegroundd(keys=keys, source_key=keys[0], margin=10)
    )
    
    # Training augmentations
    if is_training:
        transforms.extend([
            RandCropByPosNegLabeld(
                keys=keys + ["label"],
                label_key="label",
                spatial_size=config.spatial_size,
                pos=1,
                neg=1,
                num_samples=4,
            ),
            RandFlipd(keys=keys + ["label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys + ["label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys + ["label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=keys + ["label"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys=keys, offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=keys, prob=0.1, mean=0.0, std=0.1),
        ])
    
    transforms.append(EnsureTyped(keys=keys))
    
    return Compose(transforms)


def get_postprocessing_transforms(
    config: SegmentationConfig,
    original_keys: List[str] = ["image"],
) -> "Compose":
    """Build post-processing transform pipeline."""
    if not MONAI_AVAILABLE:
        raise RuntimeError("MONAI not available")
    
    transforms = [
        AsDiscreted(keys=["pred"], argmax=True),
    ]
    
    if config.keep_largest_component:
        transforms.append(
            KeepLargestConnectedComponentd(keys=["pred"], applied_labels=list(range(1, config.output_channels)))
        )
    
    if config.fill_holes:
        transforms.append(
            FillHolesd(keys=["pred"], applied_labels=list(range(1, config.output_channels)))
        )
    
    return Compose(transforms)


# =============================================================================
# VESSEL-SPECIFIC PREPROCESSING
# =============================================================================

class VesselEnhancementFilter:
    """
    Vesselness enhancement using Frangi/Sato filtering.
    
    Enhances tubular structures in the image to improve vessel detection.
    """
    
    def __init__(
        self,
        sigmas: List[float] = [0.5, 1.0, 2.0, 4.0],
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 15.0,
        black_ridges: bool = False,
    ):
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.black_ridges = black_ridges
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply vesselness filtering."""
        if not SCIPY_AVAILABLE:
            return image
        
        try:
            from skimage.filters import frangi
            
            # Multi-scale Frangi filtering
            vesselness = frangi(
                image,
                sigmas=self.sigmas,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                black_ridges=self.black_ridges,
            )
            
            return vesselness.astype(np.float32)
            
        except ImportError:
            logger.warning("skimage not available, using raw image")
            return image
    
    def enhance_mra(self, image: np.ndarray) -> np.ndarray:
        """Specific enhancement for MRA (bright vessels)."""
        return self(image)
    
    def enhance_cta(self, image: np.ndarray, window_center: float = 300, window_width: float = 700) -> np.ndarray:
        """Specific enhancement for CTA with bone removal."""
        # Apply window/level
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        windowed = np.clip(image, lower, upper)
        windowed = (windowed - lower) / (upper - lower)
        
        return self(windowed)


class SkullStripper:
    """
    Brain extraction / skull stripping.
    
    Uses intensity-based thresholding and morphological operations.
    For production, consider using HD-BET or SynthStrip.
    """
    
    def __init__(
        self,
        threshold_percentile: float = 10.0,
        erosion_iterations: int = 2,
        dilation_iterations: int = 3,
    ):
        self.threshold_percentile = threshold_percentile
        self.erosion_iterations = erosion_iterations
        self.dilation_iterations = dilation_iterations
    
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract brain from image.
        
        Returns:
            Tuple of (brain_extracted_image, brain_mask)
        """
        if not SCIPY_AVAILABLE:
            return image, np.ones_like(image, dtype=bool)
        
        # Initial threshold
        threshold = np.percentile(image[image > 0], self.threshold_percentile)
        mask = image > threshold
        
        # Fill holes
        mask = binary_fill_holes(mask)
        
        # Morphological cleanup
        struct = ndimage.generate_binary_structure(3, 1)
        mask = binary_opening(mask, struct, iterations=self.erosion_iterations)
        mask = binary_closing(mask, struct, iterations=self.dilation_iterations)
        
        # Keep largest component (the brain)
        labeled, num_features = ndimage.label(mask)
        if num_features > 1:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            largest_label = np.argmax(sizes) + 1
            mask = labeled == largest_label
        
        # Apply mask
        brain = image * mask
        
        return brain, mask


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

def build_segmentation_model(
    config: SegmentationConfig,
) -> "nn.Module":
    """
    Build segmentation model based on configuration.
    
    Args:
        config: Model configuration
    
    Returns:
        PyTorch model
    """
    if not MONAI_AVAILABLE or not TORCH_AVAILABLE:
        raise RuntimeError("MONAI/PyTorch not available")
    
    arch = config.architecture.lower()
    
    if arch == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
            dropout=0.1,
        )
    
    elif arch == "swinunetr":
        return SwinUNETR(
            img_size=config.spatial_size,
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
        )
    
    elif arch == "unetr":
        return UNETR(
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            img_size=config.spatial_size,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="conv",
            norm_name="instance",
            res_block=True,
        )
    
    elif arch == "segresnet":
        return SegResNet(
            spatial_dims=3,
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            init_filters=32,
            dropout_prob=0.2,
        )
    
    elif arch == "dynunet":
        # DynUNet for nnU-Net style architecture
        kernels = [[3, 3, 3]] * 5
        strides = [[1, 1, 1]] + [[2, 2, 2]] * 4
        
        return DynUNet(
            spatial_dims=3,
            in_channels=config.input_channels,
            out_channels=config.output_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name="instance",
            deep_supervision=True,
            deep_supr_num=2,
        )
    
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# =============================================================================
# INFERENCE ENGINE
# =============================================================================

class SegmentationInference:
    """
    High-level inference engine for segmentation.
    
    Handles loading, preprocessing, inference, and post-processing.
    """
    
    def __init__(
        self,
        config: SegmentationConfig,
        model_path: Optional[str] = None,
    ):
        self.config = config
        self.model_path = model_path
        
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )
        
        self.model = None
        self.preprocess = None
        self.postprocess = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize model and transforms."""
        if not MONAI_AVAILABLE or not TORCH_AVAILABLE:
            logger.error("MONAI/PyTorch not available")
            return False
        
        try:
            # Build model
            self.model = build_segmentation_model(self.config)
            
            # Load weights if provided
            if self.model_path and Path(self.model_path).exists():
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded weights from {self.model_path}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Build transforms
            self.preprocess = get_preprocessing_transforms(self.config)
            self.postprocess = get_postprocessing_transforms(self.config)
            
            self._initialized = True
            logger.info(f"SegmentationInference initialized on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, np.ndarray, Dict],
        return_probabilities: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Run segmentation inference.
        
        Args:
            image: Input image (path, array, or data dict)
            return_probabilities: Whether to return class probabilities
        
        Returns:
            Dict with 'segmentation' and optionally 'probabilities'
        """
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        
        # Prepare input
        if isinstance(image, str):
            data = {"image": image}
        elif isinstance(image, np.ndarray):
            data = {"image": image}
        else:
            data = image
        
        # Preprocess
        processed = self.preprocess(data)
        
        # Get tensor
        if isinstance(processed, dict):
            tensor = processed["image"]
        else:
            tensor = processed
        
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        
        tensor = tensor.unsqueeze(0).to(self.device)  # Add batch dim
        
        # Inference with sliding window
        output = sliding_window_inference(
            tensor,
            roi_size=self.config.spatial_size,
            sw_batch_size=self.config.batch_size,
            predictor=self.model,
            overlap=self.config.sw_overlap,
            mode="gaussian",
        )
        
        # Get predictions
        result = {}
        
        if return_probabilities:
            probs = F.softmax(output, dim=1).squeeze(0).cpu().numpy()
            result["probabilities"] = probs
        
        # Argmax for segmentation
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        result["segmentation"] = pred
        
        # Post-process
        post_data = {"pred": pred}
        post_result = self.postprocess(post_data)
        result["segmentation"] = post_result["pred"]
        
        return result
    
    def predict_batch(
        self,
        images: List[Union[str, np.ndarray]],
        num_workers: int = 4,
    ) -> List[Dict[str, np.ndarray]]:
        """Run batch inference."""
        results = []
        for image in images:
            results.append(self.predict(image))
        return results


# =============================================================================
# SPECIALIZED PIPELINES
# =============================================================================

class BrainSegmentationPipeline:
    """Complete pipeline for brain parcellation."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.config = SegmentationConfig(
            task=SegmentationTask.BRAIN_PARCELLATION,
            architecture="swinunetr",
            output_channels=len(BRAIN_LABELS),
            spatial_size=(96, 96, 96),
        )
        self.inference = SegmentationInference(self.config, model_path)
        self.skull_stripper = SkullStripper()
    
    def initialize(self) -> bool:
        return self.inference.initialize()
    
    def segment(self, image: Union[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Segment brain into parcellation regions.
        
        Returns dict with:
        - segmentation: Label map
        - brain_mask: Binary brain mask
        - label_volumes: Dict of label -> volume in mm³
        """
        # Skull strip first
        if isinstance(image, np.ndarray):
            brain, mask = self.skull_stripper(image)
            result = self.inference.predict(brain)
            result["brain_mask"] = mask
        else:
            result = self.inference.predict(image)
        
        # Calculate volumes
        voxel_volume = np.prod(self.config.voxel_spacing)
        label_volumes = {}
        for label_id, label_name in BRAIN_LABELS.items():
            count = np.sum(result["segmentation"] == label_id)
            label_volumes[label_name] = count * voxel_volume
        
        result["label_volumes"] = label_volumes
        return result


class VesselSegmentationPipeline:
    """Complete pipeline for vessel segmentation."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.config = SegmentationConfig(
            task=SegmentationTask.VESSEL_SEGMENTATION,
            architecture="unet",
            output_channels=len(VESSEL_LABELS),
            spatial_size=(128, 128, 128),
            intensity_range=(0, 500),  # CTA HU range for vessels
        )
        self.inference = SegmentationInference(self.config, model_path)
        self.enhancement = VesselEnhancementFilter()
    
    def initialize(self) -> bool:
        return self.inference.initialize()
    
    def segment(
        self,
        image: Union[str, np.ndarray],
        modality: Modality = Modality.CTA,
        enhance: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Segment vessels from angiography.
        
        Returns dict with:
        - segmentation: Label map (artery, vein, sinus)
        - vessel_mask: Binary vessel mask
        - enhanced: Vesselness-enhanced image
        """
        # Load if path
        if isinstance(image, str):
            import nibabel as nib
            nii = nib.load(image)
            image = nii.get_fdata()
        
        result = {}
        
        # Enhance vessels
        if enhance:
            if modality == Modality.CTA:
                enhanced = self.enhancement.enhance_cta(image)
            else:
                enhanced = self.enhancement.enhance_mra(image)
            result["enhanced"] = enhanced
            inference_input = enhanced
        else:
            inference_input = image
        
        # Segment
        seg_result = self.inference.predict(inference_input)
        result["segmentation"] = seg_result["segmentation"]
        result["vessel_mask"] = seg_result["segmentation"] > 0
        
        return result


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "SegmentationTask",
    "Modality",
    
    # Config
    "SegmentationConfig",
    
    # Labels
    "BRAIN_LABELS",
    "VESSEL_LABELS",
    "TUMOR_LABELS",
    
    # Transforms
    "get_preprocessing_transforms",
    "get_postprocessing_transforms",
    
    # Filters
    "VesselEnhancementFilter",
    "SkullStripper",
    
    # Models
    "build_segmentation_model",
    
    # Inference
    "SegmentationInference",
    
    # Pipelines
    "BrainSegmentationPipeline",
    "VesselSegmentationPipeline",
]
