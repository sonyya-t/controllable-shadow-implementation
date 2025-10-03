"""
Model implementations for controllable shadow generation.

Main exports:
- ShadowDiffusionModel: Complete model for training/inference (NEW - recommended)
- ShadowGenerator: Legacy high-level API
- ConditionedSDXLUNet: UNet with light conditioning
- VAEPipeline: VAE encoding/decoding pipeline
"""

# New SDXL-based implementation
from .shadow_diffusion_model import ShadowDiffusionModel, create_shadow_model
from .conditioned_unet import ConditionedSDXLUNet
from .vae_pipeline import VAEPipeline, ShadowMapConverter, MaskProcessor
from .checkpoint_manager import CheckpointManager
from .sdxl_unet import SDXLUNetForShadows, VAEWrapper

# Conditioning
from .conditioning import (
    LightParameterConditioning,
    TimestepLightEmbedding,
    AdditiveTimestepLightEmbedding,
    create_light_blob,
)

# Legacy/existing implementations
from .shadow_generator import ShadowGenerator
from .rectified_flow import RectifiedFlowModel

__all__ = [
    # NEW: Main model (recommended)
    "ShadowDiffusionModel",
    "create_shadow_model",

    # Core components
    "ConditionedSDXLUNet",
    "SDXLUNetForShadows",
    "VAEWrapper",
    "VAEPipeline",

    # Utilities
    "ShadowMapConverter",
    "MaskProcessor",
    "CheckpointManager",

    # Conditioning
    "LightParameterConditioning",
    "TimestepLightEmbedding",
    "AdditiveTimestepLightEmbedding",
    "create_light_blob",

    # Legacy
    "ShadowGenerator",
    "RectifiedFlowModel",
]
