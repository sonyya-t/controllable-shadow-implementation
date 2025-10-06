"""
Controllable Shadow Generation with Single-Step Diffusion Models

This package implements controllable shadow generation using diffusion models
trained on synthetic data from 3D rendering engines.
"""

__version__ = "0.1.0"
__author__ = "Implementation Team"
__email__ = "ai@example.com"

# Main imports for training
from .models import (
    ShadowDiffusionModel,
    create_shadow_model,
    CheckpointManager,
)

# Legacy/utils imports
from .models import ShadowGenerator
from .utils import ShadowMetrics, ImageProcessor

__all__ = [
    # NEW: Main API (use this for training)
    "ShadowDiffusionModel",
    "create_shadow_model",
    "CheckpointManager",

    # Legacy
    "ShadowGenerator",
    "ShadowMetrics",
    "ImageProcessor",
]
