"""
Controllable Shadow Generation with Single-Step Diffusion Models

This package implements controllable shadow generation using diffusion models
trained on synthetic data from 3D rendering engines.
"""

__version__, __author__, __email__ = "0.1.0", "AI Assistant", "ai@example.com"

from .models import ShadowGenerator, RectifiedFlowModel
from .utils import ShadowMetrics, ImageProcessor

__all__ = [
    "ShadowGenerator",
    "RectifiedFlowModel", 
    "ShadowMetrics",
    "ImageProcessor"
]
