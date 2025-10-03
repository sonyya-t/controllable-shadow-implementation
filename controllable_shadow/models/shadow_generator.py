"""
Main ShadowGenerator class for controllable shadow generation.

Implements the complete pipeline from object images to realistic shadows
with controllable direction, softness, and intensity.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Union, List
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

from .shadow_diffusion_model import ShadowDiffusionModel, create_shadow_model
from ..utils.image_processor import ImageProcessor


class ShadowGenerator:
    """
    Main class for controllable shadow generation.

    This class integrates the diffusion model, conditioning, and blending
    pipeline to generate realistic shadows for object images.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "auto",
                 pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 conditioning_strategy: str = "additive",
                 image_size: int = 1024):
        """
        Initialize shadow generator.

        Args:
            model_path: Path to pretrained model weights
            device: Device to run on ("auto", "cpu", "cuda")
            pretrained_model_name: HuggingFace SDXL model ID
            conditioning_strategy: "additive" or "concat"
            image_size: Image size (default: 1024)
        """
        self.device = self._get_device(device)
        self.image_size = image_size
        self.latent_size = (image_size // 8, image_size // 8)

        # Initialize components
        self.image_processor = ImageProcessor()

        # Create the SDXL-based diffusion model
        if model_path:
            self.model = create_shadow_model(
                pretrained_path=model_path,
                device=str(self.device),
                pretrained_model_name=pretrained_model_name,
                conditioning_strategy=conditioning_strategy,
                image_size=(image_size, image_size),
                latent_size=self.latent_size,
            )
        else:
            self.model = ShadowDiffusionModel(
                pretrained_model_name=pretrained_model_name,
                conditioning_strategy=conditioning_strategy,
                image_size=(image_size, image_size),
                latent_size=self.latent_size,
            )
            self.model = self.model.to(self.device)
            self.model.freeze_vae()
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return self

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'ShadowGenerator':
        """Create ShadowGenerator from pretrained weights."""
        return cls(model_path=model_path, **kwargs)
    
    def preprocess_object_image(self, image_path: str) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess object image and extract background-free version.
        
        Args:
            image_path: Path to object image
            
        Returns:
            Tuple of (processed_image, metadata)
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Remove background (simplified - in practice would use more sophisticated method)
        object_image, mask = self.image_processor.remove_background(image)
        
        # Convert to tensor and normalize
        transform = T.Compose([
            T.Resize((1024, 1024)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        object_tensor = transform(object_image)
        
        metadata = {
            'original_size': image.size,
            'object_image': object_image,
            'mask': mask
        }
        
        return object_tensor, metadata
    
    def generate_shadow_map(self,
                          object_image: Union[str, torch.Tensor],
                          mask: Optional[torch.Tensor] = None,
                          theta: float = 30.0,
                          phi: float = 60.0,
                          size: float = 4.0,
                          intensity: float = 1.0,
                          num_steps: int = 1) -> torch.Tensor:
        """
        Generate shadow map for given object image and light parameters.

        Args:
            object_image: Path to image or processed tensor (B, 3, H, W) in [-1, 1]
            mask: Optional binary mask (B, 1, H, W). If None, auto-generated.
            theta: Polar angle in degrees (vertical direction)
            phi: Azimuthal angle in degrees (horizontal direction)
            size: Light size parameter (softness control)
            intensity: Shadow intensity multiplier
            num_steps: Number of sampling steps (typically 1 for rectified flow)

        Returns:
            Generated shadow map as (B, 1, H, W) tensor in [0, 1]
        """
        self.model.eval()

        with torch.no_grad():
            # Preprocess input
            if isinstance(object_image, str):
                obj_tensor, metadata = self.preprocess_object_image(object_image)
                obj_tensor = obj_tensor.unsqueeze(0)  # Add batch dim
                mask = metadata.get('mask')
                if mask is not None:
                    mask = T.ToTensor()(mask).unsqueeze(0)
            else:
                obj_tensor = object_image

            obj_tensor = obj_tensor.to(self.device)

            # Generate mask if not provided
            if mask is None:
                mask = self.image_processor.create_mask(obj_tensor)
            mask = mask.to(self.device)

            # Ensure correct shapes
            batch_size = obj_tensor.shape[0]
            assert obj_tensor.shape[1] == 3, f"Expected 3 channels, got {obj_tensor.shape[1]}"
            assert mask.shape[1] == 1, f"Expected 1 channel mask, got {mask.shape[1]}"

            # Prepare conditioning tensors
            theta_tensor = torch.tensor([theta] * batch_size, device=self.device, dtype=torch.float32)
            phi_tensor = torch.tensor([phi] * batch_size, device=self.device, dtype=torch.float32)
            size_tensor = torch.tensor([size] * batch_size, device=self.device, dtype=torch.float32)

            # Generate shadow map using the SDXL-based model
            shadow_map = self.model.sample(
                object_image=obj_tensor,
                mask=mask,
                theta=theta_tensor,
                phi=phi_tensor,
                size=size_tensor,
                num_steps=num_steps,
                return_latent=False,
            )

            # Apply intensity scaling
            if intensity != 1.0:
                shadow_map = shadow_map * intensity
                shadow_map = torch.clamp(shadow_map, 0.0, 1.0)

        return shadow_map
    
    def blend_with_background(self,
                            object_image_path: str,
                            shadow_map: torch.Tensor,
                            background_path: str,
                            opacity: float = 0.8) -> Image.Image:
        """
        Blend object with shadow onto background image.
        
        Args:
            object_image_path: Path to original object image
            shadow_map: Generated shadow map tensor
            background_path: Path to background image
            opacity: Shadow opacity
            
        Returns:
            Final composited image
        """
        # Load images
        object_img = Image.open(object_image_path).convert('RGB')
        background_img = Image.open(background_path).convert('RGB')
        
        # Resize to match shadow map
        shadow_map = F.interpolate(shadow_map.unsqueeze(0), 
                                 size=background_img.size[::-1], 
                                 mode='bilinear', align_corners=False).squeeze(0)
        
        # Convert shadow to PIL image
        shadow_np = shadow_map[0].cpu().numpy()
        shadow_np = (shadow_np * 255).astype(np.uint8)
        shadow_img = Image.fromarray(shadow_np, mode='L')
        
        # Convert images to numpy arrays
        obj_array = np.array(object_img)
        bg_array = np.array(background_img)
        
        # Resize object to match background
        obj_resized = cv2.resize(obj_array, bg_array.shape[:2][::-1])
        
        # Create shadow layer
        shadow_array = np.array(shadow_img)
        shadow_array = np.stack([shadow_array, shadow_array, shadow_array], axis=-1)
        
        # Blend shadows onto background (darken bg pixels)
        shadow_layer = bg_array.astype(np.float32)
        shadow_mask = (shadow_array.astype(np.float32) / 255.0)
        shadow_layer = shadow_layer * (1 - shadow_mask * opacity * 0.5)
        
        # Blend object (with transparency around edges)
        object_mask = self.image_processor.create_object_mask(obj_resized)
        object_mask = object_mask[..., np.newaxis]
        
        # Final composition
        result = shadow_layer.copy()
        result = result * (1 - object_mask) + obj_resized.astype(np.float32) * object_mask
        
        # Clip and convert back
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def generate_full_pipeline(self,
                              object_image_path: str,
                              background_path: str,
                              theta: float = 30.0,
                              phi: float = 60.0,
                              size: float = 4.0,
                              intensity: float = 1.0) -> Image.Image:
        """
        Run complete shadow generation pipeline.
        
        Args:
            object_image_path: Path to object image
            background_path: Path to background image  
            theta: Polar angle in degrees
            phi: Azimuthal angle in degrees
            size: Light size parameter
            intensity: Shadow intensity
            
        Returns:
            Final composited image with shadows
        """
        # Generate shadow map
        shadow_map = self.generate_shadow_map(
            object_image_path, theta, phi, size, intensity
        )
        
        # Blend with background
        result = self.blend_with_background(
            object_image_path, shadow_map, background_path
        )
        
        return result
    
    def batch_generate(self,
                      object_paths: List[str],
                      theta_values: List[float],
                      phi_values: List[float], 
                      size_values: List[float],
                      return_maps: bool = False) -> Union[List[Image.Image], Tuple[List[Image.Image], List[torch.Tensor]]]:
        """
        Generate shadows for multiple objects with varying parameters.
        
        Args:
            object_paths: List of object image paths
            theta_values: List of theta values for each object
            phi_values: List of phi values for each object
            size_values: List of size values for each object
            return_maps: Whether to return shadow maps as well
            
        Returns:
            List of composited images (and optionally shadow maps)
        """
        shadow_maps = []
        results = []
        
        for i, obj_path in enumerate(object_paths):
            shadow_map = self.generate_shadow_map(
                obj_path,
                theta=theta_values[i],
                phi=phi_values[i], 
                size=size_values[i]
            )
            
            if return_maps:
                shadow_maps.append(shadow_map)
                
            # For batch generation, we'll just return the shadow maps
            # Real backgrounds would be provided separately
            
        if return_maps:
            return results, shadow_maps
        return results
    
