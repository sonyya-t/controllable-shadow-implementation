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

from .rectified_flow import RectifiedFlowModel  
from .conditioning import LightParameterConditioning
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
                 use_vae_cache: bool = True):
        """
        Initialize shadow generator.
        
        Args:
            model_path: Path to pretrained model weights
            device: Device to run on ("auto", "cpu", "cuda")
            use_vae_cache: Whether to cache VAE embeddings
        """
        self.device = self._get_device(device)
        self.use_vae_cache = use_vae_cache
        self._vae_cache = {}
        
        # Initialize components
        self.image_processor = ImageProcessor()
        self.light_conditioning = LightParameterConditioning(256, 10000.0)
        self.diffusion_model = self._create_model()
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
            
        # Move to device
        self.to(self.device)
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _create_model(self) -> RectifiedFlowModel:
        """Create the diffusion model."""
        return RectifiedFlowModel(
            in_channels=9,  # 4 (noise) + 4  (object) + 1 (mask)
            out_channels=4,
            conditioning_dim=768,  # 256 * 3 parameters
            embed_dim=320,
            num_res_blocks=2,
            attention_resolutions=(4, 2, 1)
        )
    
    def to(self, device: torch.device):
        """Move model to device."""
        self.device = device
        self.light_conditioning.to(device)
        self.diffusion_model.to(device)
        return self
    
    def load_model(self, model_path: str):
        """Load pretrained model weights."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.diffusion_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
        
    def save_model(self, model_path: str):
        """Save model weights."""
        checkpoint = {
            'model_state_dict': self.diffusion_model.state_dict(),
            'light_conditioning_state_dict': self.light_conditioning.state_dict()
        }
        torch.save(checkpoint, model_path)
        print(f"Saved model to {model_path}")
        
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'ShadowGenerator':
        """Create ShadowGenerator from pretrained weights."""
        generator = cls(**kwargs)
        generator.load_model(model_path)
        return generator
    
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
                          theta: float = 30.0,
                          phi: float = 60.0, 
                          size: float = 4.0,
                          intensity: float = 1.0,
                          num_steps: int = 1) -> torch.Tensor:
        """
        Generate shadow map for given object image and light parameters.
        
        Args:
            object_image: Path to image or processed tensor
            theta: Polar angle in degrees (vertical direction)
            phi: Azimuthal angle in degrees (horizontal direction)
            size: Light size parameter (softness control)
            intensity: Shadow intensity multiplier
            num_steps: Number of sampling steps (typically 1 for rectified flow)
            
        Returns:
            Generated shadow map as (1, H, W) tensor
        """
        self.diffusion_model.eval()
        
        with torch.no_grad():
            # Preprocess input
            if isinstance(object_image, str):
                obj_tensor, metadata = self.preprocess_object_image(object_image)
            else:
                obj_tensor = object_image
                
            obj_tensor = obj_tensor.unsqueeze(0).to(self.device)  # Add batch dim
            
            # Extract object mask (simplified)
            mask = self.image_processor.create_mask(obj_tensor)
            mask = mask.to(self.device)
            
            # Prepare conditioning
            theta_tensor = torch.tensor([theta], device=self.device)
            phi_tensor = torch.tensor([phi], device=self.device)  
            size_tensor = torch.tensor([size], device=self.device)
            
            conditioning = self.light_conditioning(theta_tensor, phi_tensor, size_tensor)
            
            # Cache VAE embeddings if enabled
            if self.use_vae_cache:
                cache_key = f"{hash(obj_tensor.cpu().data.tobytes())}"
                if cache_key in self._vae_cache:
                    obj_embeddings = self._vae_cache[cache_key]
                else:
                    obj_embeddings = self._encode_to_vae_space(obj_tensor)
                    self._vae_cache[cache_key] = obj_embeddings
            else:
                obj_embeddings = self._encode_to_vae_space(obj_tensor)
                
            mask_embeddings = self._encode_mask_to_latent_space(mask)
            
            # Generate shadow map
            shadow_shape = obj_embeddings.shape[1:]
            shadow_map = self.diffusion_model.sample(
                shape=shadow_shape,
                conditioning=conditioning,
                object_image=obj_embeddings,
                object_mask=mask_embeddings,
                num_steps=num_steps
            )
            
            # Decode from VAE space to pixel space
            shadow_map = self._decode_from_vae_space(shadow_map)
            
            # Apply intensity
            shadow_map = shadow_map * intensity
            
            # Take first channel only (convert 3-channel to grayscale)
            shadow_map = shadow_map[0:1]  # Keep only first channel
            
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
    
    def _encode_to_vae_space(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Encode image to VAE latent space (simplified)."""
        # In practice, this would use a proper VAE encoder
        # For now, we'll use a simple downsampling
        return F.avg_pool2d(image_tensor, kernel_size=8, stride=8)
    
    def _decode_from_vae_space(self, latent_tensor: torch.Tensor) -> torch.Tensor:
        """Decode from VAE latent space to pixel space (simplified)."""
        # In practice, this would use a proper VAE decoder
        # For now, we'll use simple upsampling
        return F.interpolate(latent_tensor, size=(1024, 1024), mode='bilinear', align_corners=False)
    
    def _encode_mask_to_latent_space(self, mask: torch.Tensor) -> torch.Tensor:
        """Encode mask to latent space."""
        return F.max_pool2d(mask, kernel_size=8, stride=8)
