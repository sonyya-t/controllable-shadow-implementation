"""
Image processing utilities for shadow generation.

Handles background removal, mask creation, and image preprocessing
for the shadow generation pipeline.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance
from typing import Tuple, Optional
import torchvision.transforms as T


class ImageProcessor:
    """
    Image processing utilities for shadow generation pipeline.
    
    Handles background removal, mask extraction, and image transformations
    needed for training and inference.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (1024, 1024)):
        self.target_size = target_size
        
    def remove_background(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Remove background from object image and return mask.
        
        This is a simplified implementation. In practice, more sophisticated
        techniques like U²-Net, remove.bg API, or learned matting would be used.
        
        Args:
            image: Input RGB image
            
        Returns:
            Tuple of (background_free_image, binary_mask)
        """
        # Get bounding box of object (simplified approach)
        bbox = self._estimate_object_bbox(image)
        
        # Create rough mask
        mask = self._create_object_mask(image, bbox)
        
        # Apply mask to remove background
        image_array = np.array(image)
        mask_array = np.array(mask)
        
        # Set background pixels to transparent/white
        image_array[mask_array == 0] = [255, 255, 255]  # White background
        
        bg_free_image = Image.fromarray(image_array)
        
        return bg_free_image, mask
    
    def _estimate_object_bbox(self, image: Image.Image) -> Tuple[int, int, int, int]:
        """
        Estimate bounding box of main object in image.
        
        Uses edge detection and contour analysis.
        """
        # Convert to grayscale
        gray = np.array(image.convert('L'))
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Fallback: assume object is in center region
            h, w = gray.shape
            return w//4, h//4, 3*w//4, 3*h//4
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = max(w, h) // 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.width - x, w + 2*padding)
        h = min(image.height - y, h + 2*padding)
        
        return x, y, x + w, y + h
    
    def _create_object_mask(self, image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Create binary mask for object based on bounding box and analysis.
        """
        # Get image array
        img_array = np.array(image)
        
        # Create initial mask with object region
        mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
        
        # Fill bounding box region
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
        
        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv21.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Use GrabCut algorithm for refined segmentation
        try:
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Mask initialization
            init_mask = np.zeros(img_array.shape[:2], np.uint8)
            init_mask[y1:y2, x1:x2] = cv2.GC_PR_FGD  # Probably foreground
            init_mask[0:20, :] = cv2.GC_BGD  # Probably background
            init_mask[-20:, :] = cv2.GC_BGD  # Probably background
            init_mask[:, 0:20] = cv2.GC_BGD  # Probably background
            init_mask[:, -20:] = cv2.GC_BGD  # Probably background
            
            # Run GrabCut
            cv2.grabCut(img_array, init_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
            
            # Extract final mask
            final_mask = np.where((init_mask == cv2.GC_FGD) | (init_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            
            return Image.fromarray(final_mask)
            
        except:
            # Fallback to simple rectangular mask
            return Image.fromarray(mask)
    
    def create_mask(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Create binary mask from image tensor.
        
        Args:
            image_tensor: Image as tensor (C, H, W)
            
        Returns:
            Binary mask as tensor (1, H, W)
        """
        # Convert to PIL for processing
        transform_to_pil = T.ToPILImage()
        
        # Normalize if needed
        if image_tensor.min() < 0:
            image_tensor = (image_tensor + 1) / 2
            
        pil_image = transform_to_pil(image_tensor)
        
        # Create mask
        _, mask_pil = self.remove_background(pil_image)
        
        # Convert back to tensor
        transform_to_tensor = T.ToTensor()
        mask_tensor = transform_to_tensor(mask_pil.convert('L'))
        
        return mask_tensor.unsqueeze(0)  # Add channel dimension
    
    def create_object_mask(self, image_array: np.ndarray) -> np.ndarray:
        """
        Create object mask from image array (for blending operations).
        
        Args:
            image_array: Image as numpy array (H, W, C)
            
        Returns:
            Binary mask as numpy array (H, W)
        """
        # Convert to HSV for better separation
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Create mask for non-white pixels
        lower_bound = np.array([0, 30, 30])
        upper_bound = np.array([180, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphopathologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate slightly for better blending
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask.astype(np.float32) / 255.0
    
    def resize_with_pad(self, image: Image.Image) -> Image.Image:
        """
        Resize image with padding to maintain aspect ratio.
        
        Args:
            image: Input image
            
        Returns:
            Resized image with padding
        """
        # Calculate scaling to fit target size
        current_size = image.size
        scale = min(self.target_size[0] / current_size[0],
                   self.target_size[1] / current_size[1])
        
        # Calculate new size
        new_size = (int(current_size[0] * scale), int(current_size[1] * scale))
        
        # Resize image
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create padded image
        padded = Image.new('RGB', self.target_size, (255, 255, 255))
        
        # Paste resized image in center
        paste_x = (self.target_size[0] - new_size[0]) // 2
        paste_y = (self.target_size[1] - new_size[1]) // 2
        padded.paste(resized, (paste_x, paste_y))
        
        return padded
    
    def normalize_image(self, image_tensor: torch.Tensor,
                       mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                       std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> torch.Tensor:
        """
        Normalize image tensor with ImageNet statistics.
        
        Args:
            image_tensor: Input image tensor (C, H, W)
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Normalized image tensor
        """
        mean_tensor = torch.tensor(mean).view(-1, 1, 1)
        std_tensor = torch.tensor(std).view(-1, 1, 1)
        
        return (image_tensor - mean_tensor) / std_tensor
    
    def denormalize_image(self, image_tensor: torch.Tensor,
                          mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                          std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> torch.Tensor:
        """
        Denormalize image tensor.
        
        Args:
            image_tensor: Normalized image tensor (C, H, W)
            mean: Normalization mean
            std: Normalization std
            
        Returns:
            Denormalized image tensor
        """
        mean_tensor = torch.tensor(mean).view(-1, 1, 1)
        std_tensor = torch.tensor(std).view(-1, 1, 1)
        
        return image_tensor * std_tensor + mean_tensor
    
    def create_shadow_blending_mask(self, shadow_map: torch.Tensor,
                                   object_mask: torch.Tensor,
                                   softness: float = 1.0) -> torch.Tensor:
        """
        Create blending mask for shadow application.
        
        Args:
            shadow_map: Generated shadow map (1, H, W)
            object_mask: Object mask (1, H, W)
            softness: Shadow softness parameter
            
        Returns:
            Blending mask (1, H, W)
        """
        # Invert shadow map (darker areas should have stronger shadow effect)
        shadow_blending = 1.0 - shadow_map
        
        # Remove shadows under object (simplified approach)
        object_mask_expanded = F.max_pool2d(object_mask, 
                                          kernel_size=int(softness * 10) + 1,
                                          stride=1, 
                                          padding=int(softness * 5))
        
        shadow_blending = shadow_spending * (1.0 - object_mask_expanded)
        
        return shadow_blending
    
    def apply_shadow_to_image(self, image: torch.Tensor,
                            shadow_map: torch.Tensor,
                            object_mask: torch.Tensor,
                            intensity: float = 0.8) -> torch.Tensor:
        """
        Apply shadow map to image.
        
        Args:
            image: Input image (3, H, W)
            shadow_map: Shadow map (1, H, W)
            object_mask: Object mask (1, H, W)
            intensity: Shadow intensity
            
        Returns:
            Image with applied shadows (3, H, W)
        """
        # Expand shadow map to RGB
        shadow_rgb = shadow_map.repeat(3, 1, 1)
        
        # Apply shadow effect (darken pixels)
        shadowed_image = image * (1 - shadow_rgb * intensity * 0.7)
        
        # Protect object pixels from being darkened
        object_mask_rgb = object_mask.repeat(3, 1, 1)
        result = shadowed_image * (1 - object_mask_rgb) + image * object_mask_rgb
        
        return result.clamp(0, 1)
