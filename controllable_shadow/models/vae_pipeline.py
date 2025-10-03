"""
VAE Pipeline for shadow generation.

Handles:
1. Grayscale ↔ RGB shadow map conversion
2. VAE encoding (objects and shadows to latent space)
3. Mask processing and resizing to latent space
4. Latent concatenation [noise + object + mask]
5. VAE decoding and post-processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from PIL import Image
import torchvision.transforms as T

from .sdxl_unet import VAEWrapper


class ShadowMapConverter:
    """
    Converts between grayscale and RGB shadow maps.

    The VAE expects RGB images, so grayscale shadow maps
    need to be replicated across 3 channels.
    """

    @staticmethod
    def grayscale_to_rgb(shadow_map: torch.Tensor) -> torch.Tensor:
        """
        Convert grayscale shadow map to RGB by replicating channels.

        Args:
            shadow_map: Grayscale shadow (B, 1, H, W) or (1, H, W)

        Returns:
            RGB shadow (B, 3, H, W) or (3, H, W)
        """
        if shadow_map.dim() == 3:  # (1, H, W)
            return shadow_map.repeat(3, 1, 1)
        elif shadow_map.dim() == 4:  # (B, 1, H, W)
            return shadow_map.repeat(1, 3, 1, 1)
        else:
            raise ValueError(f"Expected 3 or 4 dimensions, got {shadow_map.dim()}")

    @staticmethod
    def rgb_to_grayscale(shadow_map_rgb: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB shadow map back to grayscale by taking first channel.

        Args:
            shadow_map_rgb: RGB shadow (B, 3, H, W) or (3, H, W)

        Returns:
            Grayscale shadow (B, 1, H, W) or (1, H, W)
        """
        if shadow_map_rgb.dim() == 3:  # (3, H, W)
            return shadow_map_rgb[0:1, :, :]
        elif shadow_map_rgb.dim() == 4:  # (B, 3, H, W)
            return shadow_map_rgb[:, 0:1, :, :]
        else:
            raise ValueError(f"Expected 3 or 4 dimensions, got {shadow_map_rgb.dim()}")

    @staticmethod
    def normalize_shadow(shadow: torch.Tensor) -> torch.Tensor:
        """
        Normalize shadow map to [-1, 1] range (for VAE).

        Args:
            shadow: Shadow map in [0, 1] range

        Returns:
            Normalized shadow in [-1, 1]
        """
        return shadow * 2.0 - 1.0

    @staticmethod
    def denormalize_shadow(shadow: torch.Tensor) -> torch.Tensor:
        """
        Denormalize shadow map from [-1, 1] to [0, 1].

        Args:
            shadow: Shadow map in [-1, 1] range

        Returns:
            Denormalized shadow in [0, 1]
        """
        return (shadow + 1.0) / 2.0


class MaskProcessor:
    """
    Processes object masks for latent space.

    Masks need to be resized to match latent dimensions
    and properly aligned with encoded objects.
    """

    @staticmethod
    def resize_to_latent(
        mask: torch.Tensor,
        latent_size: Tuple[int, int],
        mode: str = "nearest",
    ) -> torch.Tensor:
        """
        Resize binary mask to latent space dimensions.

        Args:
            mask: Binary mask (B, 1, H, W) in {0, 1}
            latent_size: Target size (H_lat, W_lat), typically (128, 128)
            mode: Interpolation mode ("nearest" for binary masks)

        Returns:
            Resized mask (B, 1, H_lat, W_lat)
        """
        return F.interpolate(
            mask,
            size=latent_size,
            mode=mode,
            align_corners=None if mode == "nearest" else False,
        )

    @staticmethod
    def create_mask_from_alpha(image: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Create binary mask from image with alpha channel.

        Args:
            image: RGBA image (B, 4, H, W) or RGB (B, 3, H, W)
            threshold: Threshold for binary mask

        Returns:
            Binary mask (B, 1, H, W)
        """
        if image.shape[1] == 4:  # RGBA
            alpha = image[:, 3:4, :, :]
            mask = (alpha > threshold).float()
        else:  # RGB - use brightness
            gray = image.mean(dim=1, keepdim=True)
            mask = (gray > threshold).float()

        return mask

    @staticmethod
    def dilate_mask(mask: torch.Tensor, iterations: int = 1) -> torch.Tensor:
        """
        Dilate binary mask using max pooling.

        Args:
            mask: Binary mask (B, 1, H, W)
            iterations: Number of dilation iterations

        Returns:
            Dilated mask (B, 1, H, W)
        """
        dilated = mask
        for _ in range(iterations):
            dilated = F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
        return dilated


class LatentConcatenator:
    """
    Concatenates latents for 9-channel input to UNet.

    Combines: [noise(4) + object(4) + mask(1)] = 9 channels
    """

    @staticmethod
    def concatenate(
        noise_latent: torch.Tensor,
        object_latent: torch.Tensor,
        mask_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate latents along channel dimension.

        Args:
            noise_latent: Random noise (B, 4, H, W)
            object_latent: Encoded object (B, 4, H, W)
            mask_latent: Resized mask (B, 1, H, W)

        Returns:
            Concatenated latent (B, 9, H, W)
        """
        # Verify shapes
        assert noise_latent.shape == object_latent.shape, \
            f"Noise and object shapes must match: {noise_latent.shape} vs {object_latent.shape}"
        assert noise_latent.shape[0] == mask_latent.shape[0], \
            "Batch sizes must match"
        assert noise_latent.shape[2:] == mask_latent.shape[2:], \
            f"Spatial dims must match: {noise_latent.shape[2:]} vs {mask_latent.shape[2:]}"

        # Concatenate [4 + 4 + 1] = 9 channels
        concatenated = torch.cat([noise_latent, object_latent, mask_latent], dim=1)

        assert concatenated.shape[1] == 9, f"Expected 9 channels, got {concatenated.shape[1]}"

        return concatenated

    @staticmethod
    def split(concatenated: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split concatenated latent back into components.

        Args:
            concatenated: Concatenated latent (B, 9, H, W)

        Returns:
            Tuple of (noise, object, mask) latents
        """
        assert concatenated.shape[1] == 9, f"Expected 9 channels, got {concatenated.shape[1]}"

        noise = concatenated[:, 0:4, :, :]
        object_latent = concatenated[:, 4:8, :, :]
        mask = concatenated[:, 8:9, :, :]

        return noise, object_latent, mask


class VAEPipeline(nn.Module):
    """
    Complete VAE pipeline for shadow generation.

    Integrates all VAE operations:
    - Encoding objects and shadows
    - Format conversions
    - Mask processing
    - Latent concatenation
    - Decoding and post-processing
    """

    def __init__(
        self,
        pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        latent_size: Tuple[int, int] = (128, 128),
    ):
        """
        Initialize VAE pipeline.

        Args:
            pretrained_model_name: HuggingFace SDXL model ID
            latent_size: Latent space dimensions (default: 128×128 for 1024×1024 images)
        """
        super().__init__()

        self.latent_size = latent_size

        # Load frozen VAE
        self.vae = VAEWrapper(pretrained_model_name)

        # Helper modules
        self.converter = ShadowMapConverter()
        self.mask_processor = MaskProcessor()
        self.concatenator = LatentConcatenator()

        print(f"✓ VAE Pipeline initialized (latent size: {latent_size})")

    def encode_object(self, object_image: torch.Tensor) -> torch.Tensor:
        """
        Encode object image to latent space.

        Args:
            object_image: RGB image (B, 3, H, W) in [-1, 1]

        Returns:
            Latent (B, 4, H//8, W//8)
        """
        # Disable autocast for VAE (must stay in FP32)
        with torch.cuda.amp.autocast(enabled=False):
            return self.vae.encode(object_image)

    def encode_shadow(self, shadow_map: torch.Tensor) -> torch.Tensor:
        """
        Encode grayscale shadow map to latent space.

        Args:
            shadow_map: Grayscale shadow (B, 1, H, W) in [0, 1]

        Returns:
            Latent (B, 4, H//8, W//8)
        """
        # Debug: Check input
        if torch.isnan(shadow_map).any() or torch.isinf(shadow_map).any():
            print(f"\n⚠️  NaN/Inf in shadow_map INPUT: mean={shadow_map.mean()}, min={shadow_map.min()}, max={shadow_map.max()}")

        # Convert to RGB
        shadow_rgb = self.converter.grayscale_to_rgb(shadow_map)

        # Debug: Check after RGB conversion
        if torch.isnan(shadow_rgb).any() or torch.isinf(shadow_rgb).any():
            print(f"\n⚠️  NaN/Inf after RGB conversion: mean={shadow_rgb.mean()}, min={shadow_rgb.min()}, max={shadow_rgb.max()}")

        # Normalize to [-1, 1]
        shadow_normalized = self.converter.normalize_shadow(shadow_rgb)

        # Debug: Check after normalization
        if torch.isnan(shadow_normalized).any() or torch.isinf(shadow_normalized).any():
            print(f"\n⚠️  NaN/Inf after normalization: mean={shadow_normalized.mean()}, min={shadow_normalized.min()}, max={shadow_normalized.max()}")

        # Encode with autocast disabled (VAE must stay in FP32)
        with torch.cuda.amp.autocast(enabled=False):
            latent = self.vae.encode(shadow_normalized)

        # Debug: Check VAE output
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            print(f"\n⚠️  NaN/Inf in VAE ENCODE OUTPUT: mean={latent.mean()}, min={latent.min()}, max={latent.max()}")
            print(f"   Input to VAE was: mean={shadow_normalized.mean()}, std={shadow_normalized.std()}, dtype={shadow_normalized.dtype}")

        return latent

    def decode_to_shadow(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to grayscale shadow map.

        Args:
            latent: Latent (B, 4, H, W)

        Returns:
            Grayscale shadow (B, 1, H*8, W*8) in [0, 1]
        """
        # Decode to RGB
        shadow_rgb = self.vae.decode(latent)

        # Convert to grayscale
        shadow_gray = self.converter.rgb_to_grayscale(shadow_rgb)

        # Denormalize to [0, 1]
        shadow_final = self.converter.denormalize_shadow(shadow_gray)

        # Clamp to valid range
        shadow_final = torch.clamp(shadow_final, 0.0, 1.0)

        return shadow_final

    def prepare_unet_input(
        self,
        object_image: torch.Tensor,
        mask: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare 9-channel input for UNet.

        Args:
            object_image: RGB object (B, 3, H, W) in [-1, 1]
            mask: Binary mask (B, 1, H, W) in {0, 1}
            noise: Optional noise latent (B, 4, h, w). If None, generates random noise.

        Returns:
            Concatenated input (B, 9, h, w)
        """
        batch_size = object_image.shape[0]

        # 1. Encode object to latent space
        object_latent = self.encode_object(object_image)

        # 2. Resize mask to latent space
        mask_latent = self.mask_processor.resize_to_latent(
            mask, self.latent_size, mode="nearest"
        )

        # 3. Generate or use provided noise
        if noise is None:
            noise = torch.randn_like(object_latent)

        # 4. Concatenate
        unet_input = self.concatenator.concatenate(noise, object_latent, mask_latent)

        return unet_input

    def process_output(self, output_latent: torch.Tensor) -> torch.Tensor:
        """
        Process UNet output latent to final shadow map.

        Args:
            output_latent: Predicted latent (B, 4, h, w)

        Returns:
            Shadow map (B, 1, H, W) in [0, 1]
        """
        return self.decode_to_shadow(output_latent)

    def print_pipeline_summary(self):
        """Print VAE pipeline summary."""
        print("\n" + "="*60)
        print("VAE Pipeline Summary")
        print("="*60)
        print(f"Latent size:          {self.latent_size}")
        print(f"Downsampling factor:  8×")
        print(f"Image size:           ({self.latent_size[0]*8}, {self.latent_size[1]*8})")
        print()
        print("Pipeline Flow:")
        print("  Object RGB (B, 3, 1024, 1024)")
        print("    ↓ VAE Encode")
        print("  Object Latent (B, 4, 128, 128)")
        print()
        print("  Mask (B, 1, 1024, 1024)")
        print("    ↓ Resize")
        print("  Mask Latent (B, 1, 128, 128)")
        print()
        print("  Noise (B, 4, 128, 128)")
        print("    ↓")
        print("  Concatenate → (B, 9, 128, 128)")
        print("    ↓ UNet")
        print("  Shadow Latent (B, 4, 128, 128)")
        print("    ↓ VAE Decode")
        print("  Shadow RGB (B, 3, 1024, 1024)")
        print("    ↓ Extract Channel")
        print("  Shadow Gray (B, 1, 1024, 1024)")
        print("="*60 + "\n")


def test_vae_pipeline():
    """Test VAE pipeline components."""
    print("\n" + "="*60)
    print("Testing VAE Pipeline")
    print("="*60 + "\n")

    # Test 1: Shadow map conversion
    print("1. Testing grayscale ↔ RGB conversion...")
    converter = ShadowMapConverter()

    shadow_gray = torch.rand(2, 1, 256, 256)
    shadow_rgb = converter.grayscale_to_rgb(shadow_gray)
    shadow_back = converter.rgb_to_grayscale(shadow_rgb)

    print(f"   Grayscale shape: {shadow_gray.shape}")
    print(f"   RGB shape: {shadow_rgb.shape}")
    print(f"   Back to gray: {shadow_back.shape}")
    assert torch.allclose(shadow_gray, shadow_back, atol=1e-6)
    print("   ✓ Conversion is lossless")

    # Test 2: Mask processing
    print("\n2. Testing mask processing...")
    processor = MaskProcessor()

    mask = torch.randint(0, 2, (2, 1, 1024, 1024)).float()
    mask_resized = processor.resize_to_latent(mask, (128, 128))

    print(f"   Original mask: {mask.shape}")
    print(f"   Resized mask: {mask_resized.shape}")
    assert mask_resized.shape == (2, 1, 128, 128)
    print("   ✓ Mask resizing works")

    # Test 3: Latent concatenation
    print("\n3. Testing latent concatenation...")
    concat = LatentConcatenator()

    noise = torch.randn(2, 4, 128, 128)
    obj = torch.randn(2, 4, 128, 128)
    mask_lat = torch.randint(0, 2, (2, 1, 128, 128)).float()

    concatenated = concat.concatenate(noise, obj, mask_lat)
    print(f"   Noise: {noise.shape}")
    print(f"   Object: {obj.shape}")
    print(f"   Mask: {mask_lat.shape}")
    print(f"   Concatenated: {concatenated.shape}")
    assert concatenated.shape == (2, 9, 128, 128)
    print("   ✓ Concatenation produces 9 channels")

    # Test split
    n, o, m = concat.split(concatenated)
    assert torch.allclose(n, noise)
    assert torch.allclose(o, obj)
    assert torch.allclose(m, mask_lat)
    print("   ✓ Split recovers original tensors")

    # Test 4: Full pipeline (requires actual VAE)
    print("\n4. Testing full VAE pipeline...")
    print("   (Requires PyTorch and diffusers to be installed)")
    print("   Pipeline initialization: VAEPipeline(...)")

    # Create pipeline instance
    pipeline = VAEPipeline()
    pipeline.print_pipeline_summary()

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_vae_pipeline()
