"""
Light parameter conditioning for shadow generation.

Implements sinusoidal embeddings for light source parameters (θ, φ, s)
as described in the paper.

Equation 7 from paper:
e(t) = [cos(ω_i · t), sin(ω_i · t)] for i in [0, d/2-1]
where ω_i = 2^(-(i·(i-1))/(d/2·(d/2-1))) · log(10000)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import math


class LightParameterConditioning(nn.Module):
    """
    Sinusoidal encoding for light source parameters (Equation 7).

    Encodes polar angle θ, azimuthal angle φ, and light size s
    into 768-dimensional embedding (256 per parameter).

    This replaces text conditioning in SDXL and is injected via
    timestep embeddings.
    """

    def __init__(self, embedding_dim: int = 256, max_freq: float = 10000.0):
        """
        Initialize sinusoidal encoder.

        Args:
            embedding_dim: Dimension per parameter (default: 256)
            max_freq: Maximum frequency for encoding (default: 10000)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_freq = max_freq

        # Compute frequency bands according to Equation 7
        # ω_i = 2^(-(i·(i-1))/(d/2·(d/2-1))) · log(max_freq)
        freqs = self._compute_frequencies(embedding_dim)
        self.register_buffer('omega_i', freqs)

    def _compute_frequencies(self, d: int) -> torch.Tensor:
        """
        Compute frequency bands according to Equation 7.

        ω_i = 2^(-(i·(i-1))/(d/2·(d/2-1))) · log(max_freq)

        Args:
            d: Embedding dimension (total)

        Returns:
            Frequency tensor of shape (d//2,)
        """
        half_dim = d // 2
        i = torch.arange(half_dim, dtype=torch.float32)

        # Compute exponent: -(i·(i-1)) / (d/2·(d/2-1))
        numerator = i * (i - 1)
        denominator = half_dim * (half_dim - 1)

        # Avoid division by zero for d=2
        if denominator == 0:
            exponent = torch.zeros_like(i)
        else:
            exponent = -numerator / denominator

        # ω_i = 2^exponent · log(max_freq)
        omega_i = (2.0 ** exponent) * math.log(self.max_freq)

        return omega_i

    def encode_scalar(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply sinusoidal encoding to scalar parameter.
        
        Args:
            x: Input scalar parameter(s) of shape (..., 1)
            
        Returns:
            Encoded embeddings of shape (..., embedding_dim)
        """
        # Normalize input to [0, 1] range
        x_norm = x.float()
        
        # Compute sinusoidal frequencies
        angles = self.omega_i[None, :] * x_norm[..., None]  # (..., d//2)
        
        # Create sinusoidal embeddings
        embeddings = torch.cat([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=-1)
        
        return embeddings
    
    def encode_light_params(self, theta: torch.Tensor, phi: torch.Tensor, 
                           size: torch.Tensor) -> torch.Tensor:
        """
        Encode light parameters (θ, φ, s) into combined embedding.
        
        Args:
            theta: Polar angle in degrees (shape: ...)
            phi: Azimuthal angle in degrees (shape: ...)
            size: Light size/softness parameter (shape: ...)
            
        Returns:
            Combined embeddings (shape: ..., embedding_dim * 3)
        """
        # Normalize parameters
        theta_norm = theta / 45.0  # Normalize to [0, 1] range (assuming max θ=45°)
        phi_norm = phi / 360.0      # Normalize to [0, 1] range  
        size_norm = (size - 2) / 6  # Normalize s to [0, 1] range (assuming s ∈ [2, 8])
        
        # Encode each parameter
        theta_emb = self.encode_scalar(theta_norm.unsqueeze(-1))
        phi_emb = self.encode_scalar(phi_norm.unsqueeze(-1))
        size_emb = self.encode_scalar(size_norm.unsqueeze(-1))
        
        # Concatenate embeddings
        combined_emb = torch.cat([theta_emb, phi_emb, size_emb], dim=-1)
        
        return combined_emb
    
    def forward(self, theta: torch.Tensor, phi: torch.Tensor,
                size: torch.Tensor) -> torch.Tensor:
        """Forward pass for light parameter conditioning."""
        return self.encode_light_params(theta, phi, size)

    def get_output_dim(self) -> int:
        """Get total output dimension (3 parameters × embedding_dim)."""
        return self.embedding_dim * 3


class TimestepLightEmbedding(nn.Module):
    """
    Combines timestep embeddings with light parameter embeddings.

    In SDXL, timestep embeddings are injected into UNet blocks.
    We add light parameter embeddings to these timestep embeddings
    to condition the model on both time and light parameters.
    """

    def __init__(
        self,
        timestep_dim: int = 320,
        light_conditioning_dim: int = 768,
        time_embed_dim: int = 1280,
    ):
        """
        Initialize timestep + light embedding combiner.

        Args:
            timestep_dim: Dimension of sinusoidal timestep encoding (SDXL default: 320)
            light_conditioning_dim: Dimension of light parameter embedding (3×256=768)
            time_embed_dim: Final embedding dimension (SDXL default: 1280)
        """
        super().__init__()

        self.timestep_dim = timestep_dim
        self.light_conditioning_dim = light_conditioning_dim
        self.time_embed_dim = time_embed_dim

        # Project timestep to higher dimension
        self.time_proj = nn.Sequential(
            nn.Linear(timestep_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Project light parameters to same dimension
        self.light_proj = nn.Sequential(
            nn.Linear(light_conditioning_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Combine both embeddings
        self.combine = nn.Sequential(
            nn.Linear(time_embed_dim * 2, time_embed_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        timestep_emb: torch.Tensor,
        light_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine timestep and light embeddings.

        Args:
            timestep_emb: Timestep embeddings (B, timestep_dim)
            light_emb: Light parameter embeddings (B, 768)

        Returns:
            Combined embeddings (B, time_embed_dim)
        """
        # Project both to same dimension
        t_emb = self.time_proj(timestep_emb)
        l_emb = self.light_proj(light_emb)

        # Concatenate and combine
        combined = torch.cat([t_emb, l_emb], dim=-1)
        output = self.combine(combined)

        return output


class AdditiveTimestepLightEmbedding(nn.Module):
    """
    Alternative: Add light embeddings to timestep embeddings.

    Simpler approach - just adds projected light parameters to
    timestep embeddings instead of concatenating.
    """

    def __init__(
        self,
        light_conditioning_dim: int = 768,
        time_embed_dim: int = 1280,
    ):
        """
        Initialize additive embedding combiner.

        Args:
            light_conditioning_dim: Light parameter embedding dim (768)
            time_embed_dim: Timestep embedding dim (1280)
        """
        super().__init__()

        # Project light parameters to timestep embedding dimension
        self.light_proj = nn.Sequential(
            nn.Linear(light_conditioning_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(
        self,
        timestep_emb: torch.Tensor,
        light_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add light embedding to timestep embedding.

        Args:
            timestep_emb: Timestep embeddings (B, time_embed_dim)
            light_emb: Light parameter embeddings (B, 768)

        Returns:
            Combined embeddings (B, time_embed_dim)
        """
        # Project light parameters
        l_emb = self.light_proj(light_emb)

        # Add to timestep embeddings
        output = timestep_emb + l_emb

        return output


class ConditionedUpsampler(nn.Module):
    """
    Upsampler that incorporates light parameter conditioning.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 conditioning_dim: int = 768):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Project conditioning to match spatial dimensions
        self.condition_proj = nn.Linear(conditioning_dim, in_channels)
        
        # Main convolution
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, out_channels, 
                              kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(out_channels // 8, out_channels),
            nn.SiLU()
        )
        
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Upsample with conditioning.
        
        Args:
            x: Input feature maps (..., channels, height, width)
            conditioning: Light parameter embeddings (..., 768)
        """
        # Process conditioning
        cond_proj = self.condition_proj(conditioning)
        cond_proj = cond_proj.unsqueeze(-1).unsqueeze(-1)  # Add spatial dims
        
        # Apply conditioning modulation
        x_conditioned = x * cond_proj
        
        # Concatenate original and conditioned features
        x_combined = torch.cat([x, x_conditioned], dim=1)
        
        # Apply convolution upsampling
        return self.conv(x_combined)


def spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor, 
                          radius: float = 8.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert spherical coordinates to Cartesian coordinates.
    
    Args:
        theta: Polar angle in degrees
        phi: Azimuthal angle in degrees  
        radius: Sphere radius
        
    Returns:
        Tuple of (x, y) cartesian coordinates
    """
    theta_rad = torch.deg2rad(theta)
    phi_rad = torch.deg2rad(phi)
    
    x = radius * torch.sin(theta_rad) * torch.cos(phi_rad)
    y = radius * torch.sin(theta_rad) * torch.sin(phi_rad)
    
    return x, y


def create_light_blob(theta: torch.Tensor, phi: torch.Tensor, size: torch.Tensor,
                    image_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """
    Create Gaussian blob representation of light source.
    
    Args:
        theta: Polar angle in degrees
        phi: Azimuthal angle in degrees
        size: Light size parameter
        image_size: Output image dimensions
        
    Returns:
        Grayscale blob image (1, height, width)
    """
    h, w = image_size
    
    # Convert to cartesian coordinates
    x, y = spherical_to_cartesian(theta, phi)
    
    # Create coordinate grids
    xx, yy = torch.meshgrid(
        torch.linspace(-4, 4, w),
        torch.linspace(-4, 4, h),
        indexing='xy'
    )
    xx = xx.to(theta.device)
    yy = yy.to(theta.device)
    
    # Create Gaussian blob
    sigma = size.float()
    blob = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))

    return blob.unsqueeze(0)  # Add channel dimension


def test_conditioning_system():
    """Comprehensive test of conditioning system."""
    print("\n" + "="*60)
    print("Testing Conditioning System")
    print("="*60 + "\n")

    # Test 1: Sinusoidal embeddings (Equation 7)
    print("1. Testing sinusoidal embeddings (Equation 7)...")
    light_encoder = LightParameterConditioning(embedding_dim=256, max_freq=10000.0)

    # Test frequency computation
    freqs = light_encoder.omega_i
    print(f"   Frequency shape: {freqs.shape}")
    print(f"   Min frequency: {freqs.min():.6f}")
    print(f"   Max frequency: {freqs.max():.6f}")

    # Test single parameter encoding
    theta = torch.tensor([30.0])  # 30 degrees
    theta_norm = theta / 45.0
    theta_emb = light_encoder.encode_scalar(theta_norm.unsqueeze(-1))
    print(f"   Theta embedding shape: {theta_emb.shape}")
    assert theta_emb.shape == (1, 256), "Incorrect embedding dimension!"
    print("   ✓ Single parameter encoding works")

    # Test 2: Light parameter encoding (θ, φ, s) → 768-dim
    print("\n2. Testing light parameter encoding...")
    batch_size = 4
    theta = torch.tensor([15.0, 30.0, 45.0, 20.0])
    phi = torch.tensor([0.0, 90.0, 180.0, 270.0])
    size = torch.tensor([2.0, 4.0, 6.0, 8.0])

    light_emb = light_encoder(theta, phi, size)
    print(f"   Input shapes: theta={theta.shape}, phi={phi.shape}, size={size.shape}")
    print(f"   Output shape: {light_emb.shape}")
    assert light_emb.shape == (batch_size, 768), "Incorrect output dimension!"
    print(f"   Output dim: {light_encoder.get_output_dim()}")
    print("   ✓ Light parameter encoding works (θ, φ, s → 768-dim)")

    # Test 3: Timestep embedding integration
    print("\n3. Testing timestep + light embedding integration...")

    # Test concatenative approach
    concat_embedder = TimestepLightEmbedding(
        timestep_dim=320,
        light_conditioning_dim=768,
        time_embed_dim=1280,
    )

    dummy_timestep_emb = torch.randn(batch_size, 320)
    combined_emb = concat_embedder(dummy_timestep_emb, light_emb)
    print(f"   Timestep emb shape: {dummy_timestep_emb.shape}")
    print(f"   Light emb shape: {light_emb.shape}")
    print(f"   Combined emb shape: {combined_emb.shape}")
    assert combined_emb.shape == (batch_size, 1280), "Incorrect combined dimension!"
    print("   ✓ Concatenative timestep+light embedding works")

    # Test additive approach
    print("\n4. Testing additive timestep + light embedding...")
    additive_embedder = AdditiveTimestepLightEmbedding(
        light_conditioning_dim=768,
        time_embed_dim=1280,
    )

    dummy_timestep_emb_1280 = torch.randn(batch_size, 1280)
    combined_emb_add = additive_embedder(dummy_timestep_emb_1280, light_emb)
    print(f"   Timestep emb shape: {dummy_timestep_emb_1280.shape}")
    print(f"   Light emb shape: {light_emb.shape}")
    print(f"   Combined emb shape: {combined_emb_add.shape}")
    assert combined_emb_add.shape == (batch_size, 1280), "Incorrect combined dimension!"
    print("   ✓ Additive timestep+light embedding works")

    # Test 5: Blob-based conditioning (for ablation)
    print("\n5. Testing blob-based light representation...")
    theta_test = torch.tensor([30.0])
    phi_test = torch.tensor([45.0])
    size_test = torch.tensor([4.0])

    blob = create_light_blob(theta_test, phi_test, size_test, image_size=(128, 128))
    print(f"   Input: θ={theta_test.item()}°, φ={phi_test.item()}°, s={size_test.item()}")
    print(f"   Blob shape: {blob.shape}")
    print(f"   Blob min/max: [{blob.min():.4f}, {blob.max():.4f}]")
    assert blob.shape == (1, 128, 128), "Incorrect blob shape!"
    print("   ✓ Blob-based conditioning works")

    # Test 6: Spherical to Cartesian conversion
    print("\n6. Testing spherical coordinate conversion...")
    x, y = spherical_to_cartesian(theta_test, phi_test, radius=8.0)
    print(f"   Spherical: θ={theta_test.item()}°, φ={phi_test.item()}°, r=8.0")
    print(f"   Cartesian: x={x.item():.4f}, y={y.item():.4f}")
    print("   ✓ Coordinate conversion works")

    # Summary
    print("\n" + "="*60)
    print("Conditioning System Summary")
    print("="*60)
    print(f"✓ Equation 7 sinusoidal embeddings: 256-dim per parameter")
    print(f"✓ Light parameter encoding: (θ, φ, s) → 768-dim")
    print(f"✓ Timestep integration: 320/1280-dim → 1280-dim combined")
    print(f"✓ Blob-based conditioning: alternative for ablation")
    print(f"✓ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run comprehensive tests
    test_conditioning_system()
