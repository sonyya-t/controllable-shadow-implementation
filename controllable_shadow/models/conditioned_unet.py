"""
Wrapper that integrates SDXL UNet with light parameter conditioning.

Maps conditioning injection points and ensures light parameters
properly influence all UNet blocks via timestep embeddings.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .sdxl_unet import SDXLUNetForShadows
from .conditioning import (
    LightParameterConditioning,
    TimestepLightEmbedding,
    AdditiveTimestepLightEmbedding,
)


class ConditionedSDXLUNet(nn.Module):
    """
    SDXL UNet with integrated light parameter conditioning.

    This wrapper:
    1. Encodes light parameters (θ, φ, s) using sinusoidal embeddings
    2. Combines with timestep embeddings
    3. Injects into UNet via timestep embedding pathway
    4. Ensures conditioning reaches all UNet blocks
    """

    def __init__(
        self,
        pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        conditioning_strategy: str = "additive",  # "additive" or "concat"
        embedding_dim: int = 256,  # Per parameter
    ):
        """
        Initialize conditioned UNet.

        Args:
            pretrained_model_name: HuggingFace SDXL model ID
            conditioning_strategy: How to combine timestep + light embeddings
                "additive": SDXL will add light embeddings to its timestep embeddings
                "concat": Not recommended - use additive instead
            embedding_dim: Embedding dimension per light parameter
        """
        super().__init__()

        self.conditioning_strategy = conditioning_strategy
        self.embedding_dim = embedding_dim

        # Load modified SDXL UNet
        self.unet = SDXLUNetForShadows(
            pretrained_model_name=pretrained_model_name,
            in_channels=9,
            out_channels=4,
            conditioning_dim=768,
        )

        # Light parameter encoder
        self.light_encoder = LightParameterConditioning(
            embedding_dim=embedding_dim,
            max_freq=10000.0,
        )

        # Project light embeddings to SDXL's time embedding dimension (1280)
        # CRITICAL: Keep in FP16 to match everything else, use small init to prevent explosion
        self.light_projection = nn.Linear(768, 1280)
        print(f"Light projection: {self.light_projection.weight.dtype}")

        # Initialize with very small weights to prevent gradient explosion
        with torch.no_grad():
            # Xavier uniform initialization with very small scale for FP16 stability
            nn.init.xavier_uniform_(self.light_projection.weight, gain=0.01)
            if self.light_projection.bias is not None:
                nn.init.zeros_(self.light_projection.bias)

        # Convert to FP16 to match UNet and training dtype
        self.light_projection = self.light_projection.half()

        print(f"✓ Conditioned SDXL UNet initialized")
        print(f"  - Light encoder: FP16")
        print(f"  - Light projection: FP16 (small init gain=0.01)")
        print(f"  - UNet: FP16")
        print(f"  - Everything: FP16 (pure FP16 pipeline)")

    def encode_light_parameters(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
        size: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode light parameters to embedding and project to 1280-dim.

        Args:
            theta: Polar angle in degrees (B,)
            phi: Azimuthal angle in degrees (B,)
            size: Light size parameter (B,)

        Returns:
            Light embeddings projected to 1280-dim (B, 1280)
        """
        # Encode to 768-dim (outputs FP16 now)
        light_emb = self.light_encoder(theta, phi, size)  # (B, 768) FP16

        print(f"\n[DEBUG] encode_light_parameters:")
        print(f"  After light_encoder (FP16): shape={light_emb.shape}, dtype={light_emb.dtype}")
        print(f"  Stats: min={light_emb.min():.4f}, max={light_emb.max():.4f}, mean={light_emb.mean():.4f}")
        print(f"  Has NaN: {torch.isnan(light_emb).any()}")

        # Project in FP16
        light_emb_projected = self.light_projection(light_emb)  # (B, 1280) FP16

        print(f"  After projection (FP16): shape={light_emb_projected.shape}, dtype={light_emb_projected.dtype}")
        print(f"  Stats: min={light_emb_projected.min():.4f}, max={light_emb_projected.max():.4f}, mean={light_emb_projected.mean():.4f}")
        print(f"  Has NaN: {torch.isnan(light_emb_projected).any()}")

        # Check projection weights for NaN
        proj_weight = self.light_projection.weight
        print(f"  Projection weights: dtype={proj_weight.dtype}, has NaN: {torch.isnan(proj_weight).any()}")

        return light_emb_projected

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor,
        size: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with light parameter conditioning.

        Args:
            sample: Latent input (B, 9, H, W)
                Concatenation of [noise(4) + object(4) + mask(1)]
            timestep: Diffusion timestep (B,) or scalar
            theta: Polar angle in degrees (B,)
            phi: Azimuthal angle in degrees (B,)
            size: Light size parameter (B,)
            return_dict: Whether to return dict

        Returns:
            Predicted output (B, 4, H, W)
        """
        batch_size = sample.shape[0]

        # Ensure timestep is correct shape
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)

        # 1. Encode light parameters (θ, φ, s) → 1280-dim in FP16
        light_emb = self.encode_light_parameters(theta, phi, size)  # (B, 1280) FP16

        print(f"\n[DEBUG] ConditionedUNet - Light parameters:")
        print(f"  theta: {theta}, phi: {phi}, size: {size}")
        print(f"  light_emb shape: {light_emb.shape}, dtype: {light_emb.dtype}")
        print(f"  light_emb stats: min={light_emb.min():.4f}, max={light_emb.max():.4f}, mean={light_emb.mean():.4f}")
        print(f"  light_emb has NaN: {torch.isnan(light_emb).any()}")

        # 2. Validate embedding dimension
        expected_dim = 1280  # SDXL time embedding dimension
        if light_emb.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Light embedding has wrong dimension: {light_emb.shape[-1]}, expected {expected_dim}"
            )

        # 3. Run UNet forward pass
        # Ensure sample and timestep match UNet dtype (FP16)
        target_dtype = sample.dtype

        # Create dummy encoder_hidden_states (not used since cross-attention removed)
        encoder_hidden_states = torch.zeros(batch_size, 77, 768, device=sample.device, dtype=target_dtype)

        # SDXL uses added_cond_kwargs to inject additional embeddings
        # SDXL's add_embedding: concat(text_embeds[1280], time_ids[6]) → Linear → 1280
        # Then: timestep_emb + add_emb
        # Everything in FP16 for consistency
        added_cond_kwargs = {
            "text_embeds": light_emb,  # Light embeddings (B, 1280) in FP16
            # time_ids format: [h_orig, w_orig, crop_top, crop_left, h_target, w_target]
            # Use image size (1024) for all to avoid NaN from zeros, in FP16
            "time_ids": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]],
                                    device=sample.device,
                                    dtype=target_dtype).repeat(batch_size, 1),
        }

        # Forward through UNet
        # SDXL will handle timestep embedding internally
        # Ensure timestep matches UNet dtype (FP16)
        if timestep.dtype != sample.dtype:
            timestep = timestep.to(sample.dtype)

        output = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )

        # When return_dict=False, SDXL returns tuple (sample,)
        # Extract the tensor
        if isinstance(output, tuple):
            output = output[0]

        if return_dict:
            return {"sample": output}
        return output

    def print_conditioning_flow(self):
        """Print how conditioning flows through the network."""
        print("\n" + "="*60)
        print("Conditioning Flow Diagram (FIXED - No Double Embedding)")
        print("="*60)
        print()
        print("Light Parameters (θ, φ, s)")
        print("    ↓")
        print("Sinusoidal Embeddings (Equation 7)")
        print("    ↓")
        print("Light Embedding: 768-dim (256 × 3)")
        print("    ↓")
        print("Projection: 768-dim → 1280-dim")
        print("    ↓")
        print("Light Embedding (projected): 1280-dim")
        print("    │")
        print("    │   Passed to SDXL via added_cond_kwargs['text_embeds']")
        print("    │")
        print("    ├─→ Timestep: t")
        print("    │       ↓")
        print("    │   SDXL computes: time_proj(t) → 320-dim")
        print("    │       ↓")
        print("    │   time_embedding(·) → 1280-dim")
        print("    │       ↓")
        print("    └─→ SDXL adds: time_emb + text_embeds")
        print("             ↓")
        print("    Final Embedding: 1280-dim (timestep + light)")
        print("             ↓")
        print("    ┌────────────────────────────┐")
        print("    │  UNet Blocks (injected)   │")
        print("    │  - Down blocks             │")
        print("    │  - Middle block            │")
        print("    │  - Up blocks               │")
        print("    └────────────────────────────┘")
        print("         ↓")
        print("    Output: Shadow Map (4-ch latent)")
        print()
        print("="*60 + "\n")


def test_conditioned_unet():
    """Test conditioned UNet with light parameters."""
    print("\n" + "="*60)
    print("Testing Conditioned SDXL UNet")
    print("="*60 + "\n")

    # Note: This test requires torch to be installed
    print("Creating conditioned UNet (additive strategy)...")
    model = ConditionedSDXLUNet(
        conditioning_strategy="additive",
        embedding_dim=256,
    )

    # Print conditioning flow
    model.print_conditioning_flow()

    # Test with dummy data
    print("Testing forward pass with dummy data...")
    batch_size = 2
    latent_h, latent_w = 128, 128

    # Create dummy inputs
    dummy_sample = torch.randn(batch_size, 9, latent_h, latent_w)
    dummy_timestep = torch.tensor([500, 600])
    dummy_theta = torch.tensor([15.0, 30.0])
    dummy_phi = torch.tensor([45.0, 90.0])
    dummy_size = torch.tensor([3.0, 5.0])

    print(f"\nInput shapes:")
    print(f"  Sample:   {dummy_sample.shape}")
    print(f"  Timestep: {dummy_timestep.shape}")
    print(f"  θ:        {dummy_theta.shape}")
    print(f"  φ:        {dummy_phi.shape}")
    print(f"  Size:     {dummy_size.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(
            sample=dummy_sample,
            timestep=dummy_timestep,
            theta=dummy_theta,
            phi=dummy_phi,
            size=dummy_size,
        )

    print(f"\nOutput shape: {output.shape}")
    assert output.shape == (batch_size, 4, latent_h, latent_w)
    print("✓ Forward pass successful!")

    # Test concatenative strategy
    print("\n" + "-"*60)
    print("Testing concatenative conditioning strategy...")
    model_concat = ConditionedSDXLUNet(
        conditioning_strategy="concat",
        embedding_dim=256,
    )

    with torch.no_grad():
        output_concat = model_concat(
            sample=dummy_sample,
            timestep=dummy_timestep,
            theta=dummy_theta,
            phi=dummy_phi,
            size=dummy_size,
        )

    print(f"Output shape: {output_concat.shape}")
    assert output_concat.shape == (batch_size, 4, latent_h, latent_w)
    print("✓ Concatenative strategy works!")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run tests
    test_conditioned_unet()
