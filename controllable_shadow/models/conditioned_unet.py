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
                "additive": Add light embeddings to timestep embeddings
                "concat": Concatenate and project to final dimension
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

        # Timestep + light embedding combiner
        if conditioning_strategy == "additive":
            self.embedding_combiner = AdditiveTimestepLightEmbedding(
                light_conditioning_dim=768,
                time_embed_dim=1280,
            )
        elif conditioning_strategy == "concat":
            self.embedding_combiner = TimestepLightEmbedding(
                timestep_dim=320,
                light_conditioning_dim=768,
                time_embed_dim=1280,
            )
        else:
            raise ValueError(f"Unknown conditioning strategy: {conditioning_strategy}")

        print(f"✓ Conditioned SDXL UNet initialized ({conditioning_strategy} strategy)")

    def encode_light_parameters(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
        size: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode light parameters to embedding.

        Args:
            theta: Polar angle in degrees (B,)
            phi: Azimuthal angle in degrees (B,)
            size: Light size parameter (B,)

        Returns:
            Light embeddings (B, 768)
        """
        return self.light_encoder(theta, phi, size)

    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get sinusoidal timestep embeddings from SDXL UNet.

        Args:
            timesteps: Timestep values (B,)

        Returns:
            Timestep embeddings (B, 320 or 1280 depending on UNet)
        """
        # SDXL uses time_proj and time_embedding modules
        # We need to extract timestep embeddings before they're combined with other info

        # Get the timestep projection from SDXL UNet
        # This is typically a sinusoidal embedding at dimension 320
        t_emb = self.unet.unet.time_proj(timesteps)

        return t_emb

    def combine_embeddings(
        self,
        timestep_emb: torch.Tensor,
        light_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine timestep and light embeddings.

        Args:
            timestep_emb: Timestep embeddings (B, 320 or 1280)
            light_emb: Light parameter embeddings (B, 768)

        Returns:
            Combined embeddings (B, 1280)
        """
        return self.embedding_combiner(timestep_emb, light_emb)

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

        # 1. Encode light parameters (θ, φ, s) → 768-dim
        light_emb = self.encode_light_parameters(theta, phi, size)

        # 2. Get timestep embeddings from UNet
        timestep_emb = self.get_timestep_embedding(timestep)

        # 3. Combine timestep + light embeddings
        combined_emb = self.combine_embeddings(timestep_emb, light_emb)

        # 4. Process combined embedding through UNet's time_embedding module
        # This projects to the final dimension and prepares for injection
        emb = self.unet.unet.time_embedding(combined_emb)

        # 5. Run UNet forward pass with combined embeddings
        # We bypass normal timestep processing since we've already combined embeddings
        # Create dummy encoder_hidden_states (not used since cross-attention removed)
        encoder_hidden_states = torch.zeros(batch_size, 77, 768, device=sample.device)

        # Forward through UNet with modified timestep embeddings
        # We need to manually handle the forward pass to inject our custom embeddings
        output = self._forward_unet_with_custom_embeddings(
            sample, emb, encoder_hidden_states
        )

        if return_dict:
            return {"sample": output}
        return output

    def _forward_unet_with_custom_embeddings(
        self,
        sample: torch.Tensor,
        emb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through UNet with custom embeddings.

        This bypasses the normal timestep processing and directly injects
        our combined (timestep + light) embeddings.

        Args:
            sample: Input latents (B, 9, H, W)
            emb: Combined embeddings (B, 1280)
            encoder_hidden_states: Dummy cross-attention (not used)

        Returns:
            Output latents (B, 4, H, W)
        """
        # This is a simplified version - in practice, we'd need to carefully
        # inject embeddings at each UNet block
        # For now, we use the standard forward but with pre-computed embeddings

        # The actual implementation depends on diffusers version
        # We'll use a workaround: pass a dummy timestep and replace embeddings

        # Store original time_embedding module
        original_time_embedding = self.unet.unet.time_embedding

        # Create a temporary module that returns our custom embeddings
        class CustomEmbedding(nn.Module):
            def __init__(self, emb):
                super().__init__()
                self.emb = emb

            def forward(self, x):
                return self.emb

        # Temporarily replace time_embedding
        self.unet.unet.time_embedding = CustomEmbedding(emb)

        # Forward pass with dummy timestep
        dummy_timestep = torch.zeros(sample.shape[0], device=sample.device)
        output = self.unet(
            sample=sample,
            timestep=dummy_timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )

        # Restore original time_embedding
        self.unet.unet.time_embedding = original_time_embedding

        return output

    def print_conditioning_flow(self):
        """Print how conditioning flows through the network."""
        print("\n" + "="*60)
        print("Conditioning Flow Diagram")
        print("="*60)
        print()
        print("Light Parameters (θ, φ, s)")
        print("    ↓")
        print("Sinusoidal Embeddings (Equation 7)")
        print("    ↓")
        print("Light Embedding: 768-dim (256 × 3)")
        print("    ↓")
        print("    ├─→ Timestep: t → Sinusoidal → 320-dim")
        print("    │")
        print(f"    └─→ Combine ({self.conditioning_strategy})")
        print("         ↓")
        print("    Combined Embedding: 1280-dim")
        print("         ↓")
        print("    Time Embedding Module (projection)")
        print("         ↓")
        print("    Final Embedding: 1280-dim")
        print("         ↓")
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
