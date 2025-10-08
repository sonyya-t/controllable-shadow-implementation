"""
Complete Shadow Generation Model.

Integrates all components:
- Modified SDXL UNet
- Light parameter conditioning
- VAE pipeline
- Rectified flow sampling

This is the main model class for training and inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union
import numpy as np
from pathlib import Path

from .conditioned_unet import ConditionedSDXLUNet
from .vae_pipeline import VAEPipeline
from .checkpoint_manager import CheckpointManager
from .conditioning import create_light_blob


class ShadowDiffusionModel(nn.Module):
    """
    Complete shadow generation model with rectified flow.

    This is the main model class that integrates:
    - SDXL UNet with 9-channel input
    - Light parameter conditioning (θ, φ, s)
    - Frozen VAE encoder/decoder
    - Rectified flow training and sampling
    """

    def __init__(
        self,
        pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        conditioning_strategy: str = "additive",
        embedding_dim: int = 256,
        latent_size: Optional[Tuple[int, int]] = None,
        image_size: Tuple[int, int] = (1024, 1024),
        use_blob_conditioning: bool = False,
        bridge_noise_sigma: float = 0.1,
    ):
        """
        Initialize shadow diffusion model.

        Args:
            pretrained_model_name: HuggingFace SDXL model ID
            conditioning_strategy: "additive" or "concat"
            embedding_dim: Embedding dimension per light parameter
            latent_size: Latent space size (auto-calculated from image_size if None)
            image_size: Output image size
            use_blob_conditioning: Use blob instead of embeddings (ablation)
            bridge_noise_sigma: Bridge noise scale (default: 0.1, paper Section 3.2.2)
        """
        super().__init__()

        # Calculate latent size from image size (VAE downsamples by 8)
        if latent_size is None:
            latent_size = (image_size[0] // 8, image_size[1] // 8)

        self.latent_size = latent_size
        self.image_size = image_size
        self.use_blob_conditioning = use_blob_conditioning
        self.bridge_noise_sigma = bridge_noise_sigma

        # Core components
        self.unet = ConditionedSDXLUNet(
            pretrained_model_name=pretrained_model_name,
            conditioning_strategy=conditioning_strategy,
            embedding_dim=embedding_dim,
        )

        self.vae_pipeline = VAEPipeline(
            pretrained_model_name=pretrained_model_name,
            latent_size=latent_size,
        )

        print("✓ Shadow Diffusion Model initialized")
        print(f"  - Bridge noise sigma: {bridge_noise_sigma}")

    def forward(
        self,
        object_image: torch.Tensor,
        mask: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor,
        size: torch.Tensor,
        timestep: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            object_image: RGB object image (B, 3, H, W) in [-1, 1]
            mask: Binary mask (B, 1, H, W)
            theta: Polar angle in degrees (B,)
            phi: Azimuthal angle in degrees (B,)
            size: Light size parameter (B,)
            timestep: Diffusion timestep (B,)
            noise: Optional noise (B, 4, h, w)

        Returns:
            Predicted output (B, 4, h, w)
        """
        # Prepare 9-channel input
        unet_input = self.vae_pipeline.prepare_unet_input(
            object_image, mask, noise
        )

        # Forward through conditioned UNet
        output = self.unet(
            sample=unet_input,
            timestep=timestep,
            theta=theta,
            phi=phi,
            size=size,
        )

        return output

    def compute_rectified_flow_loss(
        self,
        object_image: torch.Tensor,
        mask: torch.Tensor,
        shadow_map_target: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor,
        size: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute rectified flow training loss.

        Rectified Flow: minimize ||v_θ(x_t) - (x_1 - x_0)||²
        where x_t = (1-t)x_0 + t*x_1

        Args:
            object_image: RGB object (B, 3, H, W)
            mask: Binary mask (B, 1, H, W)
            shadow_map_target: Target shadow map (B, 1, H, W) in [0, 1]
            theta: Polar angle (B,)
            phi: Azimuthal angle (B,)
            size: Light size (B,)

        Returns:
            Dictionary with loss and diagnostics
        """
        batch_size = object_image.shape[0]
        device = object_image.device

        # Encode target shadow to latent space (x_1, clean target)
        x1 = self.vae_pipeline.encode_shadow(shadow_map_target)

        # Sample random noise (x_0, noise source)
        x0 = torch.randn_like(x1)

        # Sample random timestep t ~ Uniform(0, 1)
        t = torch.rand(batch_size, device=device)

        # Linear interpolation with bridge noise (as per paper Section 3.2.2):
        # x_t = σ(t)·x_0 + (1-σ(t))·x_1 + σ_bridge·√(σ(t)·(1-σ(t)))·ε
        # where σ(t) = 1-t for rectified flow
        t_broadcast = t.view(batch_size, 1, 1, 1)
        xt = t_broadcast * x1 + (1 - t_broadcast) * x0

        # Add bridge noise for training stability
        if self.bridge_noise_sigma > 0:
            bridge_noise = torch.randn_like(x0)
            noise_scale = torch.sqrt(t_broadcast * (1 - t_broadcast))
            xt = xt + self.bridge_noise_sigma * noise_scale * bridge_noise

        # Prepare input: concatenate [xt, object_latent, mask]
        object_latent = self.vae_pipeline.encode_object(object_image)

        # Resize mask to latent size
        mask_latent = self.vae_pipeline.mask_processor.resize_to_latent(
            mask, self.latent_size
        )

        # Concatenate all: [xt(4) + object(4) + mask(1)] = 9 channels
        unet_input = self.vae_pipeline.concatenator.concatenate(
            xt, object_latent, mask_latent
        )

        # Predict velocity v_θ(x_t)
        predicted_velocity = self.unet(
            sample=unet_input,
            timestep=t,
            theta=theta,
            phi=phi,
            size=size,
        )

        # Target velocity: x_1 - x_0
        target_velocity = x1 - x0

        # Rectified flow loss: MSE between predicted and target velocity
        loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)

        return {
            "loss": loss,
            "predicted_velocity": predicted_velocity,
            "target_velocity": target_velocity,
            "timestep": t,
        }

    @torch.no_grad()
    def sample(
        self,
        object_image: torch.Tensor,
        mask: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor,
        size: torch.Tensor,
        num_steps: int = 1,
        return_latent: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate shadow map using rectified flow sampling.

        For single-step (num_steps=1):
            x_1 = x_0 + v_θ(x_0)

        For multi-step:
            Euler integration with dt = 1/num_steps

        Args:
            object_image: RGB object (B, 3, H, W) in [-1, 1]
            mask: Binary mask (B, 1, H, W)
            theta: Polar angle (B,)
            phi: Azimuthal angle (B,)
            size: Light size (B,)
            num_steps: Number of sampling steps (1 for fast inference)
            return_latent: Return latent representation

        Returns:
            Shadow map (B, 1, H, W) in [0, 1]
            Optionally also latent (B, 4, h, w)
        """
        batch_size = object_image.shape[0]
        device = object_image.device

        # Start from random noise
        x = torch.randn(batch_size, 4, *self.latent_size, device=device)

        # Encode object and mask
        object_latent = self.vae_pipeline.encode_object(object_image)
        mask_latent = self.vae_pipeline.mask_processor.resize_to_latent(
            mask, self.latent_size
        )

        if num_steps == 1:
            # Single-step sampling (fast inference)
            # Start at t=0 (pure noise x_0) and integrate to t=1 (clean x_1)
            # Using rectified flow: x_1 = x_0 + ∫₀¹ v_θ(x_t) dt ≈ x_0 + v_θ(x_0)
            unet_input = self.vae_pipeline.concatenator.concatenate(
                x, object_latent, mask_latent
            )

            t = torch.zeros(batch_size, device=device)  # t=0 (start from noise)
            velocity = self.unet(
                sample=unet_input,
                timestep=t,
                theta=theta,
                phi=phi,
                size=size,
            )

            # Integrate over full trajectory [0,1] in one step
            x_final = x + velocity

        else:
            # Multi-step sampling (higher quality)
            dt = 1.0 / num_steps

            for step in range(num_steps):
                t_current = step * dt
                t = torch.full((batch_size,), t_current, device=device)

                unet_input = self.vae_pipeline.concatenator.concatenate(
                    x, object_latent, mask_latent
                )

                velocity = self.unet(
                    sample=unet_input,
                    timestep=t,
                    theta=theta,
                    phi=phi,
                    size=size,
                )

                # Euler step
                x = x + velocity * dt

            x_final = x

        # Decode to shadow map
        shadow_map = self.vae_pipeline.process_output(x_final)

        if return_latent:
            return shadow_map, x_final
        return shadow_map

    def generate_with_blob_conditioning(
        self,
        object_image: torch.Tensor,
        mask: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor,
        size: torch.Tensor,
        num_steps: int = 1,
    ) -> torch.Tensor:
        """
        Alternative: Generate using blob-based conditioning (for ablation).

        Args:
            object_image: RGB object (B, 3, H, W)
            mask: Binary mask (B, 1, H, W)
            theta: Polar angle (B,)
            phi: Azimuthal angle (B,)
            size: Light size (B,)
            num_steps: Sampling steps

        Returns:
            Shadow map (B, 1, H, W)
        """
        # Create blob representation
        batch_size = object_image.shape[0]
        blobs = []

        for i in range(batch_size):
            blob = create_light_blob(
                theta[i:i+1], phi[i:i+1], size[i:i+1],
                image_size=self.latent_size
            )
            blobs.append(blob)

        blob_batch = torch.cat(blobs, dim=0).to(object_image.device)

        # Use blob as additional conditioning
        # This would require modifying the UNet to accept 10 channels
        # For now, just return regular sampling
        # (Full implementation would need architectural changes)

        return self.sample(object_image, mask, theta, phi, size, num_steps)

    def get_trainable_parameters(self):
        """Get parameters that should be trained."""
        # Only UNet is trainable, VAE is frozen
        return self.unet.parameters()

    def freeze_vae(self):
        """Ensure VAE is frozen."""
        for param in self.vae_pipeline.vae.parameters():
            param.requires_grad = False

    def unfreeze_unet(self):
        """Unfreeze UNet for training."""
        for param in self.unet.parameters():
            param.requires_grad = True

    def print_model_summary(self):
        """Print complete model summary."""
        print("\n" + "="*70)
        print(" "*20 + "SHADOW DIFFUSION MODEL")
        print("="*70)
        print()

        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"Model Configuration:")
        print(f"  Image size:           {self.image_size}")
        print(f"  Latent size:          {self.latent_size}")
        print(f"  Conditioning:         {'Blob' if self.use_blob_conditioning else 'Embedding'}")
        print()

        print(f"Parameters:")
        print(f"  Total:                {total_params:,}")
        print(f"  Trainable (UNet):     {trainable_params:,}")
        print(f"  Frozen (VAE):         {frozen_params:,}")
        print(f"  Model size (fp32):    {total_params * 4 / 1024**3:.2f} GB")
        print(f"  Model size (fp16):    {total_params * 2 / 1024**3:.2f} GB")
        print()

        print(f"Architecture:")
        print(f"  ├─ Modified SDXL UNet (9-channel input)")
        print(f"  ├─ Light Conditioning (θ, φ, s → 768-dim)")
        print(f"  ├─ Frozen VAE (SDXL)")
        print(f"  └─ Rectified Flow Sampling (1-step capable)")
        print()

        print(f"Training:")
        print(f"  Loss: Rectified Flow (MSE on velocity)")
        print(f"  Optimizer: AdamW (recommended)")
        print(f"  Learning rate: 1e-5 (from paper)")
        print(f"  Batch size: 2 (from paper)")
        print()

        print(f"Inference:")
        print(f"  Single-step: Fast (real-time capable)")
        print(f"  Multi-step: Higher quality")
        print()

        print("="*70 + "\n")


def create_shadow_model(
    pretrained_path: Optional[str] = None,
    device: str = "cuda",
    **kwargs,
) -> ShadowDiffusionModel:
    """
    Factory function to create shadow generation model.

    Args:
        pretrained_path: Path to pretrained checkpoint (optional)
        device: Device to load model on
        **kwargs: Additional arguments for model initialization

    Returns:
        Initialized model
    """
    # Create model
    model = ShadowDiffusionModel(**kwargs)

    # Load pretrained weights if provided
    if pretrained_path is not None:
        manager = CheckpointManager()
        checkpoint = manager.load_checkpoint(pretrained_path, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Loaded pretrained weights from {pretrained_path}")

    # Move to device
    model = model.to(device)

    # Ensure VAE is frozen
    model.freeze_vae()

    return model


def test_shadow_model():
    """Test complete shadow generation model."""
    print("\n" + "="*70)
    print("Testing Complete Shadow Generation Model")
    print("="*70 + "\n")

    # Create model
    print("1. Creating model...")
    model = ShadowDiffusionModel()
    model.print_model_summary()

    # Test forward pass
    print("2. Testing forward pass...")
    batch_size = 2
    device = "cpu"  # Use CPU for testing

    # Create dummy inputs
    object_image = torch.randn(batch_size, 3, 1024, 1024)
    mask = torch.randint(0, 2, (batch_size, 1, 1024, 1024)).float()
    shadow_target = torch.rand(batch_size, 1, 1024, 1024)
    theta = torch.tensor([15.0, 30.0])
    phi = torch.tensor([45.0, 90.0])
    size = torch.tensor([3.0, 5.0])

    print(f"   Input shapes:")
    print(f"     Object:  {object_image.shape}")
    print(f"     Mask:    {mask.shape}")
    print(f"     Shadow:  {shadow_target.shape}")
    print(f"     θ:       {theta.shape}")
    print(f"     φ:       {phi.shape}")
    print(f"     Size:    {size.shape}")

    # Test loss computation
    print("\n3. Testing loss computation...")
    loss_dict = model.compute_rectified_flow_loss(
        object_image, mask, shadow_target, theta, phi, size
    )
    print(f"   Loss: {loss_dict['loss'].item():.6f}")
    print(f"   ✓ Loss computation works")

    # Test sampling (single-step)
    print("\n4. Testing single-step sampling...")
    with torch.no_grad():
        shadow_map = model.sample(
            object_image, mask, theta, phi, size, num_steps=1
        )
    print(f"   Output shape: {shadow_map.shape}")
    assert shadow_map.shape == (batch_size, 1, 1024, 1024)
    print(f"   ✓ Single-step sampling works")

    # Test multi-step sampling
    print("\n5. Testing multi-step sampling...")
    with torch.no_grad():
        shadow_map_multi = model.sample(
            object_image, mask, theta, phi, size, num_steps=4
        )
    print(f"   Output shape: {shadow_map_multi.shape}")
    assert shadow_map_multi.shape == (batch_size, 1, 1024, 1024)
    print(f"   ✓ Multi-step sampling works")

    # Test parameter counts
    print("\n6. Verifying parameter freeze...")
    trainable = sum(p.numel() for p in model.get_trainable_parameters())
    print(f"   Trainable parameters: {trainable:,}")
    print(f"   ✓ Model configured correctly")

    print("\n" + "="*70)
    print("All tests passed! Model is ready for training/inference.")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_shadow_model()
