"""
SDXL-based UNet architecture for shadow generation.

Modified SDXL UNet:
- Cross-attention blocks removed (no text conditioning)
- First conv layer modified: 4ch -> 9ch (noise + object + mask)
- New weights zero-initialized for transfer learning
- Conditioning via light parameters instead of text
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers.models.attention_processor import AttnProcessor
import copy


class SDXLUNetForShadows(nn.Module):
    """
    SDXL UNet modified for shadow generation.

    Key modifications:
    1. Remove cross-attention blocks (no text conditioning)
    2. Modify input conv: 4ch -> 9ch (noise + object VAE + mask)
    3. Zero-initialize new conv weights
    4. Load SDXL pretrained weights for existing layers
    """

    def __init__(
        self,
        pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        in_channels: int = 9,
        out_channels: int = 4,
        conditioning_dim: int = 768,
    ):
        """
        Initialize modified SDXL UNet.

        Args:
            pretrained_model_name: HuggingFace model ID for SDXL
            in_channels: Input channels (9 = 4 noise + 4 object + 1 mask)
            out_channels: Output channels (4 for VAE latent space)
            conditioning_dim: Light parameter embedding dimension
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conditioning_dim = conditioning_dim
        self.pretrained_model_name = pretrained_model_name

        # Load base SDXL UNet (default dtype)
        print(f"Loading SDXL UNet from {pretrained_model_name}...")
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            torch_dtype=torch.float16
        )

        # Store original config
        self.config = self.unet.config

        # Modify architecture
        self._remove_cross_attention()
        self._modify_input_conv()

        print("✓ SDXL UNet loaded and modified successfully!")

    def _remove_cross_attention(self):
        """
        Remove cross-attention blocks from UNet.

        Cross-attention was used for text conditioning in original SDXL.
        We replace it with identity/no-op processors since we use scalar
        parameter conditioning injected via timestep embeddings.
        """
        print("Disabling cross-attention blocks...")

        # Create no-op processor for cross-attention
        class NoOpAttnProcessor:
            """No-op attention processor that returns input unchanged."""
            def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
                # Simply return the hidden states without any attention
                # This effectively removes cross-attention computation
                return hidden_states

        # Replace attention processors
        attn_procs = {}
        cross_attn_count = 0
        self_attn_count = 0

        for name, processor in self.unet.attn_processors.items():
            if "attn2" in name:  # attn2 is cross-attention
                # Replace with no-op to skip computation
                attn_procs[name] = NoOpAttnProcessor()
                cross_attn_count += 1
            else:  # attn1 is self-attention (keep these)
                attn_procs[name] = AttnProcessor()
                self_attn_count += 1

        self.unet.set_attn_processor(attn_procs)

        print(f"✓ Cross-attention disabled: {cross_attn_count} modules")
        print(f"✓ Self-attention active: {self_attn_count} modules")

    def _modify_input_conv(self):
        """
        Modify first convolution layer for 9-channel input.

        Original SDXL: 4 channels (latent noise)
        Modified: 9 channels (4 noise + 4 object + 1 mask)

        Strategy:
        - Create new conv layer with 9 input channels
        - Copy weights for first 4 channels from pretrained SDXL
        - Zero-initialize weights for additional 5 channels
        """
        print("Modifying input convolution layer...")

        # Get original conv_in layer
        original_conv = self.unet.conv_in
        original_weight = original_conv.weight.data  # Shape: [out_ch, 4, k, k]
        original_bias = original_conv.bias.data if original_conv.bias is not None else None

        # Create new conv layer with 9 input channels
        new_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )

        # Initialize new weights
        with torch.no_grad():
            # Zero-initialize all weights
            nn.init.zeros_(new_conv.weight)

            # Copy pretrained weights for first 4 channels
            new_conv.weight[:, :4, :, :] = original_weight

            # Additional 5 channels remain zero-initialized
            # This ensures the model starts with SDXL behavior

            # Copy bias if it exists
            if original_bias is not None:
                new_conv.bias.data = original_bias

        # Replace the conv_in layer
        self.unet.conv_in = new_conv

        print(f"Input conv modified: 4ch -> {self.in_channels}ch")
        print(f"Pretrained weights copied for first 4 channels")
        print(f"New channels ({self.in_channels - 4}) zero-initialized")

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass through modified UNet.

        Args:
            sample: Latent input (B, 9, H, W) - concatenated [noise, object, mask]
            timestep: Timestep (B,) or scalar
            encoder_hidden_states: Light parameter embeddings (B, 768)
                This replaces text embeddings in original SDXL
            added_cond_kwargs: Additional conditioning (text_embeds, time_ids for SDXL)
            return_dict: Whether to return dict or tensor

        Returns:
            Predicted noise/output (B, 4, H, W)
        """
        # Note: encoder_hidden_states is used for cross-attention in original SDXL
        # Since we removed cross-attention, this won't be used by attn2 blocks
        # But we keep the parameter for API compatibility

        # The actual conditioning happens via timestep embeddings
        # (light parameters are added to timestep embeddings before passing to UNet)

        output = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=return_dict,
        )

        if return_dict:
            return output.sample
        return output

    def get_trainable_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get parameters that should be trained.

        Returns:
            Dictionary of parameter names to parameters
        """
        trainable = {}
        for name, param in self.named_parameters():
            # All UNet parameters are trainable
            # VAE will be frozen separately
            trainable[name] = param
        return trainable

    def get_new_parameters(self) -> Dict[str, torch.nn.Parameter]:
        """
        Get only the newly initialized parameters (not from SDXL pretrained).

        Returns:
            Dictionary of new parameter names to parameters
        """
        new_params = {}

        # The new parameters are in conv_in for channels 4-8
        conv_in_weight = self.unet.conv_in.weight
        # We can't easily separate them, so return full conv_in
        new_params['conv_in.weight_new_channels'] = conv_in_weight[:, 4:, :, :]

        return new_params

    def print_architecture_summary(self):
        """Print summary of model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "="*60)
        print("SDXL UNet Architecture Summary")
        print("="*60)
        print(f"Input channels:       {self.in_channels}")
        print(f"Output channels:      {self.out_channels}")
        print(f"Conditioning dim:     {self.conditioning_dim}")
        print(f"Total parameters:     {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Model size:           {total_params * 2 / 1024**2:.2f} MB (fp16)")
        print("="*60)
        print("\nKey modifications:")
        print("  ✓ Cross-attention blocks disabled")
        print("  ✓ Input conv: 4ch -> 9ch")
        print("  ✓ New channels zero-initialized")
        print("  ✓ SDXL pretrained weights loaded")
        print("="*60 + "\n")


class VAEWrapper(nn.Module):
    """
    Wrapper for SDXL VAE encoder/decoder.

    VAE is frozen during training - only used for encoding/decoding.
    Always uses FP32 for numerical stability.
    """

    def __init__(
        self,
        pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
    ):
        """
        Initialize VAE wrapper.

        Args:
            pretrained_model_name: HuggingFace model ID for SDXL
        """
        super().__init__()

        # Load VAE in FP32 for numerical stability
        print(f"Loading SDXL VAE from {pretrained_model_name}...")
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name,
            subfolder="vae",
            torch_dtype=torch.float32,
        )

        # Freeze VAE weights
        for param in self.vae.parameters():
            param.requires_grad = False

        self.vae.eval()

        print("✓ SDXL VAE loaded and frozen (FP32 for stability)")

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent space.

        Args:
            images: RGB images (B, 3, H, W) in range [-1, 1]

        Returns:
            Latents (B, 4, H//8, W//8)
        """
        # Ensure VAE is in eval mode
        self.vae.eval()

        # Encode (VAE handles dtype internally)
        latent_dist = self.vae.encode(images).latent_dist
        latents = latent_dist.sample()

        # SDXL uses scaling factor
        latents = latents * self.vae.config.scaling_factor

        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images.

        Args:
            latents: Latents (B, 4, H, W)

        Returns:
            RGB images (B, 3, H*8, W*8) in range [-1, 1]
        """
        # Ensure VAE is in eval mode
        self.vae.eval()

        # Unscale latents
        latents = latents / self.vae.config.scaling_factor

        # Decode (VAE handles dtype internally)
        images = self.vae.decode(latents).sample

        return images

    def print_vae_summary(self):
        """Print VAE information."""
        print("\n" + "="*60)
        print("SDXL VAE Summary")
        print("="*60)
        print(f"Latent channels:      4")
        print(f"Scaling factor:       {self.vae.config.scaling_factor}")
        print(f"Downsampling factor:  8")
        print(f"Input resolution:     (B, 3, H, W)")
        print(f"Latent resolution:    (B, 4, H//8, W//8)")
        print(f"Frozen:               Yes")
        print("="*60 + "\n")


def test_sdxl_components():
    """Test function to verify SDXL components load correctly."""
    print("\n" + "="*60)
    print("Testing SDXL Components")
    print("="*60 + "\n")

    # Test UNet
    print("1. Testing UNet...")
    unet = SDXLUNetForShadows()
    unet.print_architecture_summary()

    # Test VAE
    print("\n2. Testing VAE...")
    vae = VAEWrapper()
    vae.print_vae_summary()

    # Test forward pass with dummy data
    print("3. Testing forward pass with dummy data...")
    batch_size = 1
    latent_h, latent_w = 128, 128

    # Create dummy inputs
    dummy_sample = torch.randn(batch_size, 9, latent_h, latent_w)
    dummy_timestep = torch.tensor([500])
    dummy_conditioning = torch.randn(batch_size, 77, 768)  # Dummy shape

    print(f"   Input shape:       {dummy_sample.shape}")
    print(f"   Timestep:          {dummy_timestep.shape}")
    print(f"   Conditioning:      {dummy_conditioning.shape}")

    # Forward pass
    with torch.no_grad():
        output = unet(dummy_sample, dummy_timestep, dummy_conditioning)

    print(f"   Output shape:      {output.shape}")
    print(f"   ✓ Forward pass successful!")

    # Test VAE encode/decode
    print("\n4. Testing VAE encode/decode...")
    dummy_image = torch.randn(batch_size, 3, 1024, 1024)
    print(f"   Input image:       {dummy_image.shape}")

    latents = vae.encode(dummy_image)
    print(f"   Encoded latents:   {latents.shape}")

    reconstructed = vae.decode(latents)
    print(f"   Decoded image:     {reconstructed.shape}")
    print(f"   ✓ VAE encode/decode successful!")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run tests when module is executed directly
    test_sdxl_components()
