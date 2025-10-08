"""
Centralized dtype configuration for the entire pipeline.

Use this instead of manual dtype conversions throughout the code.
"""

import torch
from typing import Optional
from contextlib import contextmanager


class DtypeConfig:
    """
    Single source of truth for all dtype decisions.

    Usage:
        # Initialize model
        model = ShadowDiffusionModel()
        model = model.to(device='cuda', dtype=DtypeConfig.MODEL_DTYPE)

        # VAE stays in FP32 (set once in VAEWrapper)
        # Everything else uses AMP automatically
    """

    # Model dtypes
    MODEL_DTYPE = torch.float32  # Default model dtype (weights)

    # Component-specific overrides (set once at initialization)
    VAE_DTYPE = torch.float32          # VAE always FP32 (stability)
    CONDITIONING_DTYPE = torch.float32 # Conditioning always FP32 (precision)

    # Training configuration
    USE_AMP = True  # Use Automatic Mixed Precision
    AMP_DTYPE = torch.float16  # AMP uses FP16

    @classmethod
    def set_amp_enabled(cls, enabled: bool):
        """Enable or disable AMP globally."""
        cls.USE_AMP = enabled

    @classmethod
    @contextmanager
    def autocast(cls):
        """
        Context manager for automatic mixed precision.

        Usage:
            with DtypeConfig.autocast():
                output = model(input)  # Automatically uses FP16 where beneficial
        """
        if cls.USE_AMP:
            with torch.cuda.amp.autocast(dtype=cls.AMP_DTYPE):
                yield
        else:
            yield  # No-op if AMP disabled

    @classmethod
    @contextmanager
    def force_fp32(cls):
        """
        Force FP32 for sensitive operations (VAE, conditioning).

        Usage:
            with DtypeConfig.force_fp32():
                embedding = compute_sinusoidal_encoding(x)  # Always FP32
        """
        with torch.cuda.amp.autocast(enabled=False):
            yield


# Example training loop using DtypeConfig
def example_training_loop():
    """
    Example showing how to use DtypeConfig in training.

    This completely eliminates manual dtype conversions.
    """
    from torch.cuda.amp import GradScaler

    # Setup
    model = None  # Your model
    optimizer = None  # Your optimizer
    dataloader = None  # Your dataloader

    # Create gradient scaler for AMP
    scaler = GradScaler(enabled=DtypeConfig.USE_AMP)

    # Training loop
    for batch in dataloader:
        optimizer.zero_grad()

        # Forward pass with AMP
        with DtypeConfig.autocast():
            loss = model(batch)

        # Backward pass (AMP handles scaling)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # That's it! No manual dtype conversions needed.


if __name__ == "__main__":
    print("Dtype Configuration")
    print("=" * 60)
    print(f"Model dtype:        {DtypeConfig.MODEL_DTYPE}")
    print(f"VAE dtype:          {DtypeConfig.VAE_DTYPE}")
    print(f"Conditioning dtype: {DtypeConfig.CONDITIONING_DTYPE}")
    print(f"Use AMP:            {DtypeConfig.USE_AMP}")
    print(f"AMP dtype:          {DtypeConfig.AMP_DTYPE}")
    print("=" * 60)
    print()
    print("Key Principles:")
    print("1. NO manual .half() or .to(dtype) calls in model code")
    print("2. Use DtypeConfig.autocast() for training/inference")
    print("3. Use DtypeConfig.force_fp32() for VAE/conditioning only")
    print("4. Let PyTorch AMP handle everything else automatically")
