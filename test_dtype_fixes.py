"""
Test script to verify all dtype and architecture fixes.

Verifies:
1. VAE FP16 conversion works correctly
2. Forward pass runs without dtype mismatches
3. Loss computation works in mixed precision
4. Sampling works correctly
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from controllable_shadow.models import ShadowDiffusionModel


def test_dtype_flow():
    """Test that dtypes flow correctly through the model."""
    print("\n" + "="*70)
    print("TESTING DTYPE FIXES")
    print("="*70 + "\n")

    # Create model
    print("1. Creating model...")
    model = ShadowDiffusionModel(
        image_size=(512, 512),  # Smaller size for testing
    )
    model = model.cuda()
    model.eval()
    print("   ✓ Model created\n")

    # Create dummy inputs
    batch_size = 2
    print("2. Creating dummy inputs...")
    object_image = torch.randn(batch_size, 3, 512, 512).cuda()
    mask = torch.ones(batch_size, 1, 512, 512).cuda()
    shadow_target = torch.rand(batch_size, 1, 512, 512).cuda()
    theta = torch.tensor([15.0, 30.0]).cuda()
    phi = torch.tensor([45.0, 90.0]).cuda()
    size = torch.tensor([3.0, 5.0]).cuda()

    print(f"   Object image: {object_image.shape}, dtype={object_image.dtype}")
    print(f"   Mask: {mask.shape}, dtype={mask.dtype}")
    print(f"   Shadow target: {shadow_target.shape}, dtype={shadow_target.dtype}")
    print(f"   Light params: θ={theta}, φ={phi}, s={size}\n")

    # Test VAE encoding
    print("3. Testing VAE encoding (should return FP16)...")
    with torch.no_grad():
        object_latent = model.vae_pipeline.encode_object(object_image)
        shadow_latent = model.vae_pipeline.encode_shadow(shadow_target)

    print(f"   Object latent: {object_latent.shape}, dtype={object_latent.dtype}")
    print(f"   Shadow latent: {shadow_latent.shape}, dtype={shadow_latent.dtype}")

    assert object_latent.dtype == torch.float16, f"Expected FP16, got {object_latent.dtype}"
    assert shadow_latent.dtype == torch.float16, f"Expected FP16, got {shadow_latent.dtype}"
    print("   ✓ VAE correctly returns FP16\n")

    # Test loss computation
    print("4. Testing loss computation...")
    model.train()
    loss_dict = model.compute_rectified_flow_loss(
        object_image, mask, shadow_target, theta, phi, size
    )

    loss = loss_dict['loss']
    print(f"   Loss: {loss.item():.6f}")
    print(f"   Loss dtype: {loss.dtype}")
    print(f"   Predicted velocity dtype: {loss_dict['predicted_velocity'].dtype}")
    print(f"   Target velocity dtype: {loss_dict['target_velocity'].dtype}")

    assert loss.dtype == torch.float32, f"Expected loss in FP32, got {loss.dtype}"
    assert loss_dict['predicted_velocity'].dtype == torch.float16, "Velocity should be FP16"
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    print("   ✓ Loss computation works correctly\n")

    # Test sampling
    print("5. Testing single-step sampling...")
    model.eval()
    with torch.no_grad():
        shadow_map = model.sample(
            object_image, mask, theta, phi, size, num_steps=1
        )

    print(f"   Shadow map: {shadow_map.shape}, dtype={shadow_map.dtype}")
    print(f"   Shadow range: [{shadow_map.min():.3f}, {shadow_map.max():.3f}]")

    assert shadow_map.shape == (batch_size, 1, 512, 512), f"Wrong shape: {shadow_map.shape}"
    assert shadow_map.min() >= 0 and shadow_map.max() <= 1, "Shadow map not in [0,1]"
    print("   ✓ Sampling works correctly\n")

    # Test multi-step sampling
    print("6. Testing multi-step sampling (4 steps)...")
    with torch.no_grad():
        shadow_map_multi = model.sample(
            object_image, mask, theta, phi, size, num_steps=4
        )

    print(f"   Shadow map: {shadow_map_multi.shape}")
    assert shadow_map_multi.shape == (batch_size, 1, 512, 512)
    print("   ✓ Multi-step sampling works\n")

    # Test backward pass
    print("7. Testing backward pass...")
    loss_dict = model.compute_rectified_flow_loss(
        object_image, mask, shadow_target, theta, phi, size
    )
    loss = loss_dict['loss']
    loss.backward()

    # Check gradients exist
    has_grads = sum(1 for p in model.get_trainable_parameters() if p.grad is not None)
    total_params = sum(1 for p in model.get_trainable_parameters())

    print(f"   Parameters with gradients: {has_grads}/{total_params}")
    assert has_grads == total_params, "Some parameters didn't get gradients!"
    print("   ✓ Backward pass works\n")

    print("="*70)
    print("ALL DTYPE TESTS PASSED! ✅")
    print("="*70)
    print()
    print("Summary:")
    print("  ✅ VAE correctly converts FP32 → FP16")
    print("  ✅ Loss computation works in mixed precision")
    print("  ✅ Forward pass has no dtype mismatches")
    print("  ✅ Sampling produces valid shadow maps")
    print("  ✅ Gradients flow correctly")
    print()


def test_memory_usage():
    """Estimate memory usage."""
    print("\n" + "="*70)
    print("MEMORY USAGE ESTIMATE")
    print("="*70 + "\n")

    model = ShadowDiffusionModel(image_size=(512, 512)).cuda()

    # Get memory before forward
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Dummy forward pass
    batch_size = 2
    object_image = torch.randn(batch_size, 3, 512, 512).cuda()
    mask = torch.ones(batch_size, 1, 512, 512).cuda()
    shadow_target = torch.rand(batch_size, 1, 512, 512).cuda()
    theta = torch.tensor([15.0, 30.0]).cuda()
    phi = torch.tensor([45.0, 90.0]).cuda()
    size = torch.tensor([3.0, 5.0]).cuda()

    loss_dict = model.compute_rectified_flow_loss(
        object_image, mask, shadow_target, theta, phi, size
    )
    loss = loss_dict['loss']
    loss.backward()

    # Get memory after
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    print(f"Peak memory usage (512×512, batch=2): {peak_memory_mb:.1f} MB")
    print()

    # Estimate for 1024×1024
    # Memory scales with H^2 * W^2 for images, H * W for latents
    # 1024×1024 is 4× larger in each dimension
    # Latents are 128×128 vs 64×64 (4× total pixels)
    # But images are 1024×1024 vs 512×512 (also 4× total)
    estimated_1024 = peak_memory_mb * 4
    print(f"Estimated memory (1024×1024, batch=2): {estimated_1024:.1f} MB = {estimated_1024/1024:.2f} GB")
    print(f"Estimated memory (1024×1024, batch=8): {estimated_1024*4:.1f} MB = {estimated_1024*4/1024:.2f} GB")
    print()

    if estimated_1024 / 1024 < 20:
        print("✅ Should fit in A100 40GB with headroom")
    else:
        print("⚠️  Might be tight on A100 40GB")

    print()


if __name__ == "__main__":
    # Run tests
    test_dtype_flow()
    test_memory_usage()

    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED - READY FOR TRAINING")
    print("="*70 + "\n")
