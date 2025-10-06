#!/usr/bin/env python3
"""
Test script to verify the NaN fix for light projection layer.
"""

import torch
import torch.nn as nn
from controllable_shadow.models import ShadowDiffusionModel

def test_light_projection_stability():
    """Test that light projection layer doesn't produce NaN."""
    print("Testing light projection layer stability...")
    
    # Create model
    model = ShadowDiffusionModel(
        conditioning_strategy="additive",
        image_size=(512, 512),
    )
    
    # Create dummy inputs
    batch_size = 2
    object_image = torch.randn(batch_size, 3, 512, 512)
    mask = torch.randint(0, 2, (batch_size, 1, 512, 512)).float()
    shadow_target = torch.rand(batch_size, 1, 512, 512)
    theta = torch.tensor([15.0, 30.0])
    phi = torch.tensor([45.0, 90.0])
    size = torch.tensor([3.0, 5.0])
    
    print(f"Input shapes:")
    print(f"  Object:  {object_image.shape}")
    print(f"  Mask:    {mask.shape}")
    print(f"  Shadow:  {shadow_target.shape}")
    print(f"  θ:       {theta.shape}")
    print(f"  φ:       {phi.shape}")
    print(f"  Size:    {size.shape}")
    
    # Test multiple forward passes
    model.train()
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-5)
    
    for step in range(5):
        print(f"\nStep {step + 1}:")
        
        # Forward pass
        loss_dict = model.compute_rectified_flow_loss(
            object_image, mask, shadow_target, theta, phi, size
        )
        
        loss = loss_dict['loss']
        print(f"  Loss: {loss.item():.6f}")
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"  ❌ NaN detected in loss!")
            return False
        
        # Check light projection weights
        light_proj_weight = model.unet.light_projection.linear.weight
        if torch.isnan(light_proj_weight).any():
            print(f"  ❌ NaN detected in light projection weights!")
            return False
        
        print(f"  ✅ No NaN detected")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        light_proj_grad = model.unet.light_projection.linear.weight.grad
        if light_proj_grad is not None and torch.isnan(light_proj_grad).any():
            print(f"  ❌ NaN detected in light projection gradients!")
            return False
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"\n✅ All tests passed! Light projection layer is stable.")
    return True

if __name__ == "__main__":
    success = test_light_projection_stability()
    if success:
        print("\n🎉 NaN fix is working correctly!")
    else:
        print("\n❌ NaN fix needs more work.")
