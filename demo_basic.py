"""
Basic Demo Script for Controllable Shadow Generation

Demonstrates generating shadows from object images with different light parameters.
Includes visualization and saves results.

Usage:
    python demo_basic.py --input your_object.png
    python demo_basic.py --input your_object.png --theta 15 --phi 270 --size 6
"""

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import time

from controllable_shadow.models import create_shadow_model
from controllable_shadow.utils.debugging import MemoryProfiler


def demo_generate_shadow(model, object_image, mask, params, output_dir):
    """Generate and visualize shadow with given parameters."""
    
    theta, phi, size = params
    theta_tensor = torch.tensor([theta], device=model.device)
    phi_tensor = torch.tensor([phi], device=model.device)
    size_tensor = torch.tensor([size], device=model.device)
    
    print(f"  Generating with θ={theta}°, φ={phi}°, s={size}")
    
    # Generate shadow
    start_time = time.time()
    with torch.no_grad():
        shadow_map = model.sample(
            object_image, mask, theta_tensor, phi_tensor, size_tensor, num_steps=1
        )
    gen_time = time.time() - start_time
    
    print(f"  ✓ Generated in {gen_time:.3f}s")
    
    return shadow_map.cpu(), gen_time


def create_visualization(object_image, mask, shadow_maps, params_list, output_path):
    """Create comprehensive visualization of results."""
    
    n_examples = len(params_list)
    fig_height = max(3, n_examples * 2.5)
    
    fig, axes = plt.subplots(4, n_examples, figsize=(4*n_examples, fig_height))
    if n_examples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (shadow_map, (theta, phi, size)) in enumerate(zip(shadow_maps, params_list)):
        
        # Object image
        obj_np = (object_image[0].cpu() + 1) / 2
        obj_np = obj_np.permute(1, 2, 0).numpy()
        axes[0, i].imshow(obj_np)
        axes[0, i].set_title(f"Object Image\nθ={theta}°, φ={phi}°, s={size}")
        axes[0, i].axis('off')
        
        # Mask
        mask_np = mask[0, 0].cpu().numpy()
        axes[1, i].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title("Object Mask")
        axes[1, i].axis('off')
        
        # Shadow map
        axes[2, i].imshow(shadow_map[0, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[2, i].set_title("Generated Shadow\n(Map Only)")
        axes[2, i].axis('off')
        
        # Composite (object + shadow)
        composite = obj_np.copy()
        shadow_alpha = shadow_map[0, 0].numpy()
        # Apply shadow as darkening effect
        mask_region = mask_np > 0.5
        for c in range(3):
            composite[mask_region, c] *= (1 - shadow_alpha[mask_region] * 0.7)
        
        axes[3, i].imshow(np.clip(composite, 0, 1))
        axes[3, i].set_title("Shadow Composite\n(Object + Shadow)")
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved: {output_path}")


def run_different_light_conditions(model, object_image, mask, output_dir):
    """Demonstrate different light parameter combinations."""
    
    print("\n🔆 Demonstrating Different Light Conditions:")
    
    # Define interesting parameter combinations
    light_scenarios = [
        # "Early morning" scenario
        (15, 90, 3, "Early Morning\n(Sun from East)"),
        
        # "Late afternoon" scenario  
        (35, 270, 5, "Late Afternoon\n(Sun from West)"),
        
        # "Overhead light"
        (5, 0, 2, "Overhead Light\n(Hard Shadows)"),
        
        # "Diffuse/cloudy"
        (25, 180, 8, "Diffuse Light\n(Soft Shadows)")
    ]
    
    shadow_maps = []
    params_clean = []
    gen_times = []
    
    for theta, phi, size, description in light_scenarios:
        print(f"\n  Scenario: {description}")
        shadow_map, gen_time = demo_generate_shadow(
            model, object_image, mask, (theta, phi, size), output_dir
        )
        shadow_maps.append(shadow_map)
        params_clean.append((theta, phi, size))
        gen_times.append(gen_time)
    
    avg_time = np.mean(gen_times)
    print(f"\n📊 Performance Summary:")
    print(f"  Average generation time: {avg_time:.3f}s per shadow")
    print(f"  Total scenarios: {len(light_scenarios)}")
    print(f"  Total time: {np.sum(gen_times):.3f}s")
    
    # Create visualization
    vis_path = output_dir / "light_scenarios_demo.png"
    create_visualization(object_image, mask, shadow_maps, params_clean, vis_path)
    
    return shadow_maps, params_clean


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Basic Controllable Shadow Demo")
    
    # Input
    parser.add_argument("--input", type=str, required=True,
                       help="Path to object image")
    parser.add_argument("--mask", type=str, default=None,
                       help="Path to mask (optional, auto-generated if not provided)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./demo_output",
                       help="Output directory")
    
    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    
    # Custom parameters (optional)
    parser.add_argument("--theta", type=float, default=None,
                       help="Custom theta angle")
    parser.add_argument("--phi", type=float, default=None,
                       help="$Custom phi angle")
    parser.add_argument("--size", type=float, default=None,
                       help="Custom light size")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🌟 BASIC CONTROLLABLE SHADOW DEMO")
    print("="*70)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Memory profiler
    profiler = MemoryProfiler()
    profiler.snapshot("Startup")
    
    # Load model
    print("\n🤖 Loading Model...")
    start_load = time.time()
    model = create_shadow_model(
        pretrained_path=args.checkpoint,
        device=args.device,
    )
    model.eval()
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.3f}s")
    profiler.snapshot("Model loaded")
    
    # Load and prepare image
    print(f"\n📁 Loading Image: {args.input}")
    
    # Load image
    img = Image.open(args.input).convert('RGB')
    transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    object_image = transform(img).unsqueeze(0).to(args.device)
    
    # Create mask
    if args.mask:
        mask_img = Image.open(args.mask).convert('L')
        mask_transform = T.Compose([
            T.Resize([1024, 1024]),
            T.ToTensor(),
        ])
        mask = mask_transform(mask_img).unsqueeze(0).to(args.device)
        mask = (mask > 0.5).float()
        print("✓ Custom mask loaded")
    else:
        # Auto-generate mask
        img_denorm = (object_image + 1) / 2
        brightness = img_denorm.mean(dim=1, keepdim=True)
        mask = (brightness < 0.9).float()
        print("✓ Auto-generated mask from image")
    
    print(f"✓ Image loaded: {object_image.shape}")
    profiler.snapshot("Image loaded")
    
    # Custom generation if parameters specified
    if all([args.theta is not None, args.phi is not None, args.size is not None]):
        print(f"\n🎯 Custom Generation:")
        print(f"  θ={args.theta}°, φ={args.phi}°, s={args.size}")
        
        shadow_map, gen_time = demo_generate_shadow(
            model, object_image, mask, 
            (args.theta, args.phi, args.size), output_dir
        )
        
        # Save custom shadow
        shadow_path = output_dir / "custom_shadow.png"
        shadow_np = shadow_map[0, 0].numpy()
        shadow_img = Image.fromarray((shadow_np * 255).astype('uint8'), mode='L')
        shadow_img.save(shadow_path)
        print(f"✓ Custom shadow saved: {shadow_path}")
        
        # Single visualization
        create_visualization(
            object_image, mask, [shadow_map], 
            [(args.theta, args.phi, args.size)], 
            output_dir / "custom_demo.png"
        )
    
    # Demonstration scenarios
    run_different_light_conditions(model, object_image, mask, output_dir)
    
    # Performance summary
    profiler.snapshot("Demo complete")
    print(f"\n📊 Memory Usage Summary:")
    profiler.print_summary()
    
    print(f"\n📁 All outputs saved to: {output_dir}")
    print("\n" + "="*70)
    print("🎉 DEMO COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

