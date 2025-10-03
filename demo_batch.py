"""
Batch Processing Demo for Controllable Shadow Generation

Demonstrates processing multiple images with varying light parameters.
Shows efficiency gains from batch processing.

Usage:
    python demo_batch.py --input_dir ./test_images
    python demo_batch.py --images img1.png img2.png img3.png --batch_size 4
"""

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import time
from typing import List, Tuple
import concurrent.futures
from tqdm import tqdm

from controllable_shadow.models import create_shadow_model
from controllable_shadow.utils.debugging import MemoryProfiler, PerformanceProfiler


def load_images_batch(image_paths: List[str], device: str, target_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess a batch of images.
    
    Args:
        image_paths: List of image file paths
        device: Target device
        target_size: Resize target size
        
    Returns:
        Tuple of (images, masks) tensors
    """
    batch_size = len(image_paths)
    images = torch.zeros((batch_size, 3, target_size, target_size), device=device)
    masks = torch.zeros((batch_size, 1, target_size, target_size), device=device)
    
    transform = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).to(device)
        images[i] = img_tensor
        
        # Create mask
        img_denorm = (img_tensor + 1) / 2
        brightness = img_denorm.mean(dim=0)
      
        mask = (brightness < 0.9).float().unsqueeze(0)
        masks[i] = mask
    
    return images, masks


def generate_batch_shadows(model, images, masks, light_params, num_steps: int = 1) -> torch.Tensor:
    """
    Generate shadows for a batch of images.
    
    Args:
        model: Shadow generation model
        images: Batch of images (B, 3, H, W)
        masks: Batch of masks (B, 1, H, W)
        light_params: Dict with batch parameters
        
    Returns:
        Generated shadow maps (B, 1, H, W)
    """
    batch_size = images.shape[0]
    
    with torch.no_grad():
        shadow_maps = model.sample(
            images,
            masks,
            light_params['theta'],
            light_params['phi'],
            light_params['size'],
            num_steps=num_steps
        )
    
    return shadow_maps


def benchmark_batch_sizes(model, input_images, input_masks, device, output_dir):
    """
    Benchmark different batch sizes to show efficiency gains.
    """
    print("\n⚡ Batch Size Benchmark:")
    
    batch_sizes = [1, 2, 4]  # Test different batch sizes
    num_images = 8  # Total images to process
    
    # Prepare test images by repeating first image
    test_images = input_images[:1].repeat(num_images, 1, 1, 1).to(device)
    test_masks = input_masks[:1].repeat(num_images, 1, 1, 1).to(device)
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n  Testing batch size: {batch_size}")
        
        # Prepare batch parameters
        theta = torch.tensor([30.0] * batch_size, device=device)
        phi = torch.tensor([45.0] * batch_size, device=device)
        size = torch.tensor([4.0] * batch_size, device=device)
        
        light_params = {
            'theta': theta,
            'phi': phi,
            'size': size
        }
        
        times = []
        
        # Warmup
        for _ in range(2):
            _ = generate_batch_shadows(model, test_images[:batch_size], test_masks[:batch_size], 
                                    {'theta': theta, 'phi': phi, 'size': size})
        
        # Benchmark runs
        for _ in range(5):
            start_time = time.time()
            
            # Process in batches
            total_shadows = 0
            for start_idx in range(0, num_images, batch_size):
                end_idx = min(start_idx + batch_size, num_images)
                batch_images = test_images[start_idx:end_idx]
                batch_masks = test_masks[start_idx:end_idx]
                
                # Adjust parameters for batch size
                actual_batch_size = end_idx - start_idx
                batch_theta = theta[:actual_batch_size]
                batch_phi = phi[:actual_batch_size]
                batch_size_tensor = size[:actual_batch_size]
                
                _ = generate_batch_shadows(model, batch_images, batch_masks,
                                        {'theta': batch_theta, 'phi': batch_phi, 'size': batch_size_tensor})
                total_shadows += actual_batch_size
            
            elapsed = time.time() - start_time
            times.append(elapsed / total_shadows)  # Per shadow time
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results[batch_size] = avg_time
        
        print(f"  ✓ Avg time per shadow: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"    Throughput: {1/avg_time:.1f} shadows/sec")
    
    # Create benchmark visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    batch_sizes_list = list(results.keys())
    times_list = list(results.values())
    
    ax1.bar(batch_sizes_list, times_list)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time per Shadow (seconds)')
    ax1.set_title('Shadow Generation Performance by Batch Size')
    ax1.grid(True, alpha=0.3)
    
    throughput = [1/t for t in times_list]
    ax2.bar(batch_sizes_list, throughput)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (shadows/second)')
    ax2.set_title('Shadow Generation Throughput')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    benchmark_plot = output_dir / "batch_benchmark.png"
    plt.savefig(benchmark_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Benchmark plot saved: {benchmark_plot}")
    
    return results


def demo_parameter_sweep(model, images, masks, output_dir):
    """
    Generate shadows with swept parameters for multiple images.
    """
    print("\n🎚️ Parameter Sweep Demo:")
    
    batch_size = len(images)
    device = images.device
    
    # Define parameter sweep ranges
    param_configs = [
        (15, 0, 3, "Horizontal Sweep - Morning Light"),
        (30, 90, 5, "Vertical Angles - Midday"),
        (45, 180, 7, "Soft Shadows - Evening"),
        (20, 270, 4, "Different Directions")
    ]
    
    all_results = []
    
    for theta, phi, size, description in param_configs:
        print(f"\n  {description}: θ={theta}°, φ={phi}°, s={size}")
        
        theta_tensor = torch.tensor([theta] * batch_size, device=device)
        phi_tensor = torch.tensor([phi] * batch_size, device=device)
        size_tensor = torch.tensor([size] * batch_size, device=device)
        
        light_params = {
            'theta': theta_tensor,
            'phi': phi_tensor,
            'size': size_tensor
        }
        
        # Generate batch
        start_time = time.time()
        shadow_maps = generate_batch_shadows(model, images, masks, light_params)
        gen_time = time.time() - start_time
        
        print(f"  ✓ Generated {batch_size} shadows in {gen_time:.3f}s")
        
        # Save individual shadows
        for i, shadow_map in enumerate(shadow_maps):
            shadow_np = shadow_map.cpu().numpy()
            shadow_img = Image.fromarray((shadow_np[0] * 255).astype('uint8'), mode='L')
            output_path = output_dir / f"shadow_batch_{theta}_{phi}_{size}_img{i}.png"
            shadow_img.save(output_path)
        
        all_results.append((shadow_maps, light_params, description))
    
    # Create batch comparison visualization
    fig, axes = plt.subplots(batch_size, len(param_configs) + 1, figsize=(4 * (len(param_configs) + 1), 4 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    if len(param_configs) == 1:
        axes = axes.reshape(-1, 1)
    
    for img_idx in range(batch_size):
        # Original image
        img_np = (images[img_idx].cpu() + 1) / 2
        img_np = img_np.permute(1, 2, 0).numpy()
        axes[img_idx, 0].imshow(img_np)
        axes[img_idx, 0].set_title(f"Original Image {img_idx}")
        axes[img_idx, 0].axis('off')
        
        # Different parameter shadows
        for param_idx, (shadow_maps, _, description) in enumerate(all_results):
            col_idx = param_idx + 1
            shadow_np = shadow_maps[img_idx, 0].cpu().numpy()
            axes[img_idx, col_idx].imshow(shadow_np, cmap='gray', vmin=0, vmax=1)
            axes[img_idx, col_idx].set_title(f"{description}\nimg {img_idx}")
            axes[img_idx, col_idx].axis('off')
    
    plt.tight_layout()
    sweep_plot = output_dir / "parameter_sweep_batch.png"
    plt.savefig(sweep_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Parameter sweep visualization saved: {sweep_plot}")
    
    return all_results


def main():
    """Main batch demo function."""
    parser = argparse.ArgumentParser(description="Batch Shadow Generation Demo")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", type=str,
                       help="Directory containing images to process")
    group.add_argument("--images", nargs='+', type=str,
                       help="Individual image file paths")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./demo_batch_output",
                       help="Output directory")
    
    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    
    # Batch processing
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run batch size benchmarking")
    
    # Processing options
    parser.add_argument("--num_steps", type=int, default=1,
                       help="Number of sampling steps")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("📦 BATCH SHADOW GENERATION DEMO")
    print("="*70)
    
    # Collect input images
    if args.input_dir:
        input_dir = Path(args.input_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [str(p) for p in input_dir.iterdir() 
                     if p.suffix.lower() in image_extensions]
    else:
        image_paths = args.images
    
    if not image_paths:
        print("❌ No images found!")
        return
    
    print(f"\n📁 Found {len(image_paths)} images")
    
    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Profilers
    profiler = MemoryProfiler()
    perf_prof = PerformanceProfiler()
    
    profiler.snapshot("Startup")
    
    # Load model
    print(f"\n🤖 Loading model on {args.device}...")
    start_load = time.time()
    model = create_shadow_model(
        pretrained_path=args.checkpoint,
        device=args.device,
    )
    model.eval()
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.3f}s")
    
    profiler.snapshot("Model loaded")
    
    # Load images in batches
    batch_indices = list(range(0, len(image_paths), args.batch_size))
    all_shadow_maps = []
    
    print(f"\n📥 Loading images in batches of {args.batch_size}...")
    
    for batch_idx, start_idx in enumerate(batch_indices):
        end_idx = min(start_idx + args.batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        print(f"  Batch {batch_idx + 1}: Loading {len(batch_paths)} images")
        
        images, masks = load_images_batch(batch_paths, args.device)
        print(f"  ✓ Loaded: {images.shape}")
        
        # Generate shadows for this batch
        print(f"  Generating shadows...")
        
        theta = torch.tensor([30.0] * len(batch_paths), device=args.device)
        phi = torch.tensor([45.0] * len(batch_paths), device=args.device)
        size = torch.tensor([4.0] * len(batch_paths), device=args.device)
        
        light_params = {
            'theta': theta,
            'psi': phi,
            'size': size
        }
        
        with torch.no_grad():
            shadow_maps = model.sample(
                images, masks, theta, phi, size, num_steps=args.num_steps
            )
        
        all_shadow_maps.extend(shadow_maps.cpu())
        
        print(f"  ✓ Generated {len(batch_paths)} shadows")
        
        # Save batch results
        for i, shadow_map in enumerate(shadow_maps):
            shadow_np = shadow_map[0].cpu().numpy()
            shadow_img = Image.fromarray((shadow_np * 255).astype('uint8'), mode='L')
            
            # Extract filename from path
            img_name = Path(batch_paths[i]).stem
            output_path = output_dir / f"shadow_{img_name}.png"
            shadow_img.save(output_path)
    
    print(f"\n✅ Generated {len(all_shadow_maps)} shadow maps total!")
    
    # Load representative batch for demos
    print(f"\n🎨 Running Demo Demonstrations...")
    representative_images = []
    representative_masks = []
    
    # Load first batch for demos
    demo_batch_paths = image_paths[:min(args.batch_size, len(image_paths))]
    demo_images, demo_masks = load_images_batch(demo_batch_paths, args.device)
    representative_images = demo_images
    representative_masks = demo_masks
    
    # Run demonstrations
    if args.benchmark:
        benchmark_batch_sizes(model, representative_images, representative_masks, 
                            args.device, output_dir)
    
    demo_parameter_sweep(model, representative_images, representative_masks, output_dir)
    
    # Performance summary
    profiler.snapshot("Demo complete")
    
    print(f"\n📊 Performance Summary:")
    print(f"  Total images processed: {len(image_paths)}")
    print(f"  Average time per shadow: {load_time/len(image_paths):.3f}s (estimate)")
    print(f"  Output directory: {output_dir}")
    
    print(f"\n📊 Memory Summary:")
    profiler.print_summary()
    
    print(f"\n📁 All outputs saved to: {output_dir}")
    print("\n" + "="*70)
    print("🎉 BATCH DEMO COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

