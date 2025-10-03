"""
Advanced Shadow Compositing Demo

Demonstrates compositing shadows onto different backgrounds with realistic blending.
Shows professional shadow integration techniques.

Usage:
    python demo_composite.py --object chair.png --background grass.jpg --shadow_props theta=30 phi=90 size=4
    python demo_composite.py --object car.png --destination ./results --shadow_maps shadow1.png shadow2.png
"""

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import torchvision.transforms as T
import time
from typing import List, Tuple, Dict
import cv2

from controllable_shadow.models import create_shadow_model


def create_realistic_composite(object_img: Image.Image, 
                            shadow_map: Image.Image, 
                            background_img: Image.Image,
                            shadow_color: Tuple[int, int, int] = (0, 0, 0),
                            shadow_opacity: float = 0.7,
                            blur_radius: int = 5) -> Image.Image:
    """
    Create realistic composite of object + shadow + background.
    
    Args:
        object_img: Object image with transparent background
        shadow_map: Grayscale shadow map
        background_img: Background image
        shadow_color: RGB color for shadow
        shadow_opacity: Shadow opacity [0-1]
        blur_radius: Gaussian blur radius for shadow softness
        
    Returns:
        Composite image
    """
    
    # Ensure images have same size
    target_size = (1024, 1024)
    object_img = object_img.resize(target_size, Image.Resampling.LANCZOS)
    background_img = background_img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Create object with alpha channel if needed
    if object_img.mode != 'RGBA':
        object_img = object_img.convert('RGBA')
    
    # Extract background from object
    background_mask = np.array(object_img)
    # Simple background detection (white/bright regions)
    brightness = np.mean(background_mask[:, :, :3], axis=2)
    mask = brightness > 220  # Threshold for background
    background_mask[:, :, 3] = (mask == False).astype(np.uint8) * 255  # Alpha channel
    
    object_with_alpha = Image.fromarray(background_mask)
    
    # Process shadow map
    shadow_np = np.array(shadow_map.resize(target_size, Image.Resampling.LANCZOS))
    shadow_np = shadow_np / 255.0  # Normalize to [0,1]
    
    # Apply blur and color
    shadow_blurred = cv2.GaussianBlur(shadow_np, (blur_radius*2+1, blur_radius*2+1), blur_radius)
    
    # Create colored shadow
    shadow_rgb = np.zeros((*shadow_blurred.shape, 3), dtype=np.uint8)
    shadow_rgb[:, :, 0] = shadow_blurred * shadow_color[0]  # R
    shadow_rgb[:, :, 1] = shadow_blurred * shadow_color[1]  # G  
    shadow_rgb[:, :, 2] = shadow_blurred * shadow_color[2]  # B
    
    shadow_img = Image.fromarray(shadow_rgb, mode='RGB')
    
    # Composite: background -> shadow -> object
    result = background_img.copy()
    
    # Blend shadow onto background
    result = Image.blend(result, shadow_img, shadow_opacity)
    
    # Composite object
    result.paste(object_with_alpha, mask=object_with_alpha.split()[-1])  # Use alpha channel
    
    return result


def demo_environmental_shadows(object_img: Image.Image, shadow_maps: Dict, backgrounds: Dict):
    """
    Demonstrate different environmental shadow scenarios.
    """
    composites = {}
    
    for env_name, (background_path, shadow_params) in backgrounds.items():
        print(f"\n🌟 Environment: {env_name}")
        
        # Load background
        bg_img = Image.open(background_path).convert('RGB')
        
        # Get shadow map for these parameters
        shadow_key = (shadow_params['theta'], shadow_params['phi'], shadow_params['size'])
        shadow_map = shadow_maps[shadow_key]
        
        # Environment-specific shadow properties
        env_settings = {
            'outdoor_sunny': {'color': (0, 0, 0), 'opacity': 0.8, 'blur': 3},
            'outdoor_overcast': {'color': (64, 64, 64), 'opacity': 0.6, 'blur': 8},
            'indoor_soft': {'color': (128, 128, 128), 'opacity': 0.5, 'blur': 12},
            'studio_hard': {'color': (0, 0, 0), 'opacity': 0.9, 'blur': 1},
        }
        
        settings = env_settings.get(env_name, env_settings['outdoor_sunny'])
        
        # Create composite
        composite = create_realistic_composite(
            object_img, shadow_map, bg_img,
            shadow_color=settings['color'],
            shadow_opacity=settings['opacity'],
            blur_radius=settings['blur']
        )
        
        composites[env_name] = composite
        print(f"  ✓ Created composite with θ={shadow_params['theta']}°, φ={shadow_params['phi']}°, s={shadow_params['size']}")
    
    return composites


def demo_shadow_size_comparison(object_img: Image.Image, background_img: Image.Image, 
                              shadow_maps_by_size: Dict[float, Image.Image]):
    """
    Demonstrate varying shadow softness on same background.
    """
    print("\n🔆 Shadow Softness Comparison:")
    
    composites = {}
    
    # Define shadow properties for each size
    size_properties = {
        2: {'opacity': 0.9, 'blur': 1, 'color': (0, 0, 0)},
        4: {'opacity': 0.7, 'blur': 5, 'color': (0, 0, 0)},
        6: {'opacity': 0.6, 'blur': 8, 'color': (0, 0, 0)},
        8: {'opacity': 0.4, 'blur': 15, 'color': (32, 32, 32)}
    }
    
    for size_value, shadow_map in shadow_maps_by_size.items():
        print(f"  Size {size_value}: Soft diffuse shadow")
        
        props = size_properties.get(size_value, size_properties[4])
        
        composite = create_realistic_composite(
            object_img, shadow_map, background_img,
            shadow_opacity=props['opacity'],
            blur_radius=props['blur'],
            shadow_color=props['color']
        )
        
        composites[f"s{size_value}"] = composite
    
    return composites


def demo_time_of_day(object_img: Image.Image, background_img: Image.Image,
                    shadow_maps_by_time: Dict[str, Image.Image]):
    """
    Demonstrate time-of-day shadow variations.
    """
    print("\n⏰ Time of Day Simulation:")
    
    composites = {}
    
    # Time-specific parameters
    time_settings = {
        'sunrise': {'theta': 15, 'phi': 90, 'color': (200, 100, 50), 'opacity': 0.6, 'blur': 6},
        'morning': {'theta': 25, 'phi': 120, 'color': (150, 100, 50), 'opacity': 0.7, 'blur': 4},
        'noon': {'theta': 5, 'phi': 180, 'color': (0, 0, 0), 'opacity': 0.8, 'blur': 2},
        'afternoon': {'theta': 30, 'phi': 240, 'color': (100, 150, 200), 'opacity': 0.7, 'blur': 4},
        'sunset': {'theta': 35, 'phi': 270, 'color': (180, 120, 80), 'opacity': 0.6, 'blur': 8}
    }
    
    for time_name, shadow_map in shadow_maps_by_time.items():
        print(f"  {time_name}: Warm atmospheric shadows")
        
        settings = time_settings.get(time_name, time_settings['noon'])
        
        # Enhance background for time of day
        bg_enhanced = background_img.copy()
        
        if time_name in ['sunrise', 'sunset']:
            # Warm tone enhancement
            enhancer = ImageEnhance.Color(bg_enhanced)
            bg_enhanced = enhancer.enhance(1.3)
            
            enhancer = ImageEnhance.Brightness(bg_enhanced)
            bg_enhanced = enhancer.enhance(0.9)
        
        elif time_name == 'noon':
            # Bright, crisp
            enhancer = ImageEnhance.Contrast(bg_enhanced)
            bg_enhanced = enhancer.enhance(1.1)
        
        elif time_name in ['morning', 'afternoon']:
            # Slight coolness
            enhancer = ImageEnhance.Color(bg_enhanced)
            bg_enhanced = enhancer.enhance(1.1)
        
        composite = create_realistic_composite(
            object_img, shadow_map, bg_enhanced,
            shadow_color=settings['color'],
            shadow_opacity=settings['opacity'],
            blur_radius=settings['blur']
        )
        
        composites[time_name] = composite
    
    return composites


def create_comparison_grid(all_composites: Dict[str, Dict[str, Image.Image]], 
                          output_path: Path):
    """
    Create comprehensive comparison grid of all composites.
    """
    # Determine grid size
    scenario_types = list(all_composites.keys())
    max_count = max(len(composites) for composites in all_composites.values())
    
    fig, axes = plt.subplots(len(scenario_types), max_count, 
                            figsize=(5*max_count, 4*len(scenario_types)))
    
    if len(scenario_types) == 1:
        axes = axes.reshape(1, -1)
    if max_count == 1:
        axes = axes.reshape(-1, 1)
    
    for row_idx, (scenario_type, composites) in enumerate(all_composites.items()):
        composite_items = list(composites.items())
        
        for col_idx in range(max_count):
            ax = axes[row_idx, col_idx]
            
            if col_idx < len(composite_items):
                name, img = composite_items[col_idx]
                
                img_np = np.array(img)
                ax.imshow(img_np)
                ax.set_title(f"{scenario_type.title()}\n{name.replace('_', ' ')}")
            else:
                ax.axis('off')
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comprehensive comparison saved: {output_path}")


def load_sample_backgrounds() -> Dict[str, str]:
    """
    Return sample background images or create them if not available.
    """
    sample_backgrounds = {
        'grass': 'grass.jpg',
        'wood': 'wood.jpg', 
        'marble': 'marble.jpg',
        'indoor': 'indoor.jpg'
    }
    
    # Check what's available
    available = {}
    
    for name, filename in sample_backgrounds.items():
        path = Path(filename)
        if path.exists():
            available[name] = str(path)
    
    # If no backgrounds found, create simple solid colors
    if not available:
        print("Creating sample backgrounds...")
        
        sample_backgrounds = {
            'grass': (34, 120, 30),
            'wood': (139, 90, 43),
            'marble': (250, 240, 230),
            'indoor': (220, 220, 220)
        }
        
        for name, color in sample_backgrounds.items():
            bg = Image.new('RGB', (1024, 1024), color)
            bg.save(f"{name}.jpg")
            available[name] = f"{name}.jpg"
    
    return available


def main():
    """Main compositing demo function."""
    parser = argparse.ArgumentParser(description="Advanced Shadow Compositing Demo")
    
    # Primary inputs
    parser.add_argument("--object", type=str, help="Object image path")
    parser.add_argument("--background", type=str, default=None,
                       help="Background image path")
    parser.add_argument("--backgrounds_dir", type=str, default=None,
                       help="Directory of background images")
    
    # Shadow inputs
    parser.add_argument("--shadow_maps", nargs='+', type=str, default=None,
                       help="Pre-generated shadow map files")
    
    # Object shadow parameters
    parser.add_argument("--theta", type=float, default=30.0,
                       help="Polar angle [0-45]")
    parser.add_argument("--phi", type=float, default=45.0,
                       help="Azimuthal angle [0-360]")
    parser.add_argument("--size", type=float, default=4.0,
                       help="Shadow softness [2-8]")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./demo_composite_output",
                       help="Output directory")
    
    # Model (for generating shadows if not provided)
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Model checkpoint for generating shadows")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    
    # Compositing options
    parser.add_argument("--shadow_color_r", type=int, default=0,
                       help="Shadow color red component")
    parser.add_argument("--shadow_color_g", type=int, default=0,
                       help="Shadow color green component")
    parser.add_argument("--shadow_color_b", type=int, default=0,
                       help="Shadow color blue component")
    parser.add_argument("--shadow_opacity", type=float, default=0.7,
                       help="Shadow opacity [0-1]")
    parser.add_argument("--shadow_blur", type=int, default=5,
                       help="Shadow blur radius")
    
    # Demo modes
    parser.add_argument("--demo_env", action="store_true",
                       help="Run environmental shadow demo")
    parser.add_argument("--demo_softness", action="store_true",
                       help="Run shadow softness comparison")
    parser.add_argument("--demo_tod", action="store_true",
                       help="Run time-of-day demo")
    parser.add_argument("--run_all_demos", action="store_true",
                       help="Run all demo modes")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎨 ADVANCED SHADOW COMPOSITING DEMO")
    print("="*70)
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if not args.run_all_demos and not any([args.demo_env, args.demo_softness, args.demo_tod]):
        args.run_all_demos = True
    
    # Load object image
    if not args.object:
        print("❌ Object image required!")
        return
    
    object_img = Image.open(args.object).convert('RGB')
    print(f"✓ Loaded object: {args.object}")
    
    # Generate or load shadow maps
    shadow_maps = {}
    
    if args.shadow_maps:
        # Load provided shadow maps
        print(f"\n📁 Loading {len(args.shadow_maps)} shadow maps...")
        
        for i, shadow_path in enumerate(args.shadow_maps):
            shadow_map = Image.open(shadow_path).convert('L')
            
            # Extract parameters from filename or use defaults
            params = (args.theta, args.phi, args.size)  # Default for single shadow
            shadow_maps[params] = shadow_map
        
        print(f"✓ Loaded {len(shadow_maps)} shadow maps")
    
    else:
        # Generate shadow maps using model
        if not args.checkpoint:
            print("❌ Either provide shadow maps or model checkpoint!")
            return
        
        print(f"\n🤖 Loading model for shadow generation...")
        model = create_shadow_model(
            pretrained_path=args.checkpoint,
            device=args.device,
        )
        model.eval()
        
        # Generate various parameter combinations
        param_combinations = [
            (30, 90, 4, "Environmental shadows"),
            (15, 90, 3, "Hard shadows"),
            (35, 180, 6, "Medium shadows"),
            (25, 270, 8, "Soft shadows"),
            (5, 0, 2, "Noon shadows"),
            (20, 45, 5, "Morning"), 
            (40, 315, 7, "Evening"),
            (30, 225, 5, "Afternoon")
        ]
        
        print(f"Generating {len(param_combinations)} shadow maps...")
        
        # Prepare object tensor
        transform = T.Compose([
            T.Resize((1024, 1024)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        object_tensor = transform(object_img).unsqueeze(0).to(args.device)
        
        # Create mask
        img_denorm = (object_tensor + 1) / 2
        brightness = img_denorm.mean(dim=1, keepdim=True)
        mask = (brightness < 0.9).float()
        
        # Generate shadows
        with torch.no_grad():
            for theta, phi, size, description in param_combinations:
                print(f"  Generating θ={theta}°, φ={phi}°, s={size}")
                
                theta_tensor = torch.tensor([theta], device=args.device)
                phi_tensor = torch.tensor([phi], device=args.device)
                size_tensor = torch.tensor([size], device=args.device)
                
                shadow_tensor = model.sample(
                    object_tensor, mask, theta_tensor, phi_tensor, size_tensor, num_steps=1
                )
                
                # Convert to PIL
                shadow_np = shadow_tensor[0, 0].cpu().numpy()
                shadow_map = Image.fromarray((shadow_np * 255).astype('uint8'), mode='L')
                
                shadow_maps[(theta, phi, size)] = shadow_map
        
        print(f"✓ Generated {len(shadow_maps)} shadow maps")
    
    # Load backgrounds
    backgrounds = None
    if args.background:
        backgrounds = {'single': args.background}
    elif args.backgrounds_dir:
        bg_dir = Path(args.backgrounds_dir)
        bg_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        backgrounds = {
            p.stem: str(p) for p in bg_dir.iterdir() 
            if p.suffix.lower() in bg_extensions
        }
    else:
        print("\n📁 Loading sample backgrounds...")
        backgrounds = load_sample_backgrounds()
    
    print(f"✓ Using {len(backgrounds)} backgrounds")
    
    all_composite_results = {}
    
    # Run demonstrations
    if args.demo_env or args.run_all_demos:
        print(f"\n🌍 Environmental Shadow Demo")
        
        # Create environmental scenarios
        env_scenarios = {
            'outdoor_sunny': {
                'background': backgrounds.get('grass', list(backgrounds.values())[0]),
                'shadow_params': {'theta': 30, 'phi': 90, 'size': 3}
            },
            'outdoor_overcast': {
                'background': backgrounds.get('wood', list(backgrounds.values())[0]),
                'shadow_params': {'theta': 25, 'phi': 180, 'size': 8}
            },
            'indoor_soft': {
                'background': backgrounds.get('indoor', list(backgrounds.values())[0]),
                'shadow_params': {'theta': 20, 'phi': 270, 'size': 6}
            },
            'studio_hard': {
                'background': backgrounds.get('marble', list(backgrounds.values())[0]),
                'shadow_params': {'theta': 35, 'phi': 45, 'size': 2}
            }
        }
        
        env_backgrounds = {
            name: scenario['background'] 
            for name, scenario in env_scenarios.items()
        }
        
        env_shadow_maps = {
            (scenario['shadow_params']['theta'], 
             scenario['shadow_params']['phi'], 
             scenario['shadow_params']['size']): shadow_maps[(
                scenario['shadow_params']['theta'],
                scenario['shadow_params']['phi'],
                scenario['shadow_params']['size']
            )]
            for name, scenario in env_scenarios.items()
            if (scenario['shadow_params']['theta'],
                scenario['shadow_params']['phi'],
                scenario['shadow_params']['size']) in shadow_maps
        }
        
        composites = demo_environmental_shadows(object_img, env_shadow_maps, backgrounds)
        
        # Save individual composites
        for name, composite in composites.items():
            output_path = output_dir / f"composite_environmental_{name}.png"
            composite.save(output_path)
            print(f"  ✓ Saved: {output_path}")
        
        all_composite_results['environmental'] = composites
    
    if args.demo_softness or args.run_all_demos:
        print(f"\n🔆 Shadow Softness Comparison")
        
        # Get sizes for softness comparison
        sizes = [2, 4, 6, 8]
        
        size_shadow_maps = {}
        for size_val in sizes:
            # Find closest shadow map for this size
            closest_params = min(shadow_maps.keys(), 
                               key=lambda x: abs(x[2] - size_val))
            size_shadow_maps[size_val] = shadow_maps[closest_params]
        
        # Use first available background
        demo_bg = list(backgrounds.values())[0]
        bg_img = Image.open(demo_bg).convert('RGB')
        
        composites = demo_shadow_size_comparison(object_img, bg_img, size_shadow_maps)
        
        # Save individual composites
        for name, composite in composites.items():
            output_path = output_dir / f"composite_softness_{name}.png"
            composite.save(output_path)
            print(f"  ✓ Saved: {output_path}")
        
        all_composite_results['softness'] = composites
    
    if args.demo_tod or args.run_all_demos:
        print(f"\n⏰ Time of Day Demo")
        
        # Time-of-day shadow parameters
        tod_scenarios = {
            'sunrise': (15, 90, 6),
            'morning': (25, 120, 5),
            'noon': (5, 180, 2),
            'afternoon': (30, 240, 4),
            'sunset': (35, 270, 8)
        }
        
        tod_shadow_maps = {}
        for time_name, params in tod_scenarios.items():
            if params in shadow_maps:
                tod_shadow_maps[time_name] = shadow_maps[params]
            else:
                print(f"  Warning: No shadow map for {time_name} with params={params}")
        
        if tod_shadow_maps:
            # Use all available backgrounds for time of day
            tod_composites = {}
            
            for bg_name, bg_path in backgrounds.items():
                bg_img = Image.open(bg_path).convert('RGB')
                
                composites = demo_time_of_day(object_img, bg_img, tod_shadow_maps)
                
                tod_composites[bg_name] = composites
            
            # Save composites
            for bg_name, composites in tod_composites.items():
                for time_name, composite in composites.items():
                    output_path = output_dir / f"composite_tod_{bg_name}_{time_name}.png"
                    composite.save(output_path)
                    print(f"  ✅ Saved: {output_path}")
            
            # Use first background result for overall demo
            if tod_composites:
                first_bg = list(tod_composites.keys())[0]
                all_composite_results['time_of_day'] = tod_composites[first_bg]
    
    # Create comprehensive comparison
    if all_composite_results:
        comparison_path = output_dir / "comprehensive_comparison.png"
        create_comparison_grid(all_composite_results, comparison_path)
    
    print(f"\n📊 Demo Summary:")
    print(f"  Total composite scenarios: {sum(len(composites) for composites in all_composite_results.values())}")
    print(f"  Shadow maps used: {len(shadow_maps)}")
    print(f"  Backgrounds used: {len(backgrounds)}")
    print(f"  Output directory: {output_dir}")
    
    print("\n" + "="*70)
    print("🎉 COMPOSITING DEMO COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
