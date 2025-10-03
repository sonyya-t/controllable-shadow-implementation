"""
Shadow Generation Demo Launcher

Convenient script to launch different demo modes with proper setup.
Handles common tasks like downloading samples, setting up directories, etc.

Usage:
    python run_demo.py basic --input your_object.png
    python run_demo.py batch --input_dir ./images
    python run_demo.py composite --object chair.png --background grass.jpg
    python run_demo.py interactive --mode gradio
    python run_demo.py all --sample_dir ./test_images
"""

import argparse
import subprocess
import sys
from pathlib import Path
import requests
from PIL import Image, ImageDraw
import json


def setup_sample_images(output_dir: Path, num_samples: int = 5):
    """
    Create sample images if not available.
    """
    sample_dir = output_dir / "sample_images"
    sample_dir.mkdir(exist_ok=True)
    
    if len(list(sample_dir.glob("*.png"))) >= num_samples:
        print(f"✓ Sample images already exist in {sample_dir}")
        return sample_dir
    
    print(f"Creating {num_samples} sample images...")
    
    # Create simple geometric shapes as sample objects
    samples = [
        ("chair", "rectangular chair"),
        ("ball", "circular ball"), 
        ("car", "simple car"),
        ("house", "house silhouette"),
        ("tree", "tree silhouette")
    ]
    
    for i, (name, description) in enumerate(samples[:num_samples]):
        # Create simple object on white background
        img = Image.new('RGB', (512, 512), 'white')
        draw = ImageDraw.Draw(img)
        
        if name == "chair":
            # Simple chair
            draw.rectangle([150, 300, 350, 450], fill='black', outline='black')
            draw.rectangle([130, 250, 370, 300], fill='black', outline='black')
        
        elif name == "ball":
            # Circle
            draw.ellipse([150, 150, 350, 350], fill='black', outline='black')
        
        elif name == "car":
            # Simple car
            draw.rectangle([100, 250, 450, 400], fill='black', outline='black')
            draw.ellipse([120, 370, 200, 420], fill='black', outline='black')
            draw.ellipse([320, 370, 400, 420], fill='black', outline='black')
        
        elif name == "house":
            # House
            draw.rectangle([200, 300, 380, 450], fill='black', outline='black')
            draw.polygon([(180, 300), (290, 200), (400, 300)], fill='black', outline='black')
        
        elif name == "tree":
            # Tree
            draw.rectangle([230, 380, 270, 450], fill='black', outline='black')  # trunk
            draw.ellipse([200, 320, 300, 380], fill='black', outline='black')   # foliage
        
        # Save image
        output_path = sample_dir / f"{name}.png"
        img.save(output_path)
        print(f"  ✓ Created {output_path}")
    
    print(f"✓ Sample images created in {sample_dir}")
    return sample_dir


def setup_sample_backgrounds(output_dir: Path):
    """
    Create sample background images.
    """
    bg_dir = output_dir / "sample_backgrounds"
    bg_dir.mkdir(exist_ok=True)
    
    backgrounds = {
        'grass.jpg': (34, 120, 30),
        'wood.jpg': (139, 90, 43),
        'marble.jpg': (250, 240, 230),
        'indoor.jpg': (220, 220, 220),
        'concrete.jpg': (180, 180, 180),
        'sand.jpg': (238, 203, 173)
    }
    
    for filename, color in backgrounds.items():
        bg_path = bg_dir / filename
        if not bg_path.exists():
            bg_img = Image.new('RGB', (1024, 1024), color)
            bg_img.save(bg_path)
            print(f"  ✓ Created {bg_path}")
    
    print(f"✓ Sample backgrounds created in {bg_dir}")
    return bg_dir


def run_demo_command(command: list, demo_name: str):
    """
    Run a demo command with error handling.
    """
    print(f"\n🚀 Running {demo_name} demo...")
    print(f"Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✓ {demo_name} demo completed successfully")
        if result.stdout:
            print("Output:", result.stdout[:500] + ("..." if len(result.stdout) > 500 else ""))
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ {demo_name} demo failed")
        print(f"Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False
    
    except FileNotFoundError as e:
        print(f"❌ Demo script not found: {e}")
        return False


def run_basic_demo(args):
    """Run basic demo."""
    command = ["python", "demo_basic.py"]
    
    # Add required arguments
    command.extend(["--input", str(args.input)])
    
    # Add optional arguments
    if args.output_dir:
        command.extend(["--output_dir", str(args.output_dir)])
    if args.checkpoint:
        command.extend(["--checkpoint", str(args.checkpoint)])
    if args.device:
        command.extend(["--device", args.device])
    if args.theta is not None:
        command.extend(["--theta", str(args.theta)])
    if args.phi is not None:
        command.extend(["--phi", str(args.phi)])
    if args.size is not None:
        command.extend(["--size", str(args.size)])
    
    return run_demo_command(command, "Basic")


def run_batch_demo(args):
    """Run batch demo."""
    command = ["python", "demo_batch.py"]
    
    # Add input argument
    if args.input_dir:
        command.extend(["--input_dir", str(args.input_dir)])
    elif args.images:
        command.extend(["--images"] + args.images)
    
    # Add optional arguments
    if args.output_dir:
        command.extend(["--output_dir", str(args.output_dir)])
    if args.checkpoint:
        command.extend(["--checkpoint", str(args.checkpoint)])
    if args.device:
        command.extend(["--device", args.device])
    if args.batch_size:
        command.extend(["--batch_size", str(args.batch_size)])
    if args.benchmark:
        command.append("--benchmark")
    
    return run_demo_command(command, "Batch")


def run_composite_demo(args):
    """Run composite demo."""
    command = ["python", "demo_composite.py"]
    
    # Add required arguments
    command.extend(["--object", str(args.object)])
    
    # Add background arguments
    if args.background:
        command.extend(["--background", str(args.background)])
    if args.backgrounds_dir:
        command.extend(["--backgrounds_dir", str(args.backgrounds_dir)])
    
    # Add shadow arguments
    if args.shadow_maps:
        command.extend(["--shadow_maps"] + args.shadow_maps)
    if args.theta is not None:
        command.extend(["--theta", str(args.theta)])
    if args.phi is not None:
        command.extend(["--phi", str(args.phi)])
    if args.size is not None:
        command.extend(["--size", str(args.size)])
    
    # Add other options
    if args.output_dir:
        command.extend(["--output_dir", str(args.output_dir)])
    if args.checkpoint:
        command.extend(["--checkpoint", str(args.checkpoint)])
    if args.device:
        command.extend(["--device", args.device])
    if args.run_all_demos:
        command.append("--run_all_demos")
    
    return run_demo_command(command, "Composite")


def run_interactive_demo(args):
    """Run interactive demo."""
    command = ["python", "demo_interactive.py"]
    
    command.extend(["--interface", args.mode])
    command.extend(["--server_name", args.server_name])
    command.extend(["--port", str(args.port)])
    
    if args.checkpoint:
        command.extend(["--checkpoint", str(args.checkpoint)])
    if args.device:
        command.extend(["--device", args.device])
    if args.debug:
        command.append("--debug")
    
    print(f"\n🌐 Starting interactive demo on http://{args.server_name}:{args.port}")
    
    # Run interactively (won't exit until killed)
    try:
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        print("\n👋 Interactive demo stopped by user")
    except Exception as e:
        print(f"❌ Interactive demo failed: {e}")
        return False
    
    return True


def run_all_demos(args):
    """Run all demos in sequence."""
    print("\n🎯 Running ALL demos...")
    
    # Setup output directory
    output_dir = Path(args.output_dir or "./demo_all_output")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup sample data
    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    if not sample_dir or not sample_dir.exists():
        sample_dir = setup_sample_images(output_dir)
    
    bg_dir = setup_sample_backgrounds(output_dir)
    
    # Collect first few sample images
    sample_images = list(sample_dir.glob("*.png"))[:3]  # Use first 3
    
    if not sample_images:
        print("❌ No sample images available!")
        return False
    
    print(f"\nUsing samples: {[img.name for img in sample_images]}")
    
    # Common arguments
    common_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        device=args.device or "cuda",
        output_dir=str(output_dir)
    )
    
    # Run demos
    success_count = 0
    total_demos = 4
    
    # 1. Basic Demo
    basic_args = argparse.Namespace(
        **vars(common_args),
        input=str(sample_images[0]),
        theta=30,
        phi=45,
        size=4
    )
    if run_basic_demo(basic_args):
        success_count += 1
    
    # 2. Batch Demo
    batch_args = argparse.Namespace(
        **vars(common_args),
        input_dir=str(sample_dir),
        batch_size=2,
        benchmark=True
    )
    if run_batch_demo(batch_args):
        success_count += 1
    
    # 3. Composite Demo
    composite_args = argparse.Namespace(
        **vars(common_args),
        object=str(sample_images[0]),
        backgrounds_dir=str(bg_dir),
        theta=30,
        phi=90,
        size=5,
        run_all_demos=True
    )
    if run_composite_demo(composite_args):
        success_count += 1
    
    # 4. Interactive Demo (quick test)
    print(f"\n🌐 Testing interactive demo (quick start/test)")
    
    # We'll just validate the script exists and can import
    try:
        import demo_interactive
        print("✓ Interactive demo script validation successful")
        success_count += 1
    except ImportError:
        print("❌ Interactive demo script validation failed")
    
    # Summary
    print(f"\n📊 Demo Summary:")
    print(f"  Completed: {success_count}/{total_demos}")
    print(f"  Success rate: {success_count/total_demos*100:.1f}%")
    print(f"  Output directory: {output_dir}")
    
    if success_count == total_demos:
        print(f"\n🎉 All demos completed successfully!")
        print(f"\nTo run interactive demo:")
        print(f"  python run_demo.py interactive")
        return True
    else:
        print(f"\n⚠️ Some demos failed. Check the output above for details.")
        return False


def main():
    """Main demo launcher."""
    parser = argparse.ArgumentParser(description="Shadow Generation Demo Launcher")
    
    # Main demo subcommands
    subparsers = parser.add_subparsers(dest='demo_type', help='Demo type to run')
    
    # Basic demo
    basic_parser = subparsers.add_parser('basic', help='Basic single-image shadow generation')
    basic_parser.add_argument('--input', type=str, required=True, help='Input object image')
    basic_parser.add_argument('--theta', type=float, help='Polar angle [0-45]')
    basic_parser.add_argument('--phi', type=float, help='Azimuthal angle [0-360]')
    basic_parser.add_argument('--size', type=float, help='Light size [2-8]')
    
    # Batch demo
    batch_parser = subparsers.add_parser('batch', help='Batch processing demo')
    batch_group = batch_parser.add_mutually_exclusive_group(required=True)
    batch_group.add_argument('--input_dir', type=str, help='Input directory')
    batch_group.add_argument('--images', nargs='+', type=str, help='Image files')
    batch_parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    batch_parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    
    # Composite demo  
    composite_parser = subparsers.add_parser('composite', help='Shadow compositing demo')
    composite_parser.add_argument('--object', type=str, required=True, help='Object image')
    bg_group = composite_parser.add_mutually_exclusive_group()
    bg_group.add_argument('--background', type=str, help='Background image')
    bg_group.add_argument('--backgrounds_dir', type=str, help='Background directory')
    composite_parser.add_argument('--shadow_maps', nargs='+', type=str, help='Shadow map files')
    composite_parser.add_argument('--theta', type=float, help='Polar angle [0-45]')
    composite_parser.add_argument('--phi', type=float, help='Azimuthal angle [0-360]')
    composite_parser.add_argument('--size', type=float, help='Light size [2-8]')
    composite_parser.add_argument('--run_all_demos', action='store_true', help='Run all composite demos')
    
    # Interactive demo
    interactive_parser = subparsers.add_parser('interactive', help='Interactive web demo')
    interactive_parser.add_argument('--mode', type=str, default='gradio', 
                                   choices=['gradio', 'streamlit'], help='Web framework')
    interactive_parser.add_argument('--server_name', type=str, default='127.0.0.1', help='Server host')
    interactive_parser.add_argument('--port', type=int, default=7860, help='Server port')
    interactive_parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    # All demos
    all_parser = subparsers.add_parser('all', help='Run all demos')
    all_parser.add_argument('--sample_dir', type=str, help='Sample images directory')
    
    # Common arguments for all demos
    for p in [parser, basic_parser, batch_parser, composite_parser, interactive_parser, all_parser]:
        p.add_argument('--checkpoint', type=str, help='Model checkpoint path')
        p.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
        p.add_argument('--output_dir', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    if not args.demo_type:
        parser.print_help()
        return
    
    print("\n" + "="*70)
    print("🎭 SHADOW GENERATION DEMO LAUNCHER")
    print("="*70)
    
    success = False
    
    if args.demo_type == 'basic':
        success = run_basic_demo(args)
    
    elif args.demo_type == 'batch':
        success = run_batch_demo(args)
    
    elif args.demo_type == 'composite':
        success = run_composite_demo(args)
    
    elif args.demo_type == 'interactive':
        success = run_interactive_demo(args)
    
    elif args.demo_type == 'all':
        success = run_all_demos(args)
    
    if success:
        print("\n✅ Demo completed successfully!")
    else:
        print("\n❌ Demo failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
