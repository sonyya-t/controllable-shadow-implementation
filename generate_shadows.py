"""
Shadow generation inference script.

Quick demo for generating shadows from object images.
"""

import torch
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

from controllable_shadow.models import create_shadow_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate shadow maps")

    # Input/Output
    parser.add_argument("--input", type=str, required=True,
                        help="Input object image path")
    parser.add_argument("--output", type=str, default="shadow_output.png",
                        help="Output shadow map path")
    parser.add_argument("--mask", type=str, default=None,
                        help="Optional mask path (auto-generated if not provided)")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use")

    # Light parameters
    parser.add_argument("--theta", type=float, default=30.0,
                        help="Polar angle in degrees [0-45]")
    parser.add_argument("--phi", type=float, default=45.0,
                        help="Azimuthal angle in degrees [0-360]")
    parser.add_argument("--size", type=float, default=4.0,
                        help="Light size/softness [2-8]")

    # Generation
    parser.add_argument("--num_steps", type=int, default=1,
                        help="Number of sampling steps (1=fast, 4+=quality)")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="Image size")

    return parser.parse_args()


def load_image(image_path: str, size: int = 1024) -> torch.Tensor:
    """
    Load and preprocess image.

    Args:
        image_path: Path to image
        size: Target size

    Returns:
        Image tensor (1, 3, H, W) in [-1, 1]
    """
    img = Image.open(image_path).convert('RGB')

    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    return transform(img).unsqueeze(0)


def create_mask_from_image(image_tensor: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    """
    Create mask by detecting white/bright background.

    Args:
        image_tensor: Image tensor (1, 3, H, W) in [-1, 1]
        threshold: Brightness threshold

    Returns:
        Mask tensor (1, 1, H, W) in {0, 1}
    """
    # Denormalize to [0, 1]
    img_denorm = (image_tensor + 1) / 2

    # Compute brightness
    brightness = img_denorm.mean(dim=1, keepdim=True)

    # Create mask (inverted: 1 for object, 0 for background)
    mask = (brightness < threshold).float()

    return mask


def save_shadow_map(shadow_tensor: torch.Tensor, output_path: str):
    """
    Save shadow map to file.

    Args:
        shadow_tensor: Shadow tensor (1, 1, H, W) in [0, 1]
        output_path: Output file path
    """
    # Convert to PIL Image
    shadow_np = shadow_tensor[0, 0].cpu().numpy()
    shadow_np = (shadow_np * 255).astype('uint8')
    shadow_img = Image.fromarray(shadow_np, mode='L')

    # Save
    shadow_img.save(output_path)
    print(f"✓ Shadow map saved to {output_path}")


def main():
    """Main inference function."""
    args = parse_args()

    print("\n" + "="*70)
    print("Shadow Generation - Inference")
    print("="*70 + "\n")

    # Validate parameters
    assert 0 <= args.theta <= 45, "Theta must be in [0, 45] degrees"
    assert 0 <= args.phi <= 360, "Phi must be in [0, 360] degrees"
    assert 2 <= args.size <= 8, "Size must be in [2, 8]"

    # Load model
    print("Loading model...")
    model = create_shadow_model(
        pretrained_path=args.checkpoint,
        device=args.device,
    )
    model.eval()
    print("✓ Model loaded\n")

    # Load input image
    print(f"Loading input image: {args.input}")
    object_image = load_image(args.input, args.image_size).to(args.device)
    print(f"✓ Image loaded: {object_image.shape}\n")

    # Load or create mask
    if args.mask:
        print(f"Loading mask: {args.mask}")
        mask = load_image(args.mask, args.image_size).to(args.device)
        mask = (mask.mean(dim=1, keepdim=True) > 0.5).float()
    else:
        print("Generating mask from image (detecting background)...")
        mask = create_mask_from_image(object_image)
        mask = mask.to(args.device)

    print(f"✓ Mask ready: {mask.shape}\n")

    # Prepare parameters
    theta = torch.tensor([args.theta], device=args.device)
    phi = torch.tensor([args.phi], device=args.device)
    size = torch.tensor([args.size], device=args.device)

    print("Generation parameters:")
    print(f"  Theta (θ):     {args.theta}° (vertical angle)")
    print(f"  Phi (φ):       {args.phi}° (horizontal direction)")
    print(f"  Size (s):      {args.size} (softness)")
    print(f"  Sampling steps: {args.num_steps}\n")

    # Generate shadow
    print("Generating shadow map...")
    with torch.no_grad():
        shadow_map = model.sample(
            object_image,
            mask,
            theta,
            phi,
            size,
            num_steps=args.num_steps,
        )

    print(f"✓ Shadow generated: {shadow_map.shape}\n")

    # Save output
    save_shadow_map(shadow_map, args.output)

    print("\n" + "="*70)
    print("Generation complete!")
    print("="*70 + "\n")

    # Optional: save visualizations
    output_dir = Path(args.output).parent
    vis_path = output_dir / f"{Path(args.output).stem}_visualization.png"

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Object
        obj_np = (object_image[0].cpu() + 1) / 2
        obj_np = obj_np.permute(1, 2, 0).numpy()
        axes[0].imshow(obj_np)
        axes[0].set_title(f"Object Image\nθ={args.theta}°, φ={args.phi}°, s={args.size}")
        axes[0].axis('off')

        # Mask
        mask_np = mask[0, 0].cpu().numpy()
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title("Object Mask")
        axes[1].axis('off')

        # Shadow
        shadow_np = shadow_map[0, 0].cpu().numpy()
        axes[2].imshow(shadow_np, cmap='gray')
        axes[2].set_title("Generated Shadow Map")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {vis_path}\n")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping visualization\n")


if __name__ == "__main__":
    main()
