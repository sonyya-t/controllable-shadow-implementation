"""
Data utilities for shadow generation.

Helper functions for dataset creation, downloading, and dataloader setup.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
from pathlib import Path

from .shadow_dataset import ShadowBenchmarkDataset, ShadowDataset, collate_fn


def create_dataloaders(
    dataset_type: str = "benchmark",
    batch_size: int = 2,
    num_workers: int = 4,
    image_size: int = 1024,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.

    Args:
        dataset_type: "benchmark" or "custom"
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        image_size: Image size
        **kwargs: Additional dataset arguments

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if dataset_type == "benchmark":
        # HuggingFace benchmark dataset
        train_dataset = ShadowBenchmarkDataset(
            split="train",
            image_size=image_size,
            **kwargs
        )

        # Benchmark only has train split, so we'll use a portion for validation
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size]
        )

    elif dataset_type == "custom":
        # Custom dataset with train/val splits
        root_dir = kwargs.get('root_dir', './data')

        train_dataset = ShadowDataset(
            root_dir=root_dir,
            image_size=image_size,
            split="train",
            **kwargs
        )

        val_dataset = ShadowDataset(
            root_dir=root_dir,
            image_size=image_size,
            split="val",
            **kwargs
        )

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"✓ Dataloaders created:")
    print(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")

    return train_loader, val_loader


def download_dataset(
    dataset_name: str = "jasperai/controllable-shadow-generation-benchmark",
    cache_dir: Optional[str] = None,
) -> str:
    """
    Download dataset from HuggingFace.

    Args:
        dataset_name: HuggingFace dataset identifier
        cache_dir: Where to cache the dataset

    Returns:
        Path to cached dataset
    """
    from datasets import load_dataset

    print(f"Downloading dataset: {dataset_name}")
    print(f"Cache directory: {cache_dir or 'default'}")

    dataset = load_dataset(
        dataset_name,
        cache_dir=cache_dir,
    )

    print(f"✓ Dataset downloaded successfully")
    print(f"  Splits: {list(dataset.keys())}")
    for split, data in dataset.items():
        print(f"  {split}: {len(data)} samples")

    return cache_dir or "~/.cache/huggingface/datasets"


def get_dataset_statistics(dataloader: DataLoader) -> Dict:
    """
    Compute dataset statistics.

    Args:
        dataloader: DataLoader to analyze

    Returns:
        Dictionary with statistics
    """
    print("Computing dataset statistics...")

    theta_values = []
    phi_values = []
    size_values = []
    num_samples = 0

    for batch in dataloader:
        theta_values.extend(batch['theta'].cpu().numpy())
        phi_values.extend(batch['phi'].cpu().numpy())
        size_values.extend(batch['size'].cpu().numpy())
        num_samples += batch['object_image'].shape[0]

    import numpy as np

    stats = {
        'num_samples': num_samples,
        'theta': {
            'mean': np.mean(theta_values),
            'std': np.std(theta_values),
            'min': np.min(theta_values),
            'max': np.max(theta_values),
        },
        'phi': {
            'mean': np.mean(phi_values),
            'std': np.std(phi_values),
            'min': np.min(phi_values),
            'max': np.max(phi_values),
        },
        'size': {
            'mean': np.mean(size_values),
            'std': np.std(size_values),
            'min': np.min(size_values),
            'max': np.max(size_values),
        },
    }

    print(f"✓ Statistics computed for {num_samples} samples")
    print(f"\nParameter Ranges:")
    print(f"  θ (theta):  [{stats['theta']['min']:.1f}, {stats['theta']['max']:.1f}]°  "
          f"(mean: {stats['theta']['mean']:.1f}°)")
    print(f"  φ (phi):    [{stats['phi']['min']:.1f}, {stats['phi']['max']:.1f}]°  "
          f"(mean: {stats['phi']['mean']:.1f}°)")
    print(f"  s (size):   [{stats['size']['min']:.1f}, {stats['size']['max']:.1f}]  "
          f"(mean: {stats['size']['mean']:.1f})")

    return stats


def visualize_batch(batch: Dict, num_samples: int = 4, save_path: Optional[str] = None):
    """
    Visualize a batch of samples.

    Args:
        batch: Batch from dataloader
        num_samples: Number of samples to show
        save_path: Where to save visualization (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    num_samples = min(num_samples, batch['object_image'].shape[0])

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Denormalize object image from [-1,1] to [0,1]
        obj_img = batch['object_image'][i].cpu()
        obj_img = (obj_img + 1) / 2
        obj_img = obj_img.permute(1, 2, 0).numpy()

        mask_img = batch['mask'][i, 0].cpu().numpy()
        shadow_img = batch['shadow_map'][i, 0].cpu().numpy()

        # Get parameters
        theta = batch['theta'][i].item()
        phi = batch['phi'][i].item()
        size = batch['size'][i].item()

        # Plot
        axes[i, 0].imshow(obj_img)
        axes[i, 0].set_title(f"Object\nθ={theta:.1f}°, φ={phi:.1f}°, s={size:.1f}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask_img, cmap='gray')
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(shadow_img, cmap='gray')
        axes[i, 2].set_title("Shadow Map")
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def test_data_utils():
    """Test data utilities."""
    print("\n" + "="*70)
    print("Testing Data Utilities")
    print("="*70 + "\n")

    try:
        # Test dataloader creation
        print("1. Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            dataset_type="benchmark",
            batch_size=4,
            num_workers=0,
            image_size=1024,
        )

        # Test statistics
        print("\n2. Computing statistics...")
        stats = get_dataset_statistics(train_loader)

        # Test visualization
        print("\n3. Testing visualization...")
        batch = next(iter(train_loader))
        visualize_batch(batch, num_samples=2, save_path="sample_batch.png")

        print("\n" + "="*70)
        print("All tests passed!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Note: Requires datasets and matplotlib:")
        print("  pip install datasets matplotlib")


if __name__ == "__main__":
    test_data_utils()
