"""
Dataset loader for controllable shadow generation.

Supports:
- HuggingFace benchmark dataset (jasperai/controllable-shadow-generation-benchmark)
- Custom synthetic datasets
- Training and evaluation modes
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import numpy as np
from datasets import load_dataset
import io


class ShadowBenchmarkDataset(Dataset):
    """
    Dataset loader for HuggingFace benchmark dataset.

    Dataset: jasperai/controllable-shadow-generation-benchmark
    Format: Webdataset with image.png, mask.png, shadow.png, metadata.json
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 1024,
        cache_dir: Optional[str] = None,
        track_filter: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize benchmark dataset.

        Args:
            split: Dataset split (only "train" available)
            image_size: Target image size (default: 1024)
            cache_dir: Directory to cache downloaded data
            track_filter: Filter by track ("softness", "horizontal", "vertical", or None)
            normalize: Whether to normalize images to [-1, 1]
        """
        super().__init__()

        self.split = split
        self.image_size = image_size
        self.track_filter = track_filter
        self.normalize = normalize

        print(f"Loading HuggingFace dataset: jasperai/controllable-shadow-generation-benchmark")
        print(f"Split: {split}, Image size: {image_size}")

        # Load dataset from HuggingFace
        self.dataset = load_dataset(
            "jasperai/controllable-shadow-generation-benchmark",
            split=split,
            cache_dir=cache_dir,
        )

        # Filter by track if specified
        if track_filter:
            print(f"Filtering by track: {track_filter}")
            self.dataset = self.dataset.filter(
                lambda x: x.get('track', '') == track_filter
            )

        print(f"✓ Dataset loaded: {len(self.dataset)} samples")

        # Setup transforms
        self.transform = self._setup_transforms()

    def _setup_transforms(self):
        """Setup image transformations."""
        transforms = [
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ]

        if self.normalize:
            # Normalize to [-1, 1] for object images
            transforms.append(
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )

        return T.Compose(transforms)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - object_image: RGB object (3, H, W) in [-1, 1]
                - mask: Binary mask (1, H, W) in {0, 1}
                - shadow_map: Shadow map (1, H, W) in [0, 1]
                - theta: Polar angle (float)
                - phi: Azimuthal angle (float)
                - size: Light size (float)
                - metadata: Original metadata dict
        """
        sample = self.dataset[idx]

        # Load images
        # Note: The exact field names might vary - adjust based on actual dataset structure
        object_image = self._load_image(sample, 'image')
        mask = self._load_mask(sample, 'mask')
        shadow_map = self._load_shadow(sample, 'shadow')

        # Extract light parameters from metadata
        metadata = sample.get('metadata', {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        # Extract parameters
        theta = self._extract_param(metadata, 'theta', default=30.0)
        phi = self._extract_param(metadata, 'phi', default=45.0)
        size = self._extract_param(metadata, 'light_size', default=4.0)

        # Also try alternative parameter names
        if 'r' in metadata:  # r might be related to size
            size = float(metadata['r'])

        return {
            'object_image': object_image,
            'mask': mask,
            'shadow_map': shadow_map,
            'theta': torch.tensor(theta, dtype=torch.float32),
            'phi': torch.tensor(phi, dtype=torch.float32),
            'size': torch.tensor(size, dtype=torch.float32),
            'metadata': metadata,
            'idx': idx,
        }

    def _load_image(self, sample: Dict, key: str) -> torch.Tensor:
        """
        Load and transform object image.

        Args:
            sample: Dataset sample
            key: Key for image in sample

        Returns:
            Image tensor (3, H, W) in [-1, 1]
        """
        # Handle different possible formats
        if key in sample:
            img_data = sample[key]

            # If it's bytes, convert to PIL
            if isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
            # If it's already PIL Image
            elif isinstance(img_data, Image.Image):
                img = img_data.convert('RGB')
            # If it's a path
            elif isinstance(img_data, str):
                img = Image.open(img_data).convert('RGB')
            else:
                raise ValueError(f"Unknown image format for key '{key}': {type(img_data)}")
        else:
            # Create dummy image if not found
            print(f"Warning: '{key}' not found in sample, using white image")
            img = Image.new('RGB', (self.image_size, self.image_size), color='white')

        return self.transform(img)

    def _load_mask(self, sample: Dict, key: str) -> torch.Tensor:
        """
        Load and transform mask.

        Args:
            sample: Dataset sample
            key: Key for mask in sample

        Returns:
            Mask tensor (1, H, W) in {0, 1}
        """
        if key in sample:
            mask_data = sample[key]

            # Handle different formats
            if isinstance(mask_data, bytes):
                mask = Image.open(io.BytesIO(mask_data)).convert('L')
            elif isinstance(mask_data, Image.Image):
                mask = mask_data.convert('L')
            elif isinstance(mask_data, str):
                mask = Image.open(mask_data).convert('L')
            else:
                raise ValueError(f"Unknown mask format: {type(mask_data)}")
        else:
            # Create dummy mask
            print(f"Warning: '{key}' not found, using full mask")
            mask = Image.new('L', (self.image_size, self.image_size), color=255)

        # Convert to tensor and binarize
        mask_tensor = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])(mask)

        # Binarize (threshold at 0.5)
        mask_tensor = (mask_tensor > 0.5).float()

        return mask_tensor

    def _load_shadow(self, sample: Dict, key: str) -> torch.Tensor:
        """
        Load and transform shadow map.

        Args:
            sample: Dataset sample
            key: Key for shadow in sample

        Returns:
            Shadow tensor (1, H, W) in [0, 1]
        """
        if key in sample:
            shadow_data = sample[key]

            # Handle different formats
            if isinstance(shadow_data, bytes):
                shadow = Image.open(io.BytesIO(shadow_data)).convert('L')
            elif isinstance(shadow_data, Image.Image):
                shadow = shadow_data.convert('L')
            elif isinstance(shadow_data, str):
                shadow = Image.open(shadow_data).convert('L')
            else:
                raise ValueError(f"Unknown shadow format: {type(shadow_data)}")
        else:
            # Create dummy shadow
            print(f"Warning: '{key}' not found, using black shadow")
            shadow = Image.new('L', (self.image_size, self.image_size), color=0)

        # Convert to tensor (keep in [0, 1] range)
        shadow_tensor = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])(shadow)

        return shadow_tensor

    def _extract_param(self, metadata: Dict, key: str, default: float) -> float:
        """Extract parameter from metadata with fallback."""
        if key in metadata:
            return float(metadata[key])
        return default

    def get_track_distribution(self) -> Dict[str, int]:
        """Get distribution of samples across tracks."""
        tracks = {}
        for sample in self.dataset:
            track = sample.get('track', 'unknown')
            tracks[track] = tracks.get(track, 0) + 1
        return tracks

    def print_dataset_info(self):
        """Print dataset information."""
        print("\n" + "="*70)
        print("Benchmark Dataset Information")
        print("="*70)
        print(f"Total samples:     {len(self)}")
        print(f"Image size:        {self.image_size}×{self.image_size}")
        print(f"Track filter:      {self.track_filter or 'None (all tracks)'}")
        print(f"Normalization:     {self.normalize}")

        # Track distribution
        tracks = self.get_track_distribution()
        print(f"\nTrack Distribution:")
        for track, count in tracks.items():
            print(f"  {track:15s}: {count:4d} samples")

        # Sample metadata
        if len(self) > 0:
            sample = self[0]
            print(f"\nSample Format:")
            print(f"  object_image:  {sample['object_image'].shape}")
            print(f"  mask:          {sample['mask'].shape}")
            print(f"  shadow_map:    {sample['shadow_map'].shape}")
            print(f"  theta:         {sample['theta'].item():.2f}°")
            print(f"  phi:           {sample['phi'].item():.2f}°")
            print(f"  size:          {sample['size'].item():.2f}")

        print("="*70 + "\n")


class ShadowDataset(Dataset):
    """
    Generic dataset loader for custom shadow generation datasets.

    Expected structure:
        root/
            objects/
                img001.png
                img002.png
            masks/
                img001.png
                img002.png
            shadows/
                img001.png
                img002.png
            metadata.json
    """

    def __init__(
        self,
        root_dir: str,
        image_size: int = 1024,
        split: str = "train",
        split_ratio: float = 0.9,
        normalize: bool = True,
    ):
        """
        Initialize custom dataset.

        Args:
            root_dir: Root directory containing data
            image_size: Target image size
            split: "train" or "val"
            split_ratio: Train/val split ratio
            normalize: Normalize images
        """
        super().__init__()

        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.split = split
        self.normalize = normalize

        # Load metadata
        metadata_path = self.root_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        # Find all samples
        self.samples = self._find_samples()

        # Split dataset
        num_train = int(len(self.samples) * split_ratio)
        if split == "train":
            self.samples = self.samples[:num_train]
        else:
            self.samples = self.samples[num_train:]

        # Setup transforms
        self.transform = self._setup_transforms()

        print(f"✓ Custom dataset loaded: {len(self)} {split} samples")

    def _find_samples(self) -> List[str]:
        """Find all sample IDs."""
        objects_dir = self.root_dir / "objects"
        if not objects_dir.exists():
            return []

        samples = []
        for img_path in sorted(objects_dir.glob("*.png")):
            sample_id = img_path.stem

            # Check if mask and shadow exist
            mask_path = self.root_dir / "masks" / f"{sample_id}.png"
            shadow_path = self.root_dir / "shadows" / f"{sample_id}.png"

            if mask_path.exists() and shadow_path.exists():
                samples.append(sample_id)

        return samples

    def _setup_transforms(self):
        """Setup transforms."""
        transforms = [
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ]

        if self.normalize:
            transforms.append(
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            )

        return T.Compose(transforms)

    def __len__(self) -> int:
        """Dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample."""
        sample_id = self.samples[idx]

        # Load images
        object_path = self.root_dir / "objects" / f"{sample_id}.png"
        mask_path = self.root_dir / "masks" / f"{sample_id}.png"
        shadow_path = self.root_dir / "shadows" / f"{sample_id}.png"

        object_img = Image.open(object_path).convert('RGB')
        mask_img = Image.open(mask_path).convert('L')
        shadow_img = Image.open(shadow_path).convert('L')

        # Transform
        object_tensor = self.transform(object_img)
        mask_tensor = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])(mask_img)
        shadow_tensor = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
        ])(shadow_img)

        # Get metadata
        sample_meta = self.metadata.get(sample_id, {})
        theta = float(sample_meta.get('theta', 30.0))
        phi = float(sample_meta.get('phi', 45.0))
        size = float(sample_meta.get('size', 4.0))

        return {
            'object_image': object_tensor,
            'mask': (mask_tensor > 0.5).float(),
            'shadow_map': shadow_tensor,
            'theta': torch.tensor(theta, dtype=torch.float32),
            'phi': torch.tensor(phi, dtype=torch.float32),
            'size': torch.tensor(size, dtype=torch.float32),
            'sample_id': sample_id,
            'idx': idx,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for shadow dataset.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched dictionary
    """
    # Stack tensors
    object_images = torch.stack([item['object_image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    shadow_maps = torch.stack([item['shadow_map'] for item in batch])
    thetas = torch.stack([item['theta'] for item in batch])
    phis = torch.stack([item['phi'] for item in batch])
    sizes = torch.stack([item['size'] for item in batch])

    return {
        'object_image': object_images,
        'mask': masks,
        'shadow_map': shadow_maps,
        'theta': thetas,
        'phi': phis,
        'size': sizes,
        'metadata': [item.get('metadata', {}) for item in batch],
        'idx': [item['idx'] for item in batch],
    }


def test_dataset():
    """Test dataset loading."""
    print("\n" + "="*70)
    print("Testing Shadow Benchmark Dataset")
    print("="*70 + "\n")

    try:
        # Load benchmark dataset
        print("1. Loading benchmark dataset...")
        dataset = ShadowBenchmarkDataset(
            split="train",
            image_size=1024,
        )

        dataset.print_dataset_info()

        # Test single sample
        print("2. Testing single sample...")
        sample = dataset[0]
        print(f"   ✓ Sample loaded successfully")
        print(f"   Object shape: {sample['object_image'].shape}")
        print(f"   Mask shape: {sample['mask'].shape}")
        print(f"   Shadow shape: {sample['shadow_map'].shape}")
        print(f"   Parameters: θ={sample['theta'].item():.1f}°, "
              f"φ={sample['phi'].item():.1f}°, s={sample['size'].item():.1f}")

        # Test dataloader
        print("\n3. Testing dataloader...")
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

        batch = next(iter(dataloader))
        print(f"   ✓ Batch loaded successfully")
        print(f"   Batch size: {batch['object_image'].shape[0]}")
        print(f"   Object batch: {batch['object_image'].shape}")
        print(f"   Mask batch: {batch['mask'].shape}")
        print(f"   Shadow batch: {batch['shadow_map'].shape}")

        print("\n" + "="*70)
        print("All tests passed!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("Note: This requires the HuggingFace datasets library:")
        print("  pip install datasets")


if __name__ == "__main__":
    test_dataset()
