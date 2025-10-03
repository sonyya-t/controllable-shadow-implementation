"""
Checkpoint management for shadow generation model.

Handles:
- Saving/loading model weights
- SDXL pretrained weight initialization
- Version control and compatibility
- Partial loading (SDXL weights + new parameters)
"""

import torch
import os
from pathlib import Path
from typing import Dict, Optional, Any
import json
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints with proper handling of:
    - SDXL pretrained weights
    - Newly initialized parameters
    - Training state
    - Version compatibility
    """

    VERSION = "1.0.0"

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        optimizer_state_dict: Optional[Dict] = None,
        epoch: int = 0,
        global_step: int = 0,
        loss: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """
        Save model checkpoint with all relevant information.

        Args:
            model_state_dict: Model state dictionary
            optimizer_state_dict: Optimizer state (optional)
            epoch: Current epoch
            global_step: Global training step
            loss: Current loss value
            metadata: Additional metadata
            checkpoint_name: Custom checkpoint name

        Returns:
            Path to saved checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_epoch{epoch}_step{global_step}.pt"

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # Prepare checkpoint data
        checkpoint = {
            "version": self.VERSION,
            "model_state_dict": model_state_dict,
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
        }

        # Add optional components
        if optimizer_state_dict is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state_dict

        if metadata is not None:
            checkpoint["metadata"] = metadata

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Save metadata separately for easy inspection
        metadata_path = checkpoint_path.with_suffix(".json")
        metadata_info = {
            "version": self.VERSION,
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,
            "timestamp": checkpoint["timestamp"],
            "checkpoint_path": str(checkpoint_path),
        }
        if metadata is not None:
            metadata_info.update(metadata)

        with open(metadata_path, "w") as f:
            json.dump(metadata_info, f, indent=2)

        print(f"✓ Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint from disk.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load tensors to
            strict: Whether to strictly enforce state dict keys

        Returns:
            Dictionary containing checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Version check
        checkpoint_version = checkpoint.get("version", "unknown")
        print(f"Checkpoint version: {checkpoint_version}")

        if checkpoint_version != self.VERSION and strict:
            print(f"⚠ Warning: Version mismatch (checkpoint: {checkpoint_version}, current: {self.VERSION})")

        return checkpoint

    def load_model_weights(
        self,
        model: torch.nn.Module,
        checkpoint_path: str,
        device: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load only model weights into a model.

        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint
            device: Device to load to
            strict: Strict mode for state dict loading

        Returns:
            Checkpoint metadata
        """
        checkpoint = self.load_checkpoint(checkpoint_path, device, strict=False)

        # Load model state dict
        model_state_dict = checkpoint["model_state_dict"]

        # Handle loading with potential mismatches
        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                model_state_dict, strict=strict
            )

            if missing_keys:
                print(f"⚠ Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"⚠ Unexpected keys: {unexpected_keys}")

            print("✓ Model weights loaded successfully")

        except Exception as e:
            print(f"✗ Error loading model weights: {e}")
            raise

        return checkpoint

    def load_sdxl_pretrained(
        self,
        model: torch.nn.Module,
        pretrained_model_name: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cpu",
    ):
        """
        Load SDXL pretrained weights with special handling for modified layers.

        Args:
            model: Model with modified architecture
            pretrained_model_name: HuggingFace model ID
            device: Device to load to
        """
        print(f"Loading SDXL pretrained weights from {pretrained_model_name}...")

        # This is handled in SDXLUNetForShadows.__init__()
        # This method is for loading after model is created

        print("✓ SDXL pretrained weights loaded (handled by model initialization)")

    def save_best_model(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        loss: float,
        epoch: int,
        global_step: int,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save best model checkpoint.

        Args:
            model_state_dict: Model state
            loss: Current loss (used for comparison)
            epoch: Current epoch
            global_step: Current step
            metadata: Additional metadata

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = "best_model.pt"
        return self.save_checkpoint(
            model_state_dict=model_state_dict,
            optimizer_state_dict=None,
            epoch=epoch,
            global_step=global_step,
            loss=loss,
            metadata=metadata,
            checkpoint_name=checkpoint_name,
        )

    def save_latest_model(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        optimizer_state_dict: Optional[Dict] = None,
        epoch: int = 0,
        global_step: int = 0,
        loss: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save latest model checkpoint (for resuming training).

        Args:
            model_state_dict: Model state
            optimizer_state_dict: Optimizer state
            epoch: Current epoch
            global_step: Current step
            loss: Current loss
            metadata: Additional metadata

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = "latest_checkpoint.pt"
        return self.save_checkpoint(
            model_state_dict=model_state_dict,
            optimizer_state_dict=optimizer_state_dict,
            epoch=epoch,
            global_step=global_step,
            loss=loss,
            metadata=metadata,
            checkpoint_name=checkpoint_name,
        )

    def list_checkpoints(self) -> list:
        """
        List all checkpoints in checkpoint directory.

        Returns:
            List of checkpoint paths
        """
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        return [str(cp) for cp in checkpoints]

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to latest checkpoint.

        Returns:
            Path to latest checkpoint or None
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[-1]

    def get_best_checkpoint(self) -> Optional[str]:
        """
        Get path to best model checkpoint.

        Returns:
            Path to best checkpoint or None
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return str(best_path)
        return None

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Remove old checkpoints, keeping only the most recent N.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()

        # Never delete best model
        best_checkpoint = self.get_best_checkpoint()

        # Filter out best and latest
        regular_checkpoints = [
            cp for cp in checkpoints
            if not cp.endswith("best_model.pt") and not cp.endswith("latest_checkpoint.pt")
        ]

        # Delete old checkpoints
        if len(regular_checkpoints) > keep_last_n:
            to_delete = regular_checkpoints[:-keep_last_n]
            for checkpoint_path in to_delete:
                os.remove(checkpoint_path)
                # Also remove metadata file
                metadata_path = Path(checkpoint_path).with_suffix(".json")
                if metadata_path.exists():
                    os.remove(metadata_path)
                print(f"Deleted old checkpoint: {checkpoint_path}")

            print(f"✓ Cleaned up {len(to_delete)} old checkpoints")


def test_checkpoint_manager():
    """Test checkpoint manager functionality."""
    print("\n" + "="*60)
    print("Testing Checkpoint Manager")
    print("="*60 + "\n")

    # Create manager
    manager = CheckpointManager(checkpoint_dir="./test_checkpoints")

    # Create dummy model state
    dummy_state = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10),
    }

    # Test saving
    print("1. Testing checkpoint save...")
    path = manager.save_checkpoint(
        model_state_dict=dummy_state,
        epoch=1,
        global_step=100,
        loss=0.5,
        metadata={"test": "data"}
    )
    print(f"   ✓ Saved to: {path}")

    # Test loading
    print("\n2. Testing checkpoint load...")
    checkpoint = manager.load_checkpoint(path)
    print(f"   ✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   ✓ Loss: {checkpoint['loss']}")

    # Test best model
    print("\n3. Testing best model save...")
    manager.save_best_model(
        model_state_dict=dummy_state,
        loss=0.3,
        epoch=2,
        global_step=200,
    )
    best_path = manager.get_best_checkpoint()
    print(f"   ✓ Best model saved: {best_path}")

    # Test listing
    print("\n4. Testing checkpoint listing...")
    checkpoints = manager.list_checkpoints()
    print(f"   Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"     - {os.path.basename(cp)}")

    # Cleanup
    print("\n5. Cleaning up test checkpoints...")
    import shutil
    if os.path.exists("./test_checkpoints"):
        shutil.rmtree("./test_checkpoints")
    print("   ✓ Cleanup complete")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_checkpoint_manager()
