"""
Training script for controllable shadow generation.

Based on paper specifications:
- AdamW optimizer with lr=1e-5
- Batch size 2
- 150k iterations
- Rectified flow loss
"""

import torch
import torch.optim as optim
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import time
from typing import Dict

from controllable_shadow.models import ShadowDiffusionModel, CheckpointManager
from controllable_shadow.data import create_dataloaders
from controllable_shadow.utils.debugging import MemoryProfiler, PerformanceProfiler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train shadow generation model")

    # Model
    parser.add_argument("--conditioning_strategy", type=str, default="additive",
                        choices=["additive", "concat"],
                        help="Conditioning strategy")

    # Data
    parser.add_argument("--dataset_type", type=str, default="benchmark",
                        choices=["benchmark", "custom"],
                        help="Dataset type")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (for custom dataset)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory for HuggingFace datasets")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="Image size (1024 for paper, use 512 to save memory)")

    # Training (from paper)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (paper uses 2)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (paper uses 1e-5)")
    parser.add_argument("--max_iterations", type=int, default=150000,
                        help="Max iterations (paper uses 150k)")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Warmup steps")

    # Optimization
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing (saves memory, slower)")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Offload VAE to CPU (saves VRAM, slower)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of dataloader workers")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save checkpoint every N iterations")

    # Logging
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log every N iterations")
    parser.add_argument("--eval_every", type=int, default=1000,
                        help="Evaluate every N iterations")

    # Device
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use")

    return parser.parse_args()


class Trainer:
    """Training manager for shadow generation model."""

    def __init__(self, args):
        """
        Initialize trainer.

        Args:
            args: Parsed command line arguments
        """
        self.args = args
        self.device = torch.device(args.device)
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

        # Setup directories
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_config()

        # Initialize components
        print("Initializing model and data...")
        self.model = self._create_model()
        self.train_loader, self.val_loader = self._create_dataloaders()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.checkpoint_manager = CheckpointManager(args.checkpoint_dir)

        # Profilers
        self.mem_profiler = MemoryProfiler()
        self.perf_profiler = PerformanceProfiler()

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

        # Resume if specified
        if args.resume_from:
            self._resume_from_checkpoint(args.resume_from)

        print(f"\n{'='*70}")
        print("Training Configuration")
        print(f"{'='*70}")
        print(f"Model: ShadowDiffusionModel ({args.conditioning_strategy})")
        print(f"Dataset: {args.dataset_type}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.lr}")
        print(f"Max iterations: {args.max_iterations}")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"Gradient checkpointing: {args.gradient_checkpointing}")
        print(f"Gradient accumulation: {args.gradient_accumulation}")
        print(f"{'='*70}\n")

    def _create_model(self) -> ShadowDiffusionModel:
        """Create shadow generation model."""
        model = ShadowDiffusionModel(
            conditioning_strategy=self.args.conditioning_strategy,
            image_size=(self.args.image_size, self.args.image_size),
        )

        model = model.to(self.device)
        model.freeze_vae()  # Ensure VAE is frozen

        # Enable gradient checkpointing if requested
        if self.args.gradient_checkpointing:
            print("\n⚡ Enabling gradient checkpointing (saves memory, ~20% slower)")
            self._enable_gradient_checkpointing(model)

        model.train()

        # Print model summary
        model.print_model_summary()

        return model

    def _enable_gradient_checkpointing(self, model):
        """Enable gradient checkpointing on UNet to save memory."""
        try:
            # Enable checkpointing on the SDXL UNet
            if hasattr(model.unet, 'unet') and hasattr(model.unet.unet, 'unet'):
                # Access the actual diffusers UNet
                unet = model.unet.unet.unet
                if hasattr(unet, 'enable_gradient_checkpointing'):
                    unet.enable_gradient_checkpointing()
                    print("✓ Gradient checkpointing enabled on SDXL UNet")
                else:
                    print("⚠ UNet doesn't support gradient_checkpointing method")
            else:
                print("⚠ Could not find UNet to enable checkpointing")
        except Exception as e:
            print(f"⚠ Failed to enable gradient checkpointing: {e}")
            print("  Continuing without it...")

    def _create_dataloaders(self):
        """Create train and validation dataloaders."""
        return create_dataloaders(
            dataset_type=self.args.dataset_type,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            image_size=self.args.image_size,
            cache_dir=self.args.cache_dir,
            root_dir=self.args.data_dir,
        )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer as per paper."""
        return optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.args.warmup_steps:
                return step / self.args.warmup_steps
            return 1.0

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _save_config(self):
        """Save training configuration."""
        config_path = self.checkpoint_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        print(f"✓ Config saved to {config_path}")

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        print(f"Resuming from checkpoint: {checkpoint_path}")

        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path, self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)

        print(f"✓ Resumed from step {self.global_step}, epoch {self.epoch}")

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Batch from dataloader

        Returns:
            Dictionary with loss and metrics
        """
        # Move batch to device (or CPU if offloading VAE)
        device = 'cpu' if self.args.cpu_offload else self.device

        object_image = batch['object_image'].to(device)
        mask = batch['mask'].to(device)
        shadow_map = batch['shadow_map'].to(device)
        theta = batch['theta'].to(self.device)  # Light params stay on GPU
        phi = batch['phi'].to(self.device)
        size = batch['size'].to(self.device)

        # Forward pass (UNet is FP16, VAE is FP32 - no autocast needed)
        loss_dict = self.model.compute_rectified_flow_loss(
            object_image, mask, shadow_map, theta, phi, size
        )

        loss = loss_dict['loss'] / self.args.gradient_accumulation

        # Backward pass (use scaler for FP16 gradients)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return {'loss': loss_dict['loss'].item()}

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            self.perf_profiler.start("batch")

            # Training step
            metrics = self.train_step(batch)

            # Gradient accumulation
            if (batch_idx + 1) % self.args.gradient_accumulation == 0:
                # Gradient clipping to prevent explosion
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

                self.global_step += 1

            epoch_loss += metrics['loss']
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.6f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                'step': self.global_step
            })

            self.perf_profiler.stop("batch")

            # Logging
            if self.global_step % self.args.log_every == 0:
                self._log_metrics(metrics)

            # Evaluation
            if self.global_step % self.args.eval_every == 0:
                val_loss = self.evaluate()
                self._log_metrics({'val_loss': val_loss}, prefix="Validation")

                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_best_checkpoint(val_loss)

            # Save checkpoint
            if self.global_step % self.args.save_every == 0:
                self._save_checkpoint()

            # Check if max iterations reached
            if self.global_step >= self.args.max_iterations:
                print(f"\n✓ Reached max iterations: {self.args.max_iterations}")
                return False  # Stop training

        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {self.epoch} - Average Loss: {avg_loss:.6f}")

        return True  # Continue training

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Evaluate on validation set.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            object_image = batch['object_image'].to(self.device)
            mask = batch['mask'].to(self.device)
            shadow_map = batch['shadow_map'].to(self.device)
            theta = batch['theta'].to(self.device)
            phi = batch['phi'].to(self.device)
            size = batch['size'].to(self.device)

            loss_dict = self.model.compute_rectified_flow_loss(
                object_image, mask, shadow_map, theta, phi, size
            )

            total_loss += loss_dict['loss'].item()
            num_batches += 1

        self.model.train()
        return total_loss / num_batches

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = "Train"):
        """Log metrics to console and file."""
        log_str = f"[{prefix}] Step {self.global_step} | "
        log_str += " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        print(log_str)

        # Also write to log file
        log_file = self.checkpoint_dir / "training.log"
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')

    def _save_checkpoint(self):
        """Save training checkpoint."""
        self.checkpoint_manager.save_latest_model(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            epoch=self.epoch,
            global_step=self.global_step,
            loss=0.0,  # Will be updated
            metadata={
                'args': vars(self.args),
                'best_loss': self.best_loss,
            }
        )

    def _save_best_checkpoint(self, val_loss: float):
        """Save best model checkpoint."""
        self.checkpoint_manager.save_best_model(
            model_state_dict=self.model.state_dict(),
            loss=val_loss,
            epoch=self.epoch,
            global_step=self.global_step,
            metadata={
                'args': vars(self.args),
            }
        )
        print(f"✓ Best model saved (val_loss: {val_loss:.6f})")

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.args.max_iterations} iterations...")
        print(f"{'='*70}\n")

        start_time = time.time()

        try:
            while self.global_step < self.args.max_iterations:
                continue_training = self.train_epoch()
                self.epoch += 1

                if not continue_training:
                    break

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            self._save_checkpoint()

        except Exception as e:
            print(f"\n\nError during training: {e}")
            raise

        finally:
            elapsed = time.time() - start_time
            print(f"\n{'='*70}")
            print(f"Training finished!")
            print(f"Total time: {elapsed/3600:.2f} hours")
            print(f"Final step: {self.global_step}")
            print(f"Best validation loss: {self.best_loss:.6f}")
            print(f"{'='*70}\n")

            # Save final checkpoint
            self._save_checkpoint()

            # Print profiling stats
            self.perf_profiler.print_summary()


def main():
    """Main training function."""
    args = parse_args()

    # Create trainer and start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
