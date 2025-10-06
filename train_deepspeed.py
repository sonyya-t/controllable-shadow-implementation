"""
Training script with DeepSpeed ZeRO-Offload for 40GB GPU.

This solves the OOM issue by offloading optimizer states to CPU RAM.
"""

import torch
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import time
from typing import Dict
import deepspeed

from controllable_shadow.models import ShadowDiffusionModel, CheckpointManager
from controllable_shadow.data import create_dataloaders
from controllable_shadow.utils.debugging import MemoryProfiler, PerformanceProfiler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train shadow generation model with DeepSpeed")

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
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size")

    # Training
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--max_iterations", type=int, default=150000,
                        help="Max iterations")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Warmup steps")

    # Optimization
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing")
    parser.add_argument("--cpu_offload", action="store_true",
                        help="Offload VAE to CPU")
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

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json",
                        help="DeepSpeed config file")

    return parser.parse_args()


def create_deepspeed_config(args):
    """Create DeepSpeed configuration for ZeRO-Offload."""
    config = {
        "train_batch_size": args.batch_size * args.gradient_accumulation,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "gradient_clipping": 1.0,
        "steps_per_print": args.log_every,

        # FP16 training
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },

        # ZeRO-Offload: Stage 2 with CPU offloading
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
        },

        # Optimizer
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },

        # Scheduler
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_steps
            }
        }
    }

    # Save config
    config_path = Path(args.checkpoint_dir) / "ds_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return str(config_path)


def train(args):
    """Main training function with DeepSpeed."""

    # Setup
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = checkpoint_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Config saved to {config_path}")

    # Initialize model
    print("Initializing model and data...")
    model = ShadowDiffusionModel(
        conditioning_strategy=args.conditioning_strategy,
        image_size=(args.image_size, args.image_size),
    )

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        print("\n⚡ Enabling gradient checkpointing")
        if hasattr(model.unet, 'unet') and hasattr(model.unet.unet, 'unet'):
            unet = model.unet.unet.unet
            if hasattr(unet, 'enable_gradient_checkpointing'):
                unet.enable_gradient_checkpointing()
                print("✓ Gradient checkpointing enabled")

    model.freeze_vae()
    model.print_model_summary()

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
        root_dir=args.data_dir,
    )

    # Create DeepSpeed config
    ds_config_path = create_deepspeed_config(args)

    # Initialize DeepSpeed
    print("\n" + "="*70)
    print("Initializing DeepSpeed with ZeRO-Offload...")
    print("="*70)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.get_trainable_parameters(),
        config=ds_config_path
    )

    print(f"\n✓ DeepSpeed initialized")
    print(f"  Optimizer states will be offloaded to CPU RAM")
    print(f"  Expected VRAM usage: ~17-20 GB (vs ~40 GB without offload)")

    # Training loop
    global_step = 0
    epoch = 0
    best_loss = float('inf')

    print(f"\n{'='*70}")
    print("Training Configuration")
    print(f"{'='*70}")
    print(f"Model: ShadowDiffusionModel ({args.conditioning_strategy})")
    print(f"Dataset: {args.dataset_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Learning rate: {args.lr}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"DeepSpeed ZeRO: Stage 2 with CPU offload")
    print(f"{'='*70}\n")

    print(f"Starting training for {args.max_iterations} iterations...")
    start_time = time.time()

    try:
        while global_step < args.max_iterations:
            model_engine.train()
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                device = model_engine.device
                object_image = batch['object_image'].to(device)
                mask = batch['mask'].to(device)
                shadow_map = batch['shadow_map'].to(device)
                theta = batch['theta'].to(device)
                phi = batch['phi'].to(device)
                size = batch['size'].to(device)

                # Forward pass
                loss_dict = model.compute_rectified_flow_loss(
                    object_image, mask, shadow_map, theta, phi, size
                )
                loss = loss_dict['loss']

                # Backward pass (DeepSpeed handles scaling and gradient accumulation)
                model_engine.backward(loss)
                model_engine.step()

                global_step += 1
                epoch_loss += loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.6f}",
                    'step': global_step
                })

                # Logging
                if global_step % args.log_every == 0:
                    log_str = f"[Train] Step {global_step} | loss: {loss.item():.6f}"
                    print(log_str)

                # Save checkpoint
                if global_step % args.save_every == 0:
                    ckpt_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                    model_engine.save_checkpoint(str(checkpoint_dir), tag=f"step_{global_step}")
                    print(f"✓ Checkpoint saved: {ckpt_path}")

                # Check if max iterations reached
                if global_step >= args.max_iterations:
                    print(f"\n✓ Reached max iterations: {args.max_iterations}")
                    break

            avg_loss = epoch_loss / num_batches
            print(f"\nEpoch {epoch} - Average Loss: {avg_loss:.6f}")
            epoch += 1

            if global_step >= args.max_iterations:
                break

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        model_engine.save_checkpoint(str(checkpoint_dir), tag="interrupted")

    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise

    finally:
        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training finished!")
        print(f"Total time: {elapsed/3600:.2f} hours")
        print(f"Final step: {global_step}")
        print(f"{'='*70}\n")

        # Save final checkpoint
        model_engine.save_checkpoint(str(checkpoint_dir), tag="final")


def main():
    """Main entry point."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
