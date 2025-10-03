"""
Debugging and visualization utilities for shadow generation model.

Provides tools for:
- Shape verification at each step
- Activation visualization
- Gradient flow checking
- Memory profiling
- Error diagnostics
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict
import time


class ShapeDebugger:
    """
    Debug tensor shapes throughout the model.

    Helps identify shape mismatches and dimension issues.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize shape debugger.

        Args:
            enabled: Whether debugging is active
        """
        self.enabled = enabled
        self.shape_log = []

    def log(self, name: str, tensor: torch.Tensor, extra_info: str = ""):
        """
        Log tensor shape.

        Args:
            name: Identifier for this tensor
            tensor: Tensor to log
            extra_info: Additional information
        """
        if not self.enabled:
            return

        shape_str = f"{name:30s} | Shape: {str(tuple(tensor.shape)):20s}"
        if extra_info:
            shape_str += f" | {extra_info}"

        self.shape_log.append(shape_str)
        print(shape_str)

    def verify_shape(
        self,
        name: str,
        tensor: torch.Tensor,
        expected_shape: Tuple[int, ...],
        allow_batch: bool = True,
    ):
        """
        Verify tensor has expected shape.

        Args:
            name: Tensor identifier
            tensor: Tensor to check
            expected_shape: Expected shape (excluding batch if allow_batch=True)
            allow_batch: Whether to ignore batch dimension

        Raises:
            AssertionError if shape doesn't match
        """
        if not self.enabled:
            return

        actual_shape = tuple(tensor.shape)

        if allow_batch:
            # Compare all dimensions except batch
            if actual_shape[1:] != expected_shape[1:]:
                raise AssertionError(
                    f"Shape mismatch for {name}:\n"
                    f"  Expected: (B, {expected_shape[1:]})\n"
                    f"  Got:      {actual_shape}"
                )
        else:
            if actual_shape != expected_shape:
                raise AssertionError(
                    f"Shape mismatch for {name}:\n"
                    f"  Expected: {expected_shape}\n"
                    f"  Got:      {actual_shape}"
                )

        self.log(name, tensor, "✓ Shape verified")

    def print_summary(self):
        """Print summary of all logged shapes."""
        print("\n" + "="*70)
        print("Shape Debug Summary")
        print("="*70)
        for entry in self.shape_log:
            print(entry)
        print("="*70 + "\n")

    def clear(self):
        """Clear shape log."""
        self.shape_log = []


class ActivationMonitor:
    """
    Monitor activations during forward pass.

    Helps identify:
    - Dead neurons
    - Exploding activations
    - Distribution shifts
    """

    def __init__(self):
        """Initialize activation monitor."""
        self.activations = defaultdict(list)
        self.hooks = []

    def register_hooks(self, model: nn.Module, layer_names: Optional[List[str]] = None):
        """
        Register forward hooks on model layers.

        Args:
            model: Model to monitor
            layer_names: Specific layers to monitor (None = all)
        """
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name].append({
                        "mean": output.mean().item(),
                        "std": output.std().item(),
                        "min": output.min().item(),
                        "max": output.max().item(),
                        "shape": tuple(output.shape),
                    })
            return hook

        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                if isinstance(module, (nn.Conv2d, nn.Linear, nn.GroupNorm)):
                    handle = module.register_forward_hook(hook_fn(name))
                    self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def print_summary(self):
        """Print activation statistics."""
        print("\n" + "="*70)
        print("Activation Monitor Summary")
        print("="*70)
        print(f"{'Layer':<40s} | {'Mean':>10s} | {'Std':>10s} | {'Min':>10s} | {'Max':>10s}")
        print("-"*70)

        for name, stats_list in self.activations.items():
            if stats_list:
                stats = stats_list[-1]  # Latest activation
                print(
                    f"{name:<40s} | "
                    f"{stats['mean']:>10.4f} | "
                    f"{stats['std']:>10.4f} | "
                    f"{stats['min']:>10.4f} | "
                    f"{stats['max']:>10.4f}"
                )

        print("="*70 + "\n")

    def check_for_issues(self):
        """Check for common activation issues."""
        issues = []

        for name, stats_list in self.activations.items():
            if stats_list:
                stats = stats_list[-1]

                # Check for dead neurons (all zeros)
                if abs(stats['mean']) < 1e-7 and stats['std'] < 1e-7:
                    issues.append(f"⚠ Dead neurons in {name}")

                # Check for exploding activations
                if abs(stats['mean']) > 1e3 or abs(stats['max']) > 1e4:
                    issues.append(f"⚠ Exploding activations in {name}")

                # Check for NaN/Inf
                if np.isnan(stats['mean']) or np.isinf(stats['mean']):
                    issues.append(f"✗ NaN/Inf in {name}")

        if issues:
            print("\n" + "="*70)
            print("Activation Issues Detected:")
            print("="*70)
            for issue in issues:
                print(issue)
            print("="*70 + "\n")
        else:
            print("✓ No activation issues detected")

        return issues


class GradientMonitor:
    """
    Monitor gradients during backward pass.

    Helps identify:
    - Vanishing gradients
    - Exploding gradients
    - Dead parameters
    """

    def __init__(self):
        """Initialize gradient monitor."""
        self.gradients = defaultdict(list)
        self.hooks = []

    def register_hooks(self, model: nn.Module):
        """
        Register backward hooks on model parameters.

        Args:
            model: Model to monitor
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                def hook_fn(name):
                    def hook(grad):
                        if grad is not None:
                            self.gradients[name].append({
                                "mean": grad.mean().item(),
                                "std": grad.std().item(),
                                "norm": grad.norm().item(),
                                "min": grad.min().item(),
                                "max": grad.max().item(),
                            })
                    return hook

                handle = param.register_hook(hook_fn(name))
                self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all gradient hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def print_summary(self):
        """Print gradient statistics."""
        print("\n" + "="*70)
        print("Gradient Monitor Summary")
        print("="*70)
        print(f"{'Parameter':<40s} | {'Mean':>10s} | {'Std':>10s} | {'Norm':>10s}")
        print("-"*70)

        for name, grad_list in self.gradients.items():
            if grad_list:
                grad_stats = grad_list[-1]
                print(
                    f"{name:<40s} | "
                    f"{grad_stats['mean']:>10.4e} | "
                    f"{grad_stats['std']:>10.4e} | "
                    f"{grad_stats['norm']:>10.4e}"
                )

        print("="*70 + "\n")

    def check_for_issues(self):
        """Check for gradient issues."""
        issues = []

        for name, grad_list in self.gradients.items():
            if grad_list:
                grad = grad_list[-1]

                # Check for vanishing gradients
                if grad['norm'] < 1e-7:
                    issues.append(f"⚠ Vanishing gradient in {name}")

                # Check for exploding gradients
                if grad['norm'] > 1e3:
                    issues.append(f"⚠ Exploding gradient in {name}")

                # Check for NaN/Inf
                if np.isnan(grad['mean']) or np.isinf(grad['mean']):
                    issues.append(f"✗ NaN/Inf gradient in {name}")

        if issues:
            print("\n" + "="*70)
            print("Gradient Issues Detected:")
            print("="*70)
            for issue in issues:
                print(issue)
            print("="*70 + "\n")
        else:
            print("✓ No gradient issues detected")

        return issues


class MemoryProfiler:
    """
    Profile memory usage during training/inference.
    """

    def __init__(self):
        """Initialize memory profiler."""
        self.snapshots = []

    def snapshot(self, label: str = ""):
        """
        Take memory snapshot.

        Args:
            label: Label for this snapshot
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3

            self.snapshots.append({
                "label": label,
                "allocated": allocated,
                "reserved": reserved,
                "max_allocated": max_allocated,
            })
        else:
            self.snapshots.append({
                "label": label,
                "allocated": 0,
                "reserved": 0,
                "max_allocated": 0,
            })

    def print_summary(self):
        """Print memory usage summary."""
        print("\n" + "="*70)
        print("Memory Profile Summary")
        print("="*70)

        if not torch.cuda.is_available():
            print("CUDA not available - no GPU memory to profile")
            print("="*70 + "\n")
            return

        print(f"{'Label':<30s} | {'Allocated':>12s} | {'Reserved':>12s} | {'Max':>12s}")
        print("-"*70)

        for snap in self.snapshots:
            print(
                f"{snap['label']:<30s} | "
                f"{snap['allocated']:>10.2f} GB | "
                f"{snap['reserved']:>10.2f} GB | "
                f"{snap['max_allocated']:>10.2f} GB"
            )

        print("="*70 + "\n")

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


class PerformanceProfiler:
    """
    Profile performance (time) of different operations.
    """

    def __init__(self):
        """Initialize performance profiler."""
        self.timings = defaultdict(list)
        self.current_timers = {}

    def start(self, label: str):
        """Start timing an operation."""
        self.current_timers[label] = time.time()

    def stop(self, label: str):
        """Stop timing an operation."""
        if label in self.current_timers:
            elapsed = time.time() - self.current_timers[label]
            self.timings[label].append(elapsed)
            del self.current_timers[label]
            return elapsed
        return None

    def print_summary(self):
        """Print timing summary."""
        print("\n" + "="*70)
        print("Performance Profile Summary")
        print("="*70)
        print(f"{'Operation':<30s} | {'Mean':>12s} | {'Std':>12s} | {'Min':>12s} | {'Max':>12s}")
        print("-"*70)

        for label, times in self.timings.items():
            if times:
                mean_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)

                print(
                    f"{label:<30s} | "
                    f"{mean_time:>10.4f} s | "
                    f"{std_time:>10.4f} s | "
                    f"{min_time:>10.4f} s | "
                    f"{max_time:>10.4f} s"
                )

        print("="*70 + "\n")


def test_debugging_tools():
    """Test debugging utilities."""
    print("\n" + "="*70)
    print("Testing Debugging Tools")
    print("="*70 + "\n")

    # Test shape debugger
    print("1. Testing Shape Debugger...")
    debugger = ShapeDebugger(enabled=True)
    dummy_tensor = torch.randn(2, 3, 256, 256)
    debugger.log("Input Tensor", dummy_tensor, "RGB image")
    debugger.verify_shape("Input Tensor", dummy_tensor, (2, 3, 256, 256))
    print("   ✓ Shape debugger works\n")

    # Test activation monitor
    print("2. Testing Activation Monitor...")
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
    )
    monitor = ActivationMonitor()
    monitor.register_hooks(model)

    with torch.no_grad():
        _ = model(dummy_tensor)

    monitor.print_summary()
    monitor.check_for_issues()
    monitor.remove_hooks()
    print("   ✓ Activation monitor works\n")

    # Test memory profiler
    print("3. Testing Memory Profiler...")
    mem_profiler = MemoryProfiler()
    mem_profiler.snapshot("Initial")
    large_tensor = torch.randn(1000, 1000, 100)
    mem_profiler.snapshot("After allocation")
    del large_tensor
    mem_profiler.snapshot("After cleanup")
    mem_profiler.print_summary()
    print("   ✓ Memory profiler works\n")

    # Test performance profiler
    print("4. Testing Performance Profiler...")
    perf_profiler = PerformanceProfiler()

    perf_profiler.start("dummy_operation")
    time.sleep(0.1)
    perf_profiler.stop("dummy_operation")

    perf_profiler.start("another_operation")
    time.sleep(0.05)
    perf_profiler.stop("another_operation")

    perf_profiler.print_summary()
    print("   ✓ Performance profiler works\n")

    print("="*70)
    print("All debugging tools tested successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_debugging_tools()
