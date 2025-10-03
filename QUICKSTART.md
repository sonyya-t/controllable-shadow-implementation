# Controllable Shadow Generation - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Controllable-shadow

# Install dependencies (requires Python 3.8-3.12)
pip install -e .

# Note: First run will download ~13GB SDXL weights from HuggingFace
```

---

## 💡 Basic Usage

### Generate a Shadow Map

```python
from controllable_shadow.models import create_shadow_model
import torch
from PIL import Image
import torchvision.transforms as T

# 1. Create model
model = create_shadow_model(device="cuda")
model.eval()

# 2. Load and preprocess object image
transform = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

object_img = Image.open("path/to/object.png").convert("RGB")
object_tensor = transform(object_img).unsqueeze(0).cuda()

# 3. Create mask (assumes white background)
mask = (object_tensor.mean(dim=1, keepdim=True) > 0.9).float()

# 4. Set light parameters
theta = torch.tensor([30.0]).cuda()   # Vertical angle (0-45°)
phi = torch.tensor([45.0]).cuda()     # Horizontal angle (0-360°)
size = torch.tensor([4.0]).cuda()     # Softness (2-8)

# 5. Generate shadow (single-step = fast!)
with torch.no_grad():
    shadow_map = model.sample(
        object_tensor, mask, theta, phi, size, num_steps=1
    )

# 6. Save result
shadow_img = T.ToPILImage()(shadow_map[0])
shadow_img.save("shadow_output.png")

print("✓ Shadow generated successfully!")
```

---

## 🎨 Light Parameter Guide

### Theta (θ) - Vertical Angle
```
θ = 0°  → Light directly overhead (short shadow)
θ = 15° → Morning/evening light
θ = 30° → Typical daylight (recommended)
θ = 45° → Low sun (long shadow)
```

### Phi (φ) - Horizontal Direction
```
φ = 0°   → Shadow points right
φ = 90°  → Shadow points down
φ = 180° → Shadow points left
φ = 270° → Shadow points up
```

### Size (s) - Shadow Softness
```
s = 2 → Hard shadow (small light source)
s = 4 → Medium shadow (recommended)
s = 6 → Soft shadow
s = 8 → Very soft shadow (large/diffuse light)
```

---

## 🔧 Advanced Usage

### Multi-Step Sampling (Higher Quality)

```python
# Trade speed for quality
shadow_map = model.sample(
    object_tensor, mask, theta, phi, size,
    num_steps=4  # More steps = better quality
)
```

### Batch Generation

```python
# Generate multiple shadows at once
batch_size = 4
object_batch = torch.stack([object_tensor] * batch_size).squeeze(1)
mask_batch = torch.stack([mask] * batch_size).squeeze(1)

theta_batch = torch.tensor([15., 30., 45., 20.]).cuda()
phi_batch = torch.tensor([0., 90., 180., 270.]).cuda()
size_batch = torch.tensor([3., 4., 5., 6.]).cuda()

shadow_maps = model.sample(
    object_batch, mask_batch,
    theta_batch, phi_batch, size_batch
)
# Output: (4, 1, 1024, 1024)
```

### Training

```python
from controllable_shadow.models import ShadowDiffusionModel
import torch.optim as optim

model = ShadowDiffusionModel().cuda()
model.train()

optimizer = optim.AdamW(
    model.get_trainable_parameters(),
    lr=1e-5
)

# Training loop
for batch in dataloader:
    loss_dict = model.compute_rectified_flow_loss(
        object_image=batch['object'],
        mask=batch['mask'],
        shadow_map_target=batch['shadow'],
        theta=batch['theta'],
        phi=batch['phi'],
        size=batch['size'],
    )

    optimizer.zero_grad()
    loss_dict['loss'].backward()
    optimizer.step()
```

---

## 🐛 Debugging

### Enable Shape Debugging

```python
from controllable_shadow.utils.debugging import ShapeDebugger

debugger = ShapeDebugger(enabled=True)

# Inside your code
debugger.log("Object tensor", object_tensor)
debugger.verify_shape("Shadow output", shadow_map, (1, 1, 1024, 1024))
debugger.print_summary()
```

### Monitor Activations

```python
from controllable_shadow.utils.debugging import ActivationMonitor

monitor = ActivationMonitor()
monitor.register_hooks(model)

output = model(...)

monitor.print_summary()
monitor.check_for_issues()  # Detects dead neurons, explosions, NaN
monitor.remove_hooks()
```

### Profile Memory

```python
from controllable_shadow.utils.debugging import MemoryProfiler

profiler = MemoryProfiler()

profiler.snapshot("Initial")
shadow_map = model.sample(...)
profiler.snapshot("After generation")

profiler.print_summary()
```

---

## 📦 Model Checkpoints

### Save Model

```python
from controllable_shadow.models import CheckpointManager

manager = CheckpointManager(checkpoint_dir="./checkpoints")

# Save training checkpoint
manager.save_checkpoint(
    model_state_dict=model.state_dict(),
    optimizer_state_dict=optimizer.state_dict(),
    epoch=10,
    global_step=5000,
    loss=0.05,
)

# Save best model
manager.save_best_model(
    model_state_dict=model.state_dict(),
    loss=0.03,
    epoch=15,
    global_step=7500,
)
```

### Load Model

```python
# Load from checkpoint
model = create_shadow_model(
    pretrained_path="./checkpoints/best_model.pt",
    device="cuda"
)

# Or manually
checkpoint = manager.load_checkpoint("./checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## ⚙️ Model Configuration

### Use Different Conditioning Strategy

```python
from controllable_shadow.models import ShadowDiffusionModel

# Additive (default, faster)
model_add = ShadowDiffusionModel(conditioning_strategy="additive")

# Concatenative (more expressive)
model_concat = ShadowDiffusionModel(conditioning_strategy="concat")
```

### Custom Latent Size

```python
# For 512×512 images
model = ShadowDiffusionModel(
    latent_size=(64, 64),  # 512÷8 = 64
    image_size=(512, 512)
)
```

---

## 🎯 Common Use Cases

### 1. Shadow Map Generation Only

```python
# Just generate the shadow map
shadow_map = model.sample(object_tensor, mask, theta, phi, size)
# shadow_map: (B, 1, H, W) in [0, 1]
```

### 2. Composite Shadow onto Background

```python
# After generating shadow_map
background = Image.open("background.jpg")
object_img = Image.open("object.png")

# Resize shadow to match background
shadow_np = shadow_map[0, 0].cpu().numpy()
shadow_resized = Image.fromarray((shadow_np * 255).astype('uint8'))
shadow_resized = shadow_resized.resize(background.size)

# Composite (manual blending)
# ... use PIL.ImageChops or custom blending
```

### 3. Sweep Light Parameters

```python
# Generate shadows for different angles
results = []

for angle in range(0, 360, 30):
    phi = torch.tensor([float(angle)]).cuda()
    shadow = model.sample(object_tensor, mask, theta, phi, size)
    results.append(shadow)

# results: List of 12 shadow maps
```

---

## 📊 Performance Tips

### Inference Optimization

```python
# Use fp16 for faster inference
model = model.half()
object_tensor = object_tensor.half()

# Disable gradient computation
with torch.no_grad():
    shadow_map = model.sample(...)

# Use compiled model (PyTorch 2.0+)
model = torch.compile(model)
```

### Batch Processing

```python
# Process multiple images together
batch_size = 8  # Adjust based on GPU memory

for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    shadows = model.sample(batch, ...)
```

---

## 🆘 Troubleshooting

### Out of Memory

```python
# Reduce batch size
batch_size = 1

# Use gradient checkpointing (during training)
model.unet.unet.enable_gradient_checkpointing()

# Use fp16
model = model.half()

# Reduce latent size (lower resolution)
model = ShadowDiffusionModel(latent_size=(64, 64))
```

### Slow Inference

```python
# Use single-step sampling
num_steps = 1  # Fastest

# Compile model (PyTorch 2.0+)
model = torch.compile(model, mode="reduce-overhead")

# Use fp16
model = model.half()
```

### Poor Shadow Quality

```python
# Increase sampling steps
num_steps = 4  # or 8, 16

# Check light parameters are in valid range
assert 0 <= theta <= 45
assert 0 <= phi <= 360
assert 2 <= size <= 8

# Ensure proper mask (binary 0/1)
mask = (mask > 0.5).float()
```

---

## 📚 Example Scripts

Check the `examples/` directory for:
- `basic_generation.py` - Simple shadow generation
- `batch_processing.py` - Process multiple images
- `parameter_sweep.py` - Explore light parameters
- `training_example.py` - Training loop template
- `checkpoint_demo.py` - Save/load checkpoints

---

## 🎓 Learn More

- **Full Documentation**: See `MODEL_ARCHITECTURE_COMPLETE.md`
- **Implementation Details**: See `IMPLEMENTATION_PROGRESS.md`
- **Paper**: [arXiv:2412.11972](https://arxiv.org/abs/2412.11972)
- **Project Page**: https://gojasper.github.io/controllable-shadow-generation-project/

---

## ⭐ Quick Reference

```python
# Minimal working example
from controllable_shadow.models import create_shadow_model
import torch

model = create_shadow_model(device="cuda")
shadow = model.sample(
    object_image,  # (B, 3, 1024, 1024) in [-1,1]
    mask,          # (B, 1, 1024, 1024) in {0,1}
    theta,         # (B,) in [0, 45]
    phi,           # (B,) in [0, 360]
    size,          # (B,) in [2, 8]
    num_steps=1    # 1=fast, 4+=quality
)
# Output: (B, 1, 1024, 1024) in [0, 1]
```

---

**Happy Shadow Generating! 🌟**

For issues or questions, check the documentation or open an issue on GitHub.
