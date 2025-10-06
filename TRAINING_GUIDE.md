# Training Guide - Controllable Shadow Generation

## 🎯 Quick Start (1500 Samples on A100 40GB)

### **Option 1: Safe Training (512×512)**
```bash
python train.py \
  --dataset_type custom \
  --data_dir /path/to/your/1500/samples \
  --image_size 512 \
  --batch_size 8 \
  --mixed_precision \
  --gradient_checkpointing \
  --max_iterations 5000 \
  --save_every 500 \
  --eval_every 500 \
  --log_every 50
```

**Expected**:
- Memory: ~8-12 GB
- Time: ~2-3 hours for 5000 iterations
- Loss should drop to <0.01 (overfitting is EXPECTED on 1500 samples)

---

### **Option 2: Full Resolution (1024×1024)**
```bash
python train.py \
  --dataset_type custom \
  --data_dir /path/to/your/1500/samples \
  --image_size 1024 \
  --batch_size 2 \
  --mixed_precision \
  --gradient_checkpointing \
  --gradient_accumulation 4 \
  --max_iterations 5000 \
  --save_every 500 \
  --eval_every 500 \
  --log_every 50
```

**Expected**:
- Memory: ~25-30 GB (75% of A100)
- Time: ~4-6 hours for 5000 iterations
- Effective batch size: 2 × 4 = 8

---

## 📁 Dataset Structure Required

```
/path/to/your/data/
├── objects/
│   ├── img_00001.png
│   ├── img_00002.png
│   └── ... (1500 images)
├── masks/
│   ├── img_00001.png
│   ├── img_00002.png
│   └── ... (1500 images)
├── shadows/
│   ├── img_00001.png
│   ├── img_00002.png
│   └── ... (1500 images)
└── metadata.json
```

### **metadata.json Format**
```json
{
  "img_00001": {
    "theta": 15.0,
    "phi": 45.0,
    "size": 4.0
  },
  "img_00002": {
    "theta": 30.0,
    "phi": 90.0,
    "size": 5.5
  }
}
```

**Parameter Ranges**:
- `theta`: 0-45 degrees (polar angle)
- `phi`: 0-360 degrees (azimuthal angle)
- `size`: 2-8 (light softness)

---

## 🔍 Pre-Training Checklist

### **1. Verify Dataset**
```bash
python -c "
from controllable_shadow.data import create_dataloaders

train_loader, val_loader = create_dataloaders(
    dataset_type='custom',
    data_dir='/path/to/your/data',
    batch_size=2,
    num_workers=0,
    image_size=512
)

print(f'Train batches: {len(train_loader)}')
print(f'Val batches: {len(val_loader)}')

batch = next(iter(train_loader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f'{k}: {v.shape}, dtype={v.dtype}')
"
```

**Expected Output**:
```
Train batches: 169  (1350 samples / 8 batch_size)
Val batches: 19     (150 samples / 8 batch_size)
object_image: torch.Size([2, 3, 512, 512]), dtype=torch.float32
mask: torch.Size([2, 1, 512, 512]), dtype=torch.float32
shadow_map: torch.Size([2, 1, 512, 512]), dtype=torch.float32
theta: torch.Size([2]), dtype=torch.float32
phi: torch.Size([2]), dtype=torch.float32
size: torch.Size([2]), dtype=torch.float32
```

---

### **2. Test Forward Pass (Optional but Recommended)**
```bash
python test_dtype_fixes.py
```

**Expected Output**:
```
======================================================================
TESTING DTYPE FIXES
======================================================================

1. Creating model...
   ✓ Model created

2. Creating dummy inputs...
   ...

7. Testing backward pass...
   Parameters with gradients: 2567/2567
   ✓ Backward pass works

======================================================================
ALL DTYPE TESTS PASSED! ✅
======================================================================
```

---

### **3. Verify CUDA & GPU**
```bash
nvidia-smi

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

**Expected**:
```
GPU: NVIDIA A100-SXM4-40GB
GPU memory: 40.0 GB
```

---

## 📊 Monitoring Training

### **What to Watch**

1. **Loss Trajectory**
   - Initial: 0.1-0.5 (random predictions)
   - After 500 iter: 0.05-0.1 (learning started)
   - After 2000 iter: 0.01-0.03 (converging)
   - After 5000 iter: <0.01 (overfitting on 1500 samples - THIS IS GOOD)

2. **GPU Utilization**
   ```bash
   watch -n 1 nvidia-smi
   ```
   - Should be 90-100% GPU-Util
   - Memory should stay constant after first 100 iterations

3. **Training Log**
   ```bash
   tail -f checkpoints/training.log
   ```

4. **Learning Rate**
   - Warmup: 0 → 1e-5 over first 1000 steps
   - Then constant at 1e-5

---

## ⚠️ Troubleshooting

### **OOM Error**
```
RuntimeError: CUDA out of memory
```

**Solutions** (try in order):
1. Reduce batch size: `--batch_size 1`
2. Reduce image size: `--image_size 512`
3. Increase gradient accumulation: `--gradient_accumulation 8`
4. Clear cache:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

### **NaN Loss**
```
⚠️  NaN/Inf detected in loss!
```

**Causes & Fixes**:
1. **Bad data**: Check shadow_map range is [0, 1]
   ```python
   assert shadow_map.min() >= 0
   assert shadow_map.max() <= 1
   ```

2. **Exploding gradients**: Add gradient clipping (already in code at line 326)

3. **Learning rate too high**: Reduce to 5e-6

---

### **Slow Training**
```
~1 iteration per second (too slow)
```

**Fixes**:
1. Enable mixed precision: `--mixed_precision` (should already be on)
2. Reduce num_workers: `--num_workers 2`
3. Use smaller images: `--image_size 512`
4. Check dataloader isn't bottleneck:
   ```python
   # Profile dataloader speed
   import time
   start = time.time()
   for i, batch in enumerate(train_loader):
       if i >= 10: break
   print(f"10 batches in {time.time()-start:.2f}s")
   ```

---

### **Dataset Not Loading**
```
KeyError: 'object_image'
```

**Check**:
1. File names match metadata.json keys
2. All three folders exist: objects/, masks/, shadows/
3. Images are PNG format
4. metadata.json is valid JSON

---

## 📈 Expected Metrics

### **For 1500 Samples (Litmus Test)**

| Iteration | Loss  | Behavior |
|-----------|-------|----------|
| 0-100     | 0.2-0.5 | Random predictions |
| 100-500   | 0.1-0.2 | Learning shapes |
| 500-1000  | 0.05-0.1 | Learning light direction |
| 1000-2000 | 0.02-0.05 | Fine-tuning |
| 2000-5000 | <0.02 | **Overfitting** (memorizing dataset) |

**Overfitting is EXPECTED and GOOD** for this litmus test!

Goal: Verify training loop works, not generalization.

---

## 🎯 Success Criteria

**Litmus Test Passes If**:
- ✅ Training runs for 5000 iterations without crashes
- ✅ Loss decreases consistently (no NaN/Inf)
- ✅ GPU memory stays below 35 GB
- ✅ Final loss < 0.01 (overfitting on 1500 samples)
- ✅ Generated shadows look reasonable (even if memorized)

**Then you can**:
1. Scale to full dataset (257k images from paper)
2. Train for 150k iterations
3. Expect actual generalization

---

## 💾 Checkpoints

Saved to `./checkpoints/`:
```
checkpoints/
├── training_config.json  # Your training args
├── training.log          # All training logs
├── latest_model.pt       # Most recent checkpoint
├── best_model.pt         # Best validation loss
└── checkpoint_5000.pt    # Periodic saves
```

**To resume**:
```bash
python train.py \
  --resume_from checkpoints/latest_model.pt \
  [... other args ...]
```

---

## 🚀 After Litmus Test

If training succeeds on 1500 samples:

1. **Get full dataset**:
   - Paper uses 257,612 training images
   - Or use HuggingFace benchmark

2. **Scale training**:
   ```bash
   python train.py \
     --dataset_type benchmark \
     --image_size 1024 \
     --batch_size 2 \
     --gradient_accumulation 4 \
     --max_iterations 150000 \
     --mixed_precision \
     --gradient_checkpointing
   ```

3. **Evaluate on tracks**:
   - Track 1: Softness control
   - Track 2: Horizontal direction
   - Track 3: Vertical direction

---

## 📞 Final Notes

- **First run**: Model will download SDXL weights (~10 GB), takes 5-10 min
- **Checkpoint every 500 iterations**: Don't lose progress
- **Watch the first 100 iterations carefully**: If loss doesn't decrease, stop and debug
- **1500 samples should overfit**: If loss stays >0.05, something is wrong

**Good luck! 🎉**
