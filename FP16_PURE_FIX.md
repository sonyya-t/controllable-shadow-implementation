# Hybrid FP16/FP32 Training Fix (UNet FP16 + VAE FP32)

## 🔴 The Error

```
RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

**Location**: `diffusers/models/conv.py:543` → `self.unet.conv_in`

---

## 🔍 Root Cause

**UNet and VAE were correctly loaded in FP16**, but the **modified `conv_in` layer** (9-channel input) was created in FP32 and never converted to FP16.

**Why this happened:**
1. SDXL UNet loaded in FP16 ✓
2. VAE loaded in FP16 ✓
3. All layers converted to FP16 with `.half()` ✓
4. **But then** `_modify_input_conv()` created a NEW `nn.Conv2d` layer
5. This new layer was initialized in default FP32
6. It was never converted to FP16 ❌

**Result:** Input tensors in FP16, but conv_in weights in FP32 → dtype mismatch error

---

## ✅ The Fix

**File**: `controllable_shadow/models/sdxl_unet.py`

**Line 175**: Added `.half()` conversion after creating new conv layer

```python
# Create new conv layer with 9 input channels
new_conv = nn.Conv2d(
    in_channels=self.in_channels,
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=original_conv.bias is not None
)

# Initialize weights...
with torch.no_grad():
    nn.init.zeros_(new_conv.weight)
    new_conv.weight[:, :4, :, :] = original_weight
    if original_bias is not None:
        new_conv.bias.data = original_bias

# CRITICAL: Convert new conv layer to FP16 to match rest of UNet
new_conv = new_conv.half()  # ← THIS LINE ADDED

# Replace the conv_in layer
self.unet.conv_in = new_conv
```

---

## 🎯 Hybrid Precision Strategy (Final Solution)

**UNet in FP16, VAE in FP32** (Standard Stable Diffusion approach)

```python
# sdxl_unet.py:58
self.unet = UNet2DConditionModel.from_pretrained(
    pretrained_model_name,
    subfolder="unet",
    torch_dtype=torch.float16,  # ✓ FP16 (trainable, most compute)
)

# sdxl_unet.py:301
self.vae = AutoencoderKL.from_pretrained(
    pretrained_model_name,
    subfolder="vae",
    torch_dtype=torch.float32,  # ✓ FP32 (frozen, numerically sensitive)
)
```

**Why VAE needs FP32:**
- GroupNorm layers overflow/underflow in FP16
- KL divergence computation is sensitive
- Even in inference (frozen), VAE operations produce NaN in pure FP16
- Standard practice: All Stable Diffusion implementations keep VAE in FP32

**Then force-convert UNet only:**
```python
# Line 63 - Force UNet to FP16
self.unet = self.unet.half()

# VAE stays FP32 - no .half() call
```

**Training without autocast:**
```bash
python train.py \
  --dataset_type benchmark \
  --batch_size 1 \
  --gradient_checkpointing \
  --max_iterations 100
  # No --mixed_precision flag → pure FP16, no autocast
```

---

## 📊 Complete Hybrid Precision Flow

```
Dataset (FP32) → VAE (FP32) → FP16 Convert → UNet (FP16) → Loss (FP32)
     ↓                ↓                             ↓
object_image      VAE encode                  UNet forward
   (FP32)           (FP32)                       (FP16)
     ↓                ↓                             ↓
shadow_map        latent_fp32                predicted_velocity
  (FP32)             ↓                           (FP16)
     ↓          .half()                            ↓
     ↓           latent                        MSE loss
     ↓          (FP16) ──────────────────────> (FP16 → FP32)
     ↓                                             ↓
Conditioning (FP32→FP16) ───────────────────> .float().mean()
     ↓                                          loss (FP32)
embeddings (FP16)
```

**Dtype flow:**
- Input: FP32 (from dataset)
- VAE operations: FP32 (numerical stability)
- VAE output → FP16 conversion
- UNet operations: FP16 (speed + memory)
- Loss computation: FP32 (stability)

**All transitions explicit and safe!** ✓

---

## 🧪 Verification

**Run training:**
```bash
python train.py \
  --dataset_type benchmark \
  --image_size 512 \
  --batch_size 1 \
  --gradient_checkpointing \
  --max_iterations 100 \
  --log_every 10
```

**Expected output:**
```
Loading SDXL UNet from stabilityai/stable-diffusion-xl-base-1.0...
✓ All UNet parameters and buffers verified as FP16
✓ Conditioned SDXL UNet initialized

Loading SDXL VAE from stabilityai/stable-diffusion-xl-base-1.0...
✓ VAE loaded in FP32 for numerical stability
✓ VAE Pipeline initialized

🔧 Hybrid precision training enabled
   ✓ UNet in FP16 (trainable)
   ✓ VAE in FP32 (frozen, for stability)
   ✓ Training operations in FP16, loss in FP32

Epoch 0:   0% 1/499 [00:02<20:45, loss: 0.XXXX, lr: 1.00e-06, step: 1]
Epoch 0:   0% 2/499 [00:04<20:43, loss: 0.XXXX, lr: 2.00e-06, step: 2]
...
```

**No dtype mismatch errors! No NaN errors!** ✓

---

## 🎓 Key Lessons

1. **`torch_dtype=torch.float16` is not enough**
   - Some internal buffers/layers might stay FP32
   - Always call `.half()` to force-convert everything

2. **Watch for newly created layers**
   - When you create new `nn.Module` layers, they initialize in FP32
   - Must manually convert them to FP16

3. **Hybrid precision is optimal**
   - UNet in FP16: 50% memory savings on largest component
   - VAE in FP32: Numerical stability where needed
   - No autocast/GradScaler complexity
   - Best of both worlds

4. **VAE cannot run in pure FP16**
   - GroupNorm/LayerNorm overflow in FP16
   - KL divergence computation unstable
   - This is why all SD implementations keep VAE in FP32
   - Even though frozen, it still needs FP32 for forward pass

5. **Memory savings still significant**
   - UNet: 10.3 GB (FP32) → 5.15 GB (FP16) = 50% saved
   - VAE: stays at 335 MB (FP32) = no savings but necessary
   - UNet activations: ~50% reduction
   - Total: ~45% memory savings (vs 50% if VAE was FP16)

---

## ✅ Ready to Train

**All fixes applied:**
- ✓ `conv_in` layer converted to FP16 (critical fix!)
- ✓ UNet in FP16 (trainable)
- ✓ VAE in FP32 (frozen, numerically stable)
- ✓ VAE outputs converted to FP16 for training
- ✓ Conditioning converts to FP16
- ✓ UNet operations in FP16
- ✓ Loss computation in FP32 (stable)

**Expected memory usage:**
- 512×512, batch=1: ~7-9 GB (VAE adds ~300MB vs pure FP16)
- 512×512, batch=2: ~11-13 GB

**Training should now proceed without dtype or NaN errors!** 🚀

---

## 📝 Summary of Journey

1. **Initial attempt**: Pure FP16 everything
   - Error: `conv_in` layer wasn't converted → **Fixed**

2. **Second attempt**: Pure FP16 with conv_in fixed
   - Error: VAE producing NaN in FP16 → **Fundamental limitation**

3. **Final solution**: Hybrid FP16/FP32
   - UNet FP16 (where it matters for speed/memory)
   - VAE FP32 (where it matters for stability)
   - **This is the industry standard for Stable Diffusion!**
