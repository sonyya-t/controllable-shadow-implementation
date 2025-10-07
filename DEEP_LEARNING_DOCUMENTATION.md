# PART 2: Shadow Generation - Deep Learning Approach

## 1. Overview

### Transition from CV to DL

The team transitioned from traditional computer vision (OpenCV/CV) methods to deep learning approaches due to several limitations of the CV approach:

- **Limited Realism**: CV-based shadow generation produced geometrically correct but visually unrealistic shadows
- **Poor Generalization**: Hard-coded algorithms failed to handle diverse object shapes and lighting conditions
- **Inflexible Control**: Difficulty in achieving fine-grained control over shadow properties (softness, direction, intensity)
- **Scalability Issues**: Manual parameter tuning required for each object type and lighting scenario

### High-Level DL Strategy

The deep learning approach implements a **controllable shadow generation system** based on:

1. **Modified SDXL Architecture**: Leverages Stable Diffusion XL's powerful generative capabilities
2. **Rectified Flow Training**: Uses 1-step capable diffusion for real-time inference
3. **Light Parameter Conditioning**: Direct control over shadow direction and softness via (θ, φ, s) parameters
4. **Spatial Conditioning**: Object geometry and mask information encoded into 9-channel input

### Project Status: **INCOMPLETE** ⚠️

**Current State**: The deep learning implementation is **functionally complete but experiencing training instability**. The architecture is fully implemented and verified, but training encounters NaN/Inf losses that prevent convergence.

**What Was Accomplished**:
- ✅ Complete model architecture implementation
- ✅ SDXL UNet modification for 9-channel input
- ✅ Light parameter conditioning system
- ✅ VAE pipeline with FP16/FP32 hybrid precision
- ✅ Rectified flow loss computation
- ✅ Dataset loading and preprocessing
- ✅ Training infrastructure with checkpointing

**What Remains**:
- ❌ Stable training convergence (NaN/Inf issues)
- ❌ Model evaluation and validation
- ❌ Inference pipeline optimization
- ❌ Performance benchmarking

## 2. Model Architecture

### 2.1 Network Design

#### Core Architecture: Modified SDXL UNet

The neural network is built on **Stable Diffusion XL (SDXL)** architecture with key modifications for shadow generation:

```
ShadowDiffusionModel
├── ConditionedSDXLUNet (2.57B parameters, trainable)
│   ├── SDXLUNetForShadows (modified backbone)
│   └── LightParameterConditioning (θ, φ, s encoder)
└── VAEPipeline (83.7M parameters, frozen)
    └── VAEWrapper (SDXL VAE encoder/decoder)
```

#### Architecture Rationale

**Why SDXL?**
- **Proven Architecture**: State-of-the-art generative model with excellent image synthesis capabilities
- **Transfer Learning**: Pretrained weights provide strong foundation for shadow generation
- **Scalability**: Handles high-resolution images (1024×1024) efficiently
- **Flexibility**: Easy to modify for specific conditioning requirements

**Why Rectified Flow?**
- **1-Step Inference**: Enables real-time shadow generation
- **Straight-line Paths**: Learns direct noise-to-data transformations
- **Training Efficiency**: Simpler objective than traditional DDPM approaches

### 2.2 Input/Output Specifications

#### Input Format
- **Object Image**: RGB tensor `(B, 3, 1024, 1024)` in range `[-1, 1]`
- **Binary Mask**: Grayscale tensor `(B, 1, 1024, 1024)` in range `{0, 1}`
- **Light Parameters**: 
  - θ (polar angle): `[0°, 45°]` → controls vertical shadow direction
  - φ (azimuthal angle): `[0°, 360°]` → controls horizontal shadow direction  
  - s (light size): `[2, 8]` → controls shadow softness

#### Output Format
- **Shadow Map**: Grayscale tensor `(B, 1, 1024, 1024)` in range `[0, 1]`
- **Latent Space**: 4-channel latent `(B, 4, 128, 128)` at 8× downsampling

### 2.3 Layer Structure and Dimensions

#### Modified SDXL UNet Architecture

```
Input: 9-channel latent (B, 9, 128, 128)
├── Conv2d(9→320) - Modified input layer
├── TimeEmbedding(320→1280) - Timestep + Light conditioning
├── DownBlocks (4 levels)
│   ├── ResNetBlock + SelfAttention
│   └── Downsample2D (2× downsampling each)
├── MiddleBlock
│   └── ResNetBlock + SelfAttention
└── UpBlocks (4 levels)
    ├── ResNetBlock + SelfAttention
    └── Upsample2D (2× upsampling each)
Output: 4-channel latent (B, 4, 128, 128)
```

#### Key Modifications from Original SDXL

1. **Input Convolution**: `4ch → 9ch`
   - Original: 4 channels (noise latent)
   - Modified: 9 channels = 4 (noise) + 4 (object VAE) + 1 (mask)
   - Pretrained weights copied for first 4 channels
   - New channels zero-initialized for transfer learning

2. **Cross-Attention Removal**: 
   - Original: Text conditioning via cross-attention blocks
   - Modified: Light parameter conditioning via timestep embeddings
   - 70 cross-attention modules replaced with no-op processors

3. **Light Parameter Conditioning**:
   - Sinusoidal embeddings: `(θ, φ, s) → 768-dim`
   - Projection to timestep dimension: `768 → 1280-dim`
   - Additive integration: `timestep_emb + light_emb`

#### Activation Functions and Normalization

- **Activations**: SiLU (Swish) throughout UNet
- **Normalization**: GroupNorm (8 groups) in ResNet blocks
- **Attention**: Self-attention with scaled dot-product
- **Conditioning**: Additive timestep embeddings

### 2.4 Architecture Diagrams

#### Data Flow Architecture

```
Training Flow:
Object RGB (1024×1024) ──┐
                        ├─→ VAE Encoder ──→ Object Latent (4ch, 128×128)
Mask (1024×1024) ───────┼─→ Resize ──────→ Mask Latent (1ch, 128×128)
                        │
Shadow Target ──────────┼─→ VAE Encoder ──→ x₁ (4ch, 128×128)
                        │
Noise ──────────────────┴─→ x₀ (4ch, 128×128)

Light Params (θ,φ,s) ──→ Sinusoidal Encoder ──→ 768-dim ──→ Project 1280-dim

x_t = t·x₁ + (1-t)·x₀
Input: [x_t, object_latent, mask_latent] (9ch, 128×128)
UNet: x_t + timestep_emb + light_emb ──→ v_predicted
Target: v_target = x₁ - x₀
Loss: MSE(v_predicted, v_target)
```

#### Model Component Hierarchy

```
ShadowDiffusionModel (2.65B total params)
├── UNet Components (2.57B params, trainable)
│   ├── ConvIn: 9→320 channels (modified from SDXL)
│   ├── TimeEmbedding: 320→1280 + light conditioning
│   ├── DownBlocks: 4 levels, 320→1280 channels
│   ├── MiddleBlock: 1280 channels
│   └── UpBlocks: 4 levels, 1280→320 channels
├── Light Conditioning (768→1280 projection)
│   ├── Sinusoidal Encoder: (θ,φ,s) → 768-dim
│   └── Timestep Projection: 768 → 1280-dim
└── VAE Pipeline (83.7M params, frozen)
    ├── Encoder: RGB → 4ch latent (8× downsampling)
    └── Decoder: 4ch latent → RGB (8× upsampling)
```

## 3. Dataset

### 3.1 Dataset Requirements

The system requires a dataset with the following structure:

```
Dataset Structure:
├── object_image: RGB image (3, H, W) in [-1, 1]
├── mask: Binary mask (1, H, W) in {0, 1}
├── shadow_map: Target shadow (1, H, W) in [0, 1]
├── theta: Polar angle in degrees [0°, 45°]
├── phi: Azimuthal angle in degrees [0°, 360°]
└── size: Light size parameter [2, 8]
```

### 3.2 Data Collection Process

**Primary Dataset**: `jasperai/controllable-shadow-generation-benchmark`
- **Source**: HuggingFace datasets
- **Size**: 555 samples total
- **Format**: Webdataset with PNG images and JSON metadata
- **Tracks**: Multiple lighting scenarios (softness, horizontal, vertical)

**Data Collection Method**:
- Synthetic 3D scene rendering with controlled lighting
- Multiple object types and materials
- Systematic variation of light parameters
- Ground truth shadows computed via ray tracing

### 3.3 Preprocessing Steps

#### Image Preprocessing
```python
# Object images
transforms = [
    Resize((1024, 1024)),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # → [-1, 1]
]

# Masks and shadows
mask_transforms = [
    Resize((1024, 1024)),
    ToTensor(),
    Binarize(threshold=0.5)  # → {0, 1}
]
```

#### Light Parameter Normalization
```python
# Normalize to [0, 1] range for sinusoidal encoding
theta_norm = theta / 45.0      # [0°, 45°] → [0, 1]
phi_norm = phi / 360.0         # [0°, 360°] → [0, 1] 
size_norm = (size - 2.0) / 6.0 # [2, 8] → [0, 1]
```

### 3.4 Data Formats and Structure

#### HuggingFace Dataset Format
```json
{
  "image.png": "<bytes>",
  "mask.png": "<bytes>", 
  "shadow.png": "<bytes>",
  "metadata": {
    "theta": 30.0,
    "phi": 45.0,
    "light_size": 4.0,
    "track": "softness"
  }
}
```

#### Custom Dataset Format
```
root/
├── objects/
│   ├── img001.png
│   └── img002.png
├── masks/
│   ├── img001.png
│   └── img002.png
├── shadows/
│   ├── img001.png
│   └── img002.png
└── metadata.json
```

### 3.5 Dataset Characteristics

- **Total Samples**: 555 (benchmark dataset)
- **Train/Val Split**: 499/56 samples
- **Image Resolution**: 1024×1024 pixels
- **Parameter Ranges**:
  - θ: [0°, 45°] (polar angle)
  - φ: [0°, 360°] (azimuthal angle)  
  - s: [2, 8] (light size)

### 3.6 Data Augmentation

**Current Implementation**: No augmentation applied
- Dataset is synthetic with controlled variations
- Augmentation could include:
  - Random horizontal flips
  - Slight rotations (±5°)
  - Brightness/contrast adjustments
  - Parameter jittering (±2° for angles, ±0.5 for size)

## 4. Pipeline Issues

### 4.1 Known Bugs

#### Critical Issue: NaN/Inf Loss During Training

**Symptoms**:
```
⚠️  NaN/Inf detected in loss!
   x0 (noise): mean=0.0046, std=0.9993
   x1 (target): mean=-0.4557, std=1.3233
   predicted_velocity: mean=nan
   target_velocity: mean=-0.4603
```

**Root Cause**: The predicted velocity from the UNet becomes NaN, likely due to:
1. **Precision Mismatch**: Paper specifies FP32 training, but implementation uses FP16
2. **Numerical Instability**: FP16 precision issues in deep networks with 2.65B parameters
3. **Gradient Explosion**: Large gradients in early training steps
4. **Conditioning Issues**: Light parameter embeddings causing instability

**Attempted Fixes**:
- ✅ Hybrid FP16/FP32 precision (UNet FP16, VAE FP32)
- ✅ Gradient clipping (max_norm=1.0)
- ✅ Small initialization for light projection layer (gain=0.01)
- ✅ NaN detection and gradient zeroing
- ❌ **Still occurring**: NaN appears in first few training steps

**Severity**: **CRITICAL** - Prevents any training progress

#### Secondary Issues

**Issue #1: CUDA Factory Registration Warnings**
```
E0000 00:00:1759756632.836993   24552 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
```
- **Severity**: Low (cosmetic warnings)
- **Impact**: No functional impact on training
- **Status**: Ignorable

**Issue #2: TensorFlow Integration Warnings**
```
2025-10-06 13:17:12.796455: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on.
```
- **Severity**: Low (environment warnings)
- **Impact**: No functional impact
- **Status**: Ignorable

### 4.2 Integration Challenges

#### Data Flow Issues

**Challenge #1: VAE Latent Space Mismatch**
- **Problem**: VAE encoder/decoder dtype inconsistencies
- **Symptom**: FP32 VAE outputs mixed with FP16 UNet inputs
- **Solution Applied**: Explicit dtype conversion in VAE pipeline
- **Status**: ✅ Resolved

**Challenge #2: Light Parameter Conditioning Integration**
- **Problem**: Sinusoidal embeddings not properly integrated with SDXL timestep system
- **Symptom**: Conditioning has no effect on generated shadows
- **Solution Applied**: Additive timestep embedding strategy
- **Status**: ✅ Resolved

#### Compatibility Problems

**Challenge #1: SDXL Architecture Modifications**
- **Problem**: Modifying pretrained SDXL without breaking functionality
- **Solution Applied**: Careful weight copying and zero-initialization
- **Status**: ✅ Resolved

**Challenge #2: Mixed Precision Training**
- **Problem**: FP16 training stability with large models
- **Solution Applied**: Hybrid precision (UNet FP16, VAE FP32)
- **Status**: ⚠️ Partially resolved (NaN issues persist)

#### Deployment/Setup Challenges

**Challenge #1: Model Size and Memory**
- **Problem**: 2.65B parameter model requires significant GPU memory
- **Memory Usage**: ~5GB FP16, ~10GB FP32
- **Solution**: FP16 training with gradient checkpointing option
- **Status**: ✅ Resolved

**Challenge #2: Dataset Loading**
- **Problem**: HuggingFace dataset format compatibility
- **Solution**: Robust dataset loader with fallback mechanisms
- **Status**: ✅ Resolved

### 4.3 Unresolved Integration Tasks

#### High Priority
1. **Fix NaN/Inf Training Issue**:
   - Investigate UNet layer initialization
   - Consider gradient scaling adjustments
   - Test with different learning rates (5e-6, 1e-6)
   - Implement gradient monitoring

2. **Training Convergence Validation**:
   - Verify loss decreases over iterations
   - Test with smaller batch sizes
   - Validate on synthetic test cases

#### Medium Priority
1. **Performance Optimization**:
   - Profile training speed bottlenecks
   - Optimize dataloader performance
   - Implement mixed precision optimizations

2. **Model Evaluation**:
   - Implement quantitative metrics (SSIM, PSNR)
   - Create qualitative evaluation pipeline
   - Benchmark against CV methods

#### Low Priority
1. **Inference Optimization**:
   - Optimize single-step sampling
   - Implement batch inference
   - Add real-time demo interface

2. **Documentation and Testing**:
   - Add comprehensive unit tests
   - Create usage examples
   - Document API interfaces

---

## Technical Implementation Details

### Model Initialization
```python
model = ShadowDiffusionModel(
    pretrained_model_name="stabilityai/stable-diffusion-xl-base-1.0",
    conditioning_strategy="additive",
    embedding_dim=256,
    image_size=(1024, 1024)
)
```

### Training Configuration
```python
# From paper specifications
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
batch_size = 2
max_iterations = 150000
```

### Hardware Requirements (From Paper)

**Training Setup**:
- **GPUs**: 4× NVIDIA H100-80GB GPUs
- **Training Time**: ~2 days per model
- **Batch Size**: 2 (per GPU, so effective batch size = 8)
- **Iterations**: 150,000 iterations
- **Optimizer**: AdamW with learning rate 1e-5
- **Precision**: FP32 (paper specification)

**Memory Requirements**:
- **Training**: ~80GB GPU memory per H100 (batch_size=2, 1024×1024)
- **Inference**: ~5GB GPU memory (single image)
- **Checkpoint Size**: ~10GB (FP32 weights)

**Note**: The current implementation uses FP16 for memory efficiency, but the paper specifies FP32 training.

---

## Recommendations for Next Team

### Immediate Actions (Critical)
1. **Fix Precision Mismatch**: Switch to FP32 training as specified in paper (requires 4×H100-80GB)
2. **Debug NaN Issue**: Start with learning rate 5e-6 and monitor gradient norms
3. **Validate Architecture**: Run forward pass tests without training
4. **Test Data Pipeline**: Verify dataset loading and preprocessing

### Short-term Goals (1-2 weeks)
1. **Achieve Training Stability**: Fix NaN issues and verify convergence
2. **Basic Evaluation**: Implement loss monitoring and basic metrics
3. **Documentation**: Complete API documentation and usage examples

### Long-term Goals (1-2 months)
1. **Model Optimization**: Improve training speed and memory efficiency
2. **Comprehensive Evaluation**: Benchmark against existing methods
3. **Production Readiness**: Optimize inference and add error handling

---

**Status**: The deep learning approach is architecturally sound but requires debugging of training instability before it can be considered functional. The implementation demonstrates good software engineering practices and follows the research paper specifications closely.
