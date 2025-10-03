# Controllable Shadow Generation with Single-Step Diffusion Models

This project implements the research described in ["Controllable Shadow Generation with Single-Step Diffusion Models from Synthetic Data"](https://arxiv.org/abs/2412.11972).

## Overview

This implementation provides a fast, controllable, and background-free shadow generation system for 2D object images using single-step diffusion models. Our model enables precise control over shadow direction, softness, and intensity parameters.

## Key Features

- **Single-Step Generation**: Uses rectified flow objective for fast inference
- **Controllable Parameters**: 
  - θ (polar angle): Control vertical shadow direction
  - φ (azimuthal angle): Control horizontal shadow direction  
  - s (light size): Control shadow softness
  - I (intensity): Control shadow intensity
- **Background-Free**: Generate shadows that can be composited onto any background
- **Training on Synthetic Data**: Uses 3D-rendered synthetic dataset for training

## Installation

```bash
git clone <repository-url>
cd Controllable-shadow
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from controllable_shadow import ShadowGenerator

# Initialize the model
generator = ShadowGenerator.from_pretrained("checkpoints/shadow-model")

# Generate shadow for an object image
object_image = "path/to/object.jpg"
shadow_map = generator.generate(
    object_image=object_image,
    theta=30,  # vertical direction (degrees)
    phi=60,    # horizontal direction (degrees)  
    s=4,       # softness (light size)
    steps=1    # single step generation
)

# Blend with background
result = generator.blend_with_background(
    object_image, shadow_map, "path/to/background.jpg"
)
```

## Project Structure

```
controllable_shadow/
├── models/           # Model implementations
├── data/            # Dataset handling
├── utils/           # Utility functions
├── scripts/         # Training/inference scripts
├── demos/           # Demo applications
└── tests/           # Unit tests
```

## Model Architecture

- **Backbone**: SDXL architecture without cross-attention blocks
- **Objective**: Rectified Flow for single-step generation
- **Conditioning**: Light parameters (θ, φ, s) via sinusoidal embeddings
- **Input**: Object image + binary mask → Shadow map prediction

## Training

The model is trained on a synthetic dataset created using 3D rendering engines. See `scripts/train.py` for training configuration.

## Evaluation

Benchmark evaluation on three tracks:
- Track 1: Softness control (varying s)
- Track 2: Horizontal direction control (varying φ)  
- Track 3: Vertical direction control (varying θ)

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tasar2024controllable,
  title={Controllable Shadow Generation with Single-Step Diffusion Models from Synthetic Data},
  author={Tasar, Onur and Chadebec, Clement and Aubin, Benjamin},
  journal={arXiv preprint arXiv:2412.11972},
  year={2024}
}
```
