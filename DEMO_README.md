# 🌟 Controllable Shadow Generation - Demo Suite

A comprehensive collection of demo scripts showcasing the controllable shadow generation model.

## 🚀 Quick Start

The easiest way to try all demos:

```bash
python run_demo.py all
```

This will automatically:
- Create sample images and backgrounds
- Run basic, batch, composite, and interactive demos
- Generate comprehensive outputs

## 📋 Demo Scripts Overview

### 1. `run_demo.py` - Demo Launcher ⭐ **START HERE**
Convenient launcher for all demo modes with automatic setup.

```bash
# Run all demos automatically
python run_demo.py all

# Run specific demo types
python run_demo.py basic --input your_object.png
python run_demo.py batch --input_dir ./images --benchmark
python run_demo.py composite --object chair.png --background grass.jpg
python run_demo.py interactive --mode gradio --port 7860
```

### 2. `demo_basic.py` - Single Image Demo
Generate shadows for individual object images with various light parameters.

**Features:**
- Demonstration of different light scenarios (morning, noon, evening, soft)
- Automatic mask generation from background detection
- Comprehensive visualization with before/after comparisons
- Performance timing and memory profiling

```bash
python demo_basic.py --input chair.png
python demo_basic.py --input car.png --theta 15 --phi 270 --size 6
```

### 3. `demo_batch.py` - Batch Processing Demo
Process multiple images efficiently with benchmark comparisons.

**Features:**
- Batch size performance benchmarking (1, 2, 4 images)
- Parameter sweep demonstrations
- Memory and throughput analysis
- Batch output management

```bash
python demo_batch.py --input_dir ./test_images --batch_size 4 --benchmark
python demo_batch.py --images img1.png img2.png img3.png
```

### 4. `demo_composite.py` - Advanced Compositing Demo
Professional shadow integration with realistic environmental effects.

**Features:**
- Environmental shadow scenarios (outdoor sunny, overcast, indoor, studio)
- Time-of-day simulations (sunrise, noon, sunset)
- Shadow softness comparisons
- Realistic color and opacity blending
- Background enhancement for different lighting conditions

```bash
python demo_composite.py --object chair.png --background grass.jpg --shadow_props theta=30 phi=90 size=4
python demo_composite.py --object car.png --run_all_demos  # Run all scenarios
```

### 5. `demo_interactive.py` - Web Interface Demo
Interactive Gradio/Streamlit web interface for real-time shadow generation.

**Features:**
- Live parameter adjustment with sliders
- Instant preview and compositing
- Preset lighting scenarios
- Built-in sample images
- Download capabilities
- Auto-mask detection

```bash
python demo_interactive.py --mode gradio --port 7860
python demo_interactive.py --mode streamlit
```

### 6. `generate_shadows.py` - Basic Inference (Existing)
Simple command-line shadow generation (already implemented).

```bash
python generate_shadows.py --input object.png --output shadow.png --theta 30 --phi 45 --size 4
```

## 🎛️ Light Parameter Guide

### Parameters Explained

- **θ (Theta)**: Vertical light angle (0-45°)
  - `0°`: Light directly overhead (short shadows)
  - `15°`: Early morning/evening light
  - `30°`: Typical daylight (recommended)
  - `45°`: Low sun/early morning/late evening (long shadows)

- **φ (Phi)**: Horizontal light direction (0-360°)
  - `0°`: Light from right → shadow points left
  - `90°`: Light from top → shadow points down
  - `180°`: Light from left → shadow points right
  - `270°`: Light from below → shadow points up

- **s (Size)**: Shadow softness (2-8)
  - `2`: Hard shadows (small/bright light source)
  - `4`: Medium shadows (recommended default)
  - `6`: Soft shadows (larger/diffuse light)
  - `8`: Very soft shadows (overcast sky)

## 📊 Demo Outputs

Each demo creates organized output directories:

```
demo_output/
├── basic_demo/
│   ├── light_scenarios_demo.png      # Multi-scenario comparison
│   ├── custom_demo.png              # Single scenario visualization
│   └── custom_shadow.png             # Raw shadow map
├── batch_output/
│   ├── batch_benchmark.png          # Performance comparison
│   ├── parameter_sweep_batch.png     # Multi-parameter results
│   └── shadow_*.png                 # Individual batch results
├── composite_output/
│   ├── comprehensive_comparison.png   # All scenarios grid
│   ├── environmental_*.png           # Environmental scenarios
│   ├── softness_*.png               # Shadow softness comparison
│   └── time_of_day_*.png            # Time-of-day variations
└── sample_images/
    ├── chair.png, ball.png, car.png  # Auto-generated samples
    └── ...
```

## 🎨 Demo Scenarios

### Environmental Shadows
- **Outdoor Sunny**: Hard shadows, high contrast
- **Outdoor Overcast**: Soft, diffuse shadows
- **Indoor Soft**: Medium shadows with warm tone
- **Studio Hard**: Precise, dark shadows

### Time of Day
- **Sunrise**: Warm, soft shadows from east
- **Morning**: Medium contrast, clear direction
- **Noon**: Short, crisp overhead shadows
- **Afternoon**: Medium shadows, cool tone
- **Sunset**: Warm, long shadows from west

### Shadow Characteristics
- **Hard**: Defined edges, high contrast (studio lighting)
- **Medium**: Balanced softness (daylight)
- **Soft**: Diffuse edges, low contrast (overcast sky)

## 💡 Usage Examples

### Quick Testing
```bash
# Generate sample images and run all demos
python run_demo.py all

# Quick single test
python demo_basic.py --input sample_images/chair.png
```

### Professional Workflow
```bash
# 1. Batch process multiple objects
python demo_batch.py --input_dir ./product_images --batch_size 4

# 2. Create environmental variations
python demo_composite.py --object product.png --run_all_demos

# 3. Fine-tune with interactive interface
python demo_interactive.py --port 7860
```

### Development & Testing
```bash
# Performance benchmarking
python demo_batch.py --input_dir ./test_images --benchmark

#import parameter_testing
python demo_basic.py --input test.png --theta 20 --phi 180 --size 6

# Quality comparison
python demo_composite.py --object test.png --demo_softness --demo_tod
```

## 🔧 Configuration

### Common Arguments
```bash
--checkpoint path/to/model.pt    # Model checkpoint
--device cuda|cpu               # Processing device
--output_dir ./results          # Output location
```

### Advanced Options
```bash
--num_steps 1|4|8              # Quality vs speed trade-off
--mixed_precision              # Memory optimization
--batch_size 2|4|8             # Processing efficiency
```

## 🎯 Performance Tips

### For Inference Speed:
- Use `num_steps=1` for fastest generation
- Enable mixed precision (`--device cuda`)
- Process in batches when possible
- Use GPU acceleration

### For Quality:
- Increase `num_steps` to 4-8 for better quality
- Ensure good input mask quality
- Use appropriate light parameters for scene
- Increase sampling steps for final output

### For Interactive Use:
- Start with Gradio for better performance
- Use sample images initially for testing
- Adjust parameters incrementally
- Save working parameter combinations

## 🐛 Troubleshooting

### Common Issues:
1. **Out of Memory**: Use smaller batch size or `--device cpu`
2. **Slow Generation**: Ensure GPU is available, reduce image size
3. **Poor Shadows**: Check input image quality, try different light parameters
4. **Import Errors**: Install requirements: `pip install -r requirements.txt`

### Debug Mode:
```bash
python demo_interactive.py --debug
python run_demo.py all --debug
```

## 📚 Next Steps

After running demos:
1. **Train Custom Model**: Use `train.py` with your data
2. **Create Datasets**: Generate synthetic training data
3. **Batch Processing**: Implement production pipelines
4. **API Integration**: Deploy model as web service
5. **Quality Benchmarking**: Evaluate on custom metrics

---

**Happy Shadow Generating! 🌟**

For technical questions or issues, refer to the main documentation or open a GitHub issue.

