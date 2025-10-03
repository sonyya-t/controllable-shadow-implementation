"""
Interactive Shadow Generation Demo

Creates an interactive web interface for real-time shadow generation.
Uses Gradio for a user-friendly web interface.

Usage:
    python demo_interactive.py
    python demo_interactive.py --server_name 0.0.0.0 --port 7860
    python demo_interactive.py --streamlit  # Use Streamlit instead
"""

import torch
import argparse
import gradio as gr
import streamlit as st
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
from PIL import Image, ImageOps
import json
import time

from controllable_shadow.models import create_shadow_model


class InteractiveShadowGenerator:
    """Interactive shadow generation interface."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize interactive generator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run model on
        """
        self.device = device
        self.model = None
        self.checkpoint_path = checkpoint_path
        self.model_loaded = False
        
    def load_model(self):
        """Load shadow generation model."""
        if self.model_loaded:
            return
        
        print("Loading shadow generation model...")
        self.model = create_shadow_model(
            pretrained_path=self.checkpoint_path,
            device=self.device,
        )
        self.model.eval()
        self.model_loaded = True
        print("✓ Model loaded successfully")
    
    def generate_shadow(self, 
                       object_image: Image.Image, 
                       mask_image: Optional[Image.Image], 
                       theta: float, 
                       phi: float, 
                       size: float,
                       num_steps: int = 1) -> Tuple[Image.Image, str]:
        """
        Generate shadow for given inputs.
        
        Args:
            object_image: Input object image
            mask_image: Optional mask image
            theta: Polar angle [0-45]
            phi: Azimuthal angle [0-360]
            size: Light size [2-8]
            num_steps: Sampling steps
            
        Returns:
            Tuple of (shadow_image, info_string)
        """
        if not self.model_loaded:
            self.load_model()
        
        # Prepare inputs
        object_img = object_image.convert('RGB').resize((1024, 1024), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        object_tensor = transform(object_img).unsqueeze(0).to(self.device)
        
        # Create or use mask
        if mask_image is not None:
            mask_img = mask_image.convert('L').resize((1024, 1024), Image.Resampling.NEAREST)
            mask_tensor = transform(mask_img)[:1].unsqueeze(0).to(self.device)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            # Auto-generate mask
            img_denorm = (object_tensor + 1) / 2
            brightness = img_denorm.mean(dim=1, keepdim=True)
            mask_tensor = (brightness < 0.9).float()
        
        # Prepare light parameters
        theta_tensor = torch.tensor([theta], device=self.device)
        phi_tensor = torch.tensor([phi], device=self.device)
        size_tensor = torch.tensor([size], device=self.device)
        
        # Generate shadow
        start_time = time.time()
        with torch.no_grad():
            shadow_tensor = self.model.sample(
                object_tensor, mask_tensor, theta_tensor, phi_tensor, size_tensor,
                num_steps=num_steps
            )
        gen_time = time.time() - start_time
        
        # Convert to image
        shadow_np = shadow_tensor[0, 0].cpu().numpy()
        shadow_img = Image.fromarray((shadow_np * 255).astype('uint8'), mode='L')
        
        # Create info string
        info = f"Generated in {gen_time:.3f}s with θ={theta}°, φ={phi}°, s={size}"
        
        return shadow_img, info


def create_gradio_interface():
    """Create Gradio web interface."""
    
    # Initialize generator
    generator = InteractiveShadowGenerator()
    
    # Load sample images
    sample_dir = Path("./sample_images") if Path("./sample_images").exists() else None
    
    def process_image(object_img, mask_img, theta, phi, size, steps):
        """Process image and generate shadow."""
        try:
            shadow_img, info = generator.generate_shadow(
                object_img, mask_img, theta, phi, size, steps
            )
            return shadow_img, info, None
        except Exception as e:
            return None, f"Error: {str(e)}", str(e)
    
    def create_composite(object_img, shadow_img, opacity):
        """Create composite of object and shadow."""
        if object_img is None or shadow_img is None:
            return None
        
        try:
            # Resize to match
            object_resized = object_img.resize((1024, 1024), Image.Resampling.LANCZOS)
            shadow_resized = shadow_img.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            # Convert shadow to RGB
            shadow_rgb = Image.new('RGB', shadow_resized.size, (0, 0, 0))
            shadow_rgb.putalpha(shadow_resized)
            
            # Composite
            composite = object_resized.copy()
            composite = composite.convert('RGBA')
            composite.paste(shadow_rgb, mask=shadow_rgb.split()[-1])
            composite = composite.convert('RGB')
            
            return composite
            
        except Exception as e:
            return None
    
    with gr.Blocks(
        title="Controllable Shadow Generation",
        theme=gr.themes.Soft(),
        css="""
        .main {
            max-width: 1200px !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🌟 Controllable Shadow Generation
        
        Generate realistic shadows for object images with precise control over light direction, softness, and intensity.
        
        ### How to use:
        1. **Upload an object image** (JPEG/PNG with preferably white background)
        2. **Adjust light parameters**:
           - **θ (Theta)**: Vertical light angle (0° = overhead, 45° = low sun)
           - **φ (Phi)**: Horizontal light direction (0° = right, 90° = down, 180° = left)
           - **s (Size)**: Shadow softness (2 = hard, 8 = very soft)
        3. **Customize output**: Adjust sampling steps and shadow opacity
        4. **Generate**: Click "Generate Shadow" to create shadows
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                
                # Input section
                gr.Markdown("### 📁 Input")
                
                object_input = gr.Image(
                    label="Object Image",
                    type="pil",
                    height=300,
                    help="Upload an object image (preferably with white background)"
                )
                
                mask_input = gr.Image(
                    label="Mask (Optional)",
                    type="pil",
                    height=200,
                    help="Optional: Upload custom mask (will auto-detect from object if not provided)"
                )
                
                # Parameter controls
                gr.Markdown("### 🎛️ Light Parameters")
                
                with gr.Row():
                    theta_slider = gr.Slider(
                        minimum=0, maximum=45, step=1, value=30,
                        label="θ (Theta) - Vertical Angle",
                        info="0° = overhead, 45° = low sun"
                    )
                    phi_slider = gr.Slider(
                        minimum=0, maximum=360, step=5, value=45,
                        label="φ (Phi) - Horizontal Direction", 
                        info="0° = right, 90° = down, 180° = left, 270° = up"
                    )
                
                size_slider = gr.Slider(
                    minimum=2, maximum=8, step=0.5, value=4,
                    label="s (Size) - Shadow Softness",
                    info="2 = hard shadows, 8 = very soft shadows"
                )
                
                # Generation options
                gr.Markdown("### ⚙️ Options")
                
                num_steps = gr.Slider(
                    minimum=1, maximum=8, step=1, value=1,
                    label="Sampling Steps",
                    info="1 = fast, 4+ = higher quality"
                )
                
                opacity_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, step=0.1, value=0.7,
                    label="Shadow Opacity",
                    info="Shadow darkness/intensity"
                )
                
                generate_btn = gr.Button("🎯 Generate Shadow", variant="primary", size="lg")
                
                # Preset buttons
                gr.Markdown("### 🎨 Presets")
                
                with gr.Row():
                    preset_morning = gr.Button("🌅 Morning", size="sm")
                    preset_noon = gr.Button("☀️ Noon", size="sm")
                    preset_evening = gr.Button("🌇 Evening", size="sm")
                    preset_soft = gr.Button("☁️ Soft", size="sm")
                
            with gr.Column(scale=2):
                
                # Output section
                gr.Markdown("### 🖼️ Results")
                
                shadow_output = gr.Image(
                    label="Generated Shadow Map",
                    type="pil",
                    height=300,
                    interactive=False
                )
                
                composite_output = gr.Image(
                    label="Shadow Composite",
                    type="pil", 
                    height=300,
                    interactive=False
                )
                
                # Info text
                info_output = gr.Textbox(
                    label="Generation Info",
                    interactive=False,
                    lines=2
                )
                
                # Download buttons
                with gr.Row():
                    download_shadow = gr.DownloadButton(
                        "💾 Download Shadow",
                        value=None,
                        size="sm"
                    )
                    download_composite = gr.DownloadButton(
                        "💾 Download Composite",
                        value=None,
                        size="sm"
                    )
        
        # Event handlers
        def preset_handler(btn_name):
            """Handle preset button clicks."""
            presets = {
                "morning": (15, 90, 3, "Morning light from East"),
                "noon": (5, 180, 2, "Overhead noon sun"),
                "evening": (35, 270, 6, "Eveninc light from West"),
                "soft": (25, 180, 8, "Soft diffuse light")
            }
            
            if btn_name in presets:
                theta, phi, size, desc = presets[btn_name]
                return theta, phi, size, desc
            return None, None, None, None
        
        preset_morning.click(
            lambda: preset_handler("morning"),
            outputs=[theta_slider, phi_slider, size_slider]
        )
        preset_noon.click(
            lambda: preset_handler("noon"),
            outputs=[theta_slider, phi_slider, size_slider]
        )
        preset_evening.click(
            lambda: preset_handler("evening"),
            outputs=[theta_slider, phi_slider, size_slider]
        )
        preset_soft.click(
            lambda: preset_handler("soft"),
            outputs=[theta_slider, phi_slider, size_slider]
        )
        
        # Main generation pipeline
        generate_btn.click(
            fn=process_image,
            inputs=[object_input, mask_input, theta_slider, phi_slider, size_slider, num_steps],
            outputs=[shadow_output, info_output]
        )
        
        # Composite generation
        [object_input, shadow_output, opacity_slider].component_event(
            fn=create_composite,
            inputs=[object_input, shadow_output, opacity_slider],
            outputs=composite_output
        )
        
        # Auto-download
        shadow_output.change(
            lambda x: x,
            inputs=[shadow_output],
            outputs=[download_shadow]
        )
        
        composite_output.change(
            lambda x: x,
            inputs=[composite_output],
            outputs=[download_composite]
        )
        
        # Sample images
        if sample_dir and sample_dir.exists():
            sample_images = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg"))
            if sample_images:
                
                gr.Markdown("### 📸 Sample Images")
                
                sample_gallery = gr.Gallery(
                    value=[str(img) for img in sample_images[:5]],  # First 5 samples
                    label="Click to use",
                    height=150,
                    columns=5,
                    rows=1
                )
                
                def select_sample(img):
                    if img:
                        return gr.update(value=img)
                    return gr.update()
                
                sample_gallery.select(
                    fn=select_sample,
                    inputs=[sample_gallery],
                    outputs=[object_input]
                )
    
    return demo


def create_streamlit_interface():
    """Create Streamlit interface (alternative to Gradio)."""
    
    st.set_page_config(
        page_title="Controllable Shadow Generation",
        page_icon="🌟",
        layout="wide"
    )
    
    st.title("🌟 Controllable Shadow Generation")
    st.markdown("Generate realistic shadows for object images with precise control over light parameters.")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        st.session_state.generator = InteractiveShadowGenerator()
    
    generator = st.session_state.generator
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Input
        st.subheader("Input")
        uploaded_file = st.file_uploader(
            "Upload Object Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an object image with preferably white background"
        )
        
        uploaded_mask = st.file_uploader(
            "Upload Mask (Optional)",
            type=['png', 'jpg', 'jpeg'],
            help="Optional custom mask"
        )
        
        # Parameters
        st.subheader("Light Parameters")
        
        theta = st.slider(
            "θ (Theta) - Vertical Angle",
            min_value=0.0,
            max_value=45.0,
            value=30.0,
            step=1.0,
            help="0° = overhead light, 45° = low sun"
        )
        
        phi = st.slider(
            "φ (Phi) - Horizontal Direction",
            min_value=0.0,
            max_value=360.0,
            value=45.0,
            step=5.0,
            help="0° = right, 90° = down, 180° = left, 270° = up"
        )
        
        size = st.slider(
            "s (Size) - Shadow Softness",
            min_value=2.0,
            max_value=8.0,
            value=4.0,
            step=0.5,
            help="2 = hard shadows, 8 = very soft shadows"
        )
        
        # Options
        st.subheader("Options")
        
        num_steps = st.slider(
            "Sampling Steps",
            min_value=1,
            max_value=8,
            value=1,
            help="1 = fast, 4+ = higher quality"
        )
        
        shadow_opacity = st.slider(
            "Shadow Opacity",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            help="Shadow darkness"
        )
        
        # Presets
        st.subheader("🎨 Presets")
        
        col1, col2 = st.columns(2)
        
        if col1.button("🌅 Morning"):
            st.session_state.theta = 15
            st.session_state.phi = 90
            st.session_state.size = 3
            st.rerun()
        
        if col1.button("☀️ Noon"):
            st.session_state.theta = 5
            st.session_state.phi = 180
            st.session_state.size = 2
            st.rerun()
        
        if col2.button("🌇 Evening"):
            st.session_state.theta = 35
            st.session_state.phi = 270
            st.session_state.size = 6
            st.rerun()
        
        if col2.button("☁️ Soft"):
            st.session_state.theta = 25
            st.session_state.phi = 180
            st.session_state.size = 8
            st.rerun()
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📁 Input Image")
        
        if uploaded_file:
            object_img = Image.open(uploaded_file).convert('RGB')
            st.image(object_img, caption="Object Image", use_column_width=True)
            
            # Display mask if provided
            if uploaded_mask:
                mask_img = Image.open(uploaded_mask).convert('L')
                st.image(mask_img, caption="Custom Mask", use_column_width=True)
            else:
                mask_img = None
                st.info("No mask provided - will auto-detect from object")
        
        else:
            object_img = None
            mask_img = None
            st.info("Please upload an object image to get started")
    
    with col2:
        st.header("🎯 Generated Shadow")
        
        if object_img and st.button("Generate Shadow", type="primary"):
            with st.spinner("Generating shadow..."):
                shadow_img, info = generator.generate_shadow(
                    object_img, mask_img, theta, phi, size, num_steps
                )
                
                if shadow_img:
                    st.image(shadow_img, caption="Generated Shadow Map", use_column_width=True)
                    
                    # Composite creation
                    composite_img = create_composite(object_img, shadow_img, shadow_opacity)
                    if composite_img:
                        st.image(composite_img, caption="Shadow Composite", use_column_width=True)
                    
                    st.success(info)
                    
                    # Download options
                    col_down1, col_down2 = st.columns(2)
                    
                    with col_down1:
                        shadow_bytes = shadow_img.tobytes()
                        st.download_button(
                            "💾 Download Shadow",
                            data=shadow_bytes,
                            file_name=f"shadow_theta{theta}_phi{phi}_size{size}.png",
                            mime="image/png"
                        )
                    
                    with col_down2:
                        if composite_img:
                            composite_bytes = composite_img.tobytes()
                            st.download_button(
                                "💾 Download Composite",
                                data=composite_bytes,
                                file_name=f"composite_theta{theta}_phi{phi}_size{size}.png",
                                mime="image/png"
                            )
                else:
                    st.error("Failed to generate shadow")


def create_composite(object_img, shadow_img, opacity):
    """Create composite for Streamlit."""
    if object_img is None or shadow_img is None:
        return None
    
    try:
        # Resize to match
        object_resized = object_img.resize((1024, 1024), Image.Resampling.LANCZOS)
        shadow_resized = shadow_img.resize((1024, 1024), Image.Resampling.LANCZOS)
        
        # Create colored shadow
        shadow_array = np.array(shadow_resized)
        shadow_rgb = np.stack([shadow_array] * 3, axis=-1)
        shadow_rgb = shadow_rgb * opacity
        
        # Convert back to PIL
        shadow_rgb_img = Image.fromarray(shadow_rgb.astype(np.uint8))
        
        # Composite
        result = Image.blend(object_resized, shadow_rgb_img, shadow_opacity)
        
        return result
        
    except Exception as e:
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Interactive Shadow Generation Demo")
    
    parser.add_argument("--interface", type=str, default="gradio",
                       choices=["gradio", "streamlit"],
                       help="Web interface framework")
    parser.add_argument("--server_name", type=str, default="127.0.0.1",
                       help="Server hostname")
    parser.add_argument("--port", type=int, default=7860,
                       help="Server port")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 INTERACTIVE SHADOW GENERATION DEMO")
    print("="*70)
    
    print(f"Starting {args.interface} interface...")
    print(f"Access at: http://{args.server_name}:{args.port}")
    
    if args.interface == "gradio":
        demo = create_gradio_interface()
        
        demo.launch(
            server_name=args.server_name,
            server_port=args.port,
            debug=args.debug,
            share=False
        )
    
    elif args.interface == "streamlit":
        print("Note: Streamlit will be handled by running 'streamlit run demo_interactive.py'")
        st.run()


if __name__ == "__main__":
    main()

