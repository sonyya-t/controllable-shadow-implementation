"""
Rectified Flow implementation for single-step shadow generation.

Implements the rectified flow objective described in the paper for
fast single-step diffusion model inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import math


class RectifiedFlowModel(nn.Module):
    """
    Single-step diffusion model using rectified flow objective.
    
    Based on SDXL architecture but modified for shadow generation
    with rectified flow training objective.
    """
    
    def __init__(self, 
                 in_channels: int = 9,  # 4 (noise) + 4 (object) + 1 (mask)
                 out_channels: int = 4,
                 conditioning_dim: int = 768,
                 num_timesteps: int = 1000,
                 embed_dim: int = 320,
                 num_res_blocks: int = 2,
                 attention_resolutions: Tuple[int] = (4, 2, 1)):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conditioning_dim = conditioning_dim
        self.num_timesteps = num_timesteps
        self.embed_dim = embed_dim
        
        # Time embedding
        self.time_embed = SinusoidalPositionEmbedding(embed_dim)
        
        # Input projection  
        self.input_proj = nn.Conv2d(in_channels, embed_dim, 3, padding=1)
        
        # Downsample blocks
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(DownBlock(embed_dim, embed_dim, conditioning_dim))
        self.down_blocks.append(DownBlock(embed_dim, embed_dim * 2, conditioning_dim))
        self.down_blocks.append(DownBlock(embed_dim * 2, embed_dim * 4, conditioning_dim))
        
        # Middle block
        self.middle_block = MiddleBlock(embed_dim * 4, conditioning_dim, attention_resolutions)
        
        # Upsample blocks
        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(UpBlock(embed_dim * 4, embed_dim * 2, conditioning_dim))
        self.up_blocks.append(UpBlock(embed_dim * 2, embed_dim, conditioning_dim))
        self.up_blocks.append(UpBlock(embed_dim, embed_dim, conditioning_dim))
        
        # Output projection
        self.output_proj = nn.Conv2d(embed_dim, out_channels, 3, padding=1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                
    def forward(self, 
                x: torch.Tensor, 
                timestep: torch.Tensor,
                conditioning: torch.Tensor,
                object_image: torch.Tensor,
                object_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for rectified flow.
        
        Args:
            x: Noisy latents (batch, channels, height, width)
            timestep: Time step (batch,)
            conditioning: Light parameter embeddings (batch, 768)
            object_image: Object image embeddings (batch, channels, height, width)
            object_mask: Object mask (batch, channels, height, width)
            
        Returns:
            Predicted output (batch, out_channels, height, width)
        """
        # Encode time
        time_emb = self.time_embed(timestep)
        
        # Combine conditioning with time embedding
        cond_emb = conditioning + time_emb
        
        # Concatenate inputs
        h = torch.cat([x, object_image, object_mask], dim=1)
        h = self.input_proj(h)
        
        # Downsample path
        h_outs = [h]
        for down_block in self.down_blocks:
            h = down_block(h, cond_emb)
            h_outs.append(h)
            
        # Middle block
        h = self.middle_block(h, cond_emb)
        
        # Upsample path
        for i, up_block in enumerate(self.up_blocks):
            h = up_block(torch.cat([h, h_outs[-(i+1)]], dim=1), cond_emb)
            
        # Output projection
        output = self.output_proj(h)
        
        return output
    
    def compute_loss(self, 
                    x0: torch.Tensor, 
                    x1: torch.Tensor,
                    timestep: torch.Tensor,
                    conditioning: torch.Tensor,
                    object_image: torch.Tensor,
                    object_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute rectified flow training loss.
        
        Args:
            x0: Clean shadow map (batch, channels, height, width)
            x1: Noise sample (batch, channels, height, width)
            timestep: Random time step (batch,)
            conditioning: Light parameter embeddings (batch, 768)
            object_image: Object image embeddings (batch, channels, height, width)
            object_mask: Object mask (batch, channels, height, width)
            
        Returns:
            Dictionary containing loss and intermediate values
        """
        batch_size = x0.shape[0]
        device = x0.device
        
        # Sample random time step
        t = torch.rand(batch_size, device=device) if timestep is None else timestep
        t = t.view(batch_size, 1, 1, 1)  # Expand for broadcasting
        
        # Linearly interpolate between x0 and x1
        xt = (1 - t) * x0 + t * x1
        
        # Predict the direction vector
        prediction = self.forward(xt, t.squeeze(), conditioning, object_image, object_mask)
        
        # Rectified flow loss: minimize ||prediction - (x1 - x0)||^2
        target = x1 - x0
        loss = F.mse_loss(prediction, target)
        
        return {
            'loss': loss,
            'prediction': prediction,
            'target': target,
            't': t.squeeze()
        }
    
    @torch.no_grad()
    def sample(self, 
               shape: Tuple[int, int, int],
               conditioning: torch.Tensor,
               object_image: torch.Tensor,
               object_mask: torch.Tensor,
               num_steps: int = 1) -> torch.Tensor:
        """
        Generate sample using rectified flow.
        
        Args:
            shape: Output shape (channels, height, width)
            conditioning: Light parameter embeddings (batch, 768)
            object_image: Object image embeddings (batch, channels, height, width)
            object_mask: Object mask (batch, channels, height, width)
            num_steps: Number of sampling steps (usually 1 for rectified flow)
            
        Returns:
            Generated shadow map (batch, channels, height, width)
        """
        batch_size = conditioning.shape[0]
        device = conditioning.device
        
        # Start from random noise
        x = torch.randn(batch_size, *shape, device=device)
        
        if num_steps == 1:
            # Single step sampling
            t = torch.ones(batch_size, device=device)
            prediction = self.forward(x, t, conditioning, object_image, object_mask)
            # Move in predicted direction
            x1 = x + prediction
            return x1
        else:
            # Multi-step sampling (if needed)
            dt = 1.0 / num_steps
            for i in range(num_steps):
                t = torch.ones(batch_size, device=device) * (1 - i * dt)
                prediction = self.forward(x, t, conditioning, object_image, object_mask)
                x = x + prediction * dt
                
        return x


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embeddings for timesteps."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timestep.device) * -emb)
        emb = timestep[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


class ResBlock(nn.Module):
    """Residual block with conditioning."""
    
    def __init__(self, in_channels: int, out_channels: int, conditioning_dim: int):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(out_channels // 8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(out_channels // 8, out_channels)
        )
        
        self.condition_proj = nn.Linear(conditioning_dim, out_channels)
        self.residual_proj = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        
        h = self.main_path(x)
        
        # Apply conditioning modulation
        cond = self.condition_proj(conditioning)
        cond = cond.view(*cond.shape[:2], 1, 1)
        h = h * (1 + cond)
        
        return h + residual


class DownBlock(nn.Module):
    """Downsampling block."""
    
    def __init__(self, in_channels: int, out_channels: int, conditioning_dim: int):
        super().__init__()
        self.res_block = ResBlock(in_channels, out_channels, conditioning_dim)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x, conditioning)
        return self.downsample(x)


class UpBlock(nn.Module):
    """Upsampling block."""
    
    def __init__(self, in_channels: int, out_channels: int, conditioning_dim: int):
        super().__init__()
        self.res_block = ResBlock(in_channels, out_channels, conditioning_dim)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 
                                          kernel_size=3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x, conditioning)
        return self.upsample(x)


class MiddleBlock(nn.Module):
    """Middle block with attention."""
    
    def __init__(self, channels: int, conditioning_dim: int, attention_resolutions: Tuple[int]):
        super().__init__()
        self.res_block = ResBlock(channels, channels, conditioning_dim)
        
        # Self-attention layer
        self.attention = SpatialSelfAttention(channels, conditioning_dim)
        
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        x = self.res_block(x, conditioning)
        x = self.attention(x, conditioning)
        return x


class SpatialSelfAttention(nn.Module):
    """Spatial self-attention mechanism."""
    
    def __init__(self, channels: int, conditioning_dim: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Conv2d(channels, channels, 1)
        self.to_v = nn.Conv2d(channels, channels, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # Reshape for multi-head attention
        q = self.to_q(x).view(b, self.num_heads, self.head_dim, h, w)
        k = self.to_k(x).view(b, self.num_heads, self.head_dim, h, w)
        v = self.to_v(x).view(b, self.num_heads, self.head_dim, h, w)
        
        # Compute attention
        q = q.view(b, self.num_heads, self.head_dim, h * w).transpose(-2, -1)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w).transpose(-2, -1)
        
        attention = torch.matmul(q, k) * self.scale
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.transpose(-2, -1).contiguous().view(b, c, h, w)
        
        return self.to_out(out) + x
