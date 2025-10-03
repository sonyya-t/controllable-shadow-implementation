"""Debug script to understand the shape issue."""

import torch
from controllable_shadow.models.conditioned_unet import ConditionedSDXLUNet

# Create model
model = ConditionedSDXLUNet(conditioning_strategy="additive")

# Create dummy inputs
batch_size = 1
sample = torch.randn(batch_size, 9, 128, 128)
timestep = torch.tensor([500])
theta = torch.tensor([15.0])
phi = torch.tensor([45.0])
size = torch.tensor([3.0])

print("="*70)
print("Debugging Shape Flow")
print("="*70)

# Step 1: Get light embeddings
light_emb = model.encode_light_parameters(theta, phi, size)
print(f"\n1. Light embeddings shape: {light_emb.shape}")

# Step 2: Get timestep embeddings
timestep_emb = model.get_timestep_embedding(timestep)
print(f"2. Timestep embeddings shape: {timestep_emb.shape}")

# Step 3: Combine
combined_emb = model.combine_embeddings(timestep_emb, light_emb)
print(f"3. Combined embeddings shape: {combined_emb.shape}")

# Step 4: Check what time_embedding module expects/outputs
print(f"\n4. Checking SDXL time_embedding module...")
unet = model.unet.unet

# Test time_proj
dummy_t = torch.tensor([500.0])
t_proj_out = unet.time_proj(dummy_t)
print(f"   time_proj output: {t_proj_out.shape}")

# Test time_embedding
t_emb_out = unet.time_embedding(t_proj_out)
print(f"   time_embedding output: {t_emb_out.shape}")

# Step 5: Try forward with hook
print(f"\n5. Testing forward pass with hook...")

def debug_hook(module, input, output):
    print(f"   Hook called!")
    print(f"   Input to time_embedding: {input[0].shape if isinstance(input, tuple) else input.shape}")
    print(f"   Output from time_embedding: {output.shape}")
    print(f"   Our custom emb shape: {combined_emb.shape}")
    return combined_emb

hook_handle = unet.time_embedding.register_forward_hook(debug_hook)

try:
    print(f"\n6. Calling UNet forward...")
    encoder_hidden_states = torch.zeros(batch_size, 77, 768)
    added_cond_kwargs = {
        "text_embeds": torch.zeros(batch_size, 1280),
        "time_ids": torch.zeros(batch_size, 6),
    }

    output = model.unet(
        sample=sample,
        timestep=torch.zeros(batch_size),
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )

    print(f"\n✓ Success! Output shape: {output.shape}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    hook_handle.remove()

print("\n" + "="*70)
