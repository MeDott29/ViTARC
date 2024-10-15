import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super(VisionTransformer, self).__init__()
        
        assert img_size % patch_size == 0, "Image dimensions must be divisible by the patch size"
        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2  # 3 channels (RGB)
        
        self.patch_size = patch_size
        self.dim = dim
        
        # Linear Projection of Flattened Patches
        self.patch_embedding = nn.Linear(patch_dim, dim)
        
        # Positional Embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # Classification Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, 
                nhead=heads, 
                dim_feedforward=mlp_dim, 
                dropout=dropout
            ), 
            num_layers=depth
        )
        
        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
        B, _, _, _ = x.shape
        patch_size = self.patch_size
        
        # Flatten the patches
        x = x.reshape(B, 3, patch_size, patch_size, -1).permute(0, 4, 2, 3, 1)
        x = x.reshape(B, -1, patch_size ** 2 * 3)
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Add [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        x += self.position_embeddings
        
        # Pass through Transformer Encoder
        x = self.transformer(x)
        
        # Classification output (use the [CLS] token representation)
        cls_output = x[:, 0]
        return self.mlp_head(cls_output)

# Example usage
model = VisionTransformer()
x = torch.randn(1, 3, 224, 224)  # Batch of 1 RGB image of size 224x224
logits = model(x)
print(logits.shape)  # Should output torch.Size([1, 1000])

# Create a 3-second animation of model outputs
os.makedirs("animation_frames", exist_ok=True)
num_frames = 90  # 3 seconds at 30 frames per second

for i in range(num_frames):
    frame_path = f"animation_frames/frame_{i:03d}.png"
    if not os.path.exists(frame_path):
        # Generate random input and get the model output
        x = torch.randn(1, 3, 224, 224)
        logits = model(x).detach().numpy()
        
        # Plot the logits as a bar chart
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Logit values bar chart
        axs[0, 0].bar(range(1000), logits[0], color='blue')
        axs[0, 0].set_xlim(0, 1000)
        axs[0, 0].set_ylim(-10, 10)
        axs[0, 0].set_xlabel("Class Index")
        axs[0, 0].set_ylabel("Logit Value")
        axs[0, 0].set_title(f"Model Output Frame {i+1}")
        
        # RGB channel visualizations
        r_channel = x[0, 0].numpy()
        g_channel = x[0, 1].numpy()
        b_channel = x[0, 2].numpy()
        
        axs[0, 1].imshow(r_channel, cmap='Reds')
        axs[0, 1].set_title("Red Channel")
        axs[0, 1].axis('off')
        
        axs[1, 0].imshow(g_channel, cmap='Greens')
        axs[1, 0].set_title("Green Channel")
        axs[1, 0].axis('off')
        
        axs[1, 1].imshow(b_channel, cmap='Blues')
        axs[1, 1].set_title("Blue Channel")
        axs[1, 1].axis('off')
        
        # Save the frame
        plt.tight_layout()
        plt.savefig(frame_path)
        plt.close()

# Create a GIF from the saved frames
frame_paths = [f"animation_frames/frame_{i:03d}.png" for i in range(num_frames)]
frames = [imageio.imread(frame_path) for frame_path in frame_paths if os.path.exists(frame_path)]

imageio.mimsave('model_output_animation.gif', frames, duration=0.1)

print("Animation saved as model_output_animation.gif")