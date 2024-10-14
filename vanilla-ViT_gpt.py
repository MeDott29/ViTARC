import torch
import torch.nn as nn

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