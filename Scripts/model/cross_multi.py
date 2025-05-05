import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    """Cross-modal attention module for feature fusion"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for query, key, value
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x1, x2):
        # Ensure both inputs have the same batch size
        if x1.size(0) != x2.size(0):
            if x1.size(0) < x2.size(0):
                x1 = x1.expand(x2.size(0), -1)
            else:
                x2 = x2.expand(x1.size(0), -1)
        
        # Project queries from x1, keys and values from x2
        q = self.q_proj(x1).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x2).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x2).view(-1, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).reshape(-1, self.dim)
        out = self.out_proj(out)
        
        # Add residual connection and normalize
        out = self.norm1(x1 + out)
        out = self.norm2(out + self.dropout(out))
        
        return out