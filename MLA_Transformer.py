import torch
import torch.nn as nn

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4, q_latent_dim=32, kv_latent_dim=32): # If changing this make sure d_model % num_heads == 0
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_latent_dim = q_latent_dim
        self.kv_latent_dim = kv_latent_dim

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Project inputs to query, key, and value
        q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.out_proj(attn_output)

        return output