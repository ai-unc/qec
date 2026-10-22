import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


class RotaryEmbedding(nn.Module):
    freqs_cos: torch.Tensor
    freqs_sin: torch.Tensor

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("freqs_cos", torch.cos(emb), persistent=False)
        self.register_buffer("freqs_sin", torch.sin(emb), persistent=False)

    def forward(self, x, seq_len):
        cos = self.freqs_cos[:seq_len].to(device=x.device)
        sin = self.freqs_sin[:seq_len].to(device=x.device)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :].to(dtype=x.dtype)
    sin = sin[None, None, :, :].to(dtype=x.dtype)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class MultiHeadLatentAttention(nn.Module):
    causal_mask = torch.tensor([])

    def __init__(self, d_model = 128, num_heads = 4, dim_q_latent = 48, dim_kv_latent = 32, dim_rope = 16, dropout = 0.0, max_seq_len = 256, causal = False):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dim_q_latent = dim_q_latent # they call this q_lora_rank in the paper, but I think this is a more sensible var name
        self.dim_kv_latent = dim_kv_latent # they call this kv_lora_rank in the paper
        self.dim_rope = dim_rope
        self.max_seq_len = max_seq_len
        self.causal = causal

        self.kv_proj = nn.Linear(d_model, dim_kv_latent, bias = False)
        self.kv_norm = RMSNorm(dim_kv_latent)
        self.k_decompress = nn.Linear(dim_kv_latent, num_heads * self.head_dim, bias = False)
        self.v_decompress = nn.Linear(dim_kv_latent, num_heads * self.head_dim, bias = False)

        self.q_proj = nn.Linear(d_model, dim_q_latent, bias = False)
        self.q_decompress = nn.Linear(dim_q_latent, num_heads * self.head_dim, bias = False)

        self.k_rope_proj = nn.Linear(d_model, num_heads * dim_rope, bias = False)
        self.q_rope_proj = nn.Linear(dim_q_latent, num_heads * dim_rope, bias = False)

        self.out_proj = nn.Linear(num_heads * self.head_dim, d_model, bias = False)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(dim_rope, max_seq_len)

        if causal: # We might not want the model to be causal since looking at future qubits isn't really cheating (from my understanding at least)
            mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
            self.register_buffer("causal_mask", mask, persistent = False) # Persistent = False so the mask isn't saved to the state dict (as far as I can tell, no reason to store it)

    def forward(self, x, attention_mask):
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} > max_seq_len {self.max_seq_len}")

        kv_compressed = self.kv_norm(self.kv_proj(x))
        q_compressed = self.q_proj(x)

        k_content = self.k_decompress(kv_compressed)
        v = self.v_decompress(kv_compressed)
        q_content = self.q_decompress(q_compressed)

        k_rope = self.k_rope_proj(x)
        q_rope = self.q_rope_proj(q_compressed)

        k_content = k_content.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_content = q_content.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_rope = k_rope.view(batch_size, seq_len, self.num_heads, self.dim_rope).transpose(1, 2)
        q_rope = q_rope.view(batch_size, seq_len, self.num_heads, self.dim_rope).transpose(1, 2)

        # Rotary embedding
        cos, sin = self.rope(x, seq_len)
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        q = torch.cat((q_content, q_rope), dim=-1)
        k = torch.cat((k_content, k_rope), dim=-1)

        scale = 1.0 / math.sqrt(q.size(-1))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if self.causal:
            scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))

        if attention_mask is not None:
           padding_mask_additive = (1 - attention_mask).unsqueeze(1).unsqueeze(2) * float('-inf')
           scores = scores + padding_mask_additive

        attn_weights = self.attn_dropout(F.softmax(scores, dim = -1))

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.residual_dropout(self.out_proj(out))
