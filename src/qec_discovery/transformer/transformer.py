import math

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn


class MultiHeadAttention(nn.Module):
    d_model: int
    num_heads: int

    def setup(self):
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = self.d_model // self.num_heads

        self.W_q = nn.Dense(self.d_model)
        self.W_k = nn.Dense(self.d_model)
        self.W_v = nn.Dense(self.d_model)
        self.W_o = nn.Dense(self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = jnp.matmul(Q, K.swapaxes(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = jnp.where(mask == 0, -1e9, attn_scores)
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        return jnp.matmul(attn_probs, V)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.shape
        x = x.reshape(batch_size, seq_length, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size, _, seq_length, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.d_model)

    def __call__(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn_output))


class PositionWiseFeedForward(nn.Module):
    d_model: int
    d_ff: int

    def setup(self):
        self.fc1 = nn.Dense(self.d_ff)
        self.fc2 = nn.Dense(self.d_model)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    d_model: int
    max_seq_length: int

    def setup(self):
        # Built with NumPy so it's a plain constant (no params, no tracers).
        pe = np.zeros((self.max_seq_length, self.d_model), dtype=np.float32)
        position = np.arange(0, self.max_seq_length, dtype=np.float32)[:, None]
        div_term = np.exp(
            np.arange(0, self.d_model, 2, dtype=np.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe[None]  # (1, max_seq_length, d_model)

    def __call__(self, x):
        return x + jnp.asarray(self.pe[:, : x.shape[1]])


class CausalDecoderLayer(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    dropout: float

    def setup(self):
        self.self_attn = MultiHeadAttention(self.d_model, self.num_heads)
        self.feed_forward = PositionWiseFeedForward(self.d_model, self.d_ff)
        # epsilon=1e-5 matches nn.LayerNorm (Flax defaults to 1e-6)
        self.norm1 = nn.LayerNorm(epsilon=1e-5)
        self.norm2 = nn.LayerNorm(epsilon=1e-5)
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x, mask, deterministic: bool = True):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.drop(attn_output, deterministic=deterministic))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.drop(ff_output, deterministic=deterministic))