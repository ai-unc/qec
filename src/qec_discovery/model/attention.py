import jax
import jax.numpy as jnp
import flax.linen as nn


class MultiHeadLatentAttention(nn.Module):
    d_model: int = 128
    num_heads: int = 4  # d_model % num_heads must be 0
    q_latent_dim: int = 32   # unused, kept for parity with the PyTorch version
    kv_latent_dim: int = 32  # unused, kept for parity with the PyTorch version

    def setup(self):
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // self.num_heads

        self.q_proj = nn.Dense(self.d_model)
        self.k_proj = nn.Dense(self.d_model)
        self.v_proj = nn.Dense(self.d_model)
        self.out_proj = nn.Dense(self.d_model)

    def __call__(self, x):
        batch_size, seq_length, _ = x.shape

        # Project and split into heads: (B, T, D) -> (B, H, T, head_dim)
        def split_heads(t):
            t = t.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
            return t.transpose(0, 2, 1, 3)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        # Attention scores: (B, H, T, T)
        attn_scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) / (self.head_dim ** 0.5)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        # Weighted sum of values: (B, H, T, head_dim)
        attn_output = jnp.matmul(attn_weights, v)

        # Merge heads: (B, H, T, head_dim) -> (B, T, D)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.d_model)
        return self.out_proj(attn_output)


# Usage
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jnp.ones((2, 10, 128))

    model = MultiHeadLatentAttention()
    params = model.init(key, x)
    out = model.apply(params, x)
    print(out.shape)  # (2, 10, 128)