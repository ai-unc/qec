from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn

from .transformer import CausalDecoderLayer, PositionalEncoding  # assumed Flax modules
from .tokenizer import CliffordTokenizer


class PaddedEmbed(nn.Module):
    num_embeddings: int
    features: int
    padding_idx: int

    @nn.compact
    def __call__(self, ids):
        emb = nn.Embed(
            self.num_embeddings,
            self.features,
            embedding_init=nn.initializers.normal(stddev=1.0),  # matches nn.Embedding's N(0, 1)
        )(ids)
        return jnp.where((ids == self.padding_idx)[..., None], 0.0, emb)


class FactoredGateEmbedding(nn.Module):
    tokenizer: CliffordTokenizer
    d_model: int

    def setup(self):
        tok = self.tokenizer
        self.emb_op = PaddedEmbed(tok.OP_VOCAB_SIZE, self.d_model, tok.PAD)          # Gate embedding
        self.emb_qa = PaddedEmbed(tok.qubit_vocab_size, self.d_model, tok.Q_PAD)     # Qubit 1 embedding
        self.emb_qb = PaddedEmbed(tok.qubit_vocab_size, self.d_model, tok.Q_PAD)     # Qubit 2 embedding

    def __call__(self, ops, qa, qb):
        return self.emb_op(ops) + self.emb_qa(qa) + self.emb_qb(qb)


class CliffordDecoderOnly(nn.Module):
    tokenizer: CliffordTokenizer
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

    def setup(self):
        tok = self.tokenizer
        self.embedding = FactoredGateEmbedding(tok, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, tok.max_seq_length)
        self.layers = [
            CausalDecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
            for _ in range(self.num_layers)
        ]
        self.drop = nn.Dropout(self.dropout)
        self.head_op = nn.Dense(tok.OP_VOCAB_SIZE)
        self.head_qa = nn.Dense(tok.qubit_vocab_size)
        self.head_qb = nn.Dense(tok.qubit_vocab_size)

    @staticmethod
    def causal_mask(seq_len: int):
        # 1 = keep, shape (1, 1, T, T)
        return jnp.tril(jnp.ones((1, 1, seq_len, seq_len)))

    def __call__(self, ops, qa, qb, deterministic: bool = True):
        x = self.positional_encoding(self.embedding(ops, qa, qb))
        x = self.drop(x, deterministic=deterministic)
        mask = self.causal_mask(ops.shape[1])
        for layer in self.layers:
            x = layer(x, mask, deterministic=deterministic)
        return self.head_op(x), self.head_qa(x), self.head_qb(x)