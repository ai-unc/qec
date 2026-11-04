from __future__ import annotations

import torch
import torch.nn as nn

from .MHA_Transformer import CausalDecoderLayer, PositionalEncoding
from .tokenizer import CliffordTokenizer

class FactoredGateEmbedding(nn.Module):
    def __init__(self, tokenizer: CliffordTokenizer, d_model: int):
        super().__init__()
        self.emb_op = nn.Embedding(tokenizer.OP_VOCAB_SIZE, d_model, padding_idx=tokenizer.PAD)     # Gate embedding
        self.emb_qa = nn.Embedding(tokenizer.qubit_vocab_size, d_model, padding_idx=tokenizer.Q_PAD)  # Qubit 1 embedding   
        self.emb_qb = nn.Embedding(tokenizer.qubit_vocab_size, d_model, padding_idx=tokenizer.Q_PAD)  # Qubit 2 embedding

    def forward(self, ops: torch.Tensor, qa: torch.Tensor, qb: torch.Tensor):
        return self.emb_op(ops) + self.emb_qa(qa) + self.emb_qb(qb)


class CliffordDecoderOnly(nn.Module):
    def __init__(self, tokenizer: CliffordTokenizer, d_model = 128, num_heads = 4, num_layers = 2, d_ff = 256, dropout = 0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.embedding = FactoredGateEmbedding(tokenizer, d_model)
        self.positional_encoding = PositionalEncoding(d_model, tokenizer.max_seq_length)
        self.layers = nn.ModuleList([CausalDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head_op = nn.Linear(d_model, tokenizer.OP_VOCAB_SIZE)
        self.head_qa = nn.Linear(d_model, tokenizer.qubit_vocab_size)
        self.head_qb = nn.Linear(d_model, tokenizer.qubit_vocab_size)

    def causal_mask(self, seq_len: int, device: torch.device):
        # 1 = keep
        return torch.tril(torch.ones(1, 1, seq_len, seq_len, device=device))

    def forward(self, ops, qa, qb):
        x = self.dropout(self.positional_encoding(self.embedding(ops, qa, qb))) # Pos encoding and dropout
        mask = self.causal_mask(ops.size(1), ops.device) # Causal mask
        for layer in self.layers:
            x = layer(x, mask)
        return self.head_op(x), self.head_qa(x), self.head_qb(x)
