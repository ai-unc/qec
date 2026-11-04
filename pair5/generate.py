from __future__ import annotations

import torch
import torch.nn.functional as F

from pair4.evolution_prototype import CircuitCandidate

from .model import CliffordDecoderOnly
from .tokenizer import CliffordTokenizer


def _masked_sample(logits: torch.Tensor, allow: torch.Tensor, greedy: bool) -> int:
    masked = logits.masked_fill(~allow, float("-inf"))
    if greedy:
        return int(torch.argmax(masked).item())
    probs = F.softmax(masked, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_circuit(model: CliffordDecoderOnly, *, max_gates: int | None = None, greedy: bool = True, device: torch.device | None = None) -> CircuitCandidate:
    tokenizer = model.tokenizer
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    max_gates = tokenizer.max_gates if max_gates is None else max_gates
    ops = [tokenizer.BOS]
    qa = [tokenizer.Q_BOS]
    qb = [tokenizer.Q_BOS]

    for _ in range(max_gates + 1):
        op_t = torch.tensor([ops], dtype=torch.long, device=device)
        qa_t = torch.tensor([qa], dtype=torch.long, device=device)
        qb_t = torch.tensor([qb], dtype=torch.long, device=device)
        logits_op, logits_qa, logits_qb = model(op_t, qa_t, qb_t)

        next_op_logits = logits_op[0, -1]
        next_qa_logits = logits_qa[0, -1]
        next_qb_logits = logits_qb[0, -1]

        op_mask = torch.zeros(tokenizer.OP_VOCAB_SIZE, dtype=torch.bool, device=device)
        op_mask[tokenizer.OP_H] = True
        op_mask[tokenizer.OP_S] = True
        if tokenizer.n_qubits >= 2:
            op_mask[tokenizer.OP_CNOT] = True
        op_mask[tokenizer.EOS] = True

        op_id = _masked_sample(next_op_logits, op_mask, greedy)
        if op_id == tokenizer.EOS:
            ops.append(tokenizer.EOS)
            qa.append(tokenizer.Q_EOS)
            qb.append(tokenizer.Q_EOS)
            break

        _, qa_mask, _ = tokenizer.validity_masks(op_id, device=device)
        qa_id = _masked_sample(next_qa_logits, qa_mask, greedy)
        qb_mask = tokenizer.qb_mask_for_qa(op_id, qa_id, device=device)
        qb_id = _masked_sample(next_qb_logits, qb_mask, greedy)

        ops.append(op_id)
        qa.append(qa_id)
        qb.append(qb_id)
    else:
        ops.append(tokenizer.EOS)
        qa.append(tokenizer.Q_EOS)
        qb.append(tokenizer.Q_EOS)

    return tokenizer.decode(ops, qa, qb)
