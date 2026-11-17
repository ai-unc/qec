from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .model import CliffordDecoderOnly  # assumed to be a flax.linen.Module
from .stabilizers import candidate_to_stabilizers


def _masked_sample(
    logits: jnp.ndarray,
    allow: jnp.ndarray,
    greedy: bool,
    key: jax.Array | None,
) -> int:
    masked = jnp.where(allow, logits, -jnp.inf)
    if greedy:
        return int(jnp.argmax(masked))
    # categorical takes logits directly, so no explicit softmax is needed
    return int(jax.random.categorical(key, masked))


def generate_stabilizer_code(
    model: CliffordDecoderOnly,
    params,
    *,
    key: jax.Array | None = None,
    max_gates: int | None = None,
    greedy: bool = True,
) -> list[str]:
    tokenizer = model.tokenizer
    max_gates = tokenizer.max_gates if max_gates is None else max_gates

    if not greedy and key is None:
        raise ValueError("A PRNG `key` is required when greedy=False.")

    ops = [tokenizer.BOS]
    qa = [tokenizer.Q_BOS]
    qb = [tokenizer.Q_BOS]

    def next_key():
        nonlocal key
        if greedy:
            return None
        key, sub = jax.random.split(key)
        return sub

    for _ in range(max_gates + 1):
        op_t = jnp.asarray([ops], dtype=jnp.int32)
        qa_t = jnp.asarray([qa], dtype=jnp.int32)
        qb_t = jnp.asarray([qb], dtype=jnp.int32)
        logits_op, logits_qa, logits_qb = model.apply(params, op_t, qa_t, qb_t)

        # CNOT-only sampling keeps conjugated stabilizer-X Paulis in {I, X}.
        op_mask = np.zeros(tokenizer.OP_VOCAB_SIZE, dtype=bool)
        if tokenizer.n_qubits >= 2:
            op_mask[tokenizer.OP_CNOT] = True
        op_mask[tokenizer.EOS] = True
        op_mask = jnp.asarray(op_mask)

        op_id = _masked_sample(logits_op[0, -1], op_mask, greedy, next_key())
        if op_id == tokenizer.EOS:
            ops.append(tokenizer.EOS)
            qa.append(tokenizer.Q_EOS)
            qb.append(tokenizer.Q_EOS)
            break

        _, qa_mask, _ = tokenizer.validity_masks(op_id)
        qa_id = _masked_sample(logits_qa[0, -1], jnp.asarray(qa_mask), greedy, next_key())
        qb_mask = tokenizer.qb_mask_for_qa(op_id, qa_id)
        qb_id = _masked_sample(logits_qb[0, -1], jnp.asarray(qb_mask), greedy, next_key())

        ops.append(op_id)
        qa.append(qa_id)
        qb.append(qb_id)
    else:
        ops.append(tokenizer.EOS)
        qa.append(tokenizer.Q_EOS)
        qb.append(tokenizer.Q_EOS)

    return candidate_to_stabilizers(tokenizer.decode(ops, qa, qb))