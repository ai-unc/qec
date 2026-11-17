import unittest

import jax
import jax.numpy as jnp

from qec_discovery.search.models import CircuitCandidate, GateOp
from qec_discovery.transformer.generate import generate_stabilizer_code
from qec_discovery.transformer.model import CliffordDecoderOnly
from qec_discovery.transformer.tokenizer import CliffordTokenizer


def _init_params(model: CliffordDecoderOnly, key: jax.Array):
    tok = model.tokenizer
    ops = jnp.asarray([[tok.BOS]], dtype=jnp.int32)
    qa = jnp.asarray([[tok.Q_BOS]], dtype=jnp.int32)
    qb = jnp.asarray([[tok.Q_BOS]], dtype=jnp.int32)
    return model.init(key, ops, qa, qb)


class GenerateEvalTests(unittest.TestCase):
    def test_generate_stabilizer_code(self):
        key = jax.random.PRNGKey(0)
        n, k = 4, 2
        tokenizer = CliffordTokenizer(n_qubits=n, k_logical=k)
        model = CliffordDecoderOnly(
            tokenizer, d_model=64, num_heads=4, num_layers=1, d_ff=128
        )
        params = _init_params(model, key)
        stabilizers = generate_stabilizer_code(model, params, greedy=True)

        self.assertEqual(len(stabilizers), n - k)
        for pauli in stabilizers:
            self.assertEqual(len(pauli), n)
            self.assertTrue(all(c in "IX" for c in pauli))

    def test_forward_shapes(self):
        key = jax.random.PRNGKey(1)
        tokenizer = CliffordTokenizer(n_qubits=4, k_logical=2)
        model = CliffordDecoderOnly(
            tokenizer, d_model=64, num_heads=4, num_layers=1, d_ff=128
        )
        params = _init_params(model, key)
        candidate = CircuitCandidate(
            4, 2, (GateOp("CNOT", (0, 1)),)
        )
        encoded = tokenizer.encode(candidate)
        ops = jnp.expand_dims(jnp.asarray(encoded.ops, dtype=jnp.int32), 0)
        qa = jnp.expand_dims(jnp.asarray(encoded.qa, dtype=jnp.int32), 0)
        qb = jnp.expand_dims(jnp.asarray(encoded.qb, dtype=jnp.int32), 0)
        logits_op, logits_qa, logits_qb = model.apply(params, ops, qa, qb)
        L = int(encoded.ops.shape[0])
        self.assertEqual(tuple(logits_op.shape), (1, L, tokenizer.OP_VOCAB_SIZE))
        self.assertEqual(tuple(logits_qa.shape), (1, L, tokenizer.qubit_vocab_size))
        self.assertEqual(tuple(logits_qb.shape), (1, L, tokenizer.qubit_vocab_size))


if __name__ == "__main__":
    unittest.main()
