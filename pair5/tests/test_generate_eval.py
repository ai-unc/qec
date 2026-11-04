import unittest

import torch

from pair4.evolution_prototype import CircuitCandidate, GateOp

from pair5.generate import generate_stabilizer_code
from pair5.model import CliffordDecoderOnly
from pair5.tokenizer import CliffordTokenizer


class GenerateEvalTests(unittest.TestCase):
    def test_generate_stabilizer_code(self):
        torch.manual_seed(0)
        n, k = 4, 2
        tokenizer = CliffordTokenizer(n_qubits=n, k_logical=k)
        model = CliffordDecoderOnly(tokenizer, d_model=64, num_heads=4, num_layers=1, d_ff=128)
        stabilizers = generate_stabilizer_code(model, greedy=True)

        self.assertEqual(len(stabilizers), n - k)
        for pauli in stabilizers:
            self.assertEqual(len(pauli), n)
            self.assertTrue(all(c in "IX" for c in pauli))

    def test_forward_shapes(self):
        tokenizer = CliffordTokenizer(n_qubits=4, k_logical=2)
        model = CliffordDecoderOnly(tokenizer, d_model=64, num_heads=4, num_layers=1, d_ff=128)
        candidate = CircuitCandidate(
            4, 2, (GateOp("CNOT", (0, 1)),)
        )
        encoded = tokenizer.encode(candidate)
        logits_op, logits_qa, logits_qb = model(
            encoded.ops.unsqueeze(0),
            encoded.qa.unsqueeze(0),
            encoded.qb.unsqueeze(0),
        )
        L = encoded.ops.size(0)
        self.assertEqual(tuple(logits_op.shape), (1, L, tokenizer.OP_VOCAB_SIZE))
        self.assertEqual(tuple(logits_qa.shape), (1, L, tokenizer.qubit_vocab_size))
        self.assertEqual(tuple(logits_qb.shape), (1, L, tokenizer.qubit_vocab_size))


if __name__ == "__main__":
    unittest.main()
