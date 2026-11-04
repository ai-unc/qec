import unittest

import torch

from pair4.evolution_prototype import CircuitCandidate, GateOp

from pair5.generate import generate_circuit
from pair5.model import CliffordDecoderOnly
from pair5.tokenizer import CliffordTokenizer


class GenerateEvalTests(unittest.TestCase):
    def test_generate_valid_candidate(self):
        torch.manual_seed(0)
        tokenizer = CliffordTokenizer(n_qubits=4, k_logical=2)
        model = CliffordDecoderOnly(tokenizer, d_model=64, num_heads=4, num_layers=1, d_ff=128)
        candidate = generate_circuit(model, greedy=True)

        self.assertIsInstance(candidate, CircuitCandidate)
        self.assertEqual(candidate.n_qubits, 4)
        self.assertEqual(candidate.k_logical, 2)
        for gate in candidate.gates:
            self.assertIsInstance(gate, GateOp)
            if gate.op == "CNOT":
                self.assertNotEqual(gate.qubits[0], gate.qubits[1])

    def test_forward_shapes(self):
        tokenizer = CliffordTokenizer(n_qubits=4, k_logical=2)
        model = CliffordDecoderOnly(tokenizer, d_model=64, num_heads=4, num_layers=1, d_ff=128)
        candidate = CircuitCandidate(
            4, 2, (GateOp("H", (0,)), GateOp("CNOT", (0, 1)), GateOp("S", (1,)))
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
