import unittest

from pair4.evolution_prototype import CircuitCandidate, GateOp

from pair5.tokenizer import CliffordTokenizer


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tok = CliffordTokenizer(n_qubits=4, k_logical=2)

    def test_encode_decode_round_trip(self):
        gates = (
            GateOp("H", (0,)),
            GateOp("CNOT", (0, 2)),
            GateOp("S", (1,)),
        )
        candidate = CircuitCandidate(4, 2, gates)
        encoded = self.tok.encode(candidate)
        decoded = self.tok.decode(encoded.ops, encoded.qa, encoded.qb)
        self.assertEqual(decoded, candidate)

    def test_cnot_same_wire_rejected(self):
        gate = self.tok.decode_gate(
            self.tok.OP_CNOT,
            self.tok.qubit_id(1),
            self.tok.qubit_id(1),
        )
        self.assertIsNone(gate)

    def test_validity_mask_blocks_cnot_self(self):
        qa = self.tok.qubit_id(2)
        qb_mask = self.tok.qb_mask_for_qa(self.tok.OP_CNOT, qa)
        self.assertFalse(bool(qb_mask[qa].item()))
        self.assertTrue(bool(qb_mask[self.tok.qubit_id(0)].item()))

    def test_hs_requires_unused_qb(self):
        self.assertIsNone(
            self.tok.decode_gate(self.tok.OP_H, self.tok.qubit_id(0), self.tok.qubit_id(1))
        )
        self.assertEqual(
            self.tok.decode_gate(self.tok.OP_H, self.tok.qubit_id(0), self.tok.Q_UNUSED),
            GateOp("H", (0,)),
        )


if __name__ == "__main__":
    unittest.main()
