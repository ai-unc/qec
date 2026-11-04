import unittest

from pair4.evolution_prototype import CircuitCandidate, GateOp

from pair5.stabilizers import candidate_to_stabilizers


class StabilizerTests(unittest.TestCase):
    def test_empty_encoder_stabilizer_x(self):
        self.assertEqual(candidate_to_stabilizers(CircuitCandidate(2, 1)), ["XI"])

    def test_cnot_spreads_x(self):
        candidate = CircuitCandidate(2, 0, (GateOp("CNOT", (0, 1)),))
        self.assertEqual(candidate_to_stabilizers(candidate), ["XX", "IX"])

    def test_xix_example(self):
        candidate = CircuitCandidate(3, 1, (GateOp("CNOT", (0, 2)),))
        self.assertEqual(candidate_to_stabilizers(candidate), ["XIX", "IXI"])

if __name__ == "__main__":
    unittest.main()
