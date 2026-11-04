import unittest

from pair4.evolution_prototype import CircuitCandidate, GateOp

from pair5.export import candidate_to_pair2_circuit


class ExportTests(unittest.TestCase):
    def test_candidate_to_pair2_circuit(self):
        candidate = CircuitCandidate(
            4,
            2,
            (
                GateOp("H", (0,)),
                GateOp("CNOT", (0, 2)),
                GateOp("S", (1,)),
            ),
        )
        self.assertEqual(
            candidate_to_pair2_circuit(candidate),
            [("H", 0), ("CNOT", 0, 2), ("S", 1)],
        )


if __name__ == "__main__":
    unittest.main()
