from collections import Counter
from random import Random
import unittest

from pair4.evolution_prototype import (CircuitCandidate, GateOp, SamplingConfig, SearchConfig,
                   mutate_circuit, random_stream, sample_random_circuit, sample_random_gate)


class ChosenEdit(Random):
    def __init__(self, operation, index):
        super().__init__(0)
        self.operation = operation
        self.index = index

    def choice(self, options):
        if self.operation in options:
            return self.operation
        return super().choice(options)

    def randrange(self, *args):
        # Only the first range draw is the edit location; later draws sample gates.
        index, self.index = self.index, None
        return index if index is not None else super().randrange(*args)


class OperatorTests(unittest.TestCase):
    def test_invalid_gate_records_and_registers(self):
        for op, qubits in [("T", (0,)), ("H", (0, 1)), ("CNOT", (0,)),
                           ("CNOT", (1, 1)), ("S", (-1,)), ("H", (True,))]:
            with self.subTest(op=op, qubits=qubits), self.assertRaises(ValueError):
                GateOp(op, qubits)
        for n, k in [(0, 0), (2, -1), (2, 3), (True, 0)]:
            with self.subTest(n=n, k=k), self.assertRaises(ValueError):
                CircuitCandidate(n, k)
        with self.assertRaises(ValueError):
            CircuitCandidate(2, 1, (GateOp("H", (2,)),))

    def test_configuration_rejects_nonterminating_or_empty_budgets(self):
        for cap in (0, -1, True):
            with self.assertRaises(ValueError):
                SamplingConfig(max_length=cap)
        for budget in (0, -2, True):
            with self.assertRaises(ValueError):
                SearchConfig(budget=budget)
        self.assertEqual(SamplingConfig(5, 1).max_length, 40)

    def test_serialization_preserves_idle_wires_and_cnot_direction(self):
        original = CircuitCandidate(5, 1, (GateOp("CNOT", (3, 0)), GateOp("S", (0,))))
        self.assertEqual(CircuitCandidate.from_dict(original.to_dict()), original)
        empty = CircuitCandidate(5, 1)
        self.assertEqual(CircuitCandidate.from_dict(empty.to_dict()), empty)
        with self.assertRaises(ValueError):
            CircuitCandidate.from_dict({**empty.to_dict(), "schema_version": 2})

    def test_gate_distribution_and_directed_operands(self):
        rng = Random(20261028)
        gates = [sample_random_gate(5, rng) for _ in range(12000)]
        counts = Counter(gate.op for gate in gates)
        for op in ("H", "S", "CNOT"):
            self.assertAlmostEqual(counts[op] / len(gates), 1 / 3, delta=0.025)
        pairs = Counter(gate.qubits for gate in gates if gate.op == "CNOT")
        self.assertEqual(len(pairs), 20)
        for (control, target), count in pairs.items():
            self.assertNotEqual(control, target)
            self.assertAlmostEqual(count / counts["CNOT"], 1 / 20, delta=0.02)

    def test_single_qubit_sampling_and_mutation(self):
        rng = Random(19)
        candidate = CircuitCandidate(1, 0)
        counts = Counter(sample_random_gate(1, rng).op for _ in range(4000))
        self.assertAlmostEqual(counts["H"] / 4000, 0.5, delta=0.04)
        for _ in range(500):
            candidate = mutate_circuit(candidate, 6, rng).candidate
            self.assertLessEqual(len(candidate.gates), 6)
            self.assertTrue(all(gate.op in ("H", "S") for gate in candidate.gates))

    def test_length_distribution_includes_identity_and_cap(self):
        rng = Random(8)
        config = SamplingConfig(2, 1, 4)
        counts = Counter(len(sample_random_circuit(config, rng).gates) for _ in range(5000))
        self.assertEqual(set(counts), set(range(5)))
        for count in counts.values():
            self.assertAlmostEqual(count / 5000, 0.2, delta=0.035)

    def test_insert_at_both_endpoints_without_modifying_parent(self):
        parent = CircuitCandidate(2, 1, (GateOp("S", (0,)), GateOp("S", (1,))))
        for index in (0, 2):
            result = mutate_circuit(parent, 3, ChosenEdit("insert", index))
            self.assertEqual(result.operation, "insert")
            self.assertEqual(result.index, index)
            self.assertEqual(len(result.candidate.gates), 3)
            remaining = result.candidate.gates[:index] + result.candidate.gates[index + 1:]
            self.assertEqual(remaining, parent.gates)
        self.assertEqual(len(parent.gates), 2)

    def test_delete_every_position(self):
        parent = CircuitCandidate(2, 1, (GateOp("H", (0,)), GateOp("S", (0,)), GateOp("H", (1,))))
        for index in range(3):
            child = mutate_circuit(parent, 3, ChosenEdit("delete", index)).candidate
            self.assertEqual(child.gates, parent.gates[:index] + parent.gates[index + 1:])

    def test_boundary_masking_and_nonidentical_replacement(self):
        rng = Random(91)
        empty = CircuitCandidate(4, 2)
        self.assertEqual(mutate_circuit(empty, 1, rng).operation, "insert")
        full = CircuitCandidate(4, 2, (GateOp("H", (0,)),))
        for _ in range(100):
            result = mutate_circuit(full, 1, rng)
            self.assertIn(result.operation, ("replace", "delete"))
            self.assertNotEqual(result.candidate, full)
        with self.assertRaises(ValueError):
            mutate_circuit(CircuitCandidate(2, 1, (GateOp("H", (0,)),) * 2), 1, rng)

    def test_replacement_uses_conditioned_hierarchical_distribution(self):
        parent = CircuitCandidate(3, 1, (GateOp("H", (0,)),))
        rng = Random(170)
        rng.choice = lambda options: "replace"
        counts = Counter()
        for _ in range(6000):
            child = mutate_circuit(parent, 2, rng).candidate
            self.assertNotEqual(child, parent)
            counts[child.gates[0].op] += 1
        for op, probability in {"H": 0.25, "S": 0.375, "CNOT": 0.375}.items():
            self.assertAlmostEqual(counts[op] / 6000, probability, delta=0.035)

    def test_streams_are_isolated_and_reproducible(self):
        config = SamplingConfig()
        expected = sample_random_circuit(config, random_stream(42, "initial"))
        for _ in range(20):
            sample_random_circuit(config, random_stream(999, "mutation"))
        self.assertEqual(expected, sample_random_circuit(config, random_stream(42, "initial")))


if __name__ == "__main__":
    unittest.main()
