"""Unit tests for evaluate_candidate_generators."""

from __future__ import annotations

import unittest

from qec_discovery.transformer.pipeline_interface import evaluate_candidate_generators


class TestPipelineInterface(unittest.TestCase):
    def test_three_qubit_repetition_code(self) -> None:
        result = evaluate_candidate_generators(["ZZI", "IZZ"])
        self.assertTrue(result.valid)
        self.assertEqual(result.n, 3)
        self.assertEqual(result.k, 1)
        self.assertIsNotNone(result.logical_error_rate)
        assert result.logical_error_rate is not None
        self.assertAlmostEqual(result.logical_error_rate, 0.00725, delta=1e-10)

    def test_invalid_generator_count(self) -> None:
        result = evaluate_candidate_generators(["ZZI"], n=3, k=1)
        self.assertFalse(result.valid)
        self.assertIsNone(result.logical_error_rate)

    def test_anticommuting_generators(self) -> None:
        # X on qubit 0 anticommutes with Z on qubit 0.
        result = evaluate_candidate_generators(["XII", "ZII"], n=3, k=1)
        self.assertFalse(result.valid)
        self.assertIsNone(result.logical_error_rate)


if __name__ == "__main__":
    unittest.main()
