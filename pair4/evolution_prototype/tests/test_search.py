import unittest
from unittest.mock import patch

from pair4.evolution_prototype import (CircuitCandidate, DemoEvaluator, EvaluationResult, GateOp, MutationResult,
                   ObjectiveSpec, SamplingConfig, SearchConfig, one_plus_one_search, random_search)


class ScriptedEvaluator:
    def __init__(self, values=None, direction="minimize", synthetic=True):
        self.objective = ObjectiveSpec("test.scripted", direction, "1", is_synthetic=synthetic)
        self.values = iter(values) if values is not None else None
        self.calls = []

    def evaluate(self, candidate):
        self.calls.append(candidate)
        value = next(self.values) if self.values is not None else 1
        if isinstance(value, Exception):
            raise value
        return EvaluationResult(value)


class SearchTests(unittest.TestCase):
    def test_exact_budgets_and_initial_candidate_pairing(self):
        for budget in (1, 2, 17):
            config = SearchConfig(budget=budget, seed=18)
            first, second = ScriptedEvaluator(), ScriptedEvaluator()
            random = random_search(config, first)
            evolution = one_plus_one_search(config, second)
            self.assertEqual(random.status, "completed")
            self.assertEqual(evolution.status, "completed")
            self.assertEqual(len(first.calls), budget)
            self.assertEqual(len(second.calls), budget)
            self.assertEqual(first.calls[0], second.calls[0])

    def test_primary_score_precedes_structural_cost_and_lineage(self):
        initial = CircuitCandidate(2, 1, (GateOp("H", (0,)),))
        better = CircuitCandidate(2, 1, (GateOp("CNOT", (0, 1)),))
        worse = CircuitCandidate(2, 1)
        edits = [MutationResult(better, "replace", 0), MutationResult(worse, "delete", 0)]
        config = SearchConfig(SamplingConfig(2, 1, 2), 3)
        with patch("pair4.search.mutate_circuit", side_effect=edits):
            result = one_plus_one_search(config, ScriptedEvaluator([5, 4, 6]), initial_candidate=initial)
        self.assertEqual([event.accepted for event in result.events], [True, True, False])
        self.assertEqual([event.parent_index for event in result.events], [None, 1, 2])
        self.assertEqual([event.best_so_far for event in result.events], [5, 4, 4])
        self.assertEqual(result.best.candidate, better)

    def test_ties_prefer_fewer_cnots_then_fewer_gates(self):
        candidates = [
            CircuitCandidate(2, 1, (GateOp("CNOT", (0, 1)),)),
            CircuitCandidate(2, 1, (GateOp("H", (0,)), GateOp("S", (1,)))),
            CircuitCandidate(2, 1, (GateOp("H", (0,)),)),
        ]
        with patch("pair4.search.sample_random_circuit", side_effect=candidates):
            result = random_search(SearchConfig(SamplingConfig(2, 1, 3), 3), ScriptedEvaluator())
        self.assertEqual([item.candidate for item in result.archive], list(reversed(candidates)))
        self.assertEqual(result.best.candidate, candidates[-1])

    def test_small_primary_improvements_are_not_rounded(self):
        initial = CircuitCandidate(2, 1)
        child = CircuitCandidate(2, 1, (GateOp("CNOT", (0, 1)),))
        with patch("pair4.search.mutate_circuit", return_value=MutationResult(child, "insert", 0)):
            result = one_plus_one_search(SearchConfig(SamplingConfig(2, 1), 2),
                                        ScriptedEvaluator([1, 1 - 1e-13]), initial_candidate=initial)
        self.assertEqual(result.best.candidate, child)

    def test_equal_rank_can_accept_or_reject_without_replacing_best(self):
        initial = CircuitCandidate(2, 1, (GateOp("H", (0,)),))
        child = CircuitCandidate(2, 1, (GateOp("H", (1,)),))
        decisions = []
        for seed in range(40):
            with patch("pair4.search.mutate_circuit", return_value=MutationResult(child, "replace", 0)):
                result = one_plus_one_search(SearchConfig(SamplingConfig(2, 1), 2, seed),
                                            ScriptedEvaluator(), initial_candidate=initial)
            decisions.append(result.events[-1].accepted)
            self.assertEqual(result.best.candidate, initial)
        self.assertIn(True, decisions)
        self.assertIn(False, decisions)

    def test_replay_excludes_only_timing(self):
        for method in (random_search, one_plus_one_search):
            config = SearchConfig(budget=60, seed=42)
            first = method(config, DemoEvaluator())
            method(SearchConfig(budget=20, seed=999), DemoEvaluator())
            second = method(config, DemoEvaluator())
            self.assertEqual(first.manifest(), second.manifest())
            self.assertEqual([event.to_dict() for event in first.events],
                             [event.to_dict() for event in second.events])
            left, right = first.summary(), second.summary()
            left.pop("timing")
            right.pop("timing")
            self.assertEqual(left, right)

    def test_archive_is_unique_bounded_and_not_an_evaluation_cache(self):
        config = SearchConfig(SamplingConfig(2, 1, 1), 120)
        evaluator = ScriptedEvaluator()
        result = random_search(config, evaluator)
        self.assertEqual(result.evaluation_count, 120)
        self.assertEqual(len(evaluator.calls), 120)
        unique = {item.candidate for item in result.archive}
        self.assertEqual(len(unique), len(result.archive))
        self.assertLessEqual(len(unique), 10)
        self.assertLess(len(set(evaluator.calls)), 120)

    def test_failure_retains_partial_best_and_counts_failed_call(self):
        result = one_plus_one_search(SearchConfig(budget=10),
                                    ScriptedEvaluator([0.2, RuntimeError("provider offline")]))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.evaluation_count, 2)
        self.assertEqual(result.best.evaluation.value, 0.2)
        self.assertIn("provider offline", result.error)
        self.assertFalse(result.events[-1].accepted)
        self.assertIsNone(result.events[-1].evaluation)

    def test_nonfinite_scores_fail_instead_of_winning(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            result = random_search(SearchConfig(budget=2), ScriptedEvaluator([value]))
            self.assertEqual(result.status, "failed")
            self.assertEqual(result.evaluation_count, 1)
            self.assertIsNone(result.best)

    def test_estimates_and_invalid_scientific_dimensions_are_rejected(self):
        evaluator = ScriptedEvaluator()
        evaluator.objective = ObjectiveSpec("estimated", "minimize", "1", exact=False)
        result = random_search(SearchConfig(), evaluator)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.evaluation_count, 0)
        result = random_search(SearchConfig(SamplingConfig(1, 0)), ScriptedEvaluator(synthetic=False))
        self.assertEqual(result.evaluation_count, 0)
        self.assertIn("scientific searches", result.error)

    def test_objective_change_is_an_explicit_failure(self):
        class ChangingEvaluator(ScriptedEvaluator):
            def evaluate(self, candidate):
                result = super().evaluate(candidate)
                if len(self.calls) == 2:
                    self.objective = ObjectiveSpec("different", "minimize", "2", is_synthetic=True)
                return result

        result = random_search(SearchConfig(budget=10), ChangingEvaluator())
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.evaluation_count, 2)
        self.assertIn("metadata changed", result.error)
        self.assertEqual(result.best.evaluation_index, 1)

    def test_best_so_far_monotonic_in_both_directions(self):
        for direction in ("minimize", "maximize"):
            result = random_search(SearchConfig(budget=5),
                                   ScriptedEvaluator([4, 2, 5, 1, 3], direction))
            values = [event.best_so_far for event in result.events]
            self.assertEqual(values, sorted(values, reverse=(direction == "minimize")))

    def test_demo_distance_has_known_values_and_synthetic_label(self):
        evaluator = DemoEvaluator()
        target = CircuitCandidate(4, 2, evaluator.target)
        self.assertEqual(evaluator.evaluate(target).value, 0)
        self.assertEqual(evaluator.evaluate(CircuitCandidate(4, 2)).value, -6)
        extra = CircuitCandidate(4, 2, target.gates + (GateOp("S", (3,)),))
        self.assertEqual(evaluator.evaluate(extra).value, -1)
        self.assertTrue(evaluator.objective.is_synthetic)

    def test_objective_parameters_and_diagnostics_are_copied_and_frozen(self):
        original = {"noise": {"p": 0.01}, "labels": ["a"]}
        objective = ObjectiveSpec("probability", "minimize", "1", original)
        original["noise"]["p"] = 0.9
        self.assertEqual(objective.to_dict()["parameters"]["noise"]["p"], 0.01)
        with self.assertRaises(TypeError):
            objective.parameters["noise"]["p"] = 0.5
        with self.assertRaises(ValueError):
            EvaluationResult(0, {"invalid": float("nan")})


if __name__ == "__main__":
    unittest.main()
