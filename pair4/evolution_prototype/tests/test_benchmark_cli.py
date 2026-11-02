import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from pair4.evolution_prototype import DemoEvaluator, ObjectiveSpec, SearchConfig
from pair4.evolution_prototype.benchmark import run_benchmark, statistics
from pair4.evolution_prototype.export import write_run
from pair4.evolution_prototype.search import random_search


ROOT = Path(__file__).resolve().parents[2]


class BenchmarkTests(unittest.TestCase):
    def test_pairing_statistics_and_curve_exports(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "comparison"
            summary = run_benchmark(SearchConfig(budget=12), DemoEvaluator, [3, 9], directory)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["paired_runs_completed"], 2)
            self.assertEqual(len(summary["curves"]), 12)
            self.assertEqual(len(summary["runs"]), 4)
            self.assertTrue(summary["objective"]["is_synthetic"])
            for seed in (3, 9):
                first_events = []
                for method in ("random", "one_plus_one"):
                    history = (directory / f"seed_{seed}" / method / "history.jsonl").read_text().splitlines()
                    self.assertEqual(len(history), 12)
                    first_events.append(json.loads(history[0])["candidate"])
                self.assertEqual(first_events[0], first_events[1])
            self.assertEqual(len((directory / "best_so_far.csv").read_text().splitlines()), 25)
            report = (directory / "report.html").read_text()
            self.assertIn("SYNTHETIC DEMO", report)
            self.assertIn("<svg", report)
            for row in summary["paired_differences"]:
                self.assertEqual(row["evolution_advantage"], row["evolution_minus_random"])

    def test_changed_objective_fails_before_evaluating_second_method(self):
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            evaluator = DemoEvaluator()
            if calls == 2:
                evaluator.objective = ObjectiveSpec("different", "maximize", "2", is_synthetic=True)
            return evaluator

        with TemporaryDirectory() as temporary:
            summary = run_benchmark(SearchConfig(budget=4), factory, [0, 1], Path(temporary) / "results")
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["paired_runs_completed"], 0)
            self.assertEqual(summary["runs"][-1]["evaluation_count"], 0)
            self.assertIn("differs across benchmark", summary["error"])

    def test_factory_failure_preserves_completed_pair_and_partial_manifest(self):
        calls = 0

        def factory():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("offline")
            return DemoEvaluator()

        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "results"
            summary = run_benchmark(SearchConfig(budget=4), factory, [0, 1], directory)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["paired_runs_completed"], 1)
            self.assertEqual(len(summary["paired_differences"]), 1)
            self.assertTrue((directory / "seed_1/random/summary.json").exists())
            self.assertIn("offline", summary["error"])

    def test_duplicate_seeds_are_rejected_without_creating_output(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "results"
            with self.assertRaises(ValueError):
                run_benchmark(SearchConfig(), DemoEvaluator, [1, 1], directory)
            self.assertFalse(directory.exists())

    def test_quartiles_and_single_run_pilot(self):
        stats = statistics([1, 3, 7, 9])
        self.assertEqual((stats["median"], stats["q25"], stats["q75"]), (5, 2.5, 7.5))
        single = statistics([0.25])
        self.assertTrue(all(value == 0.25 for value in single.values()))

    def test_export_refuses_to_overwrite(self):
        result = random_search(SearchConfig(budget=2), DemoEvaluator())
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            write_run(result, directory)
            original = (directory / "history.jsonl").read_bytes()
            with self.assertRaises(FileExistsError):
                write_run(result, directory)
            self.assertEqual((directory / "history.jsonl").read_bytes(), original)


class CLITests(unittest.TestCase):
    def cli(self, *arguments, env=None):
        return subprocess.run([sys.executable, "-m", "pair4", *arguments], cwd=ROOT,
                              text=True, capture_output=True, env=env, timeout=20)

    def test_sample_json_and_explicit_evaluator_requirement(self):
        result = self.cli("sample", "--n", "5", "--k", "1", "--count", "2", "--seed", "42")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(len(data["candidates"]), 2)
        self.assertEqual(data["candidates"][0]["n_qubits"], 5)
        missing = self.cli("search", "--method", "random")
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("--evaluator", missing.stderr)

    def test_demo_search_trace_is_byte_reproducible(self):
        with TemporaryDirectory() as temporary:
            directories = [Path(temporary) / name for name in ("first", "second")]
            for directory in directories:
                result = self.cli("search", "--method", "one_plus_one", "--evaluator", "demo",
                                  "--seed", "42", "--budget", "15", "--output", str(directory))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("SYNTHETIC", result.stdout)
            for filename in ("history.jsonl", "manifest.json", "invocation.json"):
                self.assertEqual((directories[0] / filename).read_bytes(),
                                 (directories[1] / filename).read_bytes())

    def test_missing_provider_is_exported_as_zero_call_failure(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "missing"
            result = self.cli("search", "--method", "random", "--evaluator",
                              "pair4_missing_test_provider:make", "--output", str(directory))
            self.assertEqual(result.returncode, 1)
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["evaluation_count"], 0)
            self.assertIsNone(summary["best"])
            self.assertIn("ModuleNotFoundError", result.stderr)
            self.assertNotIn("SYNTHETIC", result.stdout)

    def test_external_factory_and_objective_configuration(self):
        with TemporaryDirectory() as temporary:
            provider = Path(temporary) / "test_pair4_provider.py"
            provider.write_text('''from pair4 import EvaluationResult, ObjectiveSpec
class External:
    def __init__(self, config):
        self.objective = ObjectiveSpec("test.external_length", "minimize", "1", config, is_synthetic=True)
    def evaluate(self, candidate):
        return EvaluationResult(len(candidate.gates), {"provider": "external test double"})
def make(config):
    return External(config)
''')
            configuration = Path(temporary) / "config.json"
            configuration.write_text(json.dumps({"test_parameter": 0.01}))
            env = dict(os.environ)
            env["PYTHONPATH"] = temporary + os.pathsep + env.get("PYTHONPATH", "")
            directory = Path(temporary) / "external"
            result = self.cli("search", "--method", "random", "--evaluator", "test_pair4_provider:make",
                              "--evaluator-config", str(configuration), "--budget", "8",
                              "--output", str(directory), env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["objective"]["metric_name"], "test.external_length")
            self.assertEqual(manifest["objective"]["parameters"]["test_parameter"], 0.01)
            self.assertEqual(manifest["objective"]["parameters"]["n_qubits"], 4)

    def test_cli_benchmark_and_existing_output_protection(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "benchmark"
            arguments = ("benchmark", "--evaluator", "demo", "--budget", "9",
                         "--seeds", "0", "1", "--output", str(directory))
            result = self.cli(*arguments)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = (directory / "summary.json").read_bytes()
            self.assertEqual(json.loads(original)["paired_runs_completed"], 2)
            repeated = self.cli(*arguments)
            self.assertEqual(repeated.returncode, 1)
            self.assertEqual((directory / "summary.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
