import csv
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pair4 import search
from pair4.backends import get_evaluator, get_validator, vectors
from pair4.evaluation import evaluate_code
from pair4.local_validation import validate_code
from pair4.mutations import mutate, replace_gate, remove_gate
from pair4.random_circuits import random_circuit

ROOT = Path(__file__).resolve().parents[2]


class ValidationTests(unittest.TestCase):
    def test_validation(self):
        for gs, expected in [(['ZII', 'IZI'], True), (['XII', 'ZII'], False),
                             (['ZII', 'ZII'], False), (['III', 'IZI'], False)]:
            self.assertEqual(validate_code(vectors(gs)), expected)
        for bad in ([], [[1, 0, 1]], [[0, 2]], [[1, 0], [1]], [[0, 0]]):
            self.assertFalse(validate_code(bad))
        self.assertFalse(get_validator('original')(vectors(search.SEED)))
        self.assertTrue(get_validator('local')(vectors(search.SEED)))

    def test_evaluator(self):
        for p in (0, 0.05, 0.2):
            self.assertAlmostEqual(evaluate_code(['ZZI', 'IZZ'], p=p), 3*p*p*(1-p)+p**3)
        self.assertAlmostEqual(evaluate_code(['+XI'], p=.05), .05)
        self.assertEqual(evaluate_code(['-ZZI', '+IZZ']), evaluate_code(['ZZI', 'IZZ']))
        for p in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                evaluate_code(['ZZI', 'IZZ'], p=p)
        with self.assertRaises(ValueError):
            evaluate_code(['XII', 'ZII'])
        with self.assertRaises(ValueError):
            evaluate_code(['ZZI', 'IZZ'], noise_model='unknown')
        with self.assertRaisesRegex(ValueError, 'Original evaluator unavailable'):
            get_evaluator('original')
        with patch('pair4.backends.importlib.import_module') as load:
            self.assertIs(get_evaluator('original'), load.return_value.evaluate_code)


class SeedMutationTests(unittest.TestCase):
    def test_seeds(self):
        self.assertEqual(search.starting_code()[0], ['+ZII', '+IZI'])
        self.assertEqual(search.starting_code('custom', ['-XII', 'IXI'])[0], ['-XII', '+IXI'])
        a = search.starting_code('random', seed=91)
        self.assertEqual(a, search.starting_code('random', seed=91))
        self.assertTrue(search.valid_code(a[0]))
        self.assertEqual(search.transform_generators([('H',0),('S',0),('S',0),('H',0)])[0], '-ZII')
        for mode, gs in [('custom', None), ('canonical', ['ZII','IZI']), ('custom', ['XII','ZII']), ('custom', ['ZZ','IZ'])]:
            with self.assertRaises(ValueError):
                search.starting_code(mode, gs)

    def test_mutations(self):
        parent = [('H', 0)]
        for seed in range(100):
            self.assertNotEqual(replace_gate(parent, 3, random.Random(seed)), parent)
            self.assertLessEqual(len(mutate(parent, 3, random.Random(seed), 1)), 1)
        self.assertEqual(parent, [('H',0)])
        self.assertEqual(len(mutate([], 3, random.Random(1), 1)), 1)
        for operation in (replace_gate, remove_gate):
            with self.assertRaises(ValueError):
                operation([])
        gates = random_circuit(100, 1, 42)
        self.assertTrue(all(g[0] in ('H', 'S') and g[1] == 0 for g in gates))
        for length, n in [(-1,3), (1,0)]:
            with self.assertRaises(ValueError):
                random_circuit(length,n)


class SearchTests(unittest.TestCase):
    def test_discovery_and_replay(self):
        for method in ('random', 'evolution'):
            report = search.run_search(method=method)
            self.assertAlmostEqual(report['results'][0]['error_rate'], search.TARGET)
            self.assertEqual(report['attempts'], 1000)
            self.assertEqual(report['results'], sorted(report['results'], key=lambda r:r['error_rate']))
            self.assertEqual(report, search.run_search(method=method))

    def test_cache_budget_and_original(self):
        with patch('pair4.search.get_evaluator', return_value=lambda *args: .05) as factory:
            evaluator = unittest.mock.Mock(return_value=.05)
            factory.return_value = evaluator
            r = search.run_search(trials=5, depth=0)
            self.assertEqual(r['attempts'], 5)
            self.assertEqual(r['unique_evaluations'], 1)
            self.assertEqual(evaluator.call_count, 2)  # Starting diagnostic plus cached search.
        r = search.run_search(trials=3, depth=0, validator_source='original')
        self.assertEqual(r['validation_rejections'], 3)
        self.assertEqual(r['results'], [])

    def test_acceptance_and_best_retention(self):
        a, b, c, d = [('H',0)], [('S',0)], [('H',1)], [('S',1)]
        outputs = [['XII','IZI'], ['YII','IZI'], ['ZII','IZI'], ['ZII','IXI']]
        # Unique keys and controlled scores isolate selection from quantum calculations.
        with patch('pair4.search.random_clifford_circuit', return_value=a), \
             patch('pair4.search.mutate', side_effect=[b,c,d]) as mutation, \
             patch('pair4.search.transform_generators', side_effect=outputs), \
             patch('pair4.search.stabilizer_span', side_effect=[{(i,)} for i in range(4)]), \
             patch('pair4.search.get_evaluator', return_value=unittest.mock.Mock(side_effect=[.5,.1,.2,.1,.3])):
            r = search.run_search(trials=4, method='evolution')
        self.assertEqual(mutation.call_args_list[0].args[0], a)
        self.assertEqual(mutation.call_args_list[1].args[0], a)  # Worse b rejected.
        self.assertEqual(mutation.call_args_list[2].args[0], c)  # Equal c accepted.
        self.assertEqual(r['results'][0]['circuit'], a)

    def test_cli_modes(self):
        for method in ('random','evolution','both'):
            with tempfile.TemporaryDirectory() as tmp:
                command = [sys.executable, str(ROOT/'pair4/search.py'), '--method',method,
                           '--trials','8','--seed-mode','random','--output-dir',tmp]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                stems = ['results_random','results_evolution'] if method == 'both' else ['results']
                reports = []
                for stem in stems:
                    reports.append(json.loads((Path(tmp)/(stem+'.json')).read_text()))
                    with (Path(tmp)/(stem+'.csv')).open() as f:
                        self.assertIn('provenance', next(csv.reader(f)))
                if method == 'both':
                    self.assertEqual(reports[0]['seed_generators'],reports[1]['seed_generators'])
        with tempfile.TemporaryDirectory() as tmp:
            signed = subprocess.run([sys.executable, '-m', 'pair4.search', '--trials', '2',
                                     '--seed-mode', 'custom', '--seed-generators', '-XII', '+IXI',
                                     '--output-dir', tmp], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(signed.returncode, 0, signed.stderr)
            self.assertEqual(json.loads((Path(tmp)/'results.json').read_text())['seed_generators'], ['-XII', '+IXI'])
        bad = subprocess.run([sys.executable,'-m','pair4.search','--evaluator-source','original'], cwd=ROOT, capture_output=True,text=True)
        self.assertNotEqual(bad.returncode,0)
        self.assertIn('Original evaluator unavailable',bad.stderr)


if __name__ == '__main__':
    unittest.main()
