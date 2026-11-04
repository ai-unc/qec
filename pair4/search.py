"""Three-qubit stabilizer search. Run: python pair4/search.py --help."""
import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pair4.backends import (ROOT, apply_circuit, get_validator, get_evaluator,
                            normalized, vectors)
from pair4.local_validation import rank_gf2, validate_code
from pair4.evaluation import evaluate_code, stabilizer_span
from pair4.random_circuits import random_circuit
from pair4.mutations import mutate

SEED = ['ZII', 'IZI']
TARGET = 0.00725
TOLERANCE = 1e-12


def random_clifford_circuit(n=3, depth=12, rng=None):
    return random_circuit(depth, n, rng=rng)


def transform_generators(circuit, seed=SEED):
    return [apply_circuit(g, circuit) for g in normalized(seed)]


def valid_code(generators, n=3, k=1):
    try:
        gs = normalized(generators)
        return len(gs) == n-k and len(gs[0]) == n+1 and validate_code(vectors(gs))
    except (TypeError, ValueError):
        return False


def exact_logical_error_rate(generators, p=0.05, noise_model='bit_flip'):
    return evaluate_code(generators, noise_model, p)


def starting_code(seed_mode='canonical', seed_generators=None, seed=42, seed_depth=12):
    if seed_mode not in ('canonical', 'custom', 'random'):
        raise ValueError('Unknown seed mode')
    if (seed_mode == 'custom') != (seed_generators is not None):
        raise ValueError('seed_generators is required only in custom mode')
    preparation = []
    gs = SEED if seed_mode != 'custom' else seed_generators
    if seed_mode == 'random':
        preparation = random_clifford_circuit(depth=seed_depth, rng=random.Random(f'{seed}:starting-code'))
        gs = transform_generators(preparation)
    gs = normalized(gs)
    if not valid_code(gs):
        raise ValueError('Starting code needs two independent commuting three-qubit generators')
    return gs, preparation


def run_search(trials=1000, depth=12, p=0.05, seed=42, *, method='random',
               max_depth=24, seed_mode='canonical', seed_generators=None,
               seed_depth=12, validator_source='local', evaluator_source='local',
               initial_code=None):
    if method not in ('random', 'evolution'):
        raise ValueError('method must be random or evolution')
    if not isinstance(trials, int) or trials < 1 or not isinstance(depth, int) or depth < 0:
        raise ValueError('trials must be positive and depth nonnegative integers')
    if not isinstance(seed_depth, int) or seed_depth < 0:
        raise ValueError('seed_depth must be nonnegative')
    if method == 'evolution' and (not isinstance(max_depth, int) or max_depth < max(1, depth)):
        raise ValueError('max_depth must be positive and at least depth')
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('p must be between 0 and 1')
    validator, evaluator = get_validator(validator_source), get_evaluator(evaluator_source)
    gs, preparation = initial_code if initial_code is not None else starting_code(seed_mode, seed_generators, seed, seed_depth)
    if not valid_code(gs):
        raise ValueError('Invalid starting code')
    initial_rate = float(evaluator(gs, 'bit_flip', p))
    if not math.isfinite(initial_rate) or not 0 <= initial_rate <= 1:
        raise ValueError('Evaluator returned an invalid starting-code probability')
    rng = random.Random(seed if method == 'random' else f'{seed}:evolution')
    results, cache = [], {}
    incumbent, incumbent_rate = None, None
    rejected = 0
    for trial in range(trials):
        circuit = (mutate(incumbent, 3, rng, max_depth) if method == 'evolution' and incumbent is not None
                   else random_clifford_circuit(depth=depth, rng=rng))
        generators = transform_generators(circuit, gs)
        if not validator(vectors(generators)):
            rejected += 1
            continue
        key = tuple(sorted(stabilizer_span(generators)))
        if key not in cache:
            rate = float(evaluator(generators, 'bit_flip', p))
            if not math.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError('Evaluator returned an invalid logical error probability')
            cache[key] = rate
            results.append(dict(generators=generators, error_rate=rate, circuit=circuit, trial=trial))
        rate = cache[key]
        if incumbent_rate is None or rate <= incumbent_rate + TOLERANCE:
            incumbent, incumbent_rate = circuit, rate
    results.sort(key=lambda r: r['error_rate'])
    return dict(results=results, method=method, attempts=trials, unique_evaluations=len(cache),
                validation_rejections=rejected, validator_source=validator_source,
                evaluator_source=evaluator_source, seed_mode=seed_mode, seed_generators=gs,
                preparation_circuit=preparation, initial_error_rate=initial_rate,
                initial_meets_target=initial_rate <= TARGET + TOLERANCE,
                target_reached=bool(results and results[0]['error_rate'] <= TARGET + TOLERANCE),
                seed=seed, depth=depth, max_depth=max_depth, seed_depth=seed_depth, p=p)


def search_codes(trials=1000, depth=12, p=0.05, seed=42, **options):
    return run_search(trials, depth, p, seed, **options)['results']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['random', 'evolution', 'both'], default='random')
    for name, default in [('trials', 1000), ('depth', 12), ('max-depth', 24), ('seed', 42), ('seed-depth', 12)]:
        parser.add_argument('--'+name, type=int, default=default)
    parser.add_argument('--p', type=float, default=0.05)
    parser.add_argument('--seed-mode', choices=['canonical', 'custom', 'random'], default='canonical')
    parser.add_argument('--seed-generators', nargs=2)
    for name in ('validator', 'evaluator'):
        parser.add_argument('--'+name+'-source', choices=['local', 'original'], default='local')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'pair4')
    parser._negative_number_matcher = re.compile(r'^-[IXYZ]+$|^-\d*\.?\d+(?:[eE][+-]?\d+)?$')
    args = vars(parser.parse_args(argv))
    output = args.pop('output_dir')
    method = args.pop('method')
    try:
        get_evaluator(args['evaluator_source'])  # Fail before baseline or searching.
        initial = starting_code(args['seed_mode'], args['seed_generators'], args['seed'], args['seed_depth'])
        baseline = exact_logical_error_rate(['ZZI', 'IZZ'])
        if abs(baseline - TARGET) > TOLERANCE:
            raise ValueError('Repetition-code baseline failed')
        print(f'Repetition-code baseline: {baseline:.4%}')
        for selected in (['random', 'evolution'] if method == 'both' else [method]):
            report = run_search(method=selected, initial_code=initial, **args)
            output.mkdir(parents=True, exist_ok=True)
            stem = 'results_'+selected if method == 'both' else 'results'
            metadata = {k: v for k, v in report.items() if k != 'results'}
            (output / (stem+'.json')).write_text(json.dumps(report, indent=2)+'\n')
            with (output / (stem+'.csv')).open('w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['rank', 'generators', 'logical_error_rate', 'circuit', 'trial', 'method', 'validator_source', 'evaluator_source', 'provenance'])
                for rank, r in enumerate(report['results'], 1):
                    writer.writerow([rank, ' '.join(r['generators']), r['error_rate'], repr(r['circuit']), r['trial'], selected, args['validator_source'], args['evaluator_source'], json.dumps(metadata)])
            print(f"{selected}: validator={args['validator_source']}, evaluator={args['evaluator_source']}; attempts={report['attempts']}, unique evaluations={report['unique_evaluations']}, rejections={report['validation_rejections']}")
            print(f"Starting code {initial[0]}: {report['initial_error_rate']:.4%}; initially meets target={report['initial_meets_target']}")
            print(f"Search reached target={report['target_reached']}")
            for r in report['results'][:5]:
                print(f"{r['generators']} -> {r['error_rate']:.4%}")
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
