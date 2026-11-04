"""Pair 4: random Clifford search for [[3,1]] stabilizer codes.
Run from repository root: python pair4/search.py
"""
import csv
import random
import sys
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT / 'pair1', ROOT / 'pair2', ROOT / 'pair3'):
    sys.path.insert(0, str(folder))

from pauli_to_bin import pauli_to_bin
from pauli_gates_stim import apply_circuit
from noise import error_probability
from syndrome import calculate_syndrome

SEED = ['ZII', 'IZI']  # [[3,1]] seed, not the full |000> state
TARGET = 0.00725


def random_clifford_circuit(n=3, depth=12, rng=None):
    rng = rng or random
    gates = []
    for _ in range(depth):
        gate = rng.choice(('H', 'S', 'CNOT'))
        if gate == 'CNOT':
            a, b = rng.sample(range(n), 2)
            gates.append((gate, a, b))
        else:
            gates.append((gate, rng.randrange(n)))
    return gates


def transform_generators(circuit, seed=SEED):
    # Pair 2 preserves a leading + or - sign.
    return [apply_circuit('+' + g, circuit) for g in seed]


def rank_gf2(rows):
    a = [list(map(int, row)) for row in rows]
    if not a:
        return 0
    rank = 0
    for col in range(len(a[0])):
        pivot = next((r for r in range(rank, len(a)) if a[r][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        for r in range(len(a)):
            if r != rank and a[r][col]:
                a[r] = [x ^ y for x, y in zip(a[r], a[rank])]
        rank += 1
        if rank == len(a):
            break
    return rank


def valid_code(generators, n=3, k=1):
    if len(generators) != n-k:
        return False
    paulis = [g.lstrip('+-') for g in generators]
    if any(len(g) != n or any(c not in 'IXYZ' for c in g) for g in paulis):
        return False
    vectors = [pauli_to_bin(g) for g in paulis]
    # Pair 1's validate_code expects binary symplectic vectors.
    # Its current gen_ind has a known bug, so independently check rank and commutation.
    if rank_gf2(vectors) != n-k:
        return False
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            a, b = vectors[i], vectors[j]
            if (sum(int(a[t])*int(b[t+n]) + int(a[t+n])*int(b[t])
                    for t in range(n)) % 2):
                return False
    return True


def stabilizer_span(generators):
    """Binary symplectic span, ignoring global phases for Pauli error equivalence."""
    vectors = [tuple(map(int, pauli_to_bin(g.lstrip('+-')))) for g in generators]
    zero = (0,) * len(vectors[0])
    span = {zero}
    for vector in vectors:
        span |= {tuple(x ^ y for x, y in zip(row, vector)) for row in tuple(span)}
    return span


def exact_logical_error_rate(generators, p=0.05, noise_model='bit_flip'):
    """Maximum-likelihood syndrome decoder, optimized for logical success.

    Enumerates all physical errors and groups by syndrome and residual
    stabilizer coset. This avoids incorrect tie-breaking by individual errors.
    """
    n = len(generators[0].lstrip('+-'))
    span = stabilizer_span(generators)
    by_syndrome = {}
    for letters in product('IXYZ', repeat=n):
        error = ''.join(letters)
        probability = float(error_probability(error, noise_model, p))
        if probability == 0:
            continue
        syndrome = tuple(map(int, calculate_syndrome(
            [g.lstrip('+-') for g in generators], error)))
        vector = tuple(map(int, pauli_to_bin(error)))
        # Canonical label of the stabilizer coset containing this error.
        coset = min(tuple(x ^ y for x, y in zip(vector, s)) for s in span)
        key = (syndrome, coset)
        by_syndrome[key] = by_syndrome.get(key, 0.0) + probability
    # For each syndrome, a decoder can recover one coset successfully.
    best = {}
    for (syndrome, _), probability in by_syndrome.items():
        best[syndrome] = max(best.get(syndrome, 0.0), probability)
    return max(0.0, 1.0 - sum(best.values()))


def search_codes(trials=1000, depth=12, p=0.05, seed=42):
    rng = random.Random(seed)
    results = []
    seen = set()
    for trial in range(trials):
        circuit = random_clifford_circuit(depth=depth, rng=rng)
        generators = transform_generators(circuit)
        if not valid_code(generators):
            continue
        # Deduplicate equivalent stabilizer spaces, rather than generator order.
        key = tuple(sorted(stabilizer_span(generators)))
        if key in seen:
            continue
        seen.add(key)
        rate = exact_logical_error_rate(generators, p)
        results.append({'generators': generators, 'error_rate': rate,
                        'circuit': circuit, 'trial': trial})
    results.sort(key=lambda r: r['error_rate'])
    return results


def main():
    baseline = exact_logical_error_rate(['ZZI', 'IZZ'])
    print(f'Repetition-code baseline: {baseline:.4%} (expected 0.7250%)')
    assert abs(baseline - TARGET) < 1e-10
    results = search_codes()
    print(f'Unique codes evaluated: {len(results)}')
    for result in results[:5]:
        print(f"{result['generators']} -> {result['error_rate']:.4%}")
    with open(ROOT / 'pair4' / 'results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['rank', 'generators', 'logical_error_rate', 'circuit', 'trial'])
        for i, r in enumerate(results, 1):
            writer.writerow([i, ' '.join(r['generators']), r['error_rate'],
                             repr(r['circuit']), r['trial']])


if __name__ == '__main__':
    main()
