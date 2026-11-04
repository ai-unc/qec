"""Exact logical recovery evaluator moved from Pair 4 search.py."""
import math
from itertools import product
from .backends import normalized, vectors, validate_code, pauli_to_bin, error_probability, calculate_syndrome

def stabilizer_span(generators):
    """Binary symplectic span, ignoring global phases for Pauli error equivalence."""
    vectors = [tuple(map(int, pauli_to_bin(g.lstrip('+-')))) for g in generators]
    zero = (0,) * len(vectors[0])
    span = {zero}
    for vector in vectors:
        span |= {tuple(x ^ y for x, y in zip(row, vector)) for row in tuple(span)}
    return span


def evaluate_code(generators, noise_model='bit_flip', p=0.05):
    """Maximum-likelihood syndrome decoder, optimized for logical success.

    Enumerates all physical errors and groups by syndrome and residual
    stabilizer coset. This avoids incorrect tie-breaking by individual errors.
    """
    generators = normalized(generators)
    if not validate_code(vectors(generators)):
        raise ValueError('Invalid stabilizer generators')
    if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('p must be finite and between 0 and 1')
    error_probability('I', noise_model, p)
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


