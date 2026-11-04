"""Explicit adapters for original collaborators and Pair 4's local backends."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Original modules use bare sibling imports. Local modules have distinct names.
for folder in ('pair1', 'pair2', 'pair3'):
    path = str(ROOT / folder)
    if path not in sys.path:
        sys.path.append(path)

from pair1.pauli_to_bin import pauli_to_bin
from pair2.pauli_gates_stim import apply_circuit
from pair3.noise import error_probability
from pair3.syndrome import calculate_syndrome
from .local_validation import validate_code


def normalized(generators):
    result = []
    for g in generators:
        if not isinstance(g, str):
            raise ValueError('Generators must be Pauli strings')
        body = g[1:] if g.startswith(('+', '-')) else g
        if not body or any(c not in 'IXYZ' for c in body):
            raise ValueError('Generators must contain only I, X, Y, Z and an optional sign')
        result.append((g[0] if g.startswith(('+', '-')) else '+') + body)
    if not result or len({len(g) for g in result}) != 1:
        raise ValueError('Generators must be nonempty and have equal lengths')
    return result


def vectors(generators):
    return [pauli_to_bin(g[1:]) for g in normalized(generators)]


def get_validator(source):
    if source == 'local':
        return validate_code
    if source == 'original':
        return importlib.import_module('pair1.validate_code').validate_code
    raise ValueError('validator_source must be local or original')


def get_evaluator(source):
    if source == 'local':
        from .evaluation import evaluate_code
        return evaluate_code
    if source != 'original':
        raise ValueError('evaluator_source must be local or original')
    try:
        module = importlib.import_module('pair3.evaluate')
    except ModuleNotFoundError as exc:
        if exc.name != 'pair3.evaluate':
            raise
        raise ValueError('Original evaluator unavailable: Pair 3 must provide pair3/evaluate.py with evaluate_code(generators, noise_model, p)') from exc
    if not callable(getattr(module, 'evaluate_code', None)):
        raise ValueError('Pair 3 does not provide callable evaluate_code')
    return module.evaluate_code
