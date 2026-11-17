# Reuses existing qec_discovery.search.baseline helpers and does not reimplement validation or logical-error-rate logic

from __future__ import annotations

from dataclasses import dataclass

from qec_discovery.search.baseline import exact_logical_error_rate, valid_code


@dataclass(frozen=True)
class EvaluationResult:
    valid: bool
    logical_error_rate: float | None
    n: int
    k: int


def evaluate_candidate_generators(generators: list[str], *, n: int = 3, k: int = 1, p: float = 0.05, noise_model: str = "bit_flip"):
    if not valid_code(generators, n=n, k=k):
        return EvaluationResult(
            valid=False,
            logical_error_rate=None,
            n=n,
            k=k,
        )
    rate = exact_logical_error_rate(generators, p=p, noise_model=noise_model)
    return EvaluationResult(
        valid=True,
        logical_error_rate=float(rate),
        n=n,
        k=k,
    )
