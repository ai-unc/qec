"""Bridge from candidate stabilizer generators to discovery-pipeline evaluation.

Canonical generator representation is unsigned Pauli strings, e.g.
``["ZZI", "IZZ"]`` for the three-qubit bit-flip repetition code. Optional
``+``/``-`` prefixes are accepted because ``baseline.valid_code`` and
``baseline.exact_logical_error_rate`` strip them internally.

This module reuses existing ``qec_discovery.search.baseline`` helpers and does
not reimplement validation or logical-error-rate logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from qec_discovery.search.baseline import exact_logical_error_rate, valid_code


@dataclass(frozen=True)
class EvaluationResult:
    """Outcome of validating and (when valid) scoring a stabilizer code."""

    valid: bool
    logical_error_rate: float | None
    n: int
    k: int


def evaluate_candidate_generators(
    generators: list[str],
    *,
    n: int = 3,
    k: int = 1,
    p: float = 0.05,
    noise_model: str = "bit_flip",
) -> EvaluationResult:
    """Validate unsigned Pauli generators and, if valid, compute exact LER.

    Parameters
    ----------
    generators:
        Candidate stabilizer generators as Pauli strings of length ``n``
        (unsigned preferred; signed prefixes allowed).
    n:
        Number of physical qubits.
    k:
        Number of logical qubits (expects ``len(generators) == n - k``).
    p:
        Physical error probability passed to the exact LER enumerator.
    noise_model:
        Noise channel name understood by ``qec_discovery.noise.models``.

    Returns
    -------
    EvaluationResult
        ``valid`` is whether the generators form a commuting, independent
        stabilizer code for ``[[n, k]]``. When invalid, ``logical_error_rate``
        is ``None`` and LER is not computed. When valid, ``logical_error_rate``
        is the exact logical error rate under ``noise_model`` at rate ``p``.
    """
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
