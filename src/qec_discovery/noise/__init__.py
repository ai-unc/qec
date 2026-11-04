"""Noise models and Pauli enumeration.

Syndrome calculation and decoding live in ``syndrome`` and ``decoder``. Importing
those loads the sampler, which needs JAX.
"""

from .enumerate_pauli import enumerate_pauli_errors
from .models import error_probability, model

__all__ = ["enumerate_pauli_errors", "error_probability", "model"]
