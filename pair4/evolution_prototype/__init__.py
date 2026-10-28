"""Pair 4: reproducible search over semantic Clifford gate sequences."""

from .evaluation import DemoEvaluator, EvaluationResult, Evaluator, ObjectiveSpec
from .models import CircuitCandidate, GateOp, MutationResult, SamplingConfig, SearchConfig
from .operators import mutate_circuit, random_stream, sample_random_circuit, sample_random_gate
from .search import SearchResult, one_plus_one_search, random_search

__all__ = [
    "CircuitCandidate", "DemoEvaluator", "EvaluationResult", "Evaluator", "GateOp",
    "MutationResult", "ObjectiveSpec", "SamplingConfig", "SearchConfig", "SearchResult",
    "mutate_circuit", "one_plus_one_search", "random_search", "random_stream",
    "sample_random_circuit", "sample_random_gate",
]
