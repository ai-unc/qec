"""Independent random search and a one-parent, one-offspring search."""

from dataclasses import dataclass
import platform
from time import perf_counter

from .evaluation import EvaluationResult, Evaluator, ObjectiveSpec
from .models import CircuitCandidate, MutationResult, SearchConfig
from .operators import mutate_circuit, random_stream, sample_random_circuit


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: CircuitCandidate
    evaluation: EvaluationResult
    evaluation_index: int

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "evaluation_index": self.evaluation_index,
        }


def ranking(item: ScoredCandidate, objective: ObjectiveSpec) -> tuple:
    """Smaller tuples always rank better; no rounding or weighted penalties."""
    value = item.evaluation.value
    primary = value if objective.direction == "minimize" else -value
    return primary, item.candidate.cnot_count, len(item.candidate.gates)


@dataclass(frozen=True)
class SearchEvent:
    index: int
    candidate: CircuitCandidate
    parent_index: int | None
    mutation: MutationResult | None
    evaluation: EvaluationResult | None
    accepted: bool
    best_so_far: float | None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "schema_version": 1, "evaluation_index": self.index,
            "candidate": self.candidate.to_dict(), "parent_index": self.parent_index,
            "mutation": self.mutation.to_dict() if self.mutation else None,
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
            "accepted": self.accepted, "best_so_far": self.best_so_far,
            "status": "failed" if self.error else "ok", "error": self.error,
        }


@dataclass(frozen=True)
class SearchResult:
    method: str
    config: SearchConfig
    objective: ObjectiveSpec | None
    status: str
    best: ScoredCandidate | None
    archive: tuple[ScoredCandidate, ...]
    events: tuple[SearchEvent, ...]
    error: str | None
    elapsed_seconds: float

    @property
    def evaluation_count(self) -> int:
        return len(self.events)

    def manifest(self) -> dict:
        return {
            "schema_version": 1, "method": self.method,
            "config": self.config.to_dict(),
            "objective": self.objective.to_dict() if self.objective else None,
            "environment": {"python": platform.python_version(), "pair4_schema": 1},
        }

    def summary(self) -> dict:
        return {
            "schema_version": 1, "method": self.method, "status": self.status,
            "evaluation_count": self.evaluation_count, "error": self.error,
            "objective": self.objective.to_dict() if self.objective else None,
            "best": self.best.to_dict() if self.best else None,
            "archive": [item.to_dict() for item in self.archive],
            "timing": {"elapsed_seconds": self.elapsed_seconds},
        }


def _check_candidate(candidate: CircuitCandidate, config: SearchConfig) -> None:
    if not isinstance(candidate, CircuitCandidate):
        raise ValueError("initial_candidate must be a CircuitCandidate")
    sampling = config.sampling
    if (candidate.n_qubits, candidate.k_logical) != (sampling.n_qubits, sampling.k_logical):
        raise ValueError("initial candidate dimensions differ from search configuration")
    if len(candidate.gates) > sampling.max_length:
        raise ValueError("initial candidate exceeds max_length")


def _run(method: str, config: SearchConfig, evaluator: Evaluator,
         initial_candidate: CircuitCandidate | None) -> SearchResult:
    started = perf_counter()
    events = []
    archive = []
    best = incumbent = None
    objective = None
    error = None
    try:
        objective = evaluator.objective
        if not isinstance(objective, ObjectiveSpec):
            objective = None
            raise ValueError("evaluator.objective must be an ObjectiveSpec")
        if not objective.exact:
            raise ValueError("v1 requires an exact deterministic evaluator")
        if not objective.is_synthetic and not 1 <= config.sampling.k_logical < config.sampling.n_qubits:
            raise ValueError("scientific searches require 1 <= k_logical < n_qubits")
        snapshot = objective.to_dict()
        initial = initial_candidate if initial_candidate is not None else sample_random_circuit(
            config.sampling, random_stream(config.seed, "initial"))
        _check_candidate(initial, config)
        proposal_rng = random_stream(config.seed, "proposals")
        mutation_rng = random_stream(config.seed, "mutation")
        acceptance_rng = random_stream(config.seed, "acceptance")
        for index in range(1, config.budget + 1):
            mutation = None
            parent_index = None
            if index == 1:
                candidate = initial
            elif method == "random":
                candidate = sample_random_circuit(config.sampling, proposal_rng)
            else:
                parent_index = incumbent.evaluation_index
                mutation = mutate_circuit(incumbent.candidate, config.sampling.max_length, mutation_rng)
                candidate = mutation.candidate
            try:
                evaluation = evaluator.evaluate(candidate)
                if not isinstance(evaluation, EvaluationResult):
                    raise ValueError("evaluate() must return an EvaluationResult")
                current_objective = evaluator.objective
                if not isinstance(current_objective, ObjectiveSpec) or current_objective.to_dict() != snapshot:
                    raise ValueError("evaluator objective metadata changed during the run")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                events.append(SearchEvent(index, candidate, parent_index, mutation, None,
                                          False, best.evaluation.value if best else None, error))
                break
            item = ScoredCandidate(candidate, evaluation, index)
            key = ranking(item, objective)
            accepted = incumbent is None or key < ranking(incumbent, objective)
            if (method == "one_plus_one" and incumbent is not None
                    and key == ranking(incumbent, objective)):
                accepted = acceptance_rng.random() < 0.5
            if accepted:
                incumbent = item
            if best is None or key < ranking(best, objective):
                best = item
            # Exact genotype deduplication is only for the archive, never a cache.
            previous = {entry.candidate: entry for entry in archive}
            if candidate not in previous or key < ranking(previous[candidate], objective):
                previous[candidate] = item
            archive = sorted(previous.values(), key=lambda entry: (
                ranking(entry, objective), entry.evaluation_index))[:10]
            events.append(SearchEvent(index, candidate, parent_index, mutation, evaluation,
                                      accepted, best.evaluation.value))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return SearchResult(method, config, objective, "failed" if error else "completed",
                        best, tuple(archive), tuple(events), error, perf_counter() - started)


def random_search(config: SearchConfig, evaluator: Evaluator, *,
                  initial_candidate: CircuitCandidate | None = None) -> SearchResult:
    return _run("random", config, evaluator, initial_candidate)


def one_plus_one_search(config: SearchConfig, evaluator: Evaluator, *,
                       initial_candidate: CircuitCandidate | None = None) -> SearchResult:
    return _run("one_plus_one", config, evaluator, initial_candidate)
