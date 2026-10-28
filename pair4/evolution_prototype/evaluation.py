"""The evaluator interface; physics remains with the other project pairs."""

from dataclasses import dataclass, field
from copy import deepcopy
import importlib
import math
from typing import Mapping, Protocol

from .models import CircuitCandidate, GateOp, freeze_json, thaw_json


@dataclass(frozen=True)
class ObjectiveSpec:
    metric_name: str
    direction: str
    version: str
    parameters: Mapping = field(default_factory=dict)
    exact: bool = True
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, str) or not self.metric_name:
            raise ValueError("metric_name must be nonempty")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("version must be nonempty")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("direction must be minimize or maximize")
        if type(self.exact) is not bool or type(self.is_synthetic) is not bool:
            raise ValueError("exact and is_synthetic must be booleans")
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a JSON mapping")
        object.__setattr__(self, "parameters", freeze_json(self.parameters))

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name, "direction": self.direction,
            "version": self.version, "parameters": thaw_json(self.parameters),
            "exact": self.exact, "is_synthetic": self.is_synthetic,
        }


@dataclass(frozen=True)
class EvaluationResult:
    value: float
    diagnostics: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError("evaluation value must be a finite number")
        object.__setattr__(self, "value", float(self.value))
        if not isinstance(self.diagnostics, Mapping):
            raise ValueError("diagnostics must be a JSON mapping")
        object.__setattr__(self, "diagnostics", freeze_json(self.diagnostics))

    def to_dict(self) -> dict:
        return {"value": self.value, "diagnostics": thaw_json(self.diagnostics)}


class Evaluator(Protocol):
    @property
    def objective(self) -> ObjectiveSpec: ...

    def evaluate(self, candidate: CircuitCandidate) -> EvaluationResult: ...


class DemoEvaluator:
    """SYNTHETIC: negative gate edit distance, never a QEC metric."""

    def __init__(self, n_qubits: int = 4, k_logical: int = 2):
        self.n_qubits = n_qubits
        self.k_logical = k_logical
        if n_qubits == 1:
            self.target = tuple(GateOp(op, (0,)) for op in ("H", "S", "H", "S", "S", "H"))
        else:
            self.target = (
                GateOp("H", (0,)), GateOp("CNOT", (0, 1)), GateOp("S", (1,)),
                GateOp("H", (1,)), GateOp("CNOT", (1, 0)), GateOp("S", (0,)),
            )
        target = CircuitCandidate(n_qubits, k_logical, self.target)
        self.objective = ObjectiveSpec(
            "demo.negative_gate_edit_distance", "maximize", "1",
            {"target": target.to_dict(), "warning": "SYNTHETIC; no quantum-performance meaning"},
            is_synthetic=True,
        )

    def evaluate(self, candidate: CircuitCandidate) -> EvaluationResult:
        if (candidate.n_qubits, candidate.k_logical) != (self.n_qubits, self.k_logical):
            raise ValueError("candidate register differs from the demo target")
        previous = list(range(len(self.target) + 1))
        for i, gate in enumerate(candidate.gates, 1):
            current = [i]
            for j, target_gate in enumerate(self.target, 1):
                current.append(min(current[-1] + 1, previous[j] + 1,
                                   previous[j - 1] + (gate != target_gate)))
            previous = current
        return EvaluationResult(-previous[-1], {"gate_edit_distance": previous[-1]})


def load_evaluator(entrypoint: str, config: Mapping) -> Evaluator:
    """Load an explicitly requested factory(config); no automatic fallback."""
    if entrypoint == "demo":
        return DemoEvaluator(config["n_qubits"], config["k_logical"])
    module_name, separator, name = entrypoint.partition(":")
    if not separator or not module_name or not name:
        raise ValueError("evaluator must be demo or module:factory")
    factory = getattr(importlib.import_module(module_name), name)
    return factory(deepcopy(dict(config)))
