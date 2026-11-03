"""Framework-independent, immutable records for Pair 4's search boundary."""

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping


def integer(name: str, value: int, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def freeze_json(value: Any) -> Any:
    """Copy JSON metadata, recursively freezing containers and rejecting NaN."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("metadata keys must be strings")
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    raise ValueError("metadata must contain finite JSON values")


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class GateOp:
    op: str
    qubits: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "qubits", tuple(self.qubits))
        if self.op not in ("H", "S", "CNOT"):
            raise ValueError("op must be H, S, or CNOT")
        if len(self.qubits) != (2 if self.op == "CNOT" else 1):
            raise ValueError(f"wrong arity for {self.op}")
        for qubit in self.qubits:
            integer("qubit", qubit)
        if self.op == "CNOT" and self.qubits[0] == self.qubits[1]:
            raise ValueError("CNOT control and target must differ")

    def to_dict(self) -> dict:
        return {"op": self.op, "qubits": list(self.qubits)}


@dataclass(frozen=True)
class CircuitCandidate:
    n_qubits: int
    k_logical: int
    gates: tuple[GateOp, ...] = ()

    def __post_init__(self) -> None:
        integer("n_qubits", self.n_qubits, 1)
        integer("k_logical", self.k_logical)
        if self.k_logical > self.n_qubits:
            raise ValueError("k_logical must be <= n_qubits")
        object.__setattr__(self, "gates", tuple(self.gates))
        for gate in self.gates:
            if not isinstance(gate, GateOp):
                raise ValueError("gates must contain GateOp records")
            if any(qubit >= self.n_qubits for qubit in gate.qubits):
                raise ValueError("gate operand is outside the register")

    @property
    def cnot_count(self) -> int:
        return sum(gate.op == "CNOT" for gate in self.gates)

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "n_qubits": self.n_qubits,
            "k_logical": self.k_logical,
            "gates": [gate.to_dict() for gate in self.gates],
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "CircuitCandidate":
        if data.get("schema_version") != 1:
            raise ValueError("unsupported candidate schema_version")
        return cls(data["n_qubits"], data["k_logical"], tuple(
            GateOp(gate["op"], tuple(gate["qubits"])) for gate in data["gates"]
        ))


@dataclass(frozen=True)
class SamplingConfig:
    n_qubits: int = 4
    k_logical: int = 2
    max_length: int | None = None

    def __post_init__(self) -> None:
        CircuitCandidate(self.n_qubits, self.k_logical)
        if self.max_length is None:
            object.__setattr__(self, "max_length", 8 * self.n_qubits)
        integer("max_length", self.max_length, 1)

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "k_logical": self.k_logical,
            "max_length": self.max_length,
            "length_distribution": "uniform_integer_0_to_max_inclusive",
            "gate_probabilities": {"H": 0.5, "S": 0.5} if self.n_qubits == 1
                else {"H": 1 / 3, "S": 1 / 3, "CNOT": 1 / 3},
            "connectivity": "all_to_all_directed",
            "input_convention": "ancillas_first_logical_inputs_last",
            "gate_order": "chronological",
        }


@dataclass(frozen=True)
class SearchConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    budget: int = 1000
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.sampling, SamplingConfig):
            raise ValueError("sampling must be a SamplingConfig")
        integer("budget", self.budget, 1)
        integer("seed", self.seed)

    def to_dict(self) -> dict:
        return {
            "sampling": self.sampling.to_dict(), "budget": self.budget,
            "seed": self.seed, "rng": "python.random.Random/MT19937",
            "stream_derivation": "sha256(pair4:v1:<seed>:<label>)",
            "stream_labels": ["initial", "proposals", "mutation", "acceptance"],
            "neutral_acceptance_probability": 0.5,
            "ranking": ["primary_objective", "fewer_CNOTs", "fewer_total_gates"],
            "cache_enabled": False,
        }


@dataclass(frozen=True)
class MutationResult:
    candidate: CircuitCandidate
    operation: str
    index: int
    old_gate: GateOp | None = None
    new_gate: GateOp | None = None

    def to_dict(self) -> dict:
        return {
            "operation": self.operation, "index": self.index,
            "old_gate": self.old_gate.to_dict() if self.old_gate else None,
            "new_gate": self.new_gate.to_dict() if self.new_gate else None,
        }
