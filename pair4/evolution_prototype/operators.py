"""Seeded random circuits and bounded single-edit mutation. No quantum math."""

import hashlib
from random import Random

from .models import CircuitCandidate, GateOp, MutationResult, SamplingConfig, integer


def random_stream(seed: int, label: str) -> Random:
    integer("seed", seed)
    digest = hashlib.sha256(f"pair4:v1:{seed}:{label}".encode("utf-8")).digest()
    return Random(int.from_bytes(digest, "big"))


def sample_random_gate(n_qubits: int, rng: Random) -> GateOp:
    integer("n_qubits", n_qubits, 1)
    op = rng.choice(("H", "S") if n_qubits == 1 else ("H", "S", "CNOT"))
    control = rng.randrange(n_qubits)
    if op != "CNOT":
        return GateOp(op, (control,))
    target = rng.randrange(n_qubits - 1)
    if target >= control:
        target += 1
    return GateOp(op, (control, target))


def sample_random_circuit(config: SamplingConfig, rng: Random) -> CircuitCandidate:
    length = rng.randrange(config.max_length + 1)
    return CircuitCandidate(config.n_qubits, config.k_logical, tuple(
        sample_random_gate(config.n_qubits, rng) for _ in range(length)
    ))


def _replacement(n_qubits: int, original: GateOp, rng: Random) -> GateOp:
    # Removing the original from this finite weighted set implements the base
    # hierarchical gate distribution conditioned on drawing a different gate.
    # Relative weights: H/S instruction n-1, CNOT instruction 1 (n >= 2).
    alternatives = []
    weights = []
    for op in ("H", "S"):
        for qubit in range(n_qubits):
            gate = GateOp(op, (qubit,))
            if gate != original:
                alternatives.append(gate)
                weights.append(max(1, n_qubits - 1))
    for control in range(n_qubits):
        for target in range(n_qubits):
            if control != target:
                gate = GateOp("CNOT", (control, target))
                if gate != original:
                    alternatives.append(gate)
                    weights.append(1)
    return rng.choices(alternatives, weights=weights, k=1)[0]


def mutate_circuit(candidate: CircuitCandidate, max_length: int,
                   rng: Random) -> MutationResult:
    integer("max_length", max_length, 1)
    length = len(candidate.gates)
    if length > max_length:
        raise ValueError("parent exceeds max_length")
    operations = []
    if length:
        operations.extend(("replace", "delete"))
    if length < max_length:
        operations.append("insert")
    operation = rng.choice(operations)
    gates = list(candidate.gates)
    old_gate = new_gate = None
    if operation == "insert":
        index = rng.randrange(length + 1)
        new_gate = sample_random_gate(candidate.n_qubits, rng)
        gates.insert(index, new_gate)
    else:
        index = rng.randrange(length)
        old_gate = gates[index]
        if operation == "delete":
            del gates[index]
        else:
            new_gate = _replacement(candidate.n_qubits, old_gate, rng)
            gates[index] = new_gate
    child = CircuitCandidate(candidate.n_qubits, candidate.k_logical, tuple(gates))
    return MutationResult(child, operation, index, old_gate, new_gate)
