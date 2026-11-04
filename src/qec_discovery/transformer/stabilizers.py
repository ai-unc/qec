from __future__ import annotations

from qec_discovery.clifford.pauli_gates import apply_circuit

from qec_discovery.search.models import CircuitCandidate


def _to_circuit(candidate: CircuitCandidate) -> list[tuple]:
    return [
        ("CNOT", gate.qubits[0], gate.qubits[1])
        for gate in candidate.gates
        if gate.op == "CNOT"
    ]


def _stabilizer_x(n_qubits: int, stabilizer: int) -> str:
    letters = ["I"] * n_qubits
    letters[stabilizer] = "X"
    return "+" + "".join(letters)


def candidate_to_stabilizers(candidate: CircuitCandidate) -> list[str]:
    n = candidate.n_qubits
    k = candidate.k_logical
    circuit = _to_circuit(candidate)
    return [
        apply_circuit(_stabilizer_x(n, stabilizer), circuit).lstrip("+-")
        for stabilizer in range(n - k)
    ]
