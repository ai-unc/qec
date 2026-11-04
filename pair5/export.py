from __future__ import annotations

from pair4.evolution_prototype import CircuitCandidate


def candidate_to_pair2_circuit(candidate: CircuitCandidate) -> list[tuple]:
    # Formats output: (CNOT, c, t) / (H, q) / (S, q)
    circuit = []
    for gate in candidate.gates:
        if gate.op == "CNOT":
            circuit.append(("CNOT", gate.qubits[0], gate.qubits[1]))
        else:
            circuit.append((gate.op, gate.qubits[0]))
    return circuit
