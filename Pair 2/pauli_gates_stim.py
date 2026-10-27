"""Same interface as pauli_gates.py, but Stim does the work instead of our lookup tables.

Stim is the standard stabilizer simulator (pip install stim).
"""

import stim

# Stim calls the controlled-NOT gate "CX".
STIM_NAMES = {"H": "H", "S": "S", "CNOT": "CX"}


def apply_gate(pauli, gate, q1, q2=None):
    """Return the Pauli string after one gate, e.g. apply_gate("+XI", "CNOT", 0, 1) -> "+XX".

    gate is "H", "S" or "CNOT". q1 is the qubit (or the CNOT control), q2 is the CNOT target.
    The input sign is optional; the output always has one.
    """
    if gate not in STIM_NAMES:
        raise ValueError(f"Unknown gate {gate!r}; use 'H', 'S' or 'CNOT'")
    qubits = [q1, q2] if gate == "CNOT" else [q1]  # H and S ignore q2, like pauli_gates.py
    circuit = stim.Circuit(f"{STIM_NAMES[gate]} {' '.join(map(str, qubits))}")
    result = stim.PauliString(pauli).after(circuit)
    return str(result).replace("_", "I")  # Stim writes the identity as "_"


def apply_circuit(pauli, circuit):
    """Apply a list of gates in order, e.g. [("H", 0), ("CNOT", 0, 1)]."""
    for gate in circuit:
        pauli = apply_gate(pauli, *gate)
    return pauli
