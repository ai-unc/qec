"""Same interface as pauli_gates.py, but Stim does the work instead of our lookup tables.

Stim is the standard stabilizer simulator (pip install stim). It is a cross-check only (D8);
the core code never needs it.
"""

import stim

from .pauli_gates import check_gate, parse_pauli

# Stim calls the controlled-NOT gate "CX".
STIM_NAMES = {"H": "H", "S": "S", "CNOT": "CX"}


def apply_gate(pauli, gate, q1, q2=None):
    """Return the Pauli string after one gate, e.g. apply_gate("+XI", "CNOT", 0, 1) -> "+XX".

    gate is "H", "S" or "CNOT". q1 is the qubit (or the CNOT control), q2 is the CNOT target.
    The input sign is optional; the output always has one. Validation is shared with
    pauli_gates.py, so both raise the same ValueError on bad input.
    """
    _, letters = parse_pauli(pauli)
    gate, qubits = check_gate(gate, len(letters), q1, q2)
    circuit = stim.Circuit(f"{STIM_NAMES[gate]} {' '.join(map(str, qubits))}")
    # Stim reads a bare string as "+" and wants "_" for the identity, so normalise first.
    text = pauli if pauli[0] in "+-" else "+" + pauli
    result = stim.PauliString(text.replace("I", "_")).after(circuit)
    return str(result).replace("_", "I")


def apply_circuit(pauli, circuit):
    """Apply a list of gates in order, e.g. [("H", 0), ("CNOT", 0, 1)]."""
    for gate in circuit:
        pauli = apply_gate(pauli, *gate)
    return pauli
