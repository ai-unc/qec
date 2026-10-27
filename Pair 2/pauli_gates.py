"""Apply Clifford gates (H, S, CNOT) to Pauli strings.

A gate U changes a Pauli P into U P U†. Pauli strings look like "+XIZ": a sign, then one
letter per qubit, with qubit 0 on the left.
"""

# What each gate does to each Pauli: letter -> (sign, new letter).
H = {"I": (+1, "I"), "X": (+1, "Z"), "Y": (-1, "Y"), "Z": (+1, "X")}
S = {"I": (+1, "I"), "X": (+1, "Y"), "Y": (-1, "X"), "Z": (+1, "Z")}

# CNOT acts on a (control, target) pair of letters.
CNOT = {
    "II": (+1, "II"), "IX": (+1, "IX"), "IY": (+1, "ZY"), "IZ": (+1, "ZZ"),
    "XI": (+1, "XX"), "XX": (+1, "XI"), "XY": (+1, "YZ"), "XZ": (-1, "YY"),
    "YI": (+1, "YX"), "YX": (+1, "YI"), "YY": (-1, "XZ"), "YZ": (+1, "XY"),
    "ZI": (+1, "ZI"), "ZX": (+1, "ZX"), "ZY": (+1, "IY"), "ZZ": (+1, "IZ"),
}


def apply_gate(pauli, gate, q1, q2=None):
    """Return the Pauli string after one gate, e.g. apply_gate("+XI", "CNOT", 0, 1) -> "+XX".

    gate is "H", "S" or "CNOT". q1 is the qubit (or the CNOT control), q2 is the CNOT target.
    The input sign is optional; the output always has one.
    """
    sign = -1 if pauli[0] == "-" else +1
    letters = list(pauli.lstrip("+-"))

    if gate in ("H", "S"):
        table = H if gate == "H" else S
        flip, letters[q1] = table[letters[q1]]
    elif gate == "CNOT":
        if q1 == q2:
            raise ValueError("CNOT needs two different qubits")
        flip, (letters[q1], letters[q2]) = CNOT[letters[q1] + letters[q2]]
    else:
        raise ValueError(f"Unknown gate {gate!r}; use 'H', 'S' or 'CNOT'")

    sign *= flip
    return ("+" if sign == 1 else "-") + "".join(letters)


def apply_circuit(pauli, circuit):
    """Apply a list of gates in order, e.g. [("H", 0), ("CNOT", 0, 1)]."""
    for gate in circuit:
        pauli = apply_gate(pauli, *gate)
    return pauli
