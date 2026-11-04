"""Apply Clifford gates (H, S, CNOT) to Pauli strings.

A gate U changes a Pauli P into U P U†. Pauli strings look like "+XIZ": a sign, then one
letter per qubit, with qubit 0 on the left. Bad input raises ValueError (D12) so a training
loop can score a broken circuit 0 instead of crashing.
"""

import operator

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

GATE_ARITY = {"H": 1, "S": 1, "CNOT": 2}


def parse_pauli(pauli):
    """Split a Pauli string like "-XIZ" into (sign, list of letters).

    The sign is optional (no sign means +) and must be a single "+" or "-". Every other
    character must be one of I, X, Y, Z. Raises ValueError otherwise.
    """
    if not isinstance(pauli, str):
        raise ValueError(f"Pauli must be a string, got {type(pauli).__name__}")
    sign = -1 if pauli[:1] == "-" else +1
    body = pauli[1:] if pauli[:1] in ("+", "-") else pauli
    if not body:
        raise ValueError(f"Pauli {pauli!r} has no qubits")
    bad = sorted(set(body) - set("IXYZ"))
    if bad:
        raise ValueError(f"Pauli {pauli!r} has invalid characters {bad}; use I, X, Y, Z "
                         "with at most one leading '+' or '-'")
    return sign, list(body)


def check_qubit(q, n, name="qubit"):
    """Return q as a plain int if it is an integer in range(n), else raise ValueError.

    Accepts NumPy or torch integers (anything with __index__) but not bool, and rejects
    negatives so they cannot silently wrap around to the last qubit.
    """
    if isinstance(q, bool):
        raise ValueError(f"{name} must be an integer, got bool")
    try:
        q = operator.index(q)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {q!r}") from None
    if not 0 <= q < n:
        raise ValueError(f"{name} {q} is out of range for {n} qubits")
    return q


def check_gate(gate, n, q1, q2=None):
    """Validate one gate on an n-qubit register and return (gate, [qubits as ints]).

    Checks the gate name, its arity (H and S take one qubit, CNOT takes two), the qubit
    range, and that CNOT's control and target differ. Raises ValueError otherwise.
    """
    if gate not in GATE_ARITY:
        raise ValueError(f"Unknown gate {gate!r}; use 'H', 'S' or 'CNOT'")
    if GATE_ARITY[gate] == 1:
        if q2 is not None:
            raise ValueError(f"{gate} takes one qubit, but a second qubit {q2!r} was given")
        return gate, [check_qubit(q1, n, "qubit")]
    if q2 is None:
        raise ValueError("CNOT needs a control and a target qubit")
    control = check_qubit(q1, n, "control")
    target = check_qubit(q2, n, "target")
    if control == target:
        raise ValueError("CNOT needs two different qubits")
    return gate, [control, target]


def apply_gate(pauli, gate, q1, q2=None):
    """Return the Pauli string after one gate, e.g. apply_gate("+XI", "CNOT", 0, 1) -> "+XX".

    gate is "H", "S" or "CNOT". q1 is the qubit (or the CNOT control), q2 is the CNOT target.
    The input sign is optional; the output always has one. Raises ValueError on bad input.
    """
    sign, letters = parse_pauli(pauli)
    gate, qubits = check_gate(gate, len(letters), q1, q2)

    if gate == "CNOT":
        c, t = qubits
        flip, (letters[c], letters[t]) = CNOT[letters[c] + letters[t]]
    else:
        flip, letters[qubits[0]] = (H if gate == "H" else S)[letters[qubits[0]]]

    sign *= flip
    return ("+" if sign == 1 else "-") + "".join(letters)


def apply_circuit(pauli, circuit):
    """Apply a list of gates in order, e.g. [("H", 0), ("CNOT", 0, 1)]."""
    for gate in circuit:
        if not isinstance(gate, (tuple, list)) or not 2 <= len(gate) <= 3:
            raise ValueError(f"Gate {gate!r} must look like ('H', 0) or ('CNOT', 0, 1)")
        pauli = apply_gate(pauli, *gate)
    return pauli
