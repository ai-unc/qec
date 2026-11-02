"""Adapters between Pair 2's internal format and the formats other pairs use.

In:  Pair 4 tuples, Pair 4 GateOp / CircuitCandidate records, and integer rows [id, a, b]
     are all turned into the internal form, a list of ("H", q), ("S", q), ("CNOT", c, t) (D4).
Out: unsigned strings for Pair 3 and [x | z] binary arrays for Pair 1.

Pair 4's records are read by attribute (duck typing), so this module never imports pair4 (D8).
"""

import operator

import numpy as np

from pair2.pauli_gates import GATE_ARITY, parse_pauli

# Integer rows are [gate id, a, b]. D5 (CNOT operand order) is still open with Arsh, so the
# order lives in this one constant: for CNOT, a is the control and b is the target.
ROW_GATE_IDS = {0: "H", 1: "S", 2: "CNOT"}
ROW_CNOT_ORDER = ("control", "target")


def _index(value, what):
    """Return value as a non-negative plain int (NumPy ints allowed, bool and floats not)."""
    if isinstance(value, bool):
        raise ValueError(f"{what} must be an integer, got bool")
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{what} must be an integer, got {value!r}") from None
    if value < 0:
        raise ValueError(f"{what} must be >= 0, got {value}")
    return value


def _gate_tuple(name, qubits):
    """Check a gate name and its operand count; return the internal tuple."""
    if name not in GATE_ARITY:
        raise ValueError(f"Unknown gate {name!r}; use 'H', 'S' or 'CNOT'")
    if len(qubits) != GATE_ARITY[name]:
        raise ValueError(f"{name} takes {GATE_ARITY[name]} qubit(s), got {len(qubits)}")
    qubits = [_index(q, "qubit") for q in qubits]
    if name == "CNOT" and qubits[0] == qubits[1]:
        raise ValueError("CNOT control and target must differ")
    return (name, *qubits)


def _one_gate(gate):
    if hasattr(gate, "op") and hasattr(gate, "qubits"):  # Pair 4 GateOp
        return _gate_tuple(gate.op, tuple(gate.qubits))
    if isinstance(gate, (str, bytes)) or not hasattr(gate, "__len__") or len(gate) == 0:
        raise ValueError(f"Cannot read gate {gate!r}")
    if isinstance(gate[0], str):  # ("H", 0) or ("CNOT", 0, 1)
        if len(gate) == 3 and gate[2] is None:  # tolerate ("H", 0, None)
            gate = gate[:2]
        return _gate_tuple(gate[0], tuple(gate[1:]))
    if len(gate) != 3:  # integer row [id, a, b]
        raise ValueError(f"Integer gate row must be [id, a, b], got {gate!r}")
    gate_id = _index(gate[0], "gate id")
    if gate_id not in ROW_GATE_IDS:
        raise ValueError(f"Unknown gate id {gate_id}; use 0 (H), 1 (S) or 2 (CNOT)")
    name, a, b = ROW_GATE_IDS[gate_id], gate[1], gate[2]
    if name != "CNOT":
        if b is None or _is_minus_one(b):
            return _gate_tuple(name, (a,))
        raise ValueError(f"{name} row must have b = -1, got {b!r}")
    control, target = (a, b) if ROW_CNOT_ORDER == ("control", "target") else (b, a)
    return _gate_tuple(name, (control, target))


def _is_minus_one(value):
    try:
        return not isinstance(value, bool) and operator.index(value) == -1
    except TypeError:
        return False


def to_internal(circuit):
    """Return the circuit as a list of ("H", q), ("S", q), ("CNOT", c, t) tuples.

    circuit may be a list of tuples, a list of Pair 4 GateOp records, a list of integer rows
    [id, a, b] (0=H, 1=S, 2=CNOT, b=-1 for one-qubit gates), or a CircuitCandidate.
    Raises ValueError on anything else, so a bad circuit can be scored 0.
    """
    if hasattr(circuit, "gates"):  # CircuitCandidate
        circuit = circuit.gates
    try:
        gates = list(circuit)
    except TypeError:
        raise ValueError(f"Circuit must be a sequence of gates, got {circuit!r}") from None
    return [_one_gate(g) for g in gates]


def unpack(circuit, n=None, k=None):
    """Return (internal gate list, n, k) from a circuit plus optional register size.

    A CircuitCandidate carries its own n_qubits and k_logical; if n or k is also given it must
    agree. For a plain gate list n is required and k defaults to 0 (start from |00...0>).
    Raises ValueError if the sizes are missing, inconsistent, or a gate is out of range.
    """
    if hasattr(circuit, "n_qubits") and hasattr(circuit, "k_logical"):
        if n is not None and n != circuit.n_qubits:
            raise ValueError(f"n={n} disagrees with the circuit's n_qubits={circuit.n_qubits}")
        if k is not None and k != circuit.k_logical:
            raise ValueError(f"k={k} disagrees with the circuit's k_logical={circuit.k_logical}")
        n, k = circuit.n_qubits, circuit.k_logical
    elif n is None:
        raise ValueError("n (number of qubits) is required for a plain gate list")
    k = 0 if k is None else k
    n, k = _index(n, "n"), _index(k, "k")
    if n < 1:
        raise ValueError("n must be at least 1")
    if k > n:
        raise ValueError(f"k={k} cannot exceed n={n}")
    gates = to_internal(circuit)
    for gate in gates:
        if any(q >= n for q in gate[1:]):
            raise ValueError(f"Gate {gate} is outside a {n}-qubit register")
    return gates, n, k


def unsigned(strings):
    """Drop the leading sign: "+ZIZ" -> "ZIZ". Takes one string or a list; returns the same shape.

    Pair 3's calculate_syndrome compares raw characters, so a leading "+" would shift every
    letter by one place and give wrong syndromes.
    """
    if isinstance(strings, str):
        return "".join(parse_pauli(strings)[1])
    return ["".join(parse_pauli(s)[1]) for s in strings]


def to_binary(strings):
    """Return the [x | z] binary form Pair 1 uses (signs dropped, Y sets both bits).

    One string of n qubits gives a length-2n array; a list gives a (len, 2n) array. Matches
    pair1.pauli_to_bin: x bits are 1 for X or Y, z bits are 1 for Z or Y.
    """
    single = isinstance(strings, str)
    rows = []
    for s in [strings] if single else strings:
        letters = parse_pauli(s)[1]
        rows.append([c in "XY" for c in letters] + [c in "ZY" for c in letters])
    arr = np.array(rows, dtype=int)
    return arr[0] if single else arr
