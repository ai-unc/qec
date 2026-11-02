"""From a circuit of Clifford gates to a stabilizer code.

Layout (D9): of n qubits, the first n-k are ancillas that start in |0>, so they start with the
stabilizers +Z_0 ... +Z_(n-k-1). The last k qubits hold the logical data. Pushing each starting
operator P through the circuit as P -> U P U† (D3) gives the code's stabilizer generators and its
logical operators. Strings are signed, qubit 0 on the left (D1, D2). Bad circuits raise
ValueError (D12) so callers can score them 0.
"""

from pair2.adapters import unpack
from pair2.pauli_gates import apply_gate


def _single(n, qubit, letter):
    """Return the n-qubit string with `letter` on `qubit` and I elsewhere, e.g. +IZI."""
    return "+" + "I" * qubit + letter + "I" * (n - qubit - 1)


def _push(start_operators, gates):
    """Send each starting operator through every gate in order."""
    out = []
    for pauli in start_operators:
        for gate in gates:
            pauli = apply_gate(pauli, *gate)
        out.append(pauli)
    return out


def simulate_circuit(circuit, n=None, k=0):
    """Return the n-k signed stabilizer generators of the code the circuit prepares.

    circuit is a list of gates (tuples, GateOp records or integer rows) or a CircuitCandidate,
    which carries its own n and k. Ancilla qubits 0..n-k-1 start as +Z_i, and each is conjugated
    through the circuit. With k=0 this is "start from |00...0> and read off the state's
    stabilizers". Example: H(0), CNOT(0,1) on 2 qubits gives ["+XX", "+ZZ"].
    """
    gates, n, k = unpack(circuit, n, None if hasattr(circuit, "k_logical") else k)
    return _push([_single(n, i, "Z") for i in range(n - k)], gates)


def logical_operators(circuit, n=None, k=0):
    """Return (logical_x, logical_z): lists of k signed strings, one pair per data qubit.

    Data qubit j sits at index n-k+j. X_bar_j and Z_bar_j are the bare X and Z on that qubit
    pushed through the circuit. They commute with every stabilizer, and X_bar_j anticommutes
    with Z_bar_j, which is what makes them logical operators and not just more stabilizers.
    """
    gates, n, k = unpack(circuit, n, None if hasattr(circuit, "k_logical") else k)
    data = range(n - k, n)
    return (_push([_single(n, q, "X") for q in data], gates),
            _push([_single(n, q, "Z") for q in data], gates))


def encode(circuit, n=None, k=0):
    """Return (stabilizers, logical_x, logical_z) in one call, from a single parse."""
    gates, n, k = unpack(circuit, n, None if hasattr(circuit, "k_logical") else k)
    data = range(n - k, n)
    return (_push([_single(n, i, "Z") for i in range(n - k)], gates),
            _push([_single(n, q, "X") for q in data], gates),
            _push([_single(n, q, "Z") for q in data], gates))
