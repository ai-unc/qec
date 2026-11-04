"""Random sequences of Clifford gates (H, S, CNOT).

A circuit is a list of gates like [("H", 0), ("CNOT", 0, 1)]. Qubits are numbered from 0 and
CNOT is (control, target), the same format as pauli_gates.py.
"""

import random

GATES = ("H", "S", "CNOT")


def validate_register(n_qubits):
    if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits < 1:
        raise ValueError('n_qubits must be a positive integer')


def random_gate(n_qubits=7, rng=None):
    """Return one random gate, e.g. ("H", 3) or ("CNOT", 2, 5).

    Each of H, S and CNOT is equally likely. A CNOT gets two different qubits, because a
    control and a target on the same qubit is not a valid gate.
    """
    rng = rng or random.Random()
    validate_register(n_qubits)
    gate = rng.choice(GATES if n_qubits > 1 else ('H', 'S'))
    if gate == "CNOT":
        control, target = rng.sample(range(n_qubits), 2)
        return (gate, control, target)
    return (gate, rng.randrange(n_qubits))


def random_circuit(length, n_qubits=7, seed=None, rng=None):
    """Return a list of `length` random gates. The same seed gives the same circuit."""
    validate_register(n_qubits)
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError('Invalid circuit dimensions')
    rng = rng or random.Random(seed)
    return [random_gate(n_qubits, rng) for _ in range(length)]
