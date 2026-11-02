"""Small random edits to a gate sequence: replace, insert or remove one gate.

A search can improve a circuit by making a small change and keeping it if the result is better.
Each function returns a new list and leaves the circuit you passed in unchanged.
"""

import random

from random_circuits import random_gate


def replace_gate(circuit, n_qubits=7, rng=None):
    """Return a copy with one randomly chosen gate swapped for a new random gate."""
    rng = rng or random.Random()
    new = list(circuit)
    new[rng.randrange(len(new))] = random_gate(n_qubits, rng)
    return new


def insert_gate(circuit, n_qubits=7, rng=None):
    """Return a copy with one new random gate added at a random position (including the ends)."""
    rng = rng or random.Random()
    new = list(circuit)
    new.insert(rng.randrange(len(new) + 1), random_gate(n_qubits, rng))
    return new


def remove_gate(circuit, rng=None):
    """Return a copy with one randomly chosen gate deleted."""
    rng = rng or random.Random()
    new = list(circuit)
    del new[rng.randrange(len(new))]
    return new


def mutate(circuit, n_qubits=7, rng=None):
    """Apply one randomly chosen mutation. An empty circuit can only get an insert."""
    rng = rng or random.Random()
    if not circuit:
        return insert_gate(circuit, n_qubits, rng)
    kind = rng.choice(("replace", "insert", "remove"))
    if kind == "replace":
        return replace_gate(circuit, n_qubits, rng)
    if kind == "insert":
        return insert_gate(circuit, n_qubits, rng)
    return remove_gate(circuit, rng)
