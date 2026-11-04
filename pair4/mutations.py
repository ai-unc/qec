"""Non-destructive replacement, insertion and removal of Clifford instructions."""
import random
if __package__:
    from .random_circuits import random_gate, validate_register
else:
    from random_circuits import random_gate, validate_register


def replace_gate(circuit, n_qubits=7, rng=None):
    rng = rng or random.Random()
    if not circuit:
        raise ValueError('Cannot replace a gate in an empty circuit')
    validate_register(n_qubits)
    new = list(circuit)
    index = rng.randrange(len(new))
    choices = [(op, q) for op in ('H', 'S') for q in range(n_qubits)]
    choices += [('CNOT', a, b) for a in range(n_qubits) for b in range(n_qubits) if a != b]
    new[index] = rng.choice([g for g in choices if g != new[index]])
    return new


def insert_gate(circuit, n_qubits=7, rng=None):
    rng = rng or random.Random()
    new = list(circuit)
    new.insert(rng.randrange(len(new) + 1), random_gate(n_qubits, rng))
    return new


def remove_gate(circuit, rng=None):
    rng = rng or random.Random()
    if not circuit:
        raise ValueError('Cannot remove a gate from an empty circuit')
    new = list(circuit)
    del new[rng.randrange(len(new))]
    return new


def mutate(circuit, n_qubits=7, rng=None, max_depth=None):
    rng = rng or random.Random()
    if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1 or len(circuit) > max_depth):
        raise ValueError('Invalid max_depth')
    if not circuit:
        return insert_gate(circuit, n_qubits, rng)
    operations = ['replace', 'remove']
    if max_depth is None or len(circuit) < max_depth:
        operations.append('insert')
    operation = rng.choice(operations)
    if operation == 'replace':
        return replace_gate(circuit, n_qubits, rng)
    if operation == 'remove':
        return remove_gate(circuit, rng)
    return insert_gate(circuit, n_qubits, rng)
