from itertools import product

def enumerate_pauli(n):
    paulis = ["I","X","Y","Z"]
    errors = []
    for p in product(paulis, repeat=n):
        errors.append("".join(p))
    return errors
print(enumerate_pauli(3))