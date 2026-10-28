import numpy as np

PAULI = np.array(['I', 'X', 'Y', 'Z'])

def enumerate_pauli_errors(n):
    total = 4**n
    num_arr = np.arange(total)

    digits = np.zeros((total, n), dtype=int)

    for qubit in range(n-1, -1, -1):
        digits[:, qubit] = num_arr % 4
        num_arr = num_arr // 4

    pauli_letters = PAULI[digits]

    errors = []
    for row in pauli_letters:
        errors.append(''.join(row))

    return np.array(errors)
