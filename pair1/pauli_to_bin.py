import numpy as np

# takes string consisting of X, Y, Z, I of length n
# returns np array of length 2n (first n represent x portion, last n represent z portion)
def pauli_to_bin(pauli_string):
    bin_x = [ 1 if c == 'X' or c == 'Y' else 0 for c in pauli_string ]
    bin_z = [ 1 if c == 'Z' or c == 'Y' else 0 for c in pauli_string ]
    bin = bin_x + bin_z

    bin_np = np.array(bin)
    return bin_np
