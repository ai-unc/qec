import numpy as np
from pauli_to_bin import pauli_to_bin

# takes two np arrays of length 2n
# returns 0 if A, B commute, returns 1 if they anticommute
def commutation_check(A, B):
    n = int(len(A) / 2)
    sum = 0
    for i in range(n):
        sum ^= A[i] * B[i + n] ^ B[i] * A[i + n]
    return sum
