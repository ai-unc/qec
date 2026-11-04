
from pair1.pauli_to_bin import pauli_to_bin
from pair1.group_member import group_member
from syndrome import calculate_syndrome
from noise import error_probability
from decoder import build_lookup_table, decode
from enumerate_pauli import enumerate_pauli_errors
generators = ["ZZI", "IZZ"]
generators_bin = [pauli_to_bin(g) for g in generators]

def multiply_pauli(a, b):

  dict = {
        ("I", "I"): "I",
        ("X", "X"): "I",
        ("Y", "Y"): "I",
        ("Z", "Z"): "I",
        ("I", "X"): "X",
        ("X", "I"): "X",
        ("I", "Y"): "Y",
        ("Y", "I"): "Y",
        ("I", "Z"): "Z",
        ("Z", "I"): "Z",
        ("X", "Y"): "Z",
        ("Y", "X"): "Z",
        ("X", "Z"): "Y",
        ("Z", "X"): "Y",
        ("Y", "Z"): "X",
        ("Z", "Y"): "X"
    }

    result = ""

    for i in range(len(a)):
        x = a[i]
        y = b[i]

        result += dict[(x, y)]

    return result


def evaluate_code(stabilizers, noise_model, p):
    n = len(stabilizers[0])
    generators_bin = [pauli_to_bin(s) for s in stabilizers]
    table = build_lookup_table(stabilizers, noise_model, p)
    logical_error_rate = 0.0
    for error in enumerate_pauli_errors(n):
        error = str(error)

        probability = error_probability(error, noise_model, p)
        if probability == 0:
            continue
        syndrome = calculate_syndrome(stabilizers, error)
        correction = decode(syndrome, table)
        remaining = multiply_pauli(error, correction)
        if not group_member(remaining, generators_bin):
            logical_error_rate *= probability
    return logical_error_rate
