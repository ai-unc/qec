from .enumerate_pauli import enumerate_pauli_errors
from .models import error_probability
from .syndrome import calculate_syndrome


def build_lookup_table(stabilizers, noise_model, p):
    # INPUT: stabilizers (list of Pauli strings, e.g. ["ZZI", "IZZ"]), noise_model (string: "bit_flip", "phase_flip", or "depolarizing"), and p
    # OUTPUT: dictionary mapping each syndrome (tuple of 0s and 1s) to the most likely Pauli error string with that syndrome

    # Assuming that all stabilizers are of the same length, we can just default to length of the first stabilizer code
    n = len(stabilizers[0])
    ml_err = {} # Most likely errors
    ml_err_prob = {} # Probability of the most likely error stored for each syndrome

    for error in enumerate_pauli_errors(n):
        error = str(error)

        # calc syndrome list for the error and then the probability of that error
        syndrome = tuple(calculate_syndrome(stabilizers, error))
        prob = error_probability(error, noise_model, p)

        # Given the above syndrome and probability of error, keep only the most likely error for each syndrome.
        if syndrome not in ml_err:
            ml_err[syndrome] = error
            ml_err_prob[syndrome] = prob
        elif prob > ml_err_prob[syndrome]:
            ml_err[syndrome] = error
            ml_err_prob[syndrome] = prob

    return ml_err


def decode(syndrome_list, table):
    # Given a syndrome, some combo of [1,0, ...], will return the most likely error that caused it
    return table[tuple(syndrome_list)]
