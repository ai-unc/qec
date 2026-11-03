from error_sampler import sample_pauli_error
from noise import model, error_probability


def calculate_syndrome(stabilizers, error_string):
    # 1 is anticommute, 0 is commute
    commute_list: list[int] = []
    for s in stabilizers:
        total = 0
        for s_op, err_op in zip(s, error_string):
            if s_op != err_op and s_op != "I" and err_op != "I":
                total += 1
        if total % 2 == 0:
            commute_list.append(0)
        else:
            commute_list.append(1)

    return commute_list
