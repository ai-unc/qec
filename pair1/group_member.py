
from pauli_to_bin import pauli_to_bin
from gen_ind import gen_ind

# input: error (Pauli string/binary vector), stabilizer group (list of binary vectors)
# returns: True if error belongs to the group, False otherwise
def group_member(error, group):
    # if error is a Pauli string convert to vector, comment out otherwise
    error = pauli_to_bin(error)

    # add the error to the group
    test_group = group + [error]

    # if the group is no longer independent after adding the vector, then error belongs to group
    return not gen_ind(test_group)