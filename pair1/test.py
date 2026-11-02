from gen_ind import gen_ind
from pauli_to_bin import pauli_to_bin

generators = ["ZZI", "IZZ"]
generators_bin = [pauli_to_bin(i) for i in generators]

print(generators_bin)
print(gen_ind(generators_bin))

