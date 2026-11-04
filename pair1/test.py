from gen_ind import gen_ind
from pauli_to_bin import pauli_to_bin
from group_member import group_member
from commutation_check import commutation_check
import numpy as np
import pytest

generators = ["ZZI", "IZZ"]
generators_bin = [pauli_to_bin(i) for i in generators]

print(generators_bin)
print(gen_ind(generators_bin))

print(group_member("ZIZ", generators_bin))

VALID_SETS = [
    ["Z"],
    ["X"],
    ["XX", "ZZ"],
    ["XI", "IX"],
    ["ZZI", "IZZ"],
    ["XXXX", "ZZZZ"],
    ["IIIXXXX", "IXXIIXX", "XIXIXIX",
     "IIIZZZZ", "IZZIIZZ", "ZIZIZIZ"],
]

@pytest.mark.parametrize("paulis", VALID_SETS)
def test_valid_returns_true(paulis):
    assert gen_ind([pauli_to_bin(p) for p in paulis])
'''
@pytest.mark.parametrize("paulis", VALID_SETS)
def test_valid_matches_reference(paulis):
    gens = [pauli_to_bin(p) for p in paulis]
    assert bool(gen_ind(gens)) == reference(gens)
'''

@pytest.mark.parametrize("paulis", VALID_SETS)
def test_valid_does_not_mutate_input(paulis):
    gens = [pauli_to_bin(p) for p in paulis]
    snapshot = [g.copy() for g in gens]
    gen_ind(gens)
    for before, after in zip(snapshot, gens):
        assert np.array_equal(before, after)

def test_inputs_commute():
    assert commutation_check(pauli_to_bin("IXIX"), pauli_to_bin("XIXI")) == 0

def test_inputs_anticommute():
    assert commutation_check(pauli_to_bin("XIX"), pauli_to_bin("XIZ")) == 1 


