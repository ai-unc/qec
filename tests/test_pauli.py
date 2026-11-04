import unittest

import numpy as np

from qec_discovery.pauli.commutation_check import commutation_check
from qec_discovery.pauli.gen_ind import gen_ind
from qec_discovery.pauli.group_member import group_member
from qec_discovery.pauli.pauli_to_bin import pauli_to_bin

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


class PauliTests(unittest.TestCase):
    def test_valid_returns_true(self):
        for paulis in VALID_SETS:
            self.assertTrue(gen_ind([pauli_to_bin(p) for p in paulis]))

    def test_valid_does_not_mutate_input(self):
        for paulis in VALID_SETS:
            gens = [pauli_to_bin(p) for p in paulis]
            snapshot = [g.copy() for g in gens]
            gen_ind(gens)
            for before, after in zip(snapshot, gens):
                self.assertTrue(np.array_equal(before, after))

    def test_inputs_commute(self):
        self.assertEqual(commutation_check(pauli_to_bin("IXIX"), pauli_to_bin("XIXI")), 0)

    def test_inputs_anticommute(self):
        self.assertEqual(commutation_check(pauli_to_bin("XIX"), pauli_to_bin("XIZ")), 1)


if __name__ == "__main__":
    generators_bin = [pauli_to_bin(i) for i in ("ZZI", "IZZ")]
    print(generators_bin)
    print(gen_ind(generators_bin))
    print(group_member("ZIZ", generators_bin))
