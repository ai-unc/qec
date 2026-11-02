"""Live demo: a circuit goes through Pair 2 and feeds Pair 1 and Pair 3 code.

Run from the repo root:  python -m pair2.demo
"""

import importlib.util
import pathlib
import sys

from pair2.adapters import to_binary, unsigned
from pair2.simulate import encode

ROOT = pathlib.Path(__file__).resolve().parents[1]

# The repetition code from Pair 4's layout: 2 ancillas first, 1 data qubit last (D9).
CIRCUIT = [("CNOT", 2, 0), ("CNOT", 2, 1)]


def load_pair1_pauli_to_bin():
    # Pair 1 is not a package, so load the file directly instead of editing their folder.
    spec = importlib.util.spec_from_file_location(
        "pair1_pauli_to_bin", ROOT / "pair1" / "pauli_to_bin.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.pauli_to_bin


def main():
    stabilizers, logical_x, logical_z = encode(CIRCUIT, n=3, k=1)
    print("Circuit:          ", CIRCUIT)
    print("Stabilizers:      ", stabilizers)
    print("Logical X, Z:     ", logical_x, logical_z)

    # Pair 3 uses bare imports, so its folder goes on sys.path here and nowhere else.
    sys.path.insert(0, str(ROOT / "pair3"))
    from syndrome import calculate_syndrome

    flat = unsigned(stabilizers)
    print("\nPair 3 syndromes (unsigned stabilizers", flat, "):")
    for error in ("XII", "IXI", "IIX", "ZII"):
        print(f"  error {error} -> syndrome {calculate_syndrome(flat, error)}")

    pauli_to_bin = load_pair1_pauli_to_bin()
    print("\nPair 1 binary [x | z] form:")
    for signed, plain in zip(stabilizers, flat):
        ours, theirs = to_binary(signed), pauli_to_bin(plain)
        print(f"  {signed} -> {ours.tolist()}  (matches pair1: {(ours == theirs).all()})")


if __name__ == "__main__":
    main()
