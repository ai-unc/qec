"""A circuit through Clifford simulation, then the Pauli and noise modules.

From the repository root, after ``pip install -e .``:

    python scripts/demo_pipeline.py
"""

from qec_discovery.clifford.adapters import to_binary, unsigned
from qec_discovery.clifford.simulate import encode
from qec_discovery.noise.syndrome import calculate_syndrome
from qec_discovery.pauli.pauli_to_bin import pauli_to_bin

# Repetition-code encoder: 2 ancillas first, 1 data qubit last.
CIRCUIT = [("CNOT", 2, 0), ("CNOT", 2, 1)]


def main():
    stabilizers, logical_x, logical_z = encode(CIRCUIT, n=3, k=1)
    print("Circuit:          ", CIRCUIT)
    print("Stabilizers:      ", stabilizers)
    print("Logical X, Z:     ", logical_x, logical_z)

    flat = unsigned(stabilizers)
    print("\nSyndromes (unsigned stabilizers", flat, "):")
    for error in ("XII", "IXI", "IIX", "ZII"):
        print(f"  error {error} -> syndrome {calculate_syndrome(flat, error)}")

    print("\nBinary [x | z] form:")
    for signed, plain in zip(stabilizers, flat):
        ours, theirs = to_binary(signed), pauli_to_bin(plain)
        print(f"  {signed} -> {ours.tolist()}  (matches pauli_to_bin: {(ours == theirs).all()})")


if __name__ == "__main__":
    main()
