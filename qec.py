"""
Our goal here is to discover a 3-qubit code that protects one qubit against bit flips,
and to check that its logical error rate under independent 5% bit-flip noise is
0.725% (the number 0.00725).

How the packages share the work:
--------------------------------
search.random_circuits   builds a random list of H, S, and CNOT gates
clifford.simulate        turns that list into the code's stabilizer checks
clifford.adapters        strips the leading + or - so the other packages can read the checks
pauli                    decides whether those checks form a valid stabilizer code
noise.models             gives the probability of each Pauli error
noise.enumerate_pauli    lists every error on n qubits
noise.syndrome           says which checks an error sets off
noise.decoder            guesses the error from that pattern
pauli.group_member       says whether the leftover after correction is harmless

There is no separate "model" yet. This file is the wiring as it calls those
functions in order. The attention layer in qec_discovery.model is *not* part of
this run.
"""

import random
import sys
from pathlib import Path

# find the qec_discovery package
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qec_discovery.clifford.adapters import to_binary, unsigned
from qec_discovery.clifford.simulate import simulate_circuit
from qec_discovery.noise.decoder import build_lookup_table, decode
from qec_discovery.noise.enumerate_pauli import enumerate_pauli_errors
from qec_discovery.noise.models import error_probability, model
from qec_discovery.noise.syndrome import calculate_syndrome
from qec_discovery.pauli.group_member import group_member
from qec_discovery.pauli.pauli_to_bin import pauli_to_bin
from qec_discovery.pauli.validate_code import validate_code
from qec_discovery.search.random_circuits import random_gate

"""n physical qubits, k of which carry logical data. The other n - k qubits are
ancillas (extra qubits the code uses for its checks). [[3, 1]] means 3 physical
qubits protecting 1 logical qubit, so there are 2 checks."""

N_QUBITS = 3 # number of physical qubits
K_LOGICAL = 1 # number of logical qubits
DEPTH = 12 # how many gates (H, S, or CNOT) each random candidate circuit contains
TRIALS = 1000
P_ERROR = 0.05
NOISE = "bit_flip"
TARGET = 3 * (P_ERROR ** 2) * (1 - P_ERROR) + P_ERROR ** 3  # 0.00725
SEED = 42

"""A known encoder for the repetition code. Qubits 0 and 1 start as ancillas in
|0>. Qubit 2 holds the data. Copying qubit 2 onto both ancillas with CNOT
produces the checks ZIZ and IZZ, which protect the same way as ZZI and IZZ.

Basically, this is basically the circuit that encodes one logical qubit 
into three physical qubits, with ZIZ and IZZ being the stabilizer generator
that detects bit-flip errors."""
REFERENCE_CIRCUIT = [("CNOT", 2, 0), ("CNOT", 2, 1)]


def stabilizers_of(circuit):
    """Read the code's checks off a circuit and returns them as plain Pauli strings.

    `simulate_circuit` starts the ancillas (qubits 0 .. n-k-1) in |0>, whose
    checks are +Z on each of those qubits, then pushes every check through the
    gates. The result is n - k Pauli strings, one per check, with a leading
    sign. `unsigned` drops that sign. The syndrome and Pauli packages compare
    letters, and a leading "+" would shift every letter by one place.
    """
    signed = simulate_circuit(circuit, n=N_QUBITS, k=K_LOGICAL)
    return unsigned(signed)


def accept_code(stabilizers):
    """Return whether `stabilizers` is a valid [[n, k]] stabilizer code.

    A valid code has exactly n - k checks. They must all commute (they can be
    measured together) and be independent (none is a product of the others).
    `validate_code` does the commutation and independence tests on the binary
    form of the Paulis. The count check is done here because `validate_code`
    does not take n and k.
    """
    if len(stabilizers) != N_QUBITS - K_LOGICAL:
        return False
    if any(len(check) != N_QUBITS for check in stabilizers):
        return False
    vectors = [pauli_to_bin(check) for check in stabilizers]
    return bool(validate_code(vectors))


def pauli_product(left, right):
    """Multiply two Pauli strings, dropping the overall phase.

    X*X = I, and the product of two different non-identity Paulis is the third
    one (X*Z = Y, up to a phase we ignore). The packages compare errors up to
    that phase, which is the right equivalence for "did this correction work?"
    """
    letters = []
    for a, b in zip(left, right):
        if a == b:
            letters.append("I")
        elif a == "I":
            letters.append(b)
        elif b == "I":
            letters.append(a)
        else:
            letters.append(({"X", "Y", "Z"} - {a, b}).pop())
    return "".join(letters)


def logical_error_rate(stabilizers, noise_model=NOISE, p=P_ERROR):
    """Probability that correction leaves a logical error.

    For every one of the 4**n Pauli errors:
      1. `error_probability` is its probability under the chosen noise.
      2. `calculate_syndrome` is the pattern of checks that anticommute with it.
         A 1 means that check fired. A 0 means it did not.
      3. `decode` looks that pattern up and returns the error the decoder
         will correct. The table keeps the most likely error for each pattern.
      4. The leftover is error times correction. If that leftover is inside
         the stabilizer group, the logical qubit is fine (`group_member`).
         If it is not, the logical qubit was flipped and this error counts
         as a failure.

    Adding the probabilities of the failures is an exact logical error rate.
    No sampling is involved. On 3 qubits there are only 64 errors.
    """
    table = build_lookup_table(stabilizers, noise_model, p)
    group = [pauli_to_bin(check) for check in stabilizers]
    failed = 0.0
    for error in enumerate_pauli_errors(len(stabilizers[0])):
        error = str(error)
        probability = error_probability(error, noise_model, p)
        if probability == 0.0:
            continue
        syndrome = calculate_syndrome(stabilizers, error)
        correction = decode(syndrome, table)
        leftover = pauli_product(error, correction)
        if not group_member(leftover, group):
            failed += probability
    return failed


def discover(trials=TRIALS, depth=DEPTH, seed=SEED):
    """Search random circuits and return them ranked from best to worst.

    Each trial draws `depth` gates from `random_gate`, converts the circuit
    into checks, drops anything `accept_code` rejects, and scores the rest.
    The same stabilizer list can appear many times, so it is scored once.
    """
    rng = random.Random(seed)
    ranked = []
    seen = set()
    for trial in range(trials):
        circuit = [random_gate(N_QUBITS, rng) for _ in range(depth)]
        stabilizers = stabilizers_of(circuit)
        if not accept_code(stabilizers):
            continue
        key = tuple(sorted(stabilizers))
        if key in seen:
            continue
        seen.add(key)
        ranked.append({
            "stabilizers": stabilizers,
            "error_rate": logical_error_rate(stabilizers),
            "circuit": circuit,
            "trial": trial,
        })
    ranked.sort(key=lambda item: item["error_rate"])
    return ranked


def show_reference():
    """Walk one known-good circuit through every stage, then score it."""
    print("Noise model")
    print(f"  {NOISE} at p={P_ERROR}: each qubit is independently {model(NOISE, P_ERROR)}")
    print(f"  target logical error rate = {TARGET:.5f} = {TARGET:.4%}")
    print()

    print("1. Circuit -> checks   (clifford.simulate)")
    print(f"  circuit {REFERENCE_CIRCUIT}")
    stabilizers = stabilizers_of(REFERENCE_CIRCUIT)
    print(f"  checks  {stabilizers}")
    print(f"  binary  {[row.tolist() for row in to_binary(stabilizers)]}")
    print()

    print("2. Validation          (pauli.validate_code)")
    print(f"  valid [[{N_QUBITS}, {K_LOGICAL}]] code: {accept_code(stabilizers)}")
    print()

    print("3. Syndromes and corrections   (noise.syndrome, noise.decoder)")
    print("  A single X on each qubit must produce a different syndrome,")
    print("  and the decoder must name that same X as the correction.")
    table = build_lookup_table(stabilizers, NOISE, P_ERROR)
    for error in ("XII", "IXI", "IIX"):
        syndrome = calculate_syndrome(stabilizers, error)
        correction = decode(syndrome, table)
        print(f"  {error} -> syndrome {syndrome} -> correction {correction}")
    print()

    print("4. Logical error rate  (enumerate, decode, pauli.group_member)")
    bit_flip = logical_error_rate(stabilizers, "bit_flip", P_ERROR)
    phase_flip = logical_error_rate(stabilizers, "phase_flip", P_ERROR)
    print(f"  bit-flip:    {bit_flip:.5f}")
    print(f"  phase-flip:  {phase_flip:.5f}")
    print("  The phase-flip number is worse than doing nothing (5%).")
    print("  These checks catch X errors. A single Z can still be a logical")
    print("  error, so the code's quantum distance is 1.")
    return stabilizers, bit_flip


def main():
    stabilizers, bit_flip = show_reference()
    if abs(bit_flip - TARGET) > 1e-10:
        raise SystemExit(
            f"reference code scored {bit_flip:.8f}, expected {TARGET:.8f}"
        )

    print()
    print("5. Search              (search.random_circuits, then the same grading)")
    print(f"  {TRIALS} random circuits, depth {DEPTH}, seed {SEED}")
    ranked = discover()
    print(f"  distinct codes scored: {len(ranked)}")
    print("  every code, best rate first. >>> marks ZIZ and IZZ:")
    target_checks = set(stabilizers)
    match = None
    for item in ranked:
        found = set(item["stabilizers"]) == target_checks
        if found:
            match = item
        mark = ">>>" if found else "   "
        print(
            f"  {mark} trial {item['trial']:4d}  {item['stabilizers']}  "
            f"{item['error_rate']:.4%}"
        )

    print()
    if match is None:
        raise SystemExit(f"search did not produce the checks {stabilizers}")
    print(
        f"Discovered {match['stabilizers']} at trial {match['trial']} "
        f"with logical error rate {match['error_rate']:.4%}."
    )
    print("Those are the repetition-code checks. The search built them from a random circuit.")


if __name__ == "__main__":
    main()
