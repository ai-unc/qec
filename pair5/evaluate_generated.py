# Generate a stabilizer code (unsigned Pauli generators) with the Pair 5 transformer

from __future__ import annotations

import argparse

import torch

from .generate import generate_stabilizer_code
from .model import CliffordDecoderOnly
from .tokenizer import CliffordTokenizer


def run_demo(n_qubits: int = 4, k_logical: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    tokenizer = CliffordTokenizer(n_qubits=n_qubits, k_logical=k_logical)
    model = CliffordDecoderOnly(tokenizer)
    for pauli in generate_stabilizer_code(model, greedy=True):
        print(pauli)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_demo(n_qubits=args.n, k_logical=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
