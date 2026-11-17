# Generate a stabilizer code (unsigned Pauli generators) with the Pair 5 transformer

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from .generate import generate_stabilizer_code
from .model import CliffordDecoderOnly
from .pipeline_interface import evaluate_candidate_generators
from .tokenizer import CliffordTokenizer


def _init_params(model: CliffordDecoderOnly, key: jax.Array):
    tok = model.tokenizer
    ops = jnp.asarray([[tok.BOS]], dtype=jnp.int32)
    qa = jnp.asarray([[tok.Q_BOS]], dtype=jnp.int32)
    qb = jnp.asarray([[tok.Q_BOS]], dtype=jnp.int32)
    return model.init(key, ops, qa, qb)


def run_demo(
    n_qubits: int = 4,
    k_logical: int = 2,
    seed: int = 0,
    *,
    greedy: bool = False,
    min_gates: int = 4,
):
    key = jax.random.PRNGKey(seed)
    key_init, key_gen = jax.random.split(key)
    tokenizer = CliffordTokenizer(n_qubits=n_qubits, k_logical=k_logical)
    model = CliffordDecoderOnly(tokenizer)
    params = _init_params(model, key_init)
    # Sampling defaults force a few CNOTs first: untrained logits heavily favor EOS.
    gen_min_gates = 0 if greedy else min_gates
    generators = generate_stabilizer_code(
        model,
        params,
        key=None if greedy else key_gen,
        greedy=greedy,
        min_gates=gen_min_gates,
    )
    print("generators:")
    for pauli in generators:
        print(f"  {pauli}")
    result = evaluate_candidate_generators(generators, n=n_qubits, k=k_logical)
    print(f"valid: {result.valid}")
    if result.logical_error_rate is None:
        print("logical_error_rate: n/a (invalid code)")
    else:
        print(f"logical_error_rate: {result.logical_error_rate:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Decode with argmax (deterministic). Default is categorical sampling.",
    )
    parser.add_argument(
        "--min-gates",
        type=int,
        default=4,
        help="When sampling, disallow EOS until this many gates (default: 4).",
    )
    args = parser.parse_args()
    run_demo(
        n_qubits=args.n,
        k_logical=args.k,
        seed=args.seed,
        greedy=args.greedy,
        min_gates=args.min_gates,
    )


if __name__ == "__main__":
    main()
