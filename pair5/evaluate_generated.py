# DEMO: generate a circuit and score it using Pair 4's DemoEvaluator.

from __future__ import annotations

import argparse

import torch

from pair4.evolution_prototype import CircuitCandidate, DemoEvaluator, GateOp

from .generate import generate_circuit
from .model import CliffordDecoderOnly
from .tokenizer import CliffordTokenizer


def _format_gate(gate: GateOp) -> str:
    if gate.op == "CNOT":
        control, target = gate.qubits
        return f"CNOT  control={control}  target={target}"
    return f"{gate.op:<4}  qubit={gate.qubits[0]}"


def _format_circuit(candidate: CircuitCandidate) -> str:
    lines = [
        "Generated circuit",
        "─────────────────",
        f"  Register : [[{candidate.n_qubits}, {candidate.k_logical}]]"
        f"  ({candidate.n_qubits} qubits, {candidate.k_logical} logical)",
        f"  Length   : {len(candidate.gates)} gate(s)",
        "",
        "  Gates (chronological):",
    ]
    if not candidate.gates:
        lines.append("    (empty — identity encoder)")
    else:
        for i, gate in enumerate(candidate.gates, 1):
            lines.append(f"    {i:>2}. {_format_gate(gate)}")
    return "\n".join(lines)


def _format_score(result, evaluator: DemoEvaluator) -> str:
    distance = result.diagnostics["gate_edit_distance"]
    target_len = len(evaluator.target)
    lines = [
        "",
        "DemoEvaluator score (synthetic)",
        "───────────────────────────────",
        f"  Metric           : {evaluator.objective.metric_name}",
        f"  Direction        : {evaluator.objective.direction}",
        f"  Score            : {result.value}",
        f"  Gate edit dist.  : {distance}  (vs {target_len}-gate demo target)",
        "",
        "  Note: This is just a DEMO using Pair 4's DemoEvaluator.",
    ]
    return "\n".join(lines)


def run_demo(
    n_qubits: int = 4,
    k_logical: int = 2,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    tokenizer = CliffordTokenizer(n_qubits=n_qubits, k_logical=k_logical)
    model = CliffordDecoderOnly(tokenizer)
    candidate = generate_circuit(model, greedy=True)
    evaluator = DemoEvaluator(n_qubits=n_qubits, k_logical=k_logical)
    result = evaluator.evaluate(candidate)

    print()
    print(f"Pair 5 generation demo  (seed={seed})")
    print("══════════════════════════════════")
    print()
    print(_format_circuit(candidate))
    print(_format_score(result, evaluator))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_demo(n_qubits=args.n, k_logical=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
