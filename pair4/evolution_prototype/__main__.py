"""Run from the repository root with python3 -m pair4."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

from .benchmark import run_benchmark, run_from_factory
from .evaluation import load_evaluator
from .export import write_json, write_run
from .models import SamplingConfig, SearchConfig, integer
from .operators import random_stream, sample_random_circuit


def _sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=4, help="physical qubits (default: 4)")
    parser.add_argument("--k", type=int, default=2, help="logical qubits (default: 2)")
    parser.add_argument("--max-length", type=int, help="positive gate cap (default: 8*n)")
    parser.add_argument("--output", type=Path, help="new output file (sample) or directory (search/benchmark)")


def _evaluator_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evaluator", required=True, help="explicit demo or external module:factory")
    parser.add_argument("--evaluator-config", type=Path, help="JSON object passed to external factory")
    parser.add_argument("--budget", type=int, default=1000, help="evaluator calls per run, including initialization")


def _output_directory(command: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parent / "runs" / f"{command}-{stamp}-{uuid4().hex[:8]}"


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pair 4: bounded Clifford circuit search")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="generate legal semantic gate sequences")
    _sampling_arguments(sample)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--count", type=int, default=1)
    search = commands.add_parser("search", help="run one search with an explicit evaluator")
    _sampling_arguments(search)
    _evaluator_arguments(search)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--method", required=True, choices=("random", "one_plus_one"))
    benchmark = commands.add_parser("benchmark", help="compare both methods across paired seeds")
    _sampling_arguments(benchmark)
    _evaluator_arguments(benchmark)
    benchmark.add_argument("--seeds", nargs="+", type=int, default=list(range(30)))
    args = parser.parse_args(argv)
    try:
        sampling = SamplingConfig(args.n, args.k, args.max_length)
        if args.command == "sample":
            integer("count", args.count, 1)
            rng = random_stream(args.seed, "initial")
            data = {"schema_version": 1, "seed": args.seed, "config": sampling.to_dict(),
                    "candidates": [sample_random_circuit(sampling, rng).to_dict() for _ in range(args.count)]}
            document = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as stream:
                    stream.write(document)
                print(f"Candidates: {args.output.resolve()}")
            else:
                print(document, end="")
            return 0
        config = SearchConfig(sampling, args.budget, args.seed if args.command == "search" else 0)
        evaluator_config = {}
        if args.evaluator_config:
            if args.evaluator == "demo":
                raise ValueError("--evaluator-config is only for external evaluators")
            evaluator_config = json.loads(args.evaluator_config.read_text(encoding="utf-8"),
                                          parse_constant=_reject_constant)
            if not isinstance(evaluator_config, dict):
                raise ValueError("evaluator configuration must be a JSON object")
        for key, value in (("n_qubits", args.n), ("k_logical", args.k)):
            if key in evaluator_config and evaluator_config[key] != value:
                raise ValueError(f"evaluator configuration conflicts with {key}")
            evaluator_config[key] = value

        def factory():
            return load_evaluator(args.evaluator, evaluator_config)

        output = args.output if args.output else _output_directory(args.command)
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        if args.command == "search":
            result = run_from_factory(args.method, config, factory)
            write_run(result, output)
            if result.objective and result.objective.is_synthetic:
                print("SYNTHETIC DEMO: scores have no quantum-performance meaning.")
            print(f"{result.status}: {result.evaluation_count} evaluator calls")
            if result.best:
                print(f"Best {result.objective.metric_name}: {result.best.evaluation.value:g}")
            error = result.error
        else:
            def progress(result):
                label = "SYNTHETIC " if result.objective and result.objective.is_synthetic else ""
                print(f"{label}seed={result.config.seed} method={result.method}: "
                      f"{result.status}, calls={result.evaluation_count}", flush=True)

            summary = run_benchmark(config, factory, args.seeds, output, progress=progress)
            print(f'{summary["paired_runs_completed"]} complete paired seeds; {summary["status"]}')
            print(f"Report: {(output / 'report.html').resolve()}")
            error = summary["error"]
        write_json(output / "invocation.json", {
            "schema_version": 1, "evaluator": args.evaluator,
            "evaluator_config": evaluator_config,
        })
        print(f"Results: {output.resolve()}")
        if error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
