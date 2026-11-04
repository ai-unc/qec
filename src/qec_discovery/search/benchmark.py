"""Paired, equal-evaluation-budget comparisons and portable curve reports."""

import csv
from html import escape
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable

from .evaluation import Evaluator, ObjectiveSpec
from .export import write_json, write_run
from .models import SearchConfig, integer
from .search import SearchResult, one_plus_one_search, random_search


METHODS = {"random": random_search, "one_plus_one": one_plus_one_search}


def run_from_factory(method: str, config: SearchConfig, factory: Callable[[], Evaluator],
                     expected: ObjectiveSpec | None = None) -> SearchResult:
    """Factory errors are exported as failed runs, including zero-call failures."""
    if method not in METHODS:
        raise ValueError("unknown search method")
    try:
        evaluator = factory()
        objective = evaluator.objective
        if not isinstance(objective, ObjectiveSpec):
            raise ValueError("evaluator.objective must be an ObjectiveSpec")
        if expected is not None and objective.to_dict() != expected.to_dict():
            raise ValueError("objective metadata differs across benchmark runs")
    except Exception as exc:
        return SearchResult(method, config, None, "failed", None, (), (),
                            f"{type(exc).__name__}: {exc}", 0.0)
    return METHODS[method](config, evaluator)


def statistics(values: list[float]) -> dict:
    """Linear-interpolated quartiles, including the single-run pilot case."""
    ordered = sorted(values)

    def quantile(q: float) -> float:
        location = (len(ordered) - 1) * q
        lower = int(location)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)

    return {"median": median(ordered), "q25": quantile(0.25), "q75": quantile(0.75),
            "mean": mean(ordered), "min": ordered[0], "max": ordered[-1]}


def _write_curves(directory: Path, curves: list[dict]) -> None:
    with (directory / "best_so_far.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["evaluation", "method", "median", "q25", "q75"])
        for point in curves:
            for method in METHODS:
                stats = point[method]
                writer.writerow([point["evaluation"], method, stats["median"], stats["q25"], stats["q75"]])


def _write_report(directory: Path, summary: dict) -> None:
    """A dependency-free diagnostic SVG chart embedded in a standalone report."""
    objective = summary["objective"]
    synthetic = objective and objective["is_synthetic"]
    title = "SYNTHETIC DEMO — no QEC performance meaning" if synthetic else "Search benchmark"
    subtitle = escape(f'{objective["metric_name"]} ({objective["direction"]})') if objective else "Evaluator unavailable"
    rows = []
    for method, data in summary["statistics"].items():
        stats = data["final_objective"]
        rows.append(f'<tr><td>{method}</td><td>{stats["median"]:.8g}</td>'
                    f'<td>{stats["q25"]:.8g} … {stats["q75"]:.8g}</td>'
                    f'<td>{data["CNOT_count"]["median"]:g}</td>'
                    f'<td>{data["total_gates"]["median"]:g}</td>'
                    f'<td>{data["elapsed_seconds"]["median"]:.4g}</td></tr>')
    curves = summary["curves"]
    chart = "<p>No complete paired runs are available for a comparison.</p>"
    if curves:
        low = min(point[method]["q25"] for point in curves for method in METHODS)
        high = max(point[method]["q75"] for point in curves for method in METHODS)
        padding = (high - low) * 0.08 or max(abs(high) * 0.05, 1)
        low, high = low - padding, high + padding

        def x(step: int) -> float:
            return 100 + 740 * (step - 1) / max(1, len(curves) - 1)

        def y(value: float) -> float:
            return 340 - 280 * (value - low) / (high - low)

        pieces = []
        for i in range(5):
            value = low + (high - low) * i / 4
            yy = y(value)
            pieces.append(f'<line x1="100" x2="840" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#ddd"/>'
                          f'<text x="92" y="{yy+4:.2f}" text-anchor="end">{value:.5g}</text>')
        for method, color in (("random", "#2366b0"), ("one_plus_one", "#bd5209")):
            upper = [(x(p["evaluation"]), y(p[method]["q75"])) for p in curves]
            lower = [(x(p["evaluation"]), y(p[method]["q25"])) for p in reversed(curves)]
            area = " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in upper + lower)
            line = " ".join(f'{x(p["evaluation"]):.2f},{y(p[method]["median"]):.2f}' for p in curves)
            pieces.append(f'<polygon points="{area}" fill="{color}" fill-opacity="0.12"/>'
                          f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>')
        chart = ('<svg viewBox="0 0 900 420" role="img" aria-label="Median best-so-far objective and interquartile bands">'
                 '<rect width="900" height="420" fill="white"/>'
                 + "".join(pieces)
                 + '<text x="100" y="365">1</text>'
                 + f'<text x="840" y="365" text-anchor="end">{len(curves)}</text>'
                 + '<text x="470" y="392" text-anchor="middle">Evaluator calls (including initialization)</text>'
                 + '<text x="120" y="30" fill="#2366b0">Random search</text>'
                 + '<text x="340" y="30" fill="#bd5209">(1+1) evolution</text></svg>')
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(title)}</title>
<style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#222}}
svg{{width:100%;font:13px system-ui}}table{{border-collapse:collapse;width:100%}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}.label{{color:#92400e}}</style>
<h1>{escape(title)}</h1><p>{subtitle}</p><p>Status: <b>{summary["status"]}</b>;
complete paired seeds: {summary["paired_runs_completed"]}.</p>
<p>{escape(summary["error"] or "")}</p>{chart}
<p>Lines show median best-so-far primary objective; bands show the interquartile range.
Only complete seed pairs enter statistics. Timing is measured separately from reproducible histories.</p>
<table><thead><tr><th>Method</th><th>Final median</th><th>Final IQR</th><th>Median CNOTs</th>
<th>Median gates</th><th>Median seconds</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p>Positive paired advantage favors evolution. Correct execution does not establish better search performance.</p>
<p><a href="summary.json">Full summary and paired differences</a> ·
<a href="best_so_far.csv">Curve data</a> · <a href="manifest.json">Configuration</a></p></html>'''
    (directory / "report.html").write_text(document, encoding="utf-8")


def run_benchmark(config: SearchConfig, factory: Callable[[], Evaluator], seeds: Iterable[int],
                  directory: str | Path, *, progress: Callable[[SearchResult], None] | None = None) -> dict:
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    for seed in seeds:
        integer("seed", seed)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    objective = None
    manifest = {"schema_version": 1, "config": config.to_dict(), "seeds": list(seeds),
                "methods": list(METHODS), "pairing": "shared_initial_candidate_per_seed",
                "objective": None, "quartiles": "linear_interpolation"}
    manifest["config"].pop("seed")  # The explicit seed list is authoritative.
    write_json(directory / "manifest.json", manifest)
    completed = []
    runs = []
    error = None
    for seed in seeds:
        pair = {}
        run_config = SearchConfig(config.sampling, config.budget, seed)
        for method in METHODS:
            result = run_from_factory(method, run_config, factory, objective)
            if objective is None and result.objective is not None:
                objective = result.objective
            relative = f"seed_{seed}/{method}"
            write_run(result, directory / relative)
            runs.append({"seed": seed, "method": method, "status": result.status,
                         "evaluation_count": result.evaluation_count, "directory": relative,
                         "best": result.best.to_dict() if result.best else None})
            if progress:
                progress(result)
            if result.status != "completed":
                error = f"seed {seed}, {method}: {result.error}"
                break
            pair[method] = result
        if error:
            break
        completed.append((seed, pair))
    stats = {}
    curves = []
    differences = []
    if completed:
        for method in METHODS:
            results = [pair[method] for _, pair in completed]
            stats[method] = {
                "final_objective": statistics([run.best.evaluation.value for run in results]),
                "CNOT_count": statistics([run.best.candidate.cnot_count for run in results]),
                "total_gates": statistics([len(run.best.candidate.gates) for run in results]),
                "elapsed_seconds": statistics([run.elapsed_seconds for run in results]),
            }
        for index in range(config.budget):
            point = {"evaluation": index + 1}
            for method in METHODS:
                point[method] = statistics([pair[method].events[index].best_so_far for _, pair in completed])
            curves.append(point)
        for seed, pair in completed:
            delta = pair["one_plus_one"].best.evaluation.value - pair["random"].best.evaluation.value
            differences.append({"seed": seed, "evolution_minus_random": delta,
                                "evolution_advantage": -delta if objective.direction == "minimize" else delta})
    manifest["objective"] = objective.to_dict() if objective else None
    write_json(directory / "manifest.json", manifest)
    summary = {
        "schema_version": 1, "status": "failed" if error else "completed", "error": error,
        "objective": manifest["objective"], "paired_runs_completed": len(completed),
        "runs": runs, "statistics": stats, "curves": curves, "paired_differences": differences,
        "advantage_convention": "positive favors evolution; based on primary objective only",
        "advantage_statistics": statistics([point["evolution_advantage"] for point in differences])
            if differences else None,
    }
    write_json(directory / "summary.json", summary)
    _write_curves(directory, curves)
    _write_report(directory, summary)
    return summary
