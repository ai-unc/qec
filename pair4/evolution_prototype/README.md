# Pair 4: Clifford circuit search

For a step-by-step visual walkthrough, read [Pair 4 explained visually](EXPLANATION.md),
with Mermaid flowcharts for generation, mutation, search, ranking, and integration.

This package implements Pair 4's random Clifford sequence generator, bounded
gate mutations, independent random search, and a mutation-only `(1+1)` search.
It uses Python's standard library, requires Python 3.10+, and was tested with
Python 3.12.3. Run commands from the repository root. No installation is needed.

The included evaluator is **synthetic**. It measures similarity to a six-gate
sequence, not error detection, code distance, or error correction. Real code
evaluation remains a dependency on the other project pairs.

## Quick start

Generate two reproducible candidate encoders:

```bash
python3 -m pair4 sample --n 5 --k 1 --seed 42 --count 2
```

Run either search with the explicitly selected demo evaluator:

```bash
python3 -m pair4 search --method random --evaluator demo --seed 42 --budget 1000
python3 -m pair4 search --method one_plus_one --evaluator demo --seed 42 --budget 1000
```

Compare both methods using 30 paired seeds (`0` through `29`) and 1,000 evaluator
calls per method per seed:

```bash
python3 -m pair4 benchmark --evaluator demo
```

Use a separate pilot before a full comparison:

```bash
python3 -m pair4 benchmark --evaluator demo --budget 100 --seeds 1000 1001
```

Search and benchmark commands print the generated output directory under
`pair4/runs/`. A benchmark also creates a standalone `report.html` with median
best-so-far curves and interquartile bands. Generated outputs and Python caches
are ignored by `pair4/.gitignore`.

Use `--output PATH` to choose a new directory (or a new JSON file for `sample`).
Existing output locations are never overwritten. `--help` describes each command.

Run the tests:

```bash
python3 -m unittest discover -s pair4/tests -v
```

## How the algorithms work

A candidate contains `n_qubits`, `k_logical`, and an immutable chronological tuple
of `GateOp` records. For example, the sequence `H(0), CNOT(0,2), S(1)` applies H
first, then CNOT with control 0 and target 2, then S on wire 1. Qubit indices start
at zero. The physical register size is stored even if a sequence is empty or
leaves high-index wires idle.

The first `n-k` wires are ancillas initialized to zero; the remaining `k` wires
hold arbitrary logical inputs. Pair 2 will conjugate the ancilla generators
`+Z_0, ..., +Z_(n-k-1)` through the circuit using `P -> U P U†`. Exactly these
`n-k` generators define the code. Conjugating all `n` Z generators of an all-zero
state instead describes one stabilizer state, not the full `[[n,k]]` code.

Clifford conjugation preserves commutation and independence, so each legal
candidate defines a valid code with this convention. Validity alone says nothing
about its distance or noise performance. Pair 4 implements no conjugation or
stabilizer linear algebra.

### Generation and mutation

- Sample length uniformly from the integers `0` through `max_length`, inclusive.
  The default cap is `8*n`; a cap of zero is rejected. The empty identity encoder
  is valid.
- Sample a gate type uniformly from H, S, CNOT, then sample its operands
  uniformly. CNOT uses a uniformly chosen ordered pair of distinct wires with
  all-to-all connectivity. For one physical qubit, sample only H or S.
- For mutation, choose equally among currently legal replace, delete, and insert
  operations. Empty sequences allow only insertion; sequences at the cap allow
  only replacement or deletion.
- Replacement chooses an existing position and a different instruction, using
  the original gate distribution conditioned on excluding the old instruction.
  It uses a finite weighted choice, not an unbounded retry loop.
- Insertion chooses among all `L+1` gaps, including both endpoints. Deletion
  chooses among the `L` existing positions. Parents remain unchanged.

Equal gate-type probabilities are different from equal probabilities for every
gate/operand tuple. Neither gives uniform sampling of Clifford transformations
or stabilizer codes. Length bounds and mutation also bias the explored codes.

### Selection and budgets

Random search draws an independent candidate on each call and retains the best.
The `(1+1)` search starts with one random parent and evaluates one mutated child
at a time. It accepts a better child, rejects a worse one, and accepts an exactly
equally ranked child with probability `1/2`. This permits some movement across
plateaus. It is an evolutionary baseline, with no population or crossover.

Both methods compare the primary evaluator value first, respecting its minimize
or maximize direction. Exact primary ties prefer fewer CNOTs, then fewer total
gates. Scores are not rounded, and no weighted structural penalty changes the
primary objective. These tie-breakers themselves affect exploration; they can
favor shrinking on plateaus and do not guarantee good search performance.

The budget counts every evaluator invocation, including initialization,
duplicates, and calls that fail. There is no evaluation cache. The archive keeps
the ten best distinct exact gate sequences; equivalent codes are not deduplicated.
Archive deduplication does not save evaluation calls. Equal-ranked archive items
retain their earliest evaluation order.

Each seed has isolated `initial`, `proposals`, `mutation`, and `acceptance`
streams. A stream uses Python's `random.Random`/MT19937, initialized with the
big-endian integer SHA-256 digest of UTF-8 `pair4:v1:<seed>:<label>`.
The two methods share an initial candidate for each paired seed, while their
subsequent proposal streams are separate. Pin the Python version for exact replay;
the manifest records it, and every generated candidate is also saved.

## Python API

```python
from pair4 import (
    DemoEvaluator, SamplingConfig, SearchConfig,
    mutate_circuit, one_plus_one_search, random_stream, sample_random_circuit,
)

sampling = SamplingConfig(n_qubits=4, k_logical=2, max_length=32)
candidate = sample_random_circuit(sampling, random_stream(42, "initial"))
edit = mutate_circuit(candidate, sampling.max_length, random_stream(42, "mutation"))

result = one_plus_one_search(
    SearchConfig(sampling=sampling, budget=1000, seed=42),
    DemoEvaluator(n_qubits=4, k_logical=2),
)
assert result.status == "completed", result.error
print(result.best.candidate.to_dict())
```

`random_search` has the same signature. Both accept an optional keyword-only
`initial_candidate` for controlled experiments. `SearchResult` contains `best`,
`archive`, `events`, `evaluation_count`, `status`, `error`, and elapsed timing.
`pair4.export.write_run(result, directory)` creates the portable exports.

## External evaluator contract and ownership

Pair 4 consumes an object satisfying this interface:

```python
from pair4 import CircuitCandidate, EvaluationResult, ObjectiveSpec

# An externally implemented evaluator exposes:
# objective: ObjectiveSpec
# evaluate(candidate: CircuitCandidate) -> EvaluationResult

objective = ObjectiveSpec(
    metric_name="harmful_undetected_probability",
    direction="minimize",
    version="your-evaluator-version",
    parameters={
        "n_qubits": 4,
        "k_logical": 2,
        "noise": {"model": "depolarizing", "p": 0.01},
        "placement": "post_encoding_data",
        "syndrome_information": "ideal",
        "enumeration": "complete",
    },
    exact=True,
    is_synthetic=False,
)
# Return the actual metric as EvaluationResult(value, diagnostics={...}).
# Raise an exception if a score is unavailable or evaluation fails.
```

The evaluator owns all scientific calculations and must be deterministic. The
objective parameters must fully identify the experiment, including relevant
noise, decoder, truncation, and convention settings. Parameters and diagnostics
are copied into immutable JSON-compatible containers. Metadata must stay fixed
within a run and across a benchmark. Version 1 rejects estimated evaluations.

The CLI loads an explicitly named `module:factory`; the factory receives one
configuration dictionary and returns a **fresh evaluator instance**. It receives
`n_qubits` and `k_logical` from the CLI plus any fields in `--evaluator-config`.
Conflicting dimensions are rejected. The entrypoint and factory configuration
are saved in `invocation.json`. The `demo` evaluator accepts no configuration
file, preventing scientific-looking parameters from being silently ignored.

Once collaborators provide a real factory, the invocation will look like this
(`team_evaluator:make` is a placeholder, not an existing module):

```bash
python3 -m pair4 benchmark --n 4 --k 2 --evaluator team_evaluator:make \
  --evaluator-config pair4/examples/depolarizing.json
```

`examples/depolarizing.json` is the proposed experiment request. The external
factory must interpret it and reflect its effective settings in `ObjectiveSpec`;
it is not an implemented Pair 3 noise model. Scientific runs require
`1 <= k < n`. Candidate construction and synthetic demos also support `k=0`
and `k=n` for diagnostics.

| Owner | Integration responsibility |
| --- | --- |
| Pair 1 | Pauli conversion, commutation, GF(2) rank, and independence validation |
| Pair 2 | Gate translation, explicit register size, forward Clifford conjugation, signed stabilizers, and any logical operators |
| Pair 3 | Noise models, enumeration, degeneracy-aware harmful-error classification, and code metrics |
| External integration provider | Compose the owning pairs' functions behind `evaluate(candidate)` |
| Pair 4 | Semantic candidates, generation, mutation, search, ranking, budgets, archives, and exports |
| Pair 5 | Token vocabulary, special tokens, padding, tensors, and transformer training |

Pair 4 never strips stabilizer signs, repairs matrices, simulates gates, samples
physical errors, builds a decoder, or assigns transformer token IDs.

The selected future scientific objective is
`sum_E p(E) * 1[E is in N(S) but not S]`, ignoring overall error phase. Stabilizer
errors are harmless and excluded, including nonidentity stabilizers. A nonzero
syndrome signals detection, not necessarily successful correction. This metric
does not equal decoder-dependent logical failure probability. Initial experiments
use `(n,k)=(4,2)`, followed by `(5,1)`, post-encoding depolarizing data noise
`p=0.01`, and ideal syndrome information. They remain pending real evaluators.

## Outputs and failure behavior

Each run directory contains:

| File | Contents |
| --- | --- |
| `manifest.json` | Schema, method, sampling/search settings, objective, PRNG derivation, and Python version |
| `history.jsonl` | Every assessed candidate, including rejections and duplicates, its parent evaluation index, edit, score/diagnostics, acceptance, and best-so-far value |
| `summary.json` | Completion status, evaluation count, best candidate, top-ten archive, error, and separate timing |
| `invocation.json` | CLI evaluator entrypoint and factory settings; at the benchmark root for benchmark runs |

Benchmark roots also contain per-seed run directories, a benchmark manifest and
summary, `best_so_far.csv`, and `report.html`. The summary includes final-score
median/IQR, CNOT/gate counts, runtime, paired raw differences, and an advantage
value whose positive sign favors evolution. Only complete seed pairs enter the
comparison statistics. Quartiles use linear interpolation.

Histories and manifests reproduce byte-for-byte in the pinned environment.
Timing and output-directory names are intentionally not part of that guarantee.
The manifests distinguish synthetic from scientific objectives. Search history
contains semantic gate records for Pair 5 to tokenize, including low-scoring and
rejected proposals; no training labels are inferred beyond the supplied metric.

Missing factories, evaluation exceptions, invalid/nonfinite results, and changed
objective metadata produce failed runs with diagnostic messages and retained
partial histories/best candidates. CLI failures exit nonzero. Benchmarks stop
on their first failed run and retain earlier outputs. Nothing silently falls
back to the demo or fabricates a numeric failure score. Process termination or
power loss during a single run is not checkpointed; histories are exported when
that run returns, including when it returns an evaluator failure.

## Validation and remaining dependencies

The automated suite covers gate/operand distributions, empty and capped
sequences, all edit positions, replacement conditioning, reproducibility,
lexicographic ranking, neutral acceptance, exact budgets, archive uniqueness,
failure retention, metadata consistency, exports, external factory loading, and
paired benchmark statistics. Fixed-seed statistical checks have broad,
predeclared tolerances; they detect gross bias rather than prove uniformity.

These scientific integration checks are **pending dependencies**, not tests that
have passed against mocked quantum calculations:

| Check | Expected owner-provided evidence |
| --- | --- |
| Code versus state | Pair 2 returns exactly `n-k` signed code generators; Pair 1 confirms commutation and GF(2) independence |
| Sign/order fixture | Chronological H, S, S, H maps initial `+Z` to `-Z` |
| Degeneracy fixture | For `S=<Z_0>` on two qubits, identity and `Z_0` are harmless; `X_1` is harmful undetected |
| Syndrome collisions | `X_0` and `X_0 X_1` have the same nonzero syndrome but differ logically for that fixture |
| Trusted code fixtures | Pair 3 reports distances 2 and 3 for independently trusted `[[4,2,2]]` and `[[5,1,3]]` descriptors/encoders |

Do not require stochastic discovery of a target code, or convergence to the demo
target, as a software correctness condition. For scientific comparisons, use
separate pilot/tuning seeds, then freeze the evaluator and budget before running
the 30 paired seeds. Compare objective curves and paired outcomes, not only the
best seed. Random search may win. Demo results establish only search mechanics.

Crossover, populations, canonical code-equivalence deduplication, circuit
simplification, restricted hardware connectivity, sampled scoring, decoder
optimization, and transformer-guided proposals are deferred.

## Research basis

- [Perplexity research report](../perplexity_research.md)
- [Gemini research report](../gemini_research.md)
- [Tandeitnik and Guerreiro, *Evolving Quantum Circuits*, §II.2](https://arxiv.org/html/2210.05058v1#S2.SS2)
- [Webster and Browne, *Engineering Quantum Error Correction Codes Using Evolutionary Algorithms*](https://arxiv.org/abs/2409.13017)
- [Gottesman, *Stabilizer Codes and Quantum Error Correction*, Chapters 3–4](https://arxiv.org/html/quant-ph/9705052v1)

The single-parent search, equal legal mutation weights, hard cap, tie-breakers,
and benchmark settings are project defaults, not claimed optima or reproductions
of a published experiment.
