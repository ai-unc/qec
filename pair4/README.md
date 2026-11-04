# Pair 4: real stabilizer-code search

Run from the repository root with the project environment (`.venv/bin/python`).
Requires NumPy, Stim, and JAX (JAX is imported by Pair 3's syndrome module).
The unused `evolution_prototype` is independent and is not imported or tested.

```bash
.venv/bin/python pair4/search.py --method both
.venv/bin/python -m pair4.search --method evolution --trials 1000 --seed 42
.venv/bin/python pair4/search.py --seed-mode custom --seed-generators ZZI IZZ
.venv/bin/python pair4/search.py --method both --seed-mode random --seed-depth 12
.venv/bin/python pair4/search.py --validator-source original
.venv/bin/python -m unittest discover -s pair4/tests -v
```

Use `--help` for all options. Defaults are random search, 1,000 attempts, depth
12, evolutionary maximum depth 24, seed 42, canonical starting generators, local
validator/evaluator, and independent bit-flip noise with probability 0.05.
`--output-dir` selects the output folder (default: Pair 4). Runs overwrite their
named CSV/JSON outputs. A single method writes `results.csv` and `results.json`;
`both` writes `results_random.*` and `results_evolution.*`.

## Starting codes

The two starting stabilizers constrain the +1 eigenspace of two independent,
commuting operators on three qubits, leaving one logical qubit. `ZII, IZI`
means the initial code space is `|00>` tensored with an arbitrary logical input.
Pair 2 conjugates these operators through each candidate Clifford circuit,
preserving their signs, independence, and commutation. Three independent
stabilizers would describe one state rather than a code encoding one qubit.

`--seed-mode custom` requires exactly two valid three-qubit strings. Other seed
modes reject `--seed-generators`. Signs are optional, for example `--seed-generators -XII +IXI`.

Random mode transforms the canonical seed through `--seed-depth` Clifford gates
using a separate deterministic stream derived from `"<seed>:starting-code"`.
Both search methods share that starting code. This changes finite-budget
exploration, not the set reachable by arbitrary Clifford circuits. Starting
from the repetition code already supplies the 0.725% target; initialization
performance is reported separately from search performance.

## Methods and accounting

Random search samples independent circuits of exactly `--depth` gates. Evolution
starts with a random circuit, then replaces, inserts, or removes one gate from
the incumbent. It accepts improvements and ties within 1e-12, rejects worse
children, and retains the best visited code separately. Empty circuits permit
insertion only; circuits at the cap permit replacement/removal only. Replacement
selects a different legal instruction uniformly. These simple choices are
baselines, not a claim that evolution outperforms random search.

Each proposal consumes one attempt, including initialization, rejected codes,
and duplicate stabilizer spaces. Duplicate spaces share cached scores; signs
are ignored for error-equivalence caching, not during Clifford simulation.
The starting code is evaluated once as a diagnostic outside the attempt budget
and the reported unique candidate evaluation count. Results rank distinct
visited spaces by logical error rate; ties retain discovery order.
Random search uses `random.Random(seed)`; evolution uses
`random.Random("<seed>:evolution")`. Reproducibility assumes the same Python and
dependency versions. CSV rows include backend/starting-code provenance; JSON
also includes settings, counts, preparation circuit, and initialization results.
Candidate circuits act on the exported starting generators; random preparation
circuits act on the canonical generators first.

## Backends and scientific assumptions

`--validator-source local` uses `local_validation.py`, a corrected adaptation of
`pair1/validate_code.py` and `pair1/gen_ind.py`: all pairwise commutation checks
and full GF(2) rank replace the originals' incomplete checks. `original` calls
the untouched Pair 1 implementation and may reject valid candidates. Starting
codes always undergo local structural validation before search.

`--evaluator-source local` uses the exact evaluator moved out of search.py. It
calls original Pair 3 noise/syndrome functions and groups physical errors by
syndrome and stabilizer coset before maximum-likelihood recovery. It assumes
independent post-encoding data errors, ideal syndrome information, and an
optimal recovery chosen for each code. The existing Pair 3 individual-error
lookup decoder is separate. No gate or measurement noise is modeled.

`--evaluator-source original` requires `pair3/evaluate.py` exposing
`evaluate_code(generators, noise_model, p) -> float`. That interface is currently
absent, so selection fails explicitly before searching. No fallback is used.
Original Pair 2's Stim conjugation is always used. Collaborator files are never
modified. The repetition-code diagnostic at p=0.05 uses the local reference
calculation independently of the selected candidate evaluator.

`search_codes(trials=1000, depth=12, p=0.05, seed=42, **options)` retains the
ranked-list interface. `run_search(...)` additionally returns settings, counts,
and initialization metadata. Options include `method`, `max_depth`,
`seed_mode`, `seed_generators`, `seed_depth`, `validator_source`, and
`evaluator_source`. `transform_generators`, `valid_code`, and
`exact_logical_error_rate` remain available. For signed custom seeds:

```python
from pair4.search import search_codes
results = search_codes(seed_mode='custom', seed_generators=['-XII', '+IXI'])
```

The canonical fixed-seed regressions reach 0.725% with both methods. This is a
regression fixture, not a guarantee for arbitrary seeds or budgets. The
repetition-code reference is `3*p**2*(1-p) + p**3`; search target attainment is
reported independently and never inferred from the reference check.
