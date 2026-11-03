# Pair 4 explained visually

Pair 4 searches for useful **Clifford encoder circuits**: ordered sequences of
H, S, and CNOT gates. It proposes circuits, asks an evaluator how good they are,
and decides which ones to retain or explore next.

This guide explains the implementation in this folder. Diagrams labeled
**future integration** describe the intended connection to the other pairs;
that physics pipeline is not implemented inside Pair 4.

> **Current status:** generation, mutation, both searches, exports, and
> benchmarking work. The bundled evaluator measures similarity to a fixed gate
> sequence. Its scores are synthetic and have no quantum-performance meaning.
> Real code-quality experiments await the other pairs' interfaces.

Mermaid diagrams render in Markdown viewers with Mermaid support, including
GitHub. If you see diagram source instead, open a Mermaid-capable preview.

## Reading map

- [1. The whole project](#1-the-whole-project)
- [2. What a candidate contains](#2-what-a-candidate-contains)
- [3. How a circuit specifies a code](#3-how-a-circuit-specifies-a-code)
- [4. Random generation](#4-random-generation)
- [5. The three mutations](#5-the-three-mutations)
- [6. Random search](#6-random-search)
- [7. Evolutionary search](#7-evolutionary-search)
- [8. How ranking works](#8-how-ranking-works)
- [9. Parent, best result, archive, and history](#9-parent-best-result-archive-and-history)
- [10. What the evaluator measures](#10-what-the-evaluator-measures)
- [11. The evaluator connection](#11-the-evaluator-connection)
- [12. Reproducibility](#12-reproducibility)
- [13. Fair benchmarking](#13-fair-benchmarking)
- [14. Output files](#14-output-files)
- [15. Failures and safeguards](#15-failures-and-safeguards)
- [16. Where the code lives](#16-where-the-code-lives)
- [17. Try it yourself](#17-try-it-yourself)
- [18. Tests, limitations, and next steps](#18-tests-limitations-and-next-steps)

## 1. The whole project

Pair 4 asks, **“Which circuit should we try next?”** The other pairs provide the
tools that interpret and evaluate those circuits.

```mermaid
flowchart TB
    P4["Pair 4: Search algorithms<br/>Generate, mutate, rank, and record candidates"]
    P2["Pair 2: Quantum simulation<br/>Clifford transformations of Pauli operators"]
    P1["Pair 1: Mathematical foundations<br/>Binary Paulis, commutation, GF(2) independence"]
    P3["Pair 3: Error evaluation<br/>Noise models and code-quality metrics"]
    P5["Pair 5: Transformer<br/>Tokenization, training, sequence generation"]
    Bridge["Future external evaluator<br/>Composes the owning pairs' tools"]
    P4 -->|"Candidate circuit"| Bridge
    Bridge --> P2
    Bridge --> P1
    Bridge --> P3
    Bridge -->|"Named metric and score"| P4
    P4 -->|"Semantic gates and scored histories"| P5
```

This shows ownership, not a claim that the connections already exist. Pair 4's
production code imports no Stim, NumPy, JAX, or PyTorch. It consumes an evaluator
interface instead of implementing another pair's subsystem.

## 2. What a candidate contains

A **candidate** is one proposed encoder:

| Field | Meaning | Example |
| --- | --- | --- |
| `n_qubits` | Number of physical qubits | `4` |
| `k_logical` | Number of logical qubits to encode | `2` |
| `gates` | Instructions in chronological application order | H, then CNOT, then S |

Physical qubits are individual wires; logical qubits are the encoded information.
Code notation `[[n,k]]` records those numbers. When distance is known, it adds
a third number: `[[n,k,d]]`. [Gottesman, §3.2](https://arxiv.org/html/quant-ph/9705052v1)

A complete exported candidate looks like this:

```json
{
  "schema_version": 1,
  "n_qubits": 4,
  "k_logical": 2,
  "gates": [
    {"op": "H", "qubits": [0]},
    {"op": "CNOT", "qubits": [0, 2]},
    {"op": "S", "qubits": [1]}
  ]
}
```

```mermaid
flowchart LR
    A["Start with four wires"]
    B["Gate 0: H on wire 0"]
    C["Gate 1: CNOT<br/>Control 0, target 2"]
    D["Gate 2: S on wire 1"]
    E["End of candidate encoder"]
    A --> B --> C --> D --> E
```

This illustrates **application order**, not a quantum circuit drawing with one
horizontal line per qubit. Indices start at zero. CNOT operands always mean
`(control, target)`. The register has four wires even though wire 3 is unused.

An empty sequence retains its register size and represents the identity encoder.
The Python records are immutable: a mutation creates a child without changing
its parent.

## 3. How a circuit specifies a code

Our convention puts the `n-k` ancillas, initialized to zero, first. The last `k`
wires hold arbitrary logical inputs. For `n=4, k=2`:

```mermaid
flowchart LR
    subgraph Inputs["Before encoding"]
        A0["Wire 0: zero-state ancilla"]
        A1["Wire 1: zero-state ancilla"]
        L0["Wire 2: logical input 0"]
        L1["Wire 3: logical input 1"]
    end
    U["Candidate Clifford encoder U"]
    Output["Two logical qubits encoded<br/>in four physical qubits"]
    A0 --> U
    A1 --> U
    L0 --> U
    L1 --> U
    U --> Output
```

A stabilizer generator specifies a codespace constraint. The ancillas initially
give `+ZIII` and `+IZII`. Pair 2 will conjugate each check through the circuit:
`P → U P U†`. This preserves commutation and independence; signs must be retained.
Exactly `n-k` independent checks leave a space of dimension `2^k`. Using all
`n` checks of an all-zero state would instead select one stabilizer state.
[Gottesman, Chapters 3–4](https://arxiv.org/html/quant-ph/9705052v1)

Thus legal candidates preserve a valid code description under this convention.
They can still have poor protection. Pair 4 proposes circuits; the external
tools determine their mathematical description and quality.

## 4. Random generation

Generation chooses a length, then chooses each instruction:

```mermaid
flowchart TD
    Config["Read n, k, positive maximum length, and random stream"]
    Length["Choose L uniformly from 0 through maximum length"]
    Remaining{"Any of the L gates left to draw?"}
    Type["Choose a legal gate type uniformly"]
    Two{"CNOT?"}
    Single["Choose one wire uniformly"]
    Double["Choose an ordered pair of distinct wires uniformly"]
    Append["Append gate to sequence"]
    Done["Return candidate with explicit n and k"]
    Config --> Length --> Remaining
    Remaining -->|"Yes"| Type --> Two
    Two -->|"No: H or S"| Single --> Append
    Two -->|"Yes"| Double --> Append
    Append --> Remaining
    Remaining -->|"No"| Done
```

The default maximum length is `8*n`. Four physical qubits therefore allow
starting lengths from 0 to 32: 33 equally likely lengths.

For `n >= 2`, H, S, and CNOT each have probability `1/3`. For `n=1`, only H and S
are available, each with probability `1/2`.

### Why choose the gate type first?

On four wires there are four H instructions, four S instructions, and twelve
directed CNOT instructions. Uniformly sampling all twenty instructions would
give CNOT probability `12/20 = 60%`. Choosing the type first keeps it at `1/3`.

This is a deliberate distribution. It does **not** uniformly sample Clifford
transformations or stabilizer codes. Different sequences can describe the same
transformation or code.

### How are distinct CNOT operands guaranteed?

The sampler picks a control, then selects among the other `n-1` wires. With
`n=5` and control 2, possible targets are `0, 1, 3, 4`. It maps a draw from
`0 ... n-2` onto that list by skipping the control index. No retry is necessary.

## 5. The three mutations

Mutation changes a candidate's description by one edit:

```mermaid
flowchart TB
    Parent["Parent<br/>H(0), CNOT(0,2), S(1)"]
    Replace["Replace middle gate<br/>H(0), H(2), S(1)"]
    Insert["Insert one gate<br/>H(0), S(2), CNOT(0,2), S(1)"]
    Delete["Delete middle gate<br/>H(0), S(1)"]
    Parent -->|"Replace"| Replace
    Parent -->|"Insert"| Insert
    Parent -->|"Delete"| Delete
```

- **Replace:** choose an existing position and a different legal instruction.
  Changing operands counts: `H(0)` can become `H(2)`.
- **Insert:** choose uniformly among all `L+1` gaps, including both endpoints.
- **Delete:** choose uniformly among the `L` existing gates.

The parent remains unchanged. Replacement uses a finite weighted set rather than
retrying until it draws something different. Its weights preserve the original
gate distribution conditioned on excluding the old instruction.

### Boundary rules

```mermaid
flowchart TD
    Parent["Read parent length L and positive cap M"]
    Empty{"L equals 0?"}
    Full{"L equals M?"}
    Insert["Insert only<br/>Probability 1"]
    AtCap["Replace or delete<br/>Probability 1/2 each"]
    Interior["Replace, insert, or delete<br/>Probability 1/3 each"]
    Child["Return new candidate and edit record"]
    Parent --> Empty
    Empty -->|"Yes"| Insert --> Child
    Empty -->|"No"| Full
    Full -->|"Yes"| AtCap --> Child
    Full -->|"No"| Interior --> Child
```

Length stays between zero and the cap. Configuration rejects a zero cap.
Equal interior insertion/deletion probabilities do not make the walk over codes
unbiased: boundaries, selection, and redundant encoders still influence it.

## 6. Random search

Random search asks, **“What happens if we keep trying fresh circuits?”** Scores
determine what is remembered, but do not guide the next draw.

```mermaid
flowchart TD
    Start["Validate configuration and evaluator"]
    Budget{"Any evaluation budget left?"}
    Draw["Draw a fresh random candidate"]
    Evaluate["Evaluate candidate<br/>One evaluator call"]
    Compare["Compare full ranking with current best"]
    Record["Update best and archive as needed<br/>Record evaluation"]
    Finish["Export results"]
    Start --> Budget
    Budget -->|"Yes"| Draw --> Evaluate --> Compare --> Record --> Budget
    Budget -->|"No"| Finish
```

The first draw uses the `initial` stream. Later draws use `proposals`. This
diagram assumes evaluation succeeds; failures are covered in section 15.

This baseline lets us test whether building on previous candidates actually
helps compared with independent sampling.

## 7. Evolutionary search

The implemented method is called **`(1+1)`**: one current parent is compared with
one proposed child at each selection step.

```mermaid
flowchart TD
    Start["Generate and evaluate one initial parent"]
    Budget{"Any evaluation budget left?"}
    Edit["Apply one legal mutation to current parent"]
    Evaluate["Evaluate child"]
    Better{"Child has a better full ranking?"}
    Equal{"Full ranking exactly equal?"}
    Coin{"Seeded draw below 0.5?"}
    Accept["Child becomes parent"]
    Reject["Keep existing parent"]
    Record["Update best and archive separately<br/>Record child, decision, and lineage"]
    Finish["Export results"]
    Start --> Budget
    Budget -->|"Yes"| Edit --> Evaluate --> Better
    Better -->|"Yes"| Accept
    Better -->|"No"| Equal
    Equal -->|"Yes"| Coin
    Equal -->|"No: child is worse"| Reject
    Coin -->|"Yes"| Accept
    Coin -->|"No"| Reject
    Accept --> Record
    Reject --> Record
    Record --> Budget
    Budget -->|"No"| Finish
```

Future mutations build on the accepted parent. Occasionally accepting an equally
ranked child permits movement across a **plateau**, where different candidates
have the same ranking. Equality includes structural costs, not just the score.

This is a mutation-only baseline with no population, crossover, or automatic
restart. It can get stuck. Gate-sequence evolution has published precedent, but
this small loop is our project choice rather than a reproduction of that paper's
algorithm. [Tandeitnik and Guerreiro, §II.2](https://arxiv.org/html/2210.05058v1#S2.SS2)

## 8. How ranking works

Both methods apply the same ordering:

```mermaid
flowchart TD
    Start["Compare candidates A and B"]
    Primary{"Primary scores different?"}
    Objective["Prefer lower for minimize<br/>Prefer higher for maximize"]
    Cnots{"CNOT counts different?"}
    FewerC["Prefer fewer CNOTs"]
    Length{"Total gate counts different?"}
    FewerG["Prefer fewer total gates"]
    Tie["Exactly equal full ranking"]
    Start --> Primary
    Primary -->|"Yes"| Objective
    Primary -->|"No"| Cnots
    Cnots -->|"Yes"| FewerC
    Cnots -->|"No"| Length
    Length -->|"Yes"| FewerG
    Length -->|"No"| Tie
```

For a probability to minimize, consider these **illustrative**, unmeasured scores:

| Candidate | Probability | CNOTs | Total gates | Result |
| --- | --- | --- | --- | --- |
| A | 0.020 | 1 | 4 | Loses to B on probability |
| B | 0.019 | 12 | 20 | Beats A despite its larger circuit |
| C | 0.019 | 9 | 30 | Beats B on CNOT count |
| D | 0.019 | 9 | 25 | Beats C on total gate count |

This is **lexicographic** ordering: later criteria matter only when earlier ones
are exactly equal. Scores are not rounded; no weighted penalty lets gate savings
outweigh worse protection.

Internally, smaller ranking tuples always win:

```text
Minimize: ( value, CNOT count, total gate count)
Maximize: (-value, CNOT count, total gate count)
```

## 9. Parent, best result, archive, and history

These are different records:

| Record | Question it answers |
| --- | --- |
| Current parent | Where does the next evolutionary mutation start? |
| Best result | What is the best-ranked candidate observed? |
| Archive | What are the ten best distinct gate sequences retained? |
| History | What happened on every evaluation? |

```mermaid
flowchart LR
    Child["Evaluated child"]
    Select["Selection decision"]
    Parent["Current parent<br/>Can move on a full tie"]
    Best["Best result<br/>Changes on strict improvement"]
    Archive["Archive<br/>Up to ten distinct sequences"]
    History["History<br/>Every assessed proposal and decision"]
    Child --> Select --> Parent
    Child --> Best
    Child --> Archive
    Child --> History
    Select --> History
```

After a neutral move from A to B, B can become the parent while A remains the
recorded best: their ranks are equal, so no strict improvement occurred.

A rejected child can still enter the archive if it ranks among the retained top
ten. Deduplication compares exact candidate sequences, not equivalent codes.
The archive is **not a cache**: drawing the same circuit again still consumes
a call and produces another history record.

## 10. What the evaluator measures

### The current demo

For multiple qubits, the fixed target sequence is:

```text
H(0), CNOT(0,1), S(1), H(1), CNOT(1,0), S(0)
```

The demo score is negative **Levenshtein edit distance** to this target. Each full
gate instruction is one item; it compares gates, not characters. An insertion,
deletion, or replacement costs one edit.

| Candidate | Distance | Demo score |
| --- | --- | --- |
| Exact target | 0 | 0 |
| Target plus one extra gate | 1 | -1 |
| Empty sequence | 6 | -6 |

The demo maximizes the score; `0` is optimal. One-qubit diagnostics use a separate
six-gate H/S-only target. A cap below six makes the full target unreachable, which
is allowed: successful execution does not require finding it.

```mermaid
flowchart LR
    Candidate["Candidate gate sequence"]
    Choice{"Explicit evaluator selection"}
    Demo["demo: sequence edit distance<br/>Implemented and synthetic"]
    Physics["External code evaluator<br/>Future scientific integration"]
    Result["Named objective, direction, and numeric value"]
    Candidate --> Choice
    Choice -->|"demo"| Demo --> Result
    Choice -->|"module:factory"| Physics --> Result
```

### The future scientific objective

We selected **harmful undetected-error probability**. This conceptual
classification belongs in the external evaluator:

```mermaid
flowchart TD
    Error["Pauli error E"]
    Syndrome{"Commutes with every stabilizer check?"}
    Detected["Nonzero syndrome<br/>Detected; correction still depends on decoding"]
    Harmless{"Equivalent to a stabilizer element<br/>up to global phase?"}
    Safe["Harmless on encoded information<br/>Exclude its probability"]
    Harmful["Zero syndrome and nontrivial logical action<br/>Include its probability"]
    Error --> Syndrome
    Syndrome -->|"No"| Detected
    Syndrome -->|"Yes: zero syndrome"| Harmless
    Harmless -->|"Yes"| Safe
    Harmless -->|"No"| Harmful
```

Ignoring overall error phase, the objective sums `p(E)` over Pauli operators
commuting with every check but outside the stabilizer group. Identity and
nonidentity stabilizer errors are excluded. This is decoder-independent; it
does not measure the outcome of recovering from detected errors.
[Webster and Browne, §II-C](https://arxiv.org/pdf/2409.13017)

| Metric | Question | Better direction |
| --- | --- | --- |
| Distance | What is the smallest weight of a harmful undetected Pauli? | Higher |
| Harmful undetected probability | How likely are harmful errors that escape detection under a specified noise model? | Lower |
| Decoder-dependent logical failure | How likely is incorrect logical information after a specified recovery? | Lower |

Weight counts affected physical qubits. Detection alone does not establish
correction. These distinctions follow the stabilizer error classification.
[Gottesman, §3.2](https://arxiv.org/html/quant-ph/9705052v1)

Pair 4 consumes the chosen metric without computing quantum errors. The demo's
instruction edit distance is entirely separate from quantum-code distance.

## 11. The evaluator connection

An **interface** is an agreement on inputs and outputs. An evaluator exposes
`objective` and `evaluate(candidate)`:

| Record | Contents |
| --- | --- |
| `ObjectiveSpec` | Metric name, direction, version, parameters, exactness, synthetic flag |
| `EvaluationResult` | Finite numeric value and optional diagnostics |

Objective metadata must stay fixed within a run and across a benchmark. Changing
noise settings midway would make scores incomparable, so metadata changes cause
an explicit failure. Version 1 requires an exact, deterministic evaluator. Here
“exact” describes the evaluation method, not arbitrary-precision arithmetic:
results are Python floats.

The next diagram is **future integration**, not implemented orchestration. The
external provider composes the owning pairs' actual APIs:

```mermaid
sequenceDiagram
    participant Search as Pair 4 search
    participant Bridge as External evaluator
    participant Sim as Pair 2 simulation
    participant Math as Pair 1 mathematics
    participant Noise as Pair 3 evaluation
    Search->>Bridge: evaluate(candidate)
    Bridge->>Sim: Interpret encoder with ancilla convention
    Sim-->>Bridge: Signed code generators
    Bridge->>Math: Convert and validate code description
    Math-->>Bridge: Mathematical representation and checks
    Bridge->>Noise: Score under fixed noise model
    Noise-->>Bridge: Harmful undetected-error probability
    Bridge-->>Search: EvaluationResult(value, diagnostics)
    Note over Search,Bridge: Pair 4 receives a score without implementing these subsystems
```

For the CLI, `module:factory` means “import this module and call this factory
function.” The function receives a configuration dictionary and returns a fresh
evaluator instance. Each benchmark run gets its own instance.

The proposed experiment uses `(n,k)=(4,2)`, then `(5,1)`, depolarizing data noise
with `p=0.01` after encoding, and ideal syndrome information.
[examples/depolarizing.json](examples/depolarizing.json) records that request;
an external factory still needs to interpret it and report its effective settings.

## 12. Reproducibility

A **seed** makes random choices repeatable in a pinned software environment.
We separate different kinds of choices into named streams:

```mermaid
flowchart LR
    Seed["Master seed: for example, 42"]
    Derive["SHA-256 of seed plus stream label"]
    Initial["initial<br/>Shared first candidate"]
    Proposals["proposals<br/>Fresh random-search candidates"]
    Mutation["mutation<br/>Evolutionary edits"]
    Acceptance["acceptance<br/>Neutral selection decisions"]
    Seed --> Derive
    Derive --> Initial
    Derive --> Proposals
    Derive --> Mutation
    Derive --> Acceptance
```

Each uses `random.Random`, initialized from the big-endian integer SHA-256 digest
of UTF-8 `pair4:v1:<seed>:<label>`. Advancing one stream does not advance another.

Both methods can therefore share their first circuit while proposing subsequent
circuits differently. The manifest records Python and configuration; the history
records every assessed circuit. Pin Python for byte-for-byte replay of histories
and manifests. Timing and output directory names vary normally.

## 13. Fair benchmarking

The question is: **With the same evaluation budget, does building on the current
parent help?**

```mermaid
flowchart TB
    Seed["One paired seed"]
    Initial["Same initial circuit for both methods"]
    Random["Random search<br/>1,000 calls including initialization"]
    Evolution["One-plus-one search<br/>1,000 calls including initialization"]
    Pair["Compare final scores and best-so-far curves"]
    Repeat["Repeat for seeds 0 through 29"]
    Report["Report medians, IQRs, costs, time, and paired differences"]
    Seed --> Initial
    Initial --> Random --> Pair
    Initial --> Evolution --> Pair
    Pair --> Repeat --> Report
```

The default successful comparison performs:

```text
30 seeds × 2 methods × 1,000 calls = 60,000 evaluator calls
```

Initialization, repeated candidates, and failed evaluator calls all consume
budget. A failed run stops that benchmark. Counting calls is clearer than
counting generations, whose costs can differ between algorithms.

| Reported quantity | Meaning |
| --- | --- |
| Best-so-far curve | Best primary score reached at each call count |
| Median | Middle outcome across seeds |
| Interquartile range, or IQR | Range from the 25th to 75th percentile |
| Structural costs | CNOT and total gate counts of selected best candidates |
| Timing | Measured separately; equal call counts can take different time |
| Paired differences | Evolutionary result compared with random result for each seed |

A positive `evolution_advantage` always favors evolution. It is
`random_score - evolutionary_score` for minimization and the opposite difference
for maximization. Only **complete seed pairs** enter comparison statistics.

Use separate pilot/tuning seeds before freezing final settings. Synthetic demo
results cannot establish better QEC performance, even if evolution wins.

## 14. Output files

Every assessed proposal creates a history record, including duplicates,
rejections, and evaluator failures. It stores the candidate, result when
available, parent evaluation index, edit, acceptance, and best primary value.

```mermaid
flowchart TB
    Run["One returned search run"]
    Manifest["manifest.json<br/>Configuration, objective, Python version"]
    History["history.jsonl<br/>One record per evaluator call"]
    Summary["summary.json<br/>Status, best, archive, error, timing"]
    CLI["CLI invocation"]
    Invocation["invocation.json<br/>Factory and supplied settings"]
    Benchmark["Benchmark root"]
    Runs["Per-seed folders for both methods"]
    Curves["best_so_far.csv<br/>Median and IQR curve data"]
    Report["report.html<br/>Standalone visual comparison"]
    Run --> Manifest
    Run --> History
    Run --> Summary
    CLI --> Invocation
    Benchmark --> Runs
    Benchmark --> Curves
    Benchmark --> Report
```

**JSONL** means one JSON object per line. Evaluation indices start at 1. A child's
`parent_index` points to the accepted candidate from which it was mutated;
that need not be the immediately preceding evaluation.

Search commands put `invocation.json` inside their run folder. Benchmarks put it
at the benchmark root, with benchmark-level manifest and summary files; per-seed
folders contain the individual run files.

These semantic gate lists can be handed to Pair 5. Token IDs, special tokens,
padding, tensors, and training-example design remain Pair 5's responsibility.

## 15. Failures and safeguards

Failure is recorded explicitly, rather than assigned a fabricated numeric score.

```mermaid
flowchart TD
    Request["Start with an explicit evaluator"]
    Ready{"Factory and objective usable?"}
    SetupFailure["Failed run with zero calls<br/>Record setup error"]
    Call["Invoke evaluator for a candidate"]
    Valid{"Finite result and unchanged objective metadata?"}
    Success["Rank candidate and continue within budget"]
    CallFailure["Failed run<br/>Count this call and retain earlier best and history"]
    Export["Export returned results<br/>CLI exits nonzero on failure"]
    Request --> Ready
    Ready -->|"No"| SetupFailure --> Export
    Ready -->|"Yes"| Call --> Valid
    Valid -->|"Yes"| Success
    Valid -->|"No or exception"| CallFailure --> Export
```

Configuration rejects invalid dimensions, nonpositive caps/budgets, invalid gate
operands, and conflicting CLI/evaluator dimensions. Scientific search requires
`1 <= k < n`; diagnostic candidates and synthetic demos can use `k=0` or `k=n`.

Missing providers never silently select the demo. Existing output locations are
not overwritten. A benchmark stops on its first failed run and retains earlier
outputs.

**Checkpoint limitation:** histories are exported when a run returns, including
after evaluator exceptions. Abrupt termination or power loss during an individual
run is not checkpointed continuously.

## 16. Where the code lives

```mermaid
flowchart TB
    CLI["__main__.py<br/>Commands and factory configuration"]
    Models["models.py<br/>Immutable candidates and configurations"]
    Operators["operators.py<br/>Sampling, mutation, random streams"]
    Evaluation["evaluation.py<br/>Evaluator contract and synthetic demo"]
    Search["search.py<br/>Loops, ranking, history, archive"]
    Benchmark["benchmark.py<br/>Paired runs, statistics, visual report"]
    Export["export.py<br/>JSON and JSONL output"]
    CLI --> Operators
    CLI --> Benchmark
    CLI --> Evaluation
    CLI --> Export
    Benchmark --> Search
    Benchmark --> Export
    Search --> Operators
    Search --> Evaluation
    Operators --> Models
    Evaluation --> Models
```

Arrows show main use relationships, not an exhaustive import graph.

| File | Start here when asking… |
| --- | --- |
| [models.py](models.py) | What is a candidate, configuration, or edit record? |
| [operators.py](operators.py) | How are gates selected and mutated? |
| [evaluation.py](evaluation.py) | What does evaluation return, and how does the demo score? |
| [search.py](search.py) | How does selection work, and what is retained? |
| [benchmark.py](benchmark.py) | How are methods compared and reports generated? |
| [export.py](export.py) | How are results written? |
| [CLI module](__main__.py) | How are command-line arguments interpreted? |
| [Package exports](__init__.py) | What can be imported directly from `pair4`? |
| [README.md](README.md) | What is the compact usage and integration reference? |

## 17. Try it yourself

Run from the repository root with Python 3.10 or newer. Pair 4 needs no
third-party dependencies.

**Inspect generated gate lists:**

```bash
python3 -m pair4 sample --n 4 --k 2 --seed 42 --count 2
```

This prints JSON; generation alone does not assess code quality.

**Run evolutionary search with synthetic scores:**

```bash
python3 -m pair4 search --method one_plus_one --evaluator demo --seed 42 --budget 100
```

Read `summary.json` for the best candidate and `history.jsonl` for the decisions.

**Run the baseline with the same initial candidate:**

```bash
python3 -m pair4 search --method random --evaluator demo --seed 42 --budget 100
```

**Generate a small visual comparison:**

```bash
python3 -m pair4 benchmark --evaluator demo --budget 100 --seeds 1000 1001
```

Open the printed `report.html` path in a browser. It contains synthetic labels,
primary-objective curves, IQR bands, and final-score and cost summaries.

**Run the default 30-seed comparison:**

```bash
python3 -m pair4 benchmark --evaluator demo
```

Output directories are created under `pair4/runs/` and ignored by Git.

**Future real-evaluator invocation:**

```bash
python3 -m pair4 benchmark --n 4 --k 2 --evaluator team_evaluator:make \
  --evaluator-config pair4/examples/depolarizing.json
```

`team_evaluator:make` is a **placeholder**, usable only once the team supplies
that module and factory. Until then it fails explicitly. The demo does not
accept `--evaluator-config`, preventing scientific-looking settings from being
silently ignored by a synthetic evaluator.

## 18. Tests, limitations, and next steps

The automated suite checks the machinery:

| Area | Examples |
| --- | --- |
| Candidate validity | Names, arity, wire ranges, distinct CNOT endpoints |
| Generation | Type/operand distributions, one-qubit behavior, length bounds |
| Mutation | End-gap insertion, deletion positions, distinct replacements, unchanged parents |
| Selection | Score priority, structural ties, neutral movement, monotonic best-so-far |
| Reproducibility | Same seeds reproduce histories despite unrelated intervening runs |
| Budgets/failures | Exact counts, partial best retention, nonfinite result rejection |
| Integration boundary | External factory loading and consistent objective metadata |
| Reporting | Exports, paired starts, quartiles, complete-pair statistics |

```bash
python3 -m unittest discover -s pair4/tests -v
```

These checks do not establish quantum-code performance. Pending scientific
integration checks include signed conjugation, exactly `n-k` independent checks,
harmless-versus-logical error classification, and trusted `[[4,2,2]]` and
`[[5,1,3]]` reference fixtures.

Length bounds limit accessible encoders, equivalent circuits can be evaluated
repeatedly, and a one-parent method can get stuck. Discovering the best possible
code or beating random search is not guaranteed.

When a real evaluator becomes available, the next experiment is a fixed-budget
comparison under the chosen noise model. Populations, crossover, circuit
simplification, hardware constraints, sampled scoring, decoder optimization,
and transformer-guided proposals remain later extensions.

### Research references

- [Original Pair 4 notes](../research.searchalgorithms.md)
- [Perplexity research](../perplexity_research.md)
- [Gemini research](../gemini_research.md)
- [Gottesman: Stabilizer Codes and Quantum Error Correction](https://arxiv.org/html/quant-ph/9705052v1)
- [Tandeitnik and Guerreiro: Evolving Quantum Circuits](https://arxiv.org/html/2210.05058v1)
- [Webster and Browne: Engineering Quantum Error Correction Codes Using Evolutionary Algorithms](https://arxiv.org/abs/2409.13017)

The implementation is the source of truth for current behavior. Values such as
`8*n`, equal legal mutation weights, the tie-breakers, and 30 paired seeds are
project defaults rather than proven optimal parameters.
