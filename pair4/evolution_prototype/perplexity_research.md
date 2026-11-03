**Start Pair 4 with bounded H/S/CNOT encoder sequences, reproducible random search, and a mutation-only \((1+1)\) evolutionary baseline.** Keep simulation, stabilizer validation, and code evaluation behind the other pairs’ interfaces; until those dependencies exist, demonstrate search mechanics—not improved error correction.

**Evidence labels:** **Published** identifies source results; **Deduction** identifies mathematical consequences; **Proposed default** identifies a project recommendation, not an established optimum. Repository status is taken from your brief, not a new repository audit.

## Representation choice

| Approach | Advantages | Limitations | First-prototype fit |
|---|---|---|---|
| **H/S/CNOT encoder sequence** | Concrete circuit genotype; every legal sequence is Clifford; conjugating a valid initial stabilizer preserves validity. Circuit-level evolution has published precedent. [Tandeitnik–Guerreiro, §II.2](https://arxiv.org/html/2210.05058v1#S2.SS2) | Many sequences represent the same code; bounded lengths and sampling distributions bias the search. | **Recommended:** directly matches Pair 4’s generation/mutation responsibilities and Pair 2’s simulation boundary. |
| **Validity-preserving stabilizer encoding** | Searches code descriptions directly; a published binary encoding supports valid generation, mutation, and crossover. [Webster–Browne, abstract](https://arxiv.org/abs/2409.13017) | Requires a specified constrained representation—not arbitrary matrix entries—and does not itself provide an encoder-cost comparison. | Credible later alternative, particularly if code quality rather than encoder structure becomes the primary objective. |
| **Arbitrary stabilizer-row mutation** | Superficially simple | Can break commutation or independence; invertible row operations alone merely change the generating basis, not the code. [Gottesman, §§3.2, 3.4, 4.1](https://arxiv.org/html/quant-ph/9705052v1) | Reject. |

**Published:** Tandeitnik–Guerreiro report evolutionary advantages over random search for their particular objectives and experiments; those results do not establish an advantage for this prototype. [§III; §IV.2, Fig. 11 and Table 1](https://arxiv.org/html/2210.05058v1#S4.SS2)

**Published:** Webster–Browne report validity-preserving binary representations and optimization under biased noise; that supports considering structured code-space search, not unrestricted matrix mutation. [Abstract](https://arxiv.org/abs/2409.13017)

**Deduction:** Circuit search is the lower-integration-risk choice here. It avoids stabilizer repair/completion entirely, while retaining a concrete object that Pair 5 could later model.

## Code construction and signs

Let \(r=n-k\). Fix ancillas on wires \(0,\ldots,r-1\), logical inputs on the remaining wires, and start with

\[
S_0=\langle +Z_0,\ldots,+Z_{r-1}\rangle,\qquad
S(U)=\langle UZ_0U^\dagger,\ldots,UZ_{r-1}U^\dagger\rangle .
\]

**Deduction:** These \(r\) generators remain commuting and independent, and their group cannot acquire \(-I\), because unitary conjugation preserves products and is invertible. Thus every syntactically legal candidate defines an \([[n,k]]\) stabilizer code; this does **not** imply useful distance. [Gottesman, §§3.2, 4.2](https://arxiv.org/html/quant-ph/9705052v1)

- **Code versus state:** \(r\) independent constraints leave a \(2^{n-r}=2^k\)-dimensional codespace; \(n\) independent constraints leave one stabilizer state, with \(k=0\). [Gottesman’s lecture notes, p. 1] [perimeterinstitute](https://perimeterinstitute.ca/personal/dgottesman/CO639-2004/Lecture4.pdf)
- **Important trap:** Simulating \(U|0\rangle^{\otimes n}\) produces a particular encoded state, not a complete description of an arbitrary logical input. Do not treat all \(n\) state stabilizers as code checks. [Gottesman, §§3.2, 4.2](https://arxiv.org/html/quant-ph/9705052v1)
- **Logical frame:** If needed, Pair 2 can expose the conjugated \(X_j,Z_j\) on logical-input wires; Pair 4 should not derive them independently. [Gottesman, §§3.2, 4.2](https://arxiv.org/html/quant-ph/9705052v1)
- **Signs:** Retain each generator’s actual Hermitian sign, \(+1\) or \(-1\). Replacing a negative generator with its positive version changes the selected eigenspace; \(\pm i\) phases are not valid Hermitian stabilizer-generator signs. [Gottesman, §3.2](https://arxiv.org/html/quant-ph/9705052v1)
- **Phase-free calculations:** Commutation, binary independence, and Pauli-error equivalence can use phase-free representations, but those do not replace signed code descriptions. Agree on that separation with Pairs 1–3. [Gottesman, §3.4](https://arxiv.org/html/quant-ph/9705052v1)

**Proposed default:** Fix the input-wire convention rather than searching over logical-input assignments. Record gates in chronological application order, with \(U=G_{L-1}\cdots G_0\), and require Pair 2 to use forward conjugation \(P\mapsto UPU^\dagger\).

The 2022 paper explicitly notes that a circuit alone does not specify its code without an input convention; its broader codeword-selection procedure need not be reproduced here. [§IV.1 and Appendix B](https://arxiv.org/html/2210.05058v1)

## Sampling and bounded mutation

All choices below are **proposed defaults**, deliberately simple and configurable.

### Candidate and sampling rules

| Item | Proposed rule |
|---|---|
| Candidate | Explicit `n`, `k`, schema version, and ordered gates |
| Gate records | `H(q)`, `S(q)`, `CNOT(control, target)`; one elementary operation per record |
| Initial experiments | \((n,k)=(4,2)\), then \((5,1)\); fix each configuration across methods |
| Length | Sample \(L\) uniformly from integers \(0,\ldots,L_{\max}\), initially \(L_{\max}=8n\) |
| Gate type | For \(n\ge2\), choose H, S, CNOT with probabilities \(1/3,1/3,1/3\) |
| Single-qubit operand | Uniform over \(0,\ldots,n-1\) |
| CNOT operands | Uniform over ordered pairs with distinct control and target |
| Connectivity | All-to-all for the first experiment; record this assumption |
| One-qubit generator | Remove CNOT and renormalize to H/S probabilities \(1/2,1/2\) |
| Empty sequence | Valid identity encoder; preserve explicit `n` |
| Invalid dimensions | Reject \(n<1\) and invalid \(k\); reserve \(k=0\) and \(k=n\) for diagnostic fixtures |

Choose distinct CNOT operands without an unbounded rejection loop. If hardware restrictions are added later, sample from an explicit ordered list of permitted directed edges.

**Deduction:** Uniform gate types are not uniform elementary instructions: there are \(n\) H choices, \(n\) S choices, and \(n(n-1)\) CNOT choices. Neither rule gives uniform sampling over Clifford transformations or stabilizer codes; call this a **specified random-circuit distribution**, not “uniform random codes.”

Record the PRNG algorithm, seed, sampling configuration, software versions, candidate serialization, and lineage. Use separate reproducible streams for candidate generation, mutation/selection, and any evaluator randomness; saving generated candidates provides stronger replayability than saving seeds alone.

### Mutation rules

| Mutation | Proposed rule |
|---|---|
| Replace | Uniform existing position; draw a legal replacement conditioned on differing from the original instruction |
| Insert | Uniform among all \(L+1\) gaps, including both endpoints; sample one legal gate |
| Delete | Uniform existing position |
| Action distribution | Equal weight among currently legal actions |
| At \(L=0\) | Insert only |
| At \(L=L_{\max}\) | Replace/delete only |
| At \(L_{\max}=0\) | Explicit immutable diagnostic case; return “no legal mutation” without retrying |
| Parent handling | Return a new candidate; never modify the parent |

Condition replacement on the finite legal instruction set rather than retrying indefinitely. A changed instruction can still produce an equivalent circuit or code; that is not a mutation failure.

**Proposed default:** Apply exactly one edit per offspring. Keep a hard length cap and use shorter length only as a secondary ranking criterion—do not introduce an arbitrary weighted fitness penalty.

**Deduction:** Insert/delete mutations still create a nonuniform walk over lengths and codes, even when their interior action weights are equal. Log length, gate-type, and duplicate distributions rather than interpreting mutation as unbiased sampling.

### Defer crossover

**Recommendation:** Defer crossover until the mutation-only baseline and evaluator are validated.

**Deduction:** Splicing legal gate sequences on the same register still produces a Clifford encoder, so it does not require stabilizer-independence repair. Its first-prototype disadvantages are additional length-bound rules and another experimental variable; a useful segment in one circuit need not retain its effect in another context. [Circuit representation precedent: Tandeitnik–Guerreiro, §II.2](https://arxiv.org/html/2210.05058v1#S2.SS2)

Do not transfer the existing matrix-row crossover/completion procedure into this prototype.

## Fitness without overclaiming

Use phase-free Pauli representations below. Let \(W\) be the binary stabilizer row space and \(W^{\perp_s}\) its symplectic orthogonal space.

| Quantity | Meaning | What it does not establish |
|---|---|---|
| **Distance** | For \(k>0\), \(d=\min_{e\in W^{\perp_s}\setminus W}\operatorname{wt}(e)\): minimum weight of a nontrivial logical Pauli. [Gottesman, §§3.2, 3.4](https://arxiv.org/html/quant-ph/9705052v1) | Noise-weighted performance or a decoder’s failure rate |
| **Harmful undetected probability** | Under a specified Pauli channel, \(P_{\mathrm{harm}}=\sum_e p(e)\mathbf1[e\in W^{\perp_s}\setminus W]\). This is a noise-weighted deduction from the stabilizer error classification. [Gottesman, §3.2](https://arxiv.org/html/quant-ph/9705052v1) | Successful recovery of detected errors |
| **Decoder-dependent logical failure** | For a specified syndrome recovery \(R_s\), \(P_L=\sum_Ep(E)\mathbf1[R_{s(E)}E\not\sim S]\), where equivalence ignores global phase. [Gottesman, §§3.2, 5.2](https://arxiv.org/html/quant-ph/9705052v1) | Decoder-independent code quality |

Consequently:

- Identity is harmless, not evidence of correction capability.
- A nonidentity stabilizer error can be harmless despite zero syndrome.
- A zero-syndrome nontrivial logical error is harmful.
- A nonzero syndrome signals detection, not successful correction.
- Errors sharing a syndrome can require the same recovery when they differ by a stabilizer, or incompatible recoveries when they differ logically. [Gottesman, §3.2](https://arxiv.org/html/quant-ph/9705052v1)

**Proposed first scientific objective:** Ask Pair 3 to supply exact harmful undetected probability for a small, explicitly specified **post-encoding data-Pauli channel with ideal syndrome information**. Describe the result as undetected-error optimization—not decoder performance or fault-tolerant encoder optimization.

If Pair 3 supplies exact distance first, run a separately labeled distance experiment. Do not silently substitute metrics.

### Minimal evaluator contract

Pair 4 should consume an agreed interface, not implement its internals.

| Direction | Minimum contents |
|---|---|
| Request | Candidate ID; explicit \(n,k\); Pair 2’s signed code descriptor; metric ID; noise-model ID/version; evaluator configuration |
| Successful result | Candidate ID; metric ID/value; minimize/maximize direction; exact/estimated status; evaluator version |
| Coverage metadata | Full enumeration or truncation rule, included probability mass, and bound/uncertainty information where applicable |
| Failure result | Explicit unavailable/invalid/failed status with reason; no fabricated numeric fitness |
| Optional results | Diagnostic distance, logical-frame information, decoder ID, shot count, confidence interval |

**Deduction:** If Pair 3 enumerates only a subset of errors, summing harmful probability over that subset gives a partial contribution, not the full probability. Do not renormalize it and relabel it as unconditional performance; Pair 3 must specify its interpretation.

**Proposed ranking:** For exact probability results, minimize lexicographically `(harmful_probability, CNOT_count, total_gate_count)`. For a separate distance experiment, maximize distance first, then minimize those structural costs; do not compare results with different evaluator configurations.

### Before evaluation exists

Pair 4 can meaningfully demonstrate:

- Reproducible generation, serialization, mutation, and bounded termination
- Random-search orchestration and evolutionary selection using clearly labeled canned test results
- Correct budget accounting, lineage, failure handling, and shortlist export
- Once Pair 2 is available, conjugation integration and invariant checks through Pair 1

It cannot yet demonstrate useful distance, lower error probability, or evolutionary superiority. Gate count and sequence diversity are engineering diagnostics, not substitutes for QEC fitness.

## Baselines and fair budgets

The following are **proposed experimental defaults**, not published performance expectations.

| Baseline | Definition |
|---|---|
| Independent random search | Draw each candidate independently from the specified initialization distribution; retain the best-so-far |
| Mutation-only \((1+1)\) evolution | Start from the same initial candidate as its paired random-search run; mutate the incumbent once per step; retain a better offspring |
| Neutral movement | On an exactly equal full ranking tuple, accept with probability \(1/2\), using the recorded selection stream |
| Output | Best candidate plus a small ranked archive; identify duplicates by serialized genotype initially |

Use a pilot to measure cost, then freeze the comparison:

- **Budget:** Initially 1,000 evaluator calls per method/run, including initialization and duplicate or semantically equivalent proposals.
- **Caching:** Disable cross-candidate caching in the first benchmark so “evaluation calls” has an unambiguous meaning; log opportunities for later caching.
- **Seeds:** Use 30 predeclared paired runs. Pair initial candidates, but keep subsequent random-generation and mutation streams separate.
- **Controls:** Same \(n,k\), bounds, initial distribution, evaluator, noise parameters, and per-evaluation accuracy/workload.
- **Stopping:** Use the fixed budget for the primary comparison; record first-target-hit time without selectively stopping favorable runs.
- **Reporting:** Best-so-far versus evaluation count, final-score distributions, target-hit counts, gate counts, and wall time—not just a best seed.
- **Tuning:** Keep tuning runs separate from final comparison seeds. For sampled evaluation, reserve an independent final reevaluation protocol.

**Deduction:** Equal generations are not necessarily equal work across search methods. Count candidate evaluations directly; report wall time as well, especially if candidate-dependent evaluation costs differ.

**Interpretation:** Validity and replay tests establish correctness. A predeclared, repeated-seed comparison can support a scoped search-improvement claim; it must also permit “no observed improvement” or random search winning.

## Integration boundaries

These are **proposed contracts**, not requests for Pair 4 to build missing dependencies.

| Pair | Pair 4 supplies/consumes | Ownership boundary |
|---|---|---|
| **1: Stabilizer mathematics** | Consume Pauli conversion, commutation, and GF(2) independence validation with documented column/qubit conventions | No Pair 4 rank, symplectic, or Pauli-math implementation. Missing validation remains a dependency. |
| **2: Clifford simulation** | Supply explicit register size, chronological gate records, and input-wire convention; consume signed conjugated stabilizers | Pair 2 owns Stim translation, conjugation direction, signs, and optional logical operators. |
| **3: Evaluation** | Supply the agreed code descriptor and evaluation configuration; consume named metrics and coverage metadata | Pair 3 owns noise probabilities, enumeration, degeneracy-aware classification, and decoder evaluation. |
| **5: Transformer** | Exchange versioned semantic gate records and candidate metadata | Pair 5 owns token IDs, vocabulary, special tokens, padding, and model architecture. No tokenizer dependency is needed initially. |

**Stim-specific integration cautions:**

- Preserve explicit `n`: Stim’s circuit qubit count is inferred from referenced qubit indices, so empty circuits and idle high-index wires need deliberate handling by Pair 2. [API, `Circuit.num_qubits`](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Circuit.num_qubits)
- Keep Pair 4’s elementary gate list separate from Stim instruction indexing: Stim can fuse adjacent instructions, making instruction count unsuitable as the genotype length. [API, `Circuit.insert`](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Circuit.insert)
- Preserve signed results and agree on forward versus inverse tableau conventions. [API, `PauliString.after`; `TableauSimulator.current_inverse_tableau`](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md)
- Pin a stable Stim version with Pair 2; the supplied development reference explicitly warns that documented features may not exist in stable releases. [API introduction](https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md)

## Acceptance tests

These are **suggested tests**, not assertions that they have been run.

### Pair 4 correctness

| Test | Acceptance criterion |
|---|---|
| Replay | Same pinned environment, configuration, and seed reproduce candidates, edits, selections, and logs |
| Gate validity | Correct arity, in-range operands, distinct CNOT endpoints |
| Boundaries | Empty sequences, one-qubit registers, maximum length, and zero-length diagnostic cap terminate correctly |
| Mutation | Insert reaches both endpoint gaps; delete can remove any position; parent remains unchanged |
| Sampling | Analytically specified probabilities; a predeclared statistical smoke test detects gross bias without requiring exact frequencies |
| Ranking | Known canned results produce the expected ordering; unavailable/NaN/error results never become winners |
| Budgets | Initialization and every assessed offspring count; fixed-budget termination is exact |
| Serialization | Round-trip retains register size, gate order, CNOT direction, and configuration |

### Cross-pair fixtures

| Fixture | Expected result and purpose |
|---|---|
| **GF(2) trap:** `ZZI`, `ZIZ`, `IZZ` | **Deduction:** commuting but only two independent generators; their support rows have real rank three and GF(2) rank two. Pair 1 must reject a claim of three independent checks. |
| **Code/state distinction:** `ZZ` versus `XX, ZZ` | **Deduction:** one check defines an \([[2,1]]\) code; two independent commuting checks define a stabilizer state. [Gottesman, §3.2](https://arxiv.org/html/quant-ph/9705052v1) |
| **Sign fixture:** apply H, S, S, H to initial `+Z` | **Deduction:** output is `-Z`, not `+Z`; tests Pair 2’s sign preservation and chronological convention. |
| **Degeneracy fixture:** \(S=\langle Z_0\rangle\) on two qubits | **Deduction:** \(I\) and \(Z_0\) are harmless; \(X_1\) is harmful undetected; \(X_0\) and \(X_0X_1\) share a nonzero syndrome but differ logically. Tests Pair 3’s classifications. |
| **Three-qubit repetition checks:** `ZZI`, `IZZ` | Full Pauli distance is one, not three: a single \(Z\) is a logical error. [Tandeitnik–Guerreiro, Appendix A](https://arxiv.org/html/2210.05058v1) |
| **\([[4,2,2]]\):** `XXXX`, `ZZZZ` | Expected distance two; a small distance-improvement target and integration fixture. [Gottesman lecture notes, “Distance 2 Codes”] [perimeterinstitute](https://perimeterinstitute.ca/personal/dgottesman/CO639-2004/Lecture5.pdf) |
| **\([[5,1,3]]\)** | A trusted descriptor/encoder should evaluate to distance three; use as an evaluator fixture, not a mandatory search-discovery test. [Gottesman, §8.2](https://arxiv.org/html/quant-ph/9705052v1) |

Pair 4 should invoke the owning pairs’ functions for these checks. A failed dependency should be visible—not replaced with a local approximation.

## Smallest useful milestone

**Deliver now:**

1. Versioned gate-sequence specification and reproducible configuration
2. Legal random generation and bounded replace/insert/delete mutation
3. Independent random search and mutation-only \((1+1)\) orchestration
4. Injected evaluator contract, ranking, budget accounting, lineage, and candidate export
5. Pair 4 unit tests and agreed cross-pair fixtures, with scientific integration tests explicitly blocked until dependencies arrive

**Defer:** crossover, populations/tournaments, direct stabilizer encodings, canonical code-equivalence deduplication, circuit simplification, hardware-specific connectivity, circuit-level noise, decoder optimization, transformer-guided proposals, and multiobjective search.

**Unresolved decisions to settle with collaborators:**

- Which exact metric can Pair 3 deliver first, and is its enumeration complete?
- What noise channel and parameter values define the first experiment?
- When will Pair 1 expose GF(2) independence validation and Pair 2 expose signed conjugation?
- What signed-Pauli format, wire order, and conjugation convention will everyone share?
- Are structural costs merely tie-breakers, or is encoder cost a separate research objective?
- Which fixtures and thresholds will be frozen before final benchmark seeds?

The first acceptance milestone should be a reproducible, bounded search pipeline with honest metric labels—not discovery of a better code.