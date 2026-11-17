# Training Pipeline

Transformer that outputs stabilizer codes as Pauli strings (e.g. `["ZZI", "IZZ"]`) and scores them with the discovery pipeline.

Currently only the small JAX demo network in `training.py`
(synthetic data). The Flax stabilizer-code transformer is **not** wired to a
training loop yet — that waits on the stabilizer tokenizer (other group).

**Scoring (works now):** any candidate generators →
`evaluate_candidate_generators` → valid? + logical error rate.

```text
(n, k) → generate n-k Pauli strings → evaluate_candidate_generators
         → if valid, get logical error rate → keep / rank
```

## Setup

From the `qec/` repo root:

```bash
pip install -e ".[train,model,stim,dev]"
```

Installs this package editable, plus JAX/Optax (training demo), Flax (model),
Stim (baseline search), and pytest/dev tools.

## How to train (what exists today)

1. Install (command above).
2. Run the demo training loop:

```bash
PYTHONPATH=src python3 -m qec_discovery.transformer.training
```

`PYTHONPATH=src` makes Python find `qec_discovery` without a broken install.
This runs `train()` in `training.py`: builds a fake regression dataset, creates
a tiny MLP, computes loss with `jax.value_and_grad`, updates weights with Optax
Adam for 200 steps, then prints:

```text
initial_loss=... final_loss=...
```

Final loss should be lower than initial. That means gradients and optimizer
updates are working.

3. Call it from Python if you want different settings:

```python
from qec_discovery.transformer.training import train

params, loss_history = train(seed=0, steps=200, learning_rate=1e-2)
# loss_history[-1] < loss_history[0]
```

Useful knobs: `steps`, `learning_rate`, `seed`, `n_samples`, `hidden_dim`.

**Not available yet:** training `CliffordDecoderOnly` (or a future Pauli
generator) on stabilizer codes. When the stabilizer tokenizer lands, training
will look like: tokenize codes → Flax forward → token loss → Optax update →
periodically call `evaluate_candidate_generators` to filter/rank (LER is not a
differentiable loss).

## Score a known code (eval bridge)

```bash
PYTHONPATH=src python3 -c "
from qec_discovery.transformer import evaluate_candidate_generators
print(evaluate_candidate_generators(['ZZI', 'IZZ']))
"
```

Runs the three-qubit repetition code through validation + exact logical error
rate. Expect `valid=True` and `logical_error_rate` ≈ `0.00725` (bit-flip,
`p=0.05`). This is **not** training — it only checks the scoring interface.

## Check the deliverables with tests

```bash
PYTHONPATH=src python3 -m unittest \
  qec_discovery.transformer.tests.test_pipeline_interface \
  qec_discovery.transformer.tests.test_training -v
```

- `test_pipeline_interface` — eval bridge on `ZZI`/`IZZ` and invalid codes  
- `test_training` — params move and loss decreases  

`-v` prints each test name as it runs.
