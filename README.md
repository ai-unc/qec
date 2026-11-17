# QEC code discovery

One package, `qec_discovery`, for generating Clifford encoders, checking the stabilizer codes they define, and scoring them under noise.

```
src/qec_discovery/
  pauli/         binary symplectic vectors, commutation, independence, membership
  clifford/      H, S, and CNOT conjugation, encoding, and format adapters
  noise/         bit-flip, phase-flip, and depolarizing noise, syndromes, decoding
  search/        circuit generation, mutation, and search
  model/         attention layer for later learned generation (JAX/Flax)
  transformer/   Pair 5 generator, eval bridge, and JAX training sandbox
```

## Setup

```bash
pip install -e ".[stim,sampling,train,model,dev]"
```

`numpy` is required. `stim` is the Clifford cross-check used by the baseline search. `jax` is used by the Pauli sampler and the transformer. `flax` is required by `qec_discovery.transformer` / `qec_discovery.model`. `optax` is required by the `train` extra (`transformer/training.py`).

## Run

```bash
python scripts/demo_pipeline.py
python scripts/discover.py
python -m qec_discovery.search sample --n 3 --k 1 --seed 42 --count 2
python3 -m qec_discovery.transformer.evaluate_generated --seed 0
```

`discover.py` writes `results/baseline.csv`. Search and benchmark runs write under `runs/`. The search command-line interface is documented in [docs/search.md](docs/search.md).

Candidate stabilizer generators are scored through
[`transformer/pipeline_interface.py`](src/qec_discovery/transformer/pipeline_interface.py).
How the eventual direct stabilizer-code transformer will use that bridge is
documented in [`transformer/training_pipeline.md`](src/qec_discovery/transformer/training_pipeline.md).

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python3 -m unittest discover -s src/qec_discovery/transformer/tests -v
```
