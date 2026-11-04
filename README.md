# QEC code discovery

One package, `qec_discovery`, for generating Clifford encoders, checking the stabilizer codes they define, and scoring them under noise.

```
src/qec_discovery/
  pauli/      binary symplectic vectors, commutation, independence, membership
  clifford/   H, S, and CNOT conjugation, encoding, and format adapters
  noise/      bit-flip, phase-flip, and depolarizing noise, syndromes, decoding
  search/     circuit generation, mutation, and search
  model/      attention layer for later learned generation
```

## Setup

```bash
pip install -e ".[stim,sampling,model,dev]"
```

`numpy` is required. `stim` is the Clifford cross-check used by the baseline search. `jax` is imported by the Pauli sampler. `torch` is imported by `qec_discovery.model`.

## Run

```bash
python scripts/demo_pipeline.py
python scripts/discover.py
python -m qec_discovery.search sample --n 3 --k 1 --seed 42 --count 2
```

`discover.py` writes `results/baseline.csv`. Search and benchmark runs write under `runs/`. The search command-line interface is documented in [docs/search.md](docs/search.md).

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
