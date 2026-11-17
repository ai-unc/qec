# Transformer (Pair 5)

JAX/Flax decoder-only model that currently generates Clifford gate sequences and
derives unsigned stabilizer Paulis. Scores them with
`evaluate_candidate_generators`. Actual Pauli-token training needs the stabilizer tokenizer — see [`training_pipeline.md`](training_pipeline.md).

From the `qec/` directory:

```bash
pip install -e ".[model,train,stim,dev]"

# Generate + evaluate (default: sample; use --greedy for argmax)
PYTHONPATH=src python3 -m qec_discovery.transformer.evaluate_generated --seed 0
PYTHONPATH=src python3 -m qec_discovery.transformer.evaluate_generated --seed 1
PYTHONPATH=src python3 -m qec_discovery.transformer.evaluate_generated --seed 0 --greedy

# JAX/Optax sandbox (synthetic MLP — not the transformer)
PYTHONPATH=src python3 -m qec_discovery.transformer.training

# Package tests
PYTHONPATH=src python3 -m unittest discover -s src/qec_discovery/transformer/tests -v
```

Flags for the demo: `--n`, `--k`, `--seed`, `--greedy`, `--min-gates`.
