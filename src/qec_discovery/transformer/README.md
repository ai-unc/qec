# Transformer (Pair 5)

JAX/Flax decoder-only model that currently generates Clifford gate sequences and
derives unsigned stabilizer Paulis. Install extras:

```bash
pip install -e ".[model,train,stim,dev]"
```

## Demo

From the repo root (with `src` on `PYTHONPATH`):

```bash
PYTHONPATH=src python3 -m qec_discovery.transformer.evaluate_generated --seed 0
PYTHONPATH=src python3 -m qec_discovery.transformer.evaluate_generated --seed 1
PYTHONPATH=src python3 -m qec_discovery.transformer.evaluate_generated --seed 0 --greedy
```

Flags: `--n`, `--k`, `--seed`, `--greedy`, `--min-gates`, `--temperature`.

Default decoding **samples** and blocks EOS until `--min-gates` CNOTs (default 4),
so different `--seed` values produce different circuits on an untrained model.
`--greedy` uses argmax (often empty / identical). Then prints validity and
logical error rate via `evaluate_candidate_generators`.

## Eval bridge and training notes

- Score candidate generators with `evaluate_candidate_generators` (see
  `pipeline_interface.py`).
- Planned direct Pauli-string generation and how it plugs into discovery eval:
  [`training_pipeline.md`](training_pipeline.md).
