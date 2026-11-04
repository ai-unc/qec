# Pair 5: Clifford decoder-only Transformer

Run commands from the `qec/` directory (so `pair4` and `pair5` import correctly).

## Demo

```bash
cd qec
python3 -m pair5.evaluate_generated --seed 0
```

Flags: `--n` (qubits), `--k` (logical qubits), `--seed`.

## Tests

```bash
cd qec
python3 -m unittest discover -s pair5/tests -v
```

If you are not in `qec/`, set `PYTHONPATH` to that folder instead:

```bash
PYTHONPATH=qec python3 -m pair5.evaluate_generated --seed 0
```
