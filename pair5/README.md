# Pair 5: Decoder-only Transformer

Run commands from the `qec/` directory.

## Demo

```bash
cd qec
python3 -m pair5.evaluate_generated --seed 0
```

Flags: `--n` (qubits), `--k` (logical qubits), `--seed`.

Currently generates a circuit and prints what we're sending to pair 2

Ex: 
```text
[("H", 0), ("CNOT", 0, 1), ("S", 1)]
```


## Tests

First just select qec
```bash
cd qec
```

Then these are the two tests:

```bash
python3 -m pair5.evaluate_generated --seed 1
```
Pick different seeds if you want to really test it, but it's entirely random rn


```bash
python3 -m unittest discover -s pair5/tests -v
```
