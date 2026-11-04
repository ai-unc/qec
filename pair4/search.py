# All of this is pseudocode so far, I'm just noting things down cause I had a sudden epiphany.
# TODO: we need to ask the other teams how we will interface with their projects. What shape can we expect their outputs to be and stuff like that.

results = []

for trial in range(num_trials):
    circuit = generate_random_circuit(...)
    stabilizers = simulate_circuit(circuit, ...)

    if not validate_code(stabilizers, ...):
        continue

    logical_error_rate = evaluate_code(
        stabilizers, noise_model="bit_flip", p=0.05
    )

    results.append({
        "circuit": circuit,
        "stabilizers": stabilizers,
        "logical_error_rate": logical_error_rate,
    })

results.sort(key=lambda result: result["logical_error_rate"])