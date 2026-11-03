import jax.random as jr
import jax.numpy as jnp
from noise import model

OPTIONS = ['I', 'X', 'Y', 'Z']


def sample_pauli_error(n_qubits, noise_model, p, key):
    error_string = ""

    if isinstance(key, int):
        key = jr.PRNGKey(key)

    dist = model(noise_model, p)

    prob_list = []
    for letter in OPTIONS:
        prob_list.append(dist[letter])
    probs = jnp.array(prob_list)

    selected = jr.choice(key, 4, shape=(n_qubits,), p=probs)
    for number in selected:
        error_string = error_string + OPTIONS[int(number)]

    return error_string
