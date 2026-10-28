import jax.random as jr
import jax.numpy as jnp
from noise import model, error_probability

OPTIONS = ['I', 'X', 'Y', 'Z']

def sample_pauli_error(n_qubits: int, noise_model, p, key):
    key = jr.PRNGKey(key) if isinstance(key, int) else key
    rand_string = ""

    for i in range(n_qubits):
        key, subkey = jr.split(key)
        rand_idx = jr.choice(subkey, len(OPTIONS))
        rand_qb = OPTIONS[int(rand_idx)]
        rand_string += rand_qb

    sample_error = error_probability(rand_string, noise_model, p)
    return sample_error
