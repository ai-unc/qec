# Minimal JAX + Optax training loop on a synthetic supervised dataset.

import jax
import jax.numpy as jnp
import optax


def init_params(key, in_dim=4, hidden_dim=16, out_dim=1):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (in_dim, hidden_dim)) * jnp.sqrt(2.0 / in_dim),
        "b1": jnp.zeros((hidden_dim,)),
        "w2": jax.random.normal(k2, (hidden_dim, out_dim)) * jnp.sqrt(2.0 / hidden_dim),
        "b2": jnp.zeros((out_dim,)),
    }


def make_synthetic_dataset(key, *, n_samples=256, in_dim=4, noise_std=0.05):
    key_x, key_w, key_noise = jax.random.split(key, 3)
    x = jax.random.normal(key_x, (n_samples, in_dim))
    w_true = jax.random.normal(key_w, (in_dim, 1))
    noise = noise_std * jax.random.normal(key_noise, (n_samples, 1))
    y = x @ w_true + noise
    return x, y


def predict(params, x):
    h = jax.nn.relu(x @ params["w1"] + params["b1"])
    return h @ params["w2"] + params["b2"]


def loss_fn(params, x, y):
    preds = predict(params, x)
    return jnp.mean((preds - y) ** 2)


def train(*, seed=0, steps=200, learning_rate=1e-2, n_samples=256, in_dim=4, hidden_dim=16):
    key = jax.random.PRNGKey(seed)
    key_data, key_params = jax.random.split(key)
    x, y = make_synthetic_dataset(key_data, n_samples=n_samples, in_dim=in_dim)
    params = init_params(key_params, in_dim=in_dim, hidden_dim=hidden_dim, out_dim=1)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    loss_history = []
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state, x, y)
        loss_history.append(float(loss))

    return params, loss_history


if __name__ == "__main__":
    _, history = train()
    print(f"initial_loss={history[0]:.6f} final_loss={history[-1]:.6f}")
