"""Unit tests for the standalone JAX/Optax training loop."""

from __future__ import annotations

import unittest

import jax.numpy as jnp

from qec_discovery.transformer.training import train


class TestTraining(unittest.TestCase):
    def test_train_decreases_loss_and_updates_params(self) -> None:
        params_short, history_short = train(seed=0, steps=1, learning_rate=1e-2)
        params_long, history_long = train(seed=0, steps=200, learning_rate=1e-2)

        self.assertGreaterEqual(len(history_long), 2)
        self.assertLess(history_long[-1], history_long[0])
        # Same seed ⇒ same initialization; more steps must move parameters.
        moved = any(
            not jnp.allclose(params_short[name], params_long[name])
            for name in params_long
        )
        self.assertTrue(moved, "expected at least one parameter tensor to change")
        self.assertEqual(len(history_short), 1)


if __name__ == "__main__":
    unittest.main()
