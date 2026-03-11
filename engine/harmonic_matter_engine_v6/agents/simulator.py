from __future__ import annotations

import jax.numpy as jnp

from harmonic_matter_engine_v6.core.jax_sph import LiquidPhysics


class PhysicsSimulator:
    """Thin wrapper around the JAX-SPH solver for demo runs."""

    def __init__(self, config: dict):
        """Create a simulator from a loaded engine config."""
        self.solver = LiquidPhysics(config)

    def run(
        self, pos: jnp.ndarray, vel: jnp.ndarray, steps: int = 10
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Advance the particle system for a fixed number of solver steps."""
        props = {"mass": jnp.ones((pos.shape[0],), dtype=jnp.float32)}
        rho = jnp.zeros((pos.shape[0],), dtype=jnp.float32)
        for _ in range(int(steps)):
            pos, vel, rho = self.solver.step(pos, vel, props)
        return pos, vel, rho
