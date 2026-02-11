from __future__ import annotations

import jax.numpy as jnp

from ..core.jax_sph import LiquidPhysics


class PhysicsSimulator:
    def __init__(self, config: dict):
        self.solver = LiquidPhysics(config)

    def run(
        self, pos: jnp.ndarray, vel: jnp.ndarray, steps: int = 10
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        props = {"mass": jnp.ones((pos.shape[0],), dtype=jnp.float32)}
        rho = jnp.zeros((pos.shape[0],), dtype=jnp.float32)
        for _ in range(int(steps)):
            pos, vel, rho = self.solver.step(pos, vel, props)
        return pos, vel, rho
