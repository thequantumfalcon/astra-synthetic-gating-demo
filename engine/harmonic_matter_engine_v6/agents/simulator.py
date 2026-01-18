from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp

from ..core.jax_sph import LiquidPhysics


class PhysicsSimulator:
    def __init__(self, config: Dict):
        self.solver = LiquidPhysics(config)

    def run(
        self, pos: jnp.ndarray, vel: jnp.ndarray, steps: int = 10
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        props = {"mass": jnp.ones((pos.shape[0],), dtype=jnp.float32)}
        rho = jnp.zeros((pos.shape[0],), dtype=jnp.float32)
        for _ in range(int(steps)):
            pos, vel, rho = self.solver.step(pos, vel, props)
        return pos, vel, rho
