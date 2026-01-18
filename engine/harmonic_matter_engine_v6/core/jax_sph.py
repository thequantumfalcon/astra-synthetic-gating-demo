from __future__ import annotations

from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp


class LiquidPhysics:
    """JAX-SPH Solver v6.0.

    Implements a demo-grade Transport Velocity Formulation (TVF) correction and
    a simple Riemann-style symmetric pressure term for shock stability.

    NOTE: Neighbor search is simplified to all-to-all within smoothing length.
    """

    def __init__(self, config: Dict):
        self.h = float(config["physics"]["smoothing_length"])
        self.c0 = float(config["physics"]["speed_of_sound"])
        self.rho0 = float(config["physics"]["rest_density"])
        self.nu = float(config["physics"]["viscosity"])
        self.p_bg = float(config["physics"].get("background_pressure", 0.0))
        self.g = jnp.array([0.0, -9.81, 0.0], dtype=jnp.float32)

    @partial(jax.jit, static_argnums=(0,))
    def kernel_poly6(self, r: jnp.ndarray) -> jnp.ndarray:
        q = r / self.h
        coef = 315.0 / (64.0 * jnp.pi * (self.h**9))
        return jnp.where(q <= 1.0, coef * (self.h**2 - r**2) ** 3, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def kernel_spiky_grad(self, r_vec: jnp.ndarray, r_len: jnp.ndarray) -> jnp.ndarray:
        """Gradient of the Spiky kernel for pressure forces.

        Shapes:
        - r_vec: (N, N, 3)
        - r_len: (N, N)
        Returns: (N, N, 3)
        """
        r_len_safe = jnp.maximum(r_len, 1e-6)
        coef = -45.0 / (jnp.pi * (self.h**6))
        factor = coef * (self.h - r_len) ** 2 / r_len_safe
        factor = jnp.where(r_len < self.h, factor, 0.0)
        return factor[..., None] * r_vec

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        pos: jnp.ndarray,
        vel: jnp.ndarray,
        properties: Dict[str, jnp.ndarray],
        dt: float = 0.01,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """One integration step using Riemann-SPH-like pressure + TVF shift."""
        diff = pos[:, None, :] - pos[None, :, :]
        dist = jnp.linalg.norm(diff, axis=-1)
        mask = dist < self.h

        w = self.kernel_poly6(dist) * mask
        mass = properties["mass"]
        rho = jnp.sum(w * mass[None, :], axis=1)
        rho = jnp.maximum(rho, 1e-6)

        pressure = self.rho0 * (self.c0**2 / 7.0) * ((rho / self.rho0) ** 7 - 1.0)
        pressure = jnp.maximum(pressure, 0.0) + self.p_bg

        grad_w = self.kernel_spiky_grad(diff, dist) * mask[..., None]
        p_term = (pressure[:, None] / (rho[:, None] ** 2)) + (
            pressure[None, :] / (rho[None, :] ** 2)
        )
        f_pressure = -jnp.sum(mass[None, :, None] * p_term[..., None] * grad_w, axis=1)

        vel_diff = vel[:, None, :] - vel[None, :, :]
        f_viscosity = self.nu * jnp.sum(
            mass[None, :, None]
            * (vel_diff / rho[None, :, None])
            * self.kernel_poly6(dist)[..., None],
            axis=1,
        )

        shift_vel = (
            -0.5 * self.h * jnp.sum(((mass / rho)[None, :, None]) * grad_w, axis=1)
        )

        acc = f_pressure + f_viscosity + self.g
        new_vel = vel + acc * dt
        new_pos = pos + (new_vel + shift_vel) * dt

        hit = (new_pos > 1.0) | (new_pos < -1.0)
        new_vel = jnp.where(hit, -new_vel * 0.6, new_vel)
        new_pos = jnp.clip(new_pos, -1.0, 1.0)

        return new_pos, new_vel, rho

    def export_step_fn(self, dt: float = 0.01):
        """Return a pure JAX function suitable for jax2tf conversion.

        Signature: (pos, vel, mass) -> (new_pos, new_vel, rho)
        Shapes should be fixed for best TFLite compatibility.
        """
        h = jnp.asarray(self.h, dtype=jnp.float32)
        c0 = jnp.asarray(self.c0, dtype=jnp.float32)
        rho0 = jnp.asarray(self.rho0, dtype=jnp.float32)
        nu = jnp.asarray(self.nu, dtype=jnp.float32)
        p_bg = jnp.asarray(self.p_bg, dtype=jnp.float32)
        g = jnp.asarray(self.g, dtype=jnp.float32)
        dt = jnp.asarray(dt, dtype=jnp.float32)

        # Inline kernels to avoid Python object references during conversion.
        def kernel_poly6(r: jnp.ndarray) -> jnp.ndarray:
            q = r / h
            coef = 315.0 / (64.0 * jnp.pi * (h**9))
            return jnp.where(q <= 1.0, coef * (h**2 - r**2) ** 3, 0.0)

        def kernel_spiky_grad(r_vec: jnp.ndarray, r_len: jnp.ndarray) -> jnp.ndarray:
            r_len_safe = jnp.maximum(r_len, 1e-6)
            coef = -45.0 / (jnp.pi * (h**6))
            factor = coef * (h - r_len) ** 2 / r_len_safe
            factor = jnp.where(r_len < h, factor, 0.0)
            return factor[..., None] * r_vec

        @jax.jit
        def step_fn(pos: jnp.ndarray, vel: jnp.ndarray, mass: jnp.ndarray):
            diff = pos[:, None, :] - pos[None, :, :]
            dist = jnp.linalg.norm(diff, axis=-1)
            mask = dist < h

            w = kernel_poly6(dist) * mask
            rho = jnp.sum(w * mass[None, :], axis=1)
            rho = jnp.maximum(rho, 1e-6)

            pressure = rho0 * (c0**2 / 7.0) * ((rho / rho0) ** 7 - 1.0)
            pressure = jnp.maximum(pressure, 0.0) + p_bg

            grad_w = kernel_spiky_grad(diff, dist) * mask[..., None]
            p_term = (pressure[:, None] / (rho[:, None] ** 2)) + (
                pressure[None, :] / (rho[None, :] ** 2)
            )
            f_pressure = -jnp.sum(
                mass[None, :, None] * p_term[..., None] * grad_w, axis=1
            )

            vel_diff = vel[:, None, :] - vel[None, :, :]
            f_viscosity = nu * jnp.sum(
                mass[None, :, None]
                * (vel_diff / rho[None, :, None])
                * kernel_poly6(dist)[..., None],
                axis=1,
            )

            shift_vel = (
                -0.5 * h * jnp.sum(((mass / rho)[None, :, None]) * grad_w, axis=1)
            )

            acc = f_pressure + f_viscosity + g
            new_vel = vel + acc * dt
            new_pos = pos + (new_vel + shift_vel) * dt

            hit = (new_pos > 1.0) | (new_pos < -1.0)
            new_vel = jnp.where(hit, -new_vel * 0.6, new_vel)
            new_pos = jnp.clip(new_pos, -1.0, 1.0)

            return new_pos, new_vel, rho

        return step_fn
