from __future__ import annotations

import numpy as np
import jax.numpy as jnp


class AudioVisualGaussianSplatting:
    """AV-GS: Represents a scene as 3D Gaussians with acoustic properties."""

    def __init__(self, num_splats: int = 1024, seed: int | None = 0):
        self.num_splats = int(num_splats)
        self._rng = np.random.default_rng(seed)

    def init_scene(self):
        """Initialize the Audio-Visual Scene Graph."""
        return {
            "xyz": self._rng.uniform(-0.5, 0.5, (self.num_splats, 3)).astype(
                np.float32
            ),
            "rotation": self._rng.uniform(0.0, 1.0, (self.num_splats, 4)).astype(
                np.float32
            ),
            "scale": self._rng.uniform(0.01, 0.05, (self.num_splats, 3)).astype(
                np.float32
            ),
            "opacity": self._rng.uniform(0.5, 1.0, (self.num_splats, 1)).astype(
                np.float32
            ),
            "impedance": self._rng.uniform(2000.0, 8000.0, (self.num_splats, 1)).astype(
                np.float32
            ),
            "absorption": self._rng.uniform(0.01, 0.9, (self.num_splats, 1)).astype(
                np.float32
            ),
            "binaural_phase": self._rng.uniform(
                0.0, 2 * np.pi, (self.num_splats, 1)
            ).astype(np.float32),
        }

    def query_acoustic_field(self, splats, query_points: jnp.ndarray):
        """Differentiable query of local acoustic impedance and absorption."""
        xyz = jnp.asarray(splats["xyz"])
        scale = jnp.asarray(splats["scale"])
        impedance = jnp.asarray(splats["impedance"])
        absorption = jnp.asarray(splats["absorption"])

        dist = jnp.linalg.norm(query_points[:, None, :] - xyz[None, :, :], axis=-1)
        sigma = jnp.mean(scale)
        sigma = jnp.maximum(sigma, 1e-6)
        weights = jnp.exp(-(dist**2) / (2.0 * sigma**2))
        weights = weights / (jnp.sum(weights, axis=1, keepdims=True) + 1e-6)

        local_Z = weights @ impedance
        local_alpha = weights @ absorption
        return local_Z, local_alpha
