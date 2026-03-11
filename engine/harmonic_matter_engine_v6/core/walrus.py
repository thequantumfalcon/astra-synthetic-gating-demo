from __future__ import annotations

import logging
import time

import numpy as np

LOGGER = logging.getLogger(__name__)


class WalrusSurrogate:
    """Wrapper for the Walrus Physics Foundation Model.

    Surrogate-mode stub for v6.0: returns prompt-shaped randomness.
    """

    def __init__(self, seed: int | None = 0):
        self.seed = seed
        self.model_name = "Walrus-1.3B"
        LOGGER.info("Loading %s (Surrogate Mode)...", self.model_name)

    def generate_initial_state(
        self, prompt: str, particle_count: int, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a deterministic surrogate particle state for a prompt."""
        LOGGER.info("[Walrus] Dreaming physics for: '%s'", prompt)
        time.sleep(0.2)
        rng = np.random.default_rng(self.seed if seed is None else seed)
        pos = rng.uniform(-0.2, 0.2, (particle_count, 3)).astype(np.float32)
        vel = rng.normal(0.0, 2.0, (particle_count, 3)).astype(np.float32)
        return pos, vel

    def steer_physics(
        self, current_state: np.ndarray, target_concept: str = "laminar flow"
    ) -> np.ndarray:
        """Return a no-op steering field for the synthetic demo pipeline."""
        _ = target_concept
        return np.zeros_like(current_state, dtype=np.float32) * 0.1
