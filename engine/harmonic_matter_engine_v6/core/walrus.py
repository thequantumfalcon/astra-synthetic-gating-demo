from __future__ import annotations

import time
from typing import Tuple

import numpy as np


class WalrusSurrogate:
    """Wrapper for the Walrus Physics Foundation Model.

    Surrogate-mode stub for v6.0: returns prompt-shaped randomness.
    """

    def __init__(self):
        self.model_name = "Walrus-1.3B"
        print(f"Loading {self.model_name} (Surrogate Mode)...")

    def generate_initial_state(
        self, prompt: str, particle_count: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        print(f"[Walrus] Dreaming physics for: '{prompt}'")
        time.sleep(0.2)
        pos = np.random.uniform(-0.2, 0.2, (particle_count, 3)).astype(np.float32)
        vel = np.random.normal(0.0, 2.0, (particle_count, 3)).astype(np.float32)
        return pos, vel

    def steer_physics(
        self, current_state: np.ndarray, target_concept: str = "laminar flow"
    ) -> np.ndarray:
        _ = target_concept
        return np.zeros_like(current_state, dtype=np.float32) * 0.1
