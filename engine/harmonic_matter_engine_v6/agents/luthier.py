from __future__ import annotations

import logging
from typing import Any

from harmonic_matter_engine_v6.core.av_gs import AudioVisualGaussianSplatting


LOGGER = logging.getLogger(__name__)


class MaterialLuthier:
    """Material Optimizer (Uses AV-GS)."""

    def __init__(self, num_splats: int = 1024):
        self.av_gs = AudioVisualGaussianSplatting(num_splats=num_splats)

    def bake_scene(self) -> dict[str, Any]:
        LOGGER.info(">>> LUTHIER: Baking Audio-Visual Gaussian Splats...")
        return self.av_gs.init_scene()
