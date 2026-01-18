from __future__ import annotations

from ..core.av_gs import AudioVisualGaussianSplatting


class MaterialLuthier:
    """Material Optimizer (Uses AV-GS)."""

    def __init__(self, num_splats: int = 1024):
        self.av_gs = AudioVisualGaussianSplatting(num_splats=num_splats)

    def bake_scene(self):
        print(">>> LUTHIER: Baking Audio-Visual Gaussian Splats...")
        return self.av_gs.init_scene()
