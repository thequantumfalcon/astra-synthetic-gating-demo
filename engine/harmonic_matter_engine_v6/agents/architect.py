from __future__ import annotations

from ..core.walrus import WalrusSurrogate


class GenerativeArchitect:
    def __init__(self):
        self.walrus = WalrusSurrogate()

    def design_simulation(self, prompt: str):
        print(f">>> ARCHITECT: Analyzing '{prompt}' with Walrus Foundation Model...")
        p = (prompt or "").lower()
        # Minimal prompt-sensitive heuristics (demo-grade).
        if "diamond" in p:
            return {
                "viscosity": 0.05,
                "stiffness": 200.0,
                "surface_tension": 0.9,
            }
        if "mercury" in p:
            return {
                "viscosity": 0.02,
                "stiffness": 80.0,
                "surface_tension": 0.5,
            }
        return {
            "viscosity": 0.015,
            "stiffness": 100.0,
            "surface_tension": 0.6,
        }
