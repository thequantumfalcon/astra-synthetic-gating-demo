"""ASTRA synthetic gating demo.

A fixed-seed toy experiment: inject a damped-sinusoid burst into synthetic
Gaussian noise, apply an amplitude gate, and report the effect on
peak-based proxy statistics. Makes no astrophysical detection claim.
"""

__all__ = [
    "astra_proof",
    "utils",
]
