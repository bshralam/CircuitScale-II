from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    geometry: List[Tuple[str, Tuple[float, float, float]]]
    charge: int = 0
    multiplicity: int = 1
    basis: str = "sto-3g"

def h2_scan(R_values: np.ndarray) -> List[MoleculeSpec]:
    specs = []
    for R in R_values:
        geom = [
            ("H", (0.0, 0.0, -R / 2)),
            ("H", (0.0, 0.0,  R / 2)),
        ]
        specs.append(MoleculeSpec(name=f"H2_R{R:.2f}", geometry=geom))
    return specs

def h4_square(delta: float, a: float = 1.0) -> MoleculeSpec:
    sx, sy = (1.0 + delta), (1.0 - delta)
    geom = [
        ("H", (-a*sx/2, -a*sy/2, 0.0)),
        ("H", ( a*sx/2, -a*sy/2, 0.0)),
        ("H", ( a*sx/2,  a*sy/2, 0.0)),
        ("H", (-a*sx/2,  a*sy/2, 0.0)),
    ]
    return MoleculeSpec(name=f"H4_sq_d{delta:+.2f}_a{a:.2f}", geometry=geom)

def h4_distortion_grid(deltas: np.ndarray, a: float = 1.0) -> List[MoleculeSpec]:
    return [h4_square(float(d), a=a) for d in deltas]
