from dataclasses import dataclass

@dataclass(frozen=True)
class VQEConfig:
    optimizer: str = "COBYLA"
    maxiter: int = 200
    shots: int = 2000
    seed: int = 7
    reps: int = 2
    p1: float = 2e-4
    p2: float = 2e-3
    p_readout: float = 0.01
