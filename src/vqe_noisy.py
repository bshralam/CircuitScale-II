import numpy as np
from qiskit_aer import AerSimulator

from .measurements import estimate_term_expectation
from .noise import make_noise_model
from .mitigation import ReadoutMitigator

def estimate_energy_noisy(backend, ansatz, theta, op, shots_per_term: int = 2000, mitigator=None) -> float:
    labels = op.paulis.to_labels()
    coeffs = op.coeffs
    E = 0.0
    for lab, c in zip(labels, coeffs):
        expP = estimate_term_expectation(
            backend=backend,
            ansatz=ansatz,
            theta=theta,
            pauli_label=str(lab),
            shots=shots_per_term,
            mitigator=mitigator,
        )
        E += (complex(c) * expP).real
    return float(E)

def spsa_optimize(energy_fn, theta0: np.ndarray, n_iter: int = 80, a: float = 0.008, c: float = 0.02,
                  alpha: float = 0.602, gamma: float = 0.101, seed: int = 7):
    rng = np.random.default_rng(seed)
    theta = theta0.astype(float).copy()
    best_theta = theta.copy()
    best_E = energy_fn(theta)

    trace = []
    for k in range(n_iter):
        ak = a / ((k + 1) ** alpha)
        ck = c / ((k + 1) ** gamma)
        delta = rng.choice([-1.0, 1.0], size=theta.shape)

        theta_plus = theta + ck * delta
        theta_minus = theta - ck * delta

        E_plus = energy_fn(theta_plus)
        E_minus = energy_fn(theta_minus)

        gk = (E_plus - E_minus) / (2.0 * ck) * (1.0 / delta)
        theta = theta - ak * gk

        E_curr = energy_fn(theta)
        if E_curr < best_E:
            best_E = E_curr
            best_theta = theta.copy()

        trace.append({
            "iter": k,
            "E_plus": E_plus,
            "E_minus": E_minus,
            "E_curr": E_curr,
            "best_E": best_E,
        })

    return best_theta, best_E, trace
