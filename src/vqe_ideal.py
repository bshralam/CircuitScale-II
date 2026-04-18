import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp, Statevector


def qubitop_to_sparsepauliop(qubit_ham, n_qubits: int) -> SparsePauliOp:
    paulis = []
    coeffs = []

    for term, coeff in qubit_ham.terms.items():
        label = ["I"] * n_qubits
        for (q, p) in term:
            label[n_qubits - 1 - q] = p
        paulis.append("".join(label))
        coeffs.append(complex(coeff))

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


def energy_statevector(theta, ansatz, operator: SparsePauliOp) -> float:
    bound = ansatz.assign_parameters(theta)
    psi = Statevector.from_instruction(bound)
    val = psi.expectation_value(operator)
    return float(np.real(val))


def run_vqe_ideal(operator: SparsePauliOp, ansatz, maxiter: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    x0 = np.zeros(ansatz.num_parameters)

    res = minimize(
        lambda x: energy_statevector(x, ansatz, operator),
        x0=x0,
        method="COBYLA",
        options={"maxiter": maxiter, "disp": False},
    )

    return float(res.fun), res
