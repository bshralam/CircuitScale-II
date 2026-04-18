import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

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

def run_vqe_ideal(operator: SparsePauliOp, ansatz, maxiter: int = 200, seed: int = 7):
    estimator = Estimator()
    optimizer = COBYLA(maxiter=maxiter)
    initial_point = np.zeros(ansatz.num_parameters)
    vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=optimizer, initial_point=initial_point)
    result = vqe.compute_minimum_eigenvalue(operator)
    return float(np.real(result.eigenvalue)), result
