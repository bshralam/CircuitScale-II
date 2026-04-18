import numpy as np
from qiskit import QuantumCircuit

def _bitstring_to_index(bitstr: str) -> int:
    return int(bitstr, 2)

def build_readout_cal_circuits(n_qubits: int):
    circs = []
    for i in range(2 ** n_qubits):
        bitstr = format(i, f"0{n_qubits}b")
        qc = QuantumCircuit(n_qubits, n_qubits)
        for qi in range(n_qubits):
            if bitstr[::-1][qi] == "1":
                qc.x(qi)
        qc.measure(range(n_qubits), range(n_qubits))
        circs.append((bitstr, qc))
    return circs

def fit_assignment_matrix(n_qubits: int, prepared_bitstrs, counts_list):
    dim = 2 ** n_qubits
    A = np.zeros((dim, dim), dtype=float)

    for prep_bitstr, counts in zip(prepared_bitstrs, counts_list):
        i = _bitstring_to_index(prep_bitstr)
        shots = sum(counts.values())
        for meas_bitstr, c in counts.items():
            j = _bitstring_to_index(meas_bitstr.replace(" ", ""))
            A[j, i] += c / shots
    return A

class ReadoutMitigator:
    def __init__(self, assignment_matrix: np.ndarray, method: str = "pinv"):
        self.A = assignment_matrix
        self.Ainv = np.linalg.pinv(self.A) if method == "pinv" else np.linalg.inv(self.A)

    def mitigate_counts(self, counts: dict, n_qubits: int) -> dict:
        dim = 2 ** n_qubits
        shots = sum(counts.values())
        p_meas = np.zeros(dim, dtype=float)

        for b, c in counts.items():
            p_meas[_bitstring_to_index(b.replace(" ", ""))] += c / shots

        p_true = self.Ainv @ p_meas
        p_true = np.clip(p_true, 0.0, 1.0)
        if p_true.sum() > 0:
            p_true /= p_true.sum()

        out = {}
        for j in range(dim):
            bitstr = format(j, f"0{n_qubits}b")
            out[bitstr] = int(round(p_true[j] * shots))
        return out
