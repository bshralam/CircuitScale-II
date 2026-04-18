import numpy as np
from qiskit import QuantumCircuit

def _basis_change_for_pauli(circ: QuantumCircuit, pauli_label: str):
    n = circ.num_qubits
    for qi in range(n):
        p = pauli_label[::-1][qi]
        if p == "X":
            circ.h(qi)
        elif p == "Y":
            circ.sdg(qi)
            circ.h(qi)

def pauli_expectation_from_counts(pauli_label: str, counts: dict) -> float:
    shots = sum(counts.values())
    if shots == 0:
        return 0.0

    active_qubits = [qi for qi, p in enumerate(pauli_label[::-1]) if p != "I"]
    if not active_qubits:
        return 1.0

    exp = 0.0
    for bitstr, c in counts.items():
        bits_q = bitstr.replace(" ", "")[::-1]
        s = sum(1 for qi in active_qubits if bits_q[qi] == "1")
        exp += (1.0 if s % 2 == 0 else -1.0) * (c / shots)
    return exp

def estimate_term_expectation(backend, ansatz, theta, pauli_label: str, shots: int, mitigator=None):
    qc = QuantumCircuit(ansatz.num_qubits)
    qc.compose(ansatz.assign_parameters(theta), inplace=True)
    _basis_change_for_pauli(qc, pauli_label)
    qc.measure_all()

    for _ in range(6):
        new_qc = qc.decompose()
        if new_qc == qc:
            break
        qc = new_qc

    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()
    if mitigator is not None:
        counts = mitigator.mitigate_counts(counts, ansatz.num_qubits)

    return pauli_expectation_from_counts(pauli_label, counts)
