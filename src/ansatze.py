from qiskit.circuit.library import EfficientSU2

def make_ansatz(n_qubits: int, reps: int = 2):
    return EfficientSU2(num_qubits=n_qubits, reps=reps, entanglement="linear")
