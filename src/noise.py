from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors import depolarizing_error, ReadoutError

def make_noise_model(p1: float = 2e-4, p2: float = 2e-3, p_readout: float = 0.01) -> NoiseModel:
    noise_model = NoiseModel()

    e1 = depolarizing_error(p1, 1)
    e2 = depolarizing_error(p2, 2)

    noise_model.add_all_qubit_quantum_error(
        e1, ["x", "y", "z", "h", "s", "sdg", "rx", "ry", "rz"]
    )
    noise_model.add_all_qubit_quantum_error(
        e2, ["cx", "cz", "swap"]
    )

    ro = ReadoutError([[1 - p_readout, p_readout],
                       [p_readout, 1 - p_readout]])
    noise_model.add_all_qubit_readout_error(ro)

    return noise_model
