from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from openfermion import count_qubits
from qiskit_aer import AerSimulator

from src.ansatze import make_ansatz
from src.vqe_ideal import qubitop_to_sparsepauliop
from src.noise import make_noise_model
from src.mitigation import build_readout_cal_circuits, fit_assignment_matrix, ReadoutMitigator
from src.vqe_noisy import estimate_energy_noisy

ref_df = pd.read_csv("data/h2/h2_refs.csv")
rows = []

for _, row in ref_df.iterrows():
    name = row["name"]

    with open(f"data/h2/{name}_ham.pkl", "rb") as f:
        qham = pickle.load(f)

    theta = np.load(f"data/h2/thetas/{name}_theta_opt.npy")
    n_qubits = count_qubits(qham)
    ansatz = make_ansatz(n_qubits, reps=2)
    op = qubitop_to_sparsepauliop(qham, n_qubits)

    noise_model = make_noise_model(p1=2e-4, p2=2e-3, p_readout=0.01)
    backend = AerSimulator(noise_model=noise_model, seed_simulator=7)

    cal = build_readout_cal_circuits(n_qubits)
    prepared = [b for (b, _) in cal]
    circuits = [qc for (_, qc) in cal]
    job = backend.run(circuits, shots=20000)
    result = job.result()
    counts_list = [result.get_counts(i) for i in range(len(circuits))]
    A = fit_assignment_matrix(n_qubits, prepared, counts_list)
    mitigator = ReadoutMitigator(A, method="pinv")

    E_raw = estimate_energy_noisy(backend, ansatz, theta, op, shots_per_term=2000, mitigator=None)
    E_mit = estimate_energy_noisy(backend, ansatz, theta, op, shots_per_term=2000, mitigator=mitigator)

    rows.append({
        "name": name,
        "param": row["param"],
        "mr_score": row["mr_score"],
        "E_FCI": row["E_FCI"],
        "E_ideal": row["E_ideal"] if "E_ideal" in row else np.nan,
        "E_noisy_raw": E_raw,
        "E_noisy_mit": E_mit,
        "raw_error": abs(E_raw - row["E_FCI"]),
        "mit_error": abs(E_mit - row["E_FCI"]),
    })

pd.DataFrame(rows).to_csv("results/csv/h2_noisy_singlepoint.csv", index=False)
print("Saved H2 noisy single-point results.")
