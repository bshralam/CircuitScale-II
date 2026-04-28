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
from src.vqe_noisy import estimate_energy_noisy, spsa_optimize

# representative geometries for saving full traces
TRACE_POINTS = {"H2_R0.70", "H2_R1.54", "H2_R3.00"}

ref_df = pd.read_csv("data/h2/h2_refs.csv")
single_df = pd.read_csv("results/csv/h2_noisy_singlepoint.csv")

single_map = {row["name"]: row for _, row in single_df.iterrows()}

rows = []
trace_dir = Path("results/csv/traces")
trace_dir.mkdir(parents=True, exist_ok=True)

for _, row in ref_df.iterrows():
    name = row["name"]
    print(f"\nStarting {name}", flush=True)
    with open(f"data/h2/{name}_ham.pkl", "rb") as f:
        qham = pickle.load(f)

    theta0 = np.load(f"data/h2/thetas/{name}_theta_opt.npy")
    n_qubits = count_qubits(qham)

    # keep same depth as your best ideal H2 run
    ansatz = make_ansatz(n_qubits, reps=4)
    op = qubitop_to_sparsepauliop(qham, n_qubits)

    noise_model = make_noise_model(p1=2e-4, p2=2e-3, p_readout=0.01)
    backend = AerSimulator(noise_model=noise_model, seed_simulator=7)

    # readout calibration
    cal = build_readout_cal_circuits(n_qubits)
    prepared = [b for (b, _) in cal]
    circuits = [qc for (_, qc) in cal]
    job = backend.run(circuits, shots=20000)
    result = job.result()
    counts_list = [result.get_counts(i) for i in range(len(circuits))]
    A = fit_assignment_matrix(n_qubits, prepared, counts_list)
    mitigator = ReadoutMitigator(A, method="pinv")

    def energy_fn(theta):
        return estimate_energy_noisy(
            backend=backend,
            ansatz=ansatz,
            theta=theta,
            op=op,
            shots_per_term=2000,
            mitigator=mitigator,
        )

    best_theta, best_E, trace = spsa_optimize(
        energy_fn=energy_fn,
        theta0=theta0,
        n_iter=80,
        a=0.008,
        c=0.02,
        alpha=0.602,
        gamma=0.101,
        seed=7,
    )
    print(f"Finished {name}: best_E = {best_E:.6f}", flush=True)

    sp = single_map[name]
    raw_error = float(sp["raw_error"])
    mit_error = float(sp["mit_error"])
    best_error = abs(best_E - row["E_FCI"])

    rows.append({
        "name": name,
        "param": row["param"],
        "mr_score": row["mr_score"],
        "E_FCI": row["E_FCI"],
        "E_noisy_raw": sp["E_noisy_raw"],
        "E_noisy_mit": sp["E_noisy_mit"],
        "E_noisy_best": best_E,
        "raw_error": raw_error,
        "mit_error": mit_error,
        "best_error": best_error,
        "improvement_vs_raw": raw_error - best_error,
        "improvement_vs_mit": mit_error - best_error,
    })

    if name in TRACE_POINTS:
        pd.DataFrame(trace).to_csv(trace_dir / f"{name}_trace.csv", index=False)

pd.DataFrame(rows).to_csv("results/csv/h2_noisy_optimized.csv", index=False)
print("Saved H2 noisy SPSA results.")
