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

# choose representative cases
targets = ["H4_sq_d+0.00_a1.00", "H4_sq_d+0.15_a1.00", "H4_sq_d+0.30_a1.00"]

ref_df = pd.read_csv("data/h4/h4_refs.csv")

rows = []
trace_dir = Path("results/traces_h4")
trace_dir.mkdir(parents=True, exist_ok=True)

for name in targets:
    print(f"\nStarting {name}", flush=True)
    row = ref_df[ref_df["name"] == name].iloc[0]

    print(f"\nStarting {name}", flush=True)

    with open(f"data/h4/{name}_ham.pkl", "rb") as f:
        qham = pickle.load(f)

    theta0 = np.load(f"data/h4/thetas/{name}_theta_opt.npy")

    n_qubits = count_qubits(qham)
    ansatz = make_ansatz(n_qubits, reps=4)
    op = qubitop_to_sparsepauliop(qham, n_qubits)

    noise_model = make_noise_model(p1=2e-4, p2=2e-3, p_readout=0.01)
    backend = AerSimulator(noise_model=noise_model, seed_simulator=7)

    # calibration
    print("  Calibrating readout...", flush=True)
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
             shots_per_term=500,
             mitigator=mitigator,
         ) 
    print("  Running SPSA...", flush=True)
    best_theta, best_E, trace = spsa_optimize(
        energy_fn=energy_fn,
        theta0=theta0,
        n_iter=20,
        a=0.008,
        c=0.02,
        alpha=0.602,
        gamma=0.101,
        seed=7,
    )

    #np.save(trace_dir / f"{name}_trace.npy", np.array(trace))
    pd.DataFrame(trace).to_csv(trace_dir / f"{name}_trace.csv", index=False)
    E_FCI = row["E_FCI"]

    rows.append({
        "name": name,
        "param": row["param"],
        "mr_score": row["mr_score"],
        "E_FCI": E_FCI,
        "E_noisy_best": best_E,
        "best_error": abs(best_E - E_FCI),
    })

    print(f"Finished {name}: best_E = {best_E:.6f}", flush=True)

Path("results/csv").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv("results/csv/h4_noisy_optimized.csv", index=False)

print("\nSaved H4 noisy SPSA results.")
