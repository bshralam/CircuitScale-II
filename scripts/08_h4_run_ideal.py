from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from openfermion import count_qubits

from src.ansatze import make_ansatz
from src.vqe_ideal import qubitop_to_sparsepauliop, run_vqe_ideal

ref_df = pd.read_csv("data/h4/h4_refs.csv")
rows = []

theta_dir = Path("data/h4/thetas")
theta_dir.mkdir(parents=True, exist_ok=True)

for _, row in ref_df.iterrows():
    name = row["name"]
    print(f"Running ideal VQE for {name}", flush=True)

    with open(f"data/h4/{name}_ham.pkl", "rb") as f:
        qham = pickle.load(f)

    n_qubits = count_qubits(qham)

    # H4 is harder than H2; use reps=4 as first baseline
    ansatz = make_ansatz(n_qubits, reps=4)
    op = qubitop_to_sparsepauliop(qham, n_qubits)

    # E_ideal, raw = run_vqe_ideal(op, ansatz, maxiter=1000, seed=7)
    # np.save(theta_dir / f"{name}_theta_opt.npy", raw.x)

    best_E = None
    best_raw = None
    best_seed = None

    for seed in [1, 2, 3, 4, 5]:
        E_try, raw_try = run_vqe_ideal(op, ansatz, maxiter=1000, seed=seed)
        if best_E is None or E_try < best_E:
            best_E = E_try
            best_raw = raw_try
            best_seed = seed

    E_ideal = best_E
    raw = best_raw

    print(f"Best seed for {name}: {best_seed}, E = {E_ideal:.6f}", flush=True)

    np.save(theta_dir / f"{name}_theta_opt.npy", raw.x)

    rows.append({
        "name": name,
        "param": row["param"],
        "mr_score": row["mr_score"],
        "E_FCI": row["E_FCI"],
        "E_ideal": E_ideal,
        "ideal_error": abs(E_ideal - row["E_FCI"]),
    })

Path("results/csv").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv("results/csv/h4_ideal.csv", index=False)

print("Saved H4 ideal VQE results.")
