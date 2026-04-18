from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from openfermion import count_qubits

from src.ansatze import make_ansatz
from src.vqe_ideal import qubitop_to_sparsepauliop, run_vqe_ideal

ref_df = pd.read_csv("data/h2/h2_refs.csv")
rows = []

theta_dir = Path("data/h2/thetas")
theta_dir.mkdir(parents=True, exist_ok=True)

for _, row in ref_df.iterrows():
    name = row["name"]
    with open(f"data/h2/{name}_ham.pkl", "rb") as f:
        qham = pickle.load(f)

    n_qubits = count_qubits(qham)
    ansatz = make_ansatz(n_qubits, reps=2)
    op = qubitop_to_sparsepauliop(qham, n_qubits)

    E_ideal, raw = run_vqe_ideal(op, ansatz, maxiter=250, seed=7)
    np.save(theta_dir / f"{name}_theta_opt.npy", raw.optimal_point)

    rows.append({
        "name": name,
        "param": row["param"],
        "mr_score": row["mr_score"],
        "E_FCI": row["E_FCI"],
        "E_ideal": E_ideal,
        "ideal_error": abs(E_ideal - row["E_FCI"]),
    })

pd.DataFrame(rows).to_csv("results/csv/h2_ideal.csv", index=False)
print("Saved ideal H2 results.")
