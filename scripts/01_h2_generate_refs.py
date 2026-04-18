from pathlib import Path
import json
import numpy as np
import pandas as pd
import pickle

from src.systems import h2_scan
from src.chemistry import compute_reference, molecular_qubit_hamiltonian

R_values = np.linspace(0.70, 3.00, 12)
specs = h2_scan(R_values)

rows = []
outdir = Path("data/h2")
outdir.mkdir(parents=True, exist_ok=True)

for spec in specs:
    ref = compute_reference(spec, ncas=2, nelecas=(1, 1))
    qham, n_qubits = molecular_qubit_hamiltonian(spec)

    rows.append({
        "name": spec.name,
        "param": float(spec.name.split("R")[1]),
        "mr_score": ref["mr_score"],
        "E_HF": ref["E_HF"],
        "E_FCI": ref["E_FCI"],
        "noon_0": float(ref["noons"][0]),
        "noon_1": float(ref["noons"][1]),
        "n_qubits": n_qubits,
    })

    with open(outdir / f"{spec.name}_ham.pkl", "wb") as f:
        pickle.dump(qham, f)

pd.DataFrame(rows).to_csv(outdir / "h2_refs.csv", index=False)
print("Saved H2 reference data.")
