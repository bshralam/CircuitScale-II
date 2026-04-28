from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from src.systems import h4_distortion_grid
from src.chemistry import compute_reference, molecular_qubit_hamiltonian

deltas = np.linspace(-0.30, 0.30, 9)
specs = h4_distortion_grid(deltas, a=1.0)

outdir = Path("data/h4")
outdir.mkdir(parents=True, exist_ok=True)

rows = []

for spec in specs:
    print(f"Running references for {spec.name}", flush=True)

    ref = compute_reference(
        spec,
        ncas=4,
        nelecas=(2, 2),
    )

    qham, n_qubits = molecular_qubit_hamiltonian(spec)

    row = {
        "name": spec.name,
        "param": float(spec.name.split("_d")[1].split("_a")[0]),
        "mr_score": ref["mr_score"],
        "E_HF": ref["E_HF"],
        "E_FCI": ref["E_FCI"],
        "E_CASSCF": ref["E_CASSCF"],
        "n_qubits": n_qubits,
    }

    for i, occ in enumerate(ref["noons"]):
        row[f"noon_{i}"] = float(occ)

    rows.append(row)

    with open(outdir / f"{spec.name}_ham.pkl", "wb") as f:
        pickle.dump(qham, f)

pd.DataFrame(rows).to_csv(outdir / "h4_refs.csv", index=False)
print("Saved H4 reference data.")
