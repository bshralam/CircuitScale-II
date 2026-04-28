from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/csv/h4_noisy_singlepoint.csv")

Path("results/figures").mkdir(parents=True, exist_ok=True)

# 1. Error vs distortion (delta)
plt.figure(figsize=(6, 4))
plt.plot(df["param"], df["raw_error"], marker="o", label="Raw")
plt.plot(df["param"], df["mit_error"], marker="o", label="Mitigated")
plt.xlabel("Distortion parameter δ")
plt.ylabel("VQE error (Ha)")
plt.title("H4 noisy VQE error vs distortion")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("results/figures/h4_error_vs_delta.png", dpi=300)
plt.close()


# 2. Error vs MR score
plt.figure(figsize=(6, 4))
plt.scatter(df["mr_score"], df["raw_error"], label="Raw")
plt.scatter(df["mr_score"], df["mit_error"], label="Mitigated")
plt.xlabel("MR score")
plt.ylabel("VQE error (Ha)")
plt.title("H4 noisy VQE error vs multireference character")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("results/figures/h4_error_vs_mr.png", dpi=300)
plt.close()

print("Saved H4 plots.")
