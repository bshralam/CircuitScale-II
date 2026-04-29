from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

Path("results/figures").mkdir(parents=True, exist_ok=True)

opt_df = pd.read_csv("results/csv/h4_noisy_optimized.csv")
single_df = pd.read_csv("results/csv/h4_noisy_singlepoint.csv")

# Keep only the geometries used in SPSA
single_sel = single_df[single_df["name"].isin(opt_df["name"])].copy()

# Merge optimized and single-point results
df = opt_df.merge(
    single_sel[["name", "E_noisy_mit", "mit_error", "E_noisy_raw", "raw_error"]],
    on="name",
    how="left",
)

df = df.sort_values("param")

# 1. Compare raw, mitigated single-point, and SPSA optimized errors
plt.figure(figsize=(6, 4))
plt.plot(df["param"], df["raw_error"], marker="o", label="Raw noisy")
plt.plot(df["param"], df["mit_error"], marker="o", label="Mitigated single-point")
plt.plot(df["param"], df["best_error"], marker="o", label="Noisy SPSA optimized")
plt.xlabel("Distortion parameter δ")
plt.ylabel("VQE error (Ha)")
plt.title("H4 noisy VQE error: single-point vs SPSA")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("results/figures/h4_spsa_vs_singlepoint_error.png", dpi=300)
plt.close()

# 2. SPSA improvement over mitigated single-point
df["spsa_improvement_vs_mit"] = df["mit_error"] - df["best_error"]

plt.figure(figsize=(6, 4))
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.plot(df["param"], df["spsa_improvement_vs_mit"], marker="o")
plt.xlabel("Distortion parameter δ")
plt.ylabel("Improvement over mitigated single-point (Ha)")
plt.title("H4 SPSA improvement vs distortion")
plt.tight_layout()
plt.savefig("results/figures/h4_spsa_improvement_vs_delta.png", dpi=300)
plt.close()

# Save merged comparison table
df.to_csv("results/csv/h4_spsa_comparison.csv", index=False)

print("Saved H4 noisy SPSA plots and comparison CSV.")
