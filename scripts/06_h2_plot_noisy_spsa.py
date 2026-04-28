from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/csv/h2_noisy_optimized.csv")

# 1. best optimized error vs R
plt.figure(figsize=(6, 4))
plt.plot(df["param"], df["best_error"], marker="o")
plt.xlabel("H-H distance R (Å)")
plt.ylabel("Optimized noisy VQE error (Ha)")
plt.title("Best noisy optimized error vs bond distance (H2)")
plt.tight_layout()
plt.savefig("results/figures/h2_best_noisy_error_vs_R.png", dpi=300)
plt.close()

# 2. improvement vs mitigated single-point
plt.figure(figsize=(6, 4))
plt.plot(df["param"], df["improvement_vs_mit"], marker="o")
plt.xlabel("H-H distance R (Å)")
plt.ylabel("Improvement over mitigated single-point (Ha)")
plt.title("SPSA improvement vs bond distance (H2)")
plt.tight_layout()
plt.savefig("results/figures/h2_spsa_improvement_vs_R.png", dpi=300)
plt.close()

# 3. representative traces
trace_dir = Path("results/csv/traces")
trace_files = [
    trace_dir / "H2_R0.70_trace.csv",
    trace_dir / "H2_R1.54_trace.csv",
    trace_dir / "H2_R3.00_trace.csv",
]

plt.figure(figsize=(7, 5))
for tf in trace_files:
    if tf.exists():
        tdf = pd.read_csv(tf)
        label = tf.stem.replace("_trace", "")
        plt.plot(tdf["iter"], tdf["best_E"], label=label)
plt.xlabel("SPSA iteration")
plt.ylabel("Best noisy energy so far (Ha)")
plt.title("Representative noisy SPSA traces (H2)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("results/figures/h2_spsa_traces.png", dpi=300)
plt.close()

print("Saved H2 noisy SPSA plots.")
