from pathlib import Path
from src.analysis import plot_metric_vs_param

plot_metric_vs_param(
    csv_path=Path("results/csv/h2_noisy_singlepoint.csv"),
    ycol="raw_error",
    out_png=Path("results/figures/h2_raw_error_vs_R.png"),
    xlabel="H-H distance R (Å)",
    ylabel="Noisy VQE error (Ha)",
    title="Raw noisy VQE error vs bond distance (H2)",
)

plot_metric_vs_param(
    csv_path=Path("results/csv/h2_noisy_singlepoint.csv"),
    ycol="mit_error",
    out_png=Path("results/figures/h2_mit_error_vs_R.png"),
    xlabel="H-H distance R (Å)",
    ylabel="Mitigated noisy VQE error (Ha)",
    title="Mitigated noisy VQE error vs bond distance (H2)",
)

print("Saved H2 plots.")
