from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric_vs_param(csv_path: Path, ycol: str, out_png: Path, xlabel: str, ylabel: str, title: str):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(6, 4))
    plt.plot(df["param"], df[ycol], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
