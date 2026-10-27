"""Exploratory analysis and figure generation."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_eda_outputs(df: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["G3"], bins=15, kde=True, ax=ax)
    ax.set(title="Distribusi Nilai Akhir (G3)", xlabel="Nilai akhir (0–20)", ylabel="Jumlah siswa")
    fig.tight_layout()
    fig.savefig(output_dir / "g3_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="studytime", y="G3", ax=ax)
    ax.set(title="Nilai Akhir berdasarkan Kategori Waktu Belajar", xlabel="Kategori waktu belajar mingguan", ylabel="Nilai akhir (G3)")
    fig.tight_layout()
    fig.savefig(output_dir / "studytime_vs_g3.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="failures", y="G3", ax=ax)
    ax.set(title="Nilai Akhir berdasarkan Riwayat Kegagalan Kelas", xlabel="Jumlah kegagalan kelas sebelumnya (kode dataset)", ylabel="Nilai akhir (G3)")
    fig.tight_layout()
    fig.savefig(output_dir / "failures_vs_g3.png", dpi=160)
    plt.close(fig)

    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Korelasi Antarvariabel Numerik")
    fig.tight_layout()
    fig.savefig(output_dir / "numeric_correlation.png", dpi=160)
    plt.close(fig)
