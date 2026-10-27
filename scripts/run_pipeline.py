"""Run EDA and regression experiments."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import load_student_data
from src.preprocessing import clean_student_data
from src.eda import create_eda_outputs
from src.modeling import evaluate_models, save_model_outputs
import pandas as pd
import matplotlib.pyplot as plt

def main():
    raw = load_student_data(ROOT / "data/raw/student-mat.csv")
    clean = clean_student_data(raw)
    (ROOT / "outputs/tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs/reports").mkdir(parents=True, exist_ok=True)
    clean.to_csv(ROOT / "outputs/tables/clean_student_data.csv", index=False)
    create_eda_outputs(clean, ROOT / "outputs/figures")

    # Experiment A: more realistic early prediction without G1/G2.
    metrics_a, fitted_a, best_a, _ = evaluate_models(clean, include_previous_grades=False)
    metrics_a.to_csv(ROOT / "outputs/tables/regression_metrics_without_G1_G2.csv", index=False)
    # Experiment B: prediction after first and second period grades are available.
    metrics_b, fitted_b, best_b, _ = evaluate_models(clean, include_previous_grades=True)
    metrics_b.to_csv(ROOT / "outputs/tables/regression_metrics_with_G1_G2.csv", index=False)

    comparison = save_model_outputs(metrics_a, fitted_a, best_a, ROOT / "outputs/tables")
    # Comparison chart for experiment A
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(comparison["actual_G3"], comparison["predicted_G3"], alpha=.75)
    ax.plot([0, 20], [0, 20], linestyle="--")
    ax.set(xlim=(0, 20), ylim=(0, 20), title=f"Aktual vs Prediksi G3 — {best_a} (tanpa G1/G2)",
           xlabel="Nilai aktual G3", ylabel="Nilai prediksi G3")
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/figures/actual_vs_predicted.png", dpi=160)
    plt.close(fig)

    report = f"""# Ringkasan Eksperimen

## Data
- Baris dataset mentah: {len(raw)}
- Baris setelah pembersihan: {len(clean)}
- Target: G3 (nilai akhir matematika, skala dataset 0–20)

## Eksperimen A — tanpa G1 dan G2
Fitur nilai periode sebelumnya tidak digunakan. Model terbaik berdasarkan MAE pada test split: **{best_a}**.

{metrics_a.to_markdown(index=False)}

## Eksperimen B — dengan G1 dan G2
Nilai periode pertama dan kedua disertakan. Ini merupakan skenario berbeda karena informasi nilai sebelumnya sudah tersedia saat prediksi.

{metrics_b.to_markdown(index=False)}

## Interpretasi
Gunakan MAE dan RMSE untuk memahami besarnya kesalahan prediksi dalam satuan nilai. R² menunjukkan proporsi variasi target yang dijelaskan model pada test set; nilai negatif dapat terjadi. Jangan menyimpulkan hubungan sebab-akibat dari korelasi.

## Batasan
Dataset berasal dari dua sekolah di Portugal dan berukuran relatif kecil. Hasil test split dapat berubah jika pembagian data berubah. Eksperimen ini bukan alat untuk mengambil keputusan penting tentang siswa secara otomatis.
"""
    (ROOT / "outputs/reports/summary.md").write_text(report, encoding="utf-8")
    print("Pipeline selesai. Periksa outputs/tables, outputs/figures, dan outputs/reports.")
    print("Model terbaik tanpa G1/G2 berdasarkan MAE:", best_a)
    print("Model terbaik dengan G1/G2 berdasarkan MAE:", best_b)

if __name__ == "__main__":
    main()
