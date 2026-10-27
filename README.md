# Student Performance Analysis & Prediction

Proyek Penambangan Data yang mengeksplorasi faktor-faktor terkait nilai akhir matematika dan membandingkan beberapa model regresi untuk memprediksi nilai akhir siswa.

> **Penting:** metrik, grafik, dan kesimpulan empiris baru tersedia setelah pipeline dijalankan. Repository ini tidak mengklaim hasil model sebelum eksperimen dilakukan.

## Hubungan dengan IBM Machine Learning Coursera
- **Exploratory Data Analysis for Machine Learning:** pemeriksaan data, statistik deskriptif, visualisasi distribusi, dan eksplorasi hubungan fitur dengan target.
- **Supervised Machine Learning: Regression:** train/test split, Linear Regression, Ridge, Random Forest Regressor, dan evaluasi MAE, RMSE, serta R².

## Tujuan
1. Memahami distribusi nilai akhir matematika.
2. Mengeksplorasi pola nilai berdasarkan fitur siswa yang tersedia.
3. Membandingkan baseline dan model regresi.
4. Mengukur kesalahan prediksi pada data test.
5. Menyajikan hasil yang dapat direproduksi.

## Dataset
Gunakan [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance), yang berisi data prestasi siswa dari dua sekolah di Portugal. Proyek menggunakan `student-mat.csv` (mata pelajaran matematika). Dataset mencakup nilai G1, G2, G3 serta informasi sosial, keluarga, dan sekolah.

Catatan metodologis: G1 dan G2 berkorelasi kuat dengan G3. Karena itu pipeline menjalankan **dua eksperimen terpisah**: tanpa G1/G2 (prediksi lebih awal) dan dengan G1/G2 (prediksi setelah nilai periode sebelumnya tersedia).

## Struktur repository
```text
config/ (opsional bila ditambahkan)
data/             instruksi dataset
docs/             metodologi, data dictionary, pertanyaan penelitian
notebooks/        notebook analisis
outputs/figures/  grafik hasil
outputs/tables/   data bersih, metrik, prediksi
outputs/reports/  ringkasan eksperimen
scripts/          pipeline end-to-end
src/              modul Python
tests/            pengujian
```

## Menjalankan di VS Code (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/run_pipeline.py
pytest -q
```

Pipeline mengunduh dataset resmi dari UCI bila `data/raw/student-mat.csv` belum ada. Jika unduhan gagal, unduh arsip dari halaman UCI, ekstrak `student-mat.csv`, dan simpan pada lokasi tersebut.

## Output
- `outputs/figures/`: distribusi nilai, grafik fitur, korelasi, aktual vs prediksi.
- `outputs/tables/`: data bersih, metrik dua skenario, tabel prediksi.
- `outputs/reports/summary.md`: ringkasan hasil aktual dari eksekusi.

## Metrik evaluasi
- **MAE:** rata-rata selisih absolut nilai prediksi dengan nilai aktual; makin rendah makin baik.
- **RMSE:** memberi penalti lebih besar pada kesalahan besar; makin rendah makin baik.
- **R²:** ukuran variasi target yang dijelaskan model pada data evaluasi; bukan persentase akurasi dan dapat bernilai negatif.

## Batasan dan etika
Data berasal dari dua sekolah di Portugal dan tidak otomatis mewakili sekolah atau siswa di Indonesia. Model tidak boleh dipakai sebagai satu-satunya dasar keputusan akademik. Hubungan yang ditemukan tidak membuktikan sebab-akibat.

## Sumber
Cortez, P. & Silva, A. (2008), *Using data mining to predict secondary school student performance*. Dataset: UCI Machine Learning Repository, Student Performance.
