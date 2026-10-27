# Metodologi

Proyek mengadaptasi alur kerja data science:
1. **Business understanding:** mempelajari faktor yang berhubungan dengan nilai akhir dan mengevaluasi kemungkinan memprediksi nilai.
2. **Data understanding / EDA:** memahami kolom, tipe data, distribusi target, dan hubungan awal antarvariabel.
3. **Data preparation:** menghapus duplikasi, menangani target yang kosong, dan menyiapkan fitur numerik/kategorikal.
4. **Modeling:** membandingkan Mean Baseline, Linear Regression, Ridge Regression, dan Random Forest Regressor.
5. **Evaluation:** menggunakan train/test split yang tetap dan metrik MAE, RMSE, serta R².
6. **Communication:** menyimpan grafik, tabel metrik, prediksi, dan laporan ringkasan.

## Dua skenario prediksi
- **Tanpa G1 dan G2:** mencoba prediksi tanpa nilai periode pertama/kedua. Lebih relevan untuk skenario prediksi lebih awal, tetapi bisa lebih sulit.
- **Dengan G1 dan G2:** memakai nilai periode pertama dan kedua untuk memprediksi nilai akhir G3. Umumnya lebih informatif, tetapi hanya adil bila nilai G1/G2 memang tersedia saat prediksi.

Kedua skenario dilaporkan terpisah agar tidak menyamakan penggunaan informasi awal dan informasi nilai yang sudah berjalan. MAE lebih rendah berarti kesalahan absolut rata-rata lebih kecil. R² tidak selalu positif dan bukan persentase akurasi.
