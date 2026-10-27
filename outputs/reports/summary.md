# Ringkasan Eksperimen

## Data
- Baris dataset mentah: 395
- Baris setelah pembersihan: 395
- Target: G3 (nilai akhir matematika, skala dataset 0–20)

## Eksperimen A — tanpa G1 dan G2
Fitur nilai periode sebelumnya tidak digunakan. Model terbaik berdasarkan MAE pada test split: **RandomForest**.

| model            |     mae |    rmse |          r2 |
|:-----------------|--------:|--------:|------------:|
| RandomForest     | 3.0412  | 3.80397 |  0.294312   |
| Ridge            | 3.39287 | 4.1934  |  0.142427   |
| LinearRegression | 3.39526 | 4.19568 |  0.141492   |
| MeanBaseline     | 3.64585 | 4.55018 | -0.00970964 |

## Eksperimen B — dengan G1 dan G2
Nilai periode pertama dan kedua disertakan. Ini merupakan skenario berbeda karena informasi nilai sebelumnya sudah tersedia saat prediksi.

| model            |     mae |    rmse |          r2 |
|:-----------------|--------:|--------:|------------:|
| RandomForest     | 1.21633 | 2.0326  |  0.798515   |
| Ridge            | 1.63907 | 2.37153 |  0.725718   |
| LinearRegression | 1.64667 | 2.37837 |  0.724134   |
| MeanBaseline     | 3.64585 | 4.55018 | -0.00970964 |

## Interpretasi
Gunakan MAE dan RMSE untuk memahami besarnya kesalahan prediksi dalam satuan nilai. R² menunjukkan proporsi variasi target yang dijelaskan model pada test set; nilai negatif dapat terjadi. Jangan menyimpulkan hubungan sebab-akibat dari korelasi.

## Batasan
Dataset berasal dari dua sekolah di Portugal dan berukuran relatif kecil. Hasil test split dapat berubah jika pembagian data berubah. Eksperimen ini bukan alat untuk mengambil keputusan penting tentang siswa secara otomatis.
