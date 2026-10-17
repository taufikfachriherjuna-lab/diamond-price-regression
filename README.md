# 💎 Diamond Price Regression

Proyek *machine learning* **regresi** untuk memprediksi harga berlian (`price`, USD) dari karakteristiknya,
menggunakan dataset **Diamonds** (53.940 baris, 10 kolom). Alur *data cleaning* dan *transformation* mengikuti
materi **Latihan: Data Cleaning dan Transformation**.

## 📊 Alur Kerja

1. **Data loading:** `diamonds.csv` dibagi 80% latih / 20% uji (`random_state=42`) menjadi `train.csv` (43.152 baris) dan `test.csv` (10.788 baris). Pembagian dilakukan **sebelum** cleaning untuk mencegah *data leakage*.
2. **Identifikasi dataset & statistik deskriptif:** `info()`, `describe(include="all")`, skewness, dan distribusi kategori.
3. **Missing value:** tidak ditemukan (0 nilai hilang). Kode pemisahan kolom batas 75%, pengisian median/modus, dan penghapusan kolom tetap dijalankan sebagai pemeriksaan.
4. **Outlier (IQR):** 3.687 baris (8,5%) dibuang dari data latih berdasarkan fitur `carat`, `depth`, `table`, `x`, `y`, `z`. Target `price` **tidak** dipangkas.
5. **Duplikasi:** 74 baris duplikat dibuang. Data latih akhir: **39.391 baris**.
6. **Encoding:** `cut`, `color`, `clarity` bersifat ordinal, sehingga dipakai `OrdinalEncoder` dengan urutan kualitas eksplisit. One-hot dan `LabelEncoder` dibandingkan di notebook; `LabelEncoder` ditolak karena mengurutkan secara alfabet (`Ideal` < `Very Good`).
7. **Standardisasi:** `StandardScaler` pada fitur (target tidak), di-*fit* hanya pada data latih.
8. **Cleaning data uji:** semua acuan (median/modus, batas IQR, encoder, scaler) berasal dari data latih. Baris uji yang identik dengan baris latih dibuang. Data uji akhir: **9.800 baris**.
9. **Model:** Linear Regression, Ridge Regression (`alpha` dari `GridSearchCV` 5-fold, hasilnya `alpha=10`), dan Gradient Boosting Regressor.
10. **Evaluasi:** MAE, MSE, RMSE, dan R². Model terbaik (RMSE terendah) disimpan ke `best_model.joblib`.

## 🏆 Hasil Evaluasi (data uji, 9.800 baris)

| Model | MAE (USD) | MSE | RMSE (USD) | R² (Test) | R² (Train) |
|---|---:|---:|---:|---:|---:|
| **Gradient Boosting** | **232,36** | **180.454** | **424,80** | **0,9846** | 0,9883 |
| Linear Regression | 715,63 | 1.119.455 | 1.058,04 | 0,9043 | 0,9067 |
| Ridge Regression | 716,28 | 1.119.938 | 1.058,27 | 0,9042 | 0,9067 |

**Model terbaik: Gradient Boosting Regressor.** Rata-rata galat prediksinya sekitar $232, jauh di bawah model linear (sekitar $716), karena hubungan fitur dan harga bersifat non-linear. Selisih R² latih dan uji kecil, sehingga tidak ada tanda *overfitting* yang berarti.

> Angka berasal dari eksekusi `tugasml.ipynb` dengan `random_state=42` (scikit-learn 1.8.0, pandas 3.0.2). Hasil bisa sedikit berbeda pada versi *library* lain.

## ⚠️ Catatan dan Keterbatasan

- **Pemangkasan outlier membuang berlian besar.** Batas IQR untuk `carat` adalah 2,00, sehingga berlian di atas 2 karat tidak ada di data latih maupun data uji. Model dan angka di atas valid untuk berlian berukuran umum, **tidak untuk berlian sangat besar**.
- **Data uji ikut difilter** (988 dari 10.788 baris dibuang oleh aturan yang sama), sehingga metrik tidak sebanding langsung dengan evaluasi pada data mentah yang utuh.
- Ridge dan Linear Regression nyaris identik karena data banyak dengan fitur sedikit, jadi regularisasi hampir tidak berpengaruh.

## 📁 Struktur Proyek

```
diamond-price-regression/
├── README.md             # Dokumentasi proyek
├── requirements.txt      # Daftar dependensi
├── diamonds.csv          # Dataset asli
├── tugasml.ipynb         # Notebook utama (cleaning, transformation, training, evaluasi)
├── train.csv             # Dihasilkan notebook (data latih sebelum cleaning)
├── test.csv              # Dihasilkan notebook (data uji sebelum cleaning)
├── best_model.joblib     # Dihasilkan notebook (model + encoder + scaler)
└── hasil_evaluasi.csv    # Dihasilkan notebook (tabel perbandingan model)
```

File yang ditandai *dihasilkan notebook* dibuat ulang setiap notebook dijalankan.

## 💻 Cara Menjalankan

```bash
git clone https://github.com/pratiwids/diamond-price-regression.git
cd diamond-price-regression

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook tugasml.ipynb  # lalu Run All
```

## 🔮 Memakai Model yang Sudah Disimpan

`best_model.joblib` berisi paket (model, `OrdinalEncoder`, `StandardScaler`, dan nama fitur), karena data baru harus
melewati encoding dan standardisasi yang sama dengan data latih.

```python
import joblib
import pandas as pd

paket = joblib.load("best_model.joblib")

def prediksi_harga(data_baru, paket):
    d = pd.DataFrame(data_baru)
    d[paket["category_features"]] = paket["ordinal_encoder"].transform(d[paket["category_features"]])
    d = d[paket["feature_names"]]
    return paket["model"].predict(paket["scaler"].transform(d))

berlian = [{"carat": 1.0, "cut": "Ideal", "color": "G", "clarity": "VS2",
            "depth": 61.5, "table": 56.0, "x": 6.4, "y": 6.4, "z": 3.95}]
print(prediksi_harga(berlian, paket))   # sekitar 6.274 USD pada eksekusi yang diverifikasi
```

> ⚠️ File `.joblib` terikat pada versi `scikit-learn` saat model dilatih (diverifikasi dengan 1.8.0). Jika versinya berbeda, jalankan ulang notebook untuk membuat model baru.

## 📚 Sumber Data

Dataset Diamonds dari repositori ggplot2 (tidyverse):
<https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv>

## 🛠️ Teknologi

Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · joblib
# diamond-price-regression
