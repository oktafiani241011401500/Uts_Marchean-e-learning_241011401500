# UTS Machine Learning – Linear Regression & Logistic Regression

- **Nama** : Oktafiani Cindriwanti Ari
- **NIM** : 241011401500
- **Kelas** : Reg B / Reg C *(coret salah satu)*
- **random_state** : 100 (2 digit terakhir NIM = `00`)

## Isi Repositori
| File | Keterangan |
|---|---|
| `tugas_UTS.ipynb` | Notebook jawaban lengkap (sudah dijalankan, beserta output dan grafik) |
| `Ads_Data.csv` | Dataset biaya iklan vs penjualan (40 data) |
| `Email_Data.csv` | Dataset email spam (60 data) |

Semua model dan metrik (`train_test_split`, `LinearRegression`, `LinearRegressionGD`, `LogisticRegression`, MAE/MSE/RMSE/R², accuracy, confusion matrix, precision, recall, F1) diimplementasikan sendiri dengan NumPy. scikit-learn hanya dipakai pada soal bonus.

## Ringkasan Hasil

### Bagian A – Linear Regression (split 80:20, 32 train / 8 test)
- Persamaan (normal equation): **ŷ = 28.4365 + 9.1358 · BiayaIklan**
- Setiap tambahan biaya iklan 1 juta Rp menaikkan penjualan rata-rata ±9.14 juta Rp.

| Data | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Training | 4.9002 | 42.2219 | 6.4978 | 0.9324 |
| Testing | 2.7510 | 11.8675 | 3.4449 | **0.9823** |

- Gradient descent (5000 iterasi): lr 0.001 terlalu lambat (b = 25.35, m = 9.62, belum konvergen); lr 0.02 pas (sama dengan normal equation); lr 0.1 divergen (loss menjadi `inf`/`nan` pada iterasi 206).
- Prediksi: 4.5 juta → 69.55 juta; 7 juta → 92.39 juta; 12 juta → 138.07 juta (ekstrapolasi, kurang bisa dipercaya).

### Bagian B – Logistic Regression (split 75:25, 45 train / 15 test; lr = 0.1, 10000 iterasi)
- Model: P(Spam) = σ(−11.8874 + 1.0296 · JumlahLink + 0.9921 · KataPromo)
- Decision boundary: x₂ = 11.9818 − 1.0378 · x₁ (kira-kira JumlahLink + KataPromo ≈ 12)

| Metrik (testing, threshold 0.5) | Nilai |
|---|---|
| Accuracy | **0.8667** |
| Precision | 0.8000 |
| Recall | 0.8000 |
| F1-score | 0.8000 |
| Confusion matrix | TN = 9, FP = 1, FN = 1, TP = 4 |

| Threshold | Precision | Recall |
|---|---|---|
| 0.3 | 0.6667 | 0.80 |
| 0.5 (dipilih) | 0.8000 | 0.80 |
| 0.7 | 1.0000 | 0.40 |

Prediksi email baru (threshold 0.5): E1 → Bukan Spam (0.0001), E2 → Bukan Spam (0.3129), E3 → Spam (0.9993), E4 → Spam (0.7475).

### Bonus – scikit-learn
- Linear Regression: hasil identik dengan implementasi sendiri.
- Logistic Regression: koefisien berbeda (regularisasi L2 default `C=1` dan gradient descent yang belum sepenuhnya konvergen), tetapi metrik testing sama (accuracy 0.8667).
