# UTS Machine Learning: Linear Regression & Logistic Regression

Repositori ini berisi laporan dan implementasi kode untuk Tugas Ujian Tengah Semester (UTS) mata kuliah **Machine Learning**. Seluruh algoritma model regresi, klasifikasi, serta metrik evaluasinya diimplementasikan mandiri menggunakan pustaka **NumPy**, lalu divalidasi dan dibandingkan dengan pustaka **Scikit-Learn**.

---

## 👤 Identitas Mahasiswa
- **Nama**  : Irzal Ihsan Nur Ali
- **NIM**   : 241011401283
- **Kelas** : Reg B
- **Random State** : `83` *(Diambil dari 2 digit terakhir NIM: 241011401283)*

---

## 📁 Struktur Repositori
```text
uts-machine-learning-241011401283/
├── README.md            # Dokumentasi, identitas, dan ringkasan hasil evaluasi model
├── tugas_UTS.ipynb      # Jupyter Notebook lengkap dengan kode, analisis, dan output/grafik
├── Ads_Data.csv         # Dataset biaya iklan dan penjualan bulanan (40 data)
└── Email_Data.csv       # Dataset filter spam email (60 data)
```

---

## 📊 Ringkasan Hasil Eksperimen

### 1. Bagian A: Linear Regression (Dataset `Ads_Data.csv`)
Tujuan eksperimen adalah memprediksi nilai **Penjualan** berdasarkan variabel **Biaya Iklan** menggunakan metode analitik (*Normal Equation*) dan optimasi iteratif (*Gradient Descent*).

- **Data Split**: 80% Training (32 data), 20% Testing (8 data) dengan `random_state = 83`.
- **Persamaan Garis Regresi**:
  $$\hat{y} = 26.7864 + 9.2549 \cdot X$$
  - **Intercept ($b$)**: `26.7864` (Estimasi penjualan dasar toko tanpa biaya iklan adalah Rp 26.786.400,-).
  - **Koefisien ($m$)**: `9.2549` (Setiap penambahan biaya iklan sebesar 1 juta Rupiah diprediksi menaikkan penjualan rata-rata sebesar Rp 9.254.900,-).
- **Hasil Metrik Evaluasi**:

| Metrik Evaluasi | Data Training | Data Testing |
|:---|:---:|:---:|
| **MAE** | 4.7189 | 4.1963 |
| **MSE** | 39.2189 | 26.4521 |
| **RMSE** | 6.2625 | **5.1432** |
| **$R^2$ (R-Squared)** | 0.9417 | **0.9442 (94.42%)** |

- **Eksperimen Learning Rate pada Gradient Descent (`n_iterations = 5000`)**:
  - `lr = 0.001`: **Terlalu Lambat** (belum konvergen, final MSE = 55.3512).
  - `lr = 0.02`: **Pas / Konvergen Optimal** (hasil koefisien dan MSE persis identik dengan Normal Equation: $b=26.7864, m=9.2549, MSE=39.2189$).
  - `lr = 0.1`: **Divergen / Overflow** (learning rate terlalu besar menyebabkan *overshooting* hingga nilai bobot meledak menjadi NaN).
- **Prediksi Nilai Baru**:
  - Biaya Iklan 4.5 juta Rp $\rightarrow$ Prediksi Penjualan: **68.43 juta Rp** (Interpolasi)
  - Biaya Iklan 7.0 juta Rp $\rightarrow$ Prediksi Penjualan: **91.57 juta Rp** (Interpolasi)
  - Biaya Iklan 12.0 juta Rp $\rightarrow$ Prediksi Penjualan: **137.84 juta Rp** (Ekstrapolasi - kurang dapat dipercaya secara mutlak karena adanya efek kejenuhan pasar / *diminishing returns*).

---

### 2. Bagian B: Logistic Regression (Dataset `Email_Data.csv`)
Tujuan eksperimen adalah mengklasifikasikan email ke dalam kategori **Spam (1)** atau **Bukan Spam (0)** berdasarkan fitur **Jumlah Link** ($x_1$) dan **Jumlah Kata Promo** ($x_2$).

- **Data Split**: 75% Training (45 data), 25% Testing (15 data) dengan `random_state = 83`.
- **Persamaan Model Probabilitas**:
  $$P(\text{Spam}) = \sigma(-12.9137 + 1.1435 \cdot \text{JumlahLink} + 1.1592 \cdot \text{KataPromo})$$
- **Persamaan Decision Boundary ($P = 0.5$)**:
  $$x_2 = 11.1402 - 0.9865 \cdot x_1$$
- **Hasil Metrik Evaluasi Data Testing (Threshold Default 0.5)**:

| Metrik | Nilai | Keterangan |
|:---|:---:|:---|
| **Accuracy** | **73.33%** | 11 dari 15 data uji berhasil diprediksi dengan benar |
| **Precision** | **62.50%** | Kemampuan model memprediksi spam tanpa banyak salah tuduh |
| **Recall** | **83.33%** | Kemampuan model menjaring email spam yang ada |
| **F1-Score** | **71.43%** | Harmonic mean antara Precision dan Recall |

- **Confusion Matrix Testing**:
  - **True Negative (TN)**: 6
  - **False Positive (FP)**: 3 *(Email bukan spam yang salah diprediksi spam)*
  - **False Negative (FN)**: 1 *(Email spam yang lolos ke inbox)*
  - **True Positive (TP)**: 5
- **Analisis Dampak Kesalahan & Pemilihan Threshold**:
  - Kesalahan **False Positive (FP) jauh lebih merugikan pengguna** karena email penting (seperti panggilan kerja, konfirmasi bisnis, kode OTP) bisa hilang/terblokir.
  - Untuk menekan angka FP, direkomendasikan memilih **threshold lebih tinggi ($\ge 0.75$)**, di mana nilai FP berkurang dari 3 menjadi 1 dan nilai **Precision meningkat drastis menjadi 83.33%**.
- **Hasil Prediksi Email Baru**:
  - **E1** (Link: 1, Promo: 2) $\rightarrow$ $P(\text{Spam}) = 0.0001 \rightarrow$ **Bukan Spam (0)**
  - **E2** (Link: 5, Promo: 6) $\rightarrow$ $P(\text{Spam}) = 0.4400 \rightarrow$ **Bukan Spam (0)**
  - **E3** (Link: 9, Promo: 10) $\rightarrow$ $P(\text{Spam}) = 0.9999 \rightarrow$ **Spam (1)**
  - **E4** (Link: 2, Promo: 11) $\rightarrow$ $P(\text{Spam}) = 0.8932 \rightarrow$ **Spam (1)**

---

### 3. Bagian C: Pemahaman Konsep
1. **Hakikat Logistic Regression**: Berperan sebagai algoritma klasifikasi diskret meskipun namanya mengandung kata regresi, di mana fungsi **sigmoid** bertugas memetakan hasil kombinasi linear riil $(-\infty, +\infty)$ ke dalam interval probabilitas $[0, 1]$.
2. **Normal Equation vs Gradient Descent**: Normal equation adalah solusi analitik tertutup satu langkah tanpa iterasi ($\mathcal{O}(n^3)$), sedangkan gradient descent adalah optimasi numerik bertahap yang sangat efisien untuk dataset masif dengan dimensi fitur yang sangat besar.
3. **Data Testing & Overfitting**: Evaluasi wajib dilakukan pada data testing untuk mengukur kemampuan generalisasi model terhadap data baru, serta mendeteksi gejala overfitting (model menghafal noise data latih).
4. **MSE vs Log Loss**: Linear regression memakai MSE karena output kontinu bersifat cembung (*convex*), sedangkan logistic regression memakai log loss (cross-entropy) agar kurva loss fungsi sigmoid tetap konveks sehingga terhindar dari jebakan lokal minimum.
5. **Variasi Hasil Antar Mahasiswa**: Perbedaan angka dipengaruhi oleh nilai `random_state` yang disesuaikan dengan 2 digit terakhir NIM tiap mahasiswa, menghasilkan subset pemisahan data training-testing yang unik.

---

### 4. Bonus (+5): Perbandingan dengan Scikit-Learn
- **Linear Regression**: Implementasi manual NumPy menghasilkan nilai yang **100% identik** dengan `sklearn.linear_model.LinearRegression` ($b = 26.7864, m = 9.2549, R^2 = 0.9442$).
- **Logistic Regression**:
  - Saat penalti dinonaktifkan (`penalty=None`), Scikit-Learn menghasilkan akurasi yang sama (**73.33%**) dengan perbedaan numerik minor karena scikit-learn menggunakan solver kuasi-Newton orde-2 (`lbfgs`).
  - Saat menggunakan pengaturan default Scikit-Learn (`penalty='l2', C=1.0`), terjadi penyusutan koefisien (*weight shrinkage*) untuk mencegah overfitting melalui regularisasi Ridge L2.
