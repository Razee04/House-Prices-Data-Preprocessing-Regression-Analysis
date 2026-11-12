# Prediksi Harga Rumah (House Prices Data Preprocessing & Regression Analysis)

## Deskripsi
Project ini menggunakan pendekatan *Machine Learning* untuk melakukan pengolahan data dan analisis prediksi harga properti/rumah (*House Prices*) menggunakan algoritma regresi. Project mencakup pemuatan data (*Data Loading*), eksplorasi data (*Exploratory Data Analysis*), pembersihan data (*Data Cleaning & Missing Value Handling*), transformasi fitur (*Encoding & Feature Scaling*), serta kajian perbandingan efektivitas dan pembaruan algoritma regresi pada berbagai studi kasus.

## Dataset
Dataset yang digunakan adalah **House Prices - Advanced Regression Techniques** yang terdiri dari 1.460 data latih (81 fitur) dan 1.459 data uji (80 fitur).

Sumber dataset: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

## Tahapan Pengolahan Data & Model
Tahapan pemrosesan data (*preprocessing*) yang diimplementasikan di dalam notebook meliputi:
1. **Inspeksi & Penanganan Missing Values:** Menghapus 4 fitur dengan tingkat kekosongan data ekstrem (`> 1.000` baris kosong), yaitu `Alley`, `PoolQC`, `Fence`, dan `MiscFeature`, sehingga menyisakan 77 fitur utama yang valid.
2. **Imputasi Data:** Mengisi nilai kosong pada variabel numerik menggunakan **Median** (tahan terhadap *outlier*) dan variabel kategorikal menggunakan **Modus** (*Mode*) hingga total *missing values* menjadi `0`.
3. **Deteksi Outlier & Visualisasi EDA:** Menggunakan pustaka `Matplotlib` dan `Seaborn` untuk memetakan distribusi harga jual (`SalePrice`) serta mendeteksi pencilan (*outliers*).
4. **Encoding & Scaling:** Melakukan transformasi variabel kategorikal menjadi numerik serta penyetaraan skala fitur sebelum masuk ke tahap pemodelan algoritma regresi (*Multiple Linear Regression*, *Random Forest Regression*, *Extreme Gradient Boosting*, dan *Artificial Neural Network*).

## Cara Menjalankan
1. Install library yang dibutuhkan:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
2. Pastikan file notebook (`.ipynb`) beserta file dataset `train.csv` dan `test.csv` berada dalam folder yang sama.
3. Jalankan Jupyter Notebook atau buka melalui Visual Studio Code:
   ```bash
   jupyter notebook
   ```
4. Jalankan seluruh sel (*Run All*) secara berurutan mulai dari tahap *Data Loading* hingga *Feature Scaling*.

---

Rahma Agustin 241011403264 05TPLP019 Teknik Informatika, Universitas Pamulang

---

## Landasan Teori: Perbandingan Efektivitas dan Pembaruan Algoritma Regresi pada Prediksi Harga Rumah

Landasan teori ini membahas perbandingan algoritma regresi yang digunakan dalam permasalahan prediksi nilai numerik (khususnya harga rumah dan komoditas), perbedaan efektivitas masing-masing metode pada berbagai studi kasus, serta pembaruan arsitektur algoritma dari pendekatan regresi linier klasik menuju metode *ensemble* dan *neural network*.

### 2.1 Tahapan Pra-Pemrosesan Data (*Data Preprocessing*)
Pra-pemrosesan data merupakan tahapan fundamental dalam *Machine Learning* untuk menjamin kualitas data sebelum dilatih oleh algoritma regresi. Dataset dunia nyata umumnya mengandung nilai kosong (*missing values*), pencilan (*outliers*), dan perbedaan rentang skala antarfitur yang dapat menurunkan performa prediksi.

Pada dataset *House Prices*, fitur numerik yang memiliki distribusi miring (*right-skewed*) diimputasi menggunakan nilai **Median** agar tidak terdistorsi oleh *outlier* properti mewah, sedangkan fitur kategorikal diimputasi menggunakan nilai **Modus**.

Rumus Imputasi Median (untuk jumlah data $n$ genap):

$$\text{Median} = \frac{X_{(n/2)} + X_{(n/2 + 1)}}{2}$$

Keterangan:
* $X_{(n/2)}$ = nilai data yang telah diurutkan pada posisi ke-$n/2$.
* $n$ = jumlah total observasi data yang valid.

### 2.2 Multiple Linear Regression (Pendekatan Regresi Klasik)
*Multiple Linear Regression* (MLR) merupakan algoritma statistik klasik yang memodelkan hubungan linier antara dua atau lebih variabel independen ($X_1, X_2, \dots, X_p$) terhadap variabel dependen numerik ($Y$).

Rumus *Multiple Linear Regression*:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon$$

Keterangan:
* $\hat{y}$ = nilai prediksi harga rumah.
* $\beta_0$ = konstanta (*intercept*).
* $\beta_1, \dots, \beta_p$ = koefisien regresi masing-masing fitur.
* $x_1, \dots, x_p$ = nilai variabel fitur properti.
* $\epsilon$ = galat (*error*).

**Perbedaan & Efektivitas pada Kasus Prediksi Harga:** Pada penelitian prediksi harga rumah yang dilakukan oleh Pratama et al. (2023), algoritma *Multiple Linear Regression* efektif digunakan apabila hubungan antarfitur properti terhadap harga bersifat linier dan telah melalui pembersihan *outlier*. Namun, kelemahan utama metode klasik ini terletak pada sensitivitasnya terhadap multikolinearitas dan ketidakmampuannya menangkap pola hubungan non-linier pada data berdimensi tinggi.

### 2.3 Pembaruan ke Ensemble Bagging: Random Forest Regression
Untuk mengatasi kelemahan model regresi linier dan *Decision Tree* tunggal yang rentan terhadap *overfitting*, dilakukan pembaruan algoritma melalui pendekatan *Ensemble Learning* berbasis *Bootstrap Aggregating* (*Bagging*), yaitu **Random Forest Regression**. Algoritma ini membangun banyak pohon keputusan (*decision trees*) secara paralel dari sampel data acak, kemudian menghitung rata-rata prediksi dari seluruh pohon tersebut.

Rumus Prediksi *Random Forest Regression*:

$$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

Keterangan:
* $\hat{y}$ = nilai prediksi akhir.
* $B$ = jumlah total pohon keputusan (*trees*) yang dibangun.
* $T_b(x)$ = hasil prediksi dari pohon ke-$b$ untuk input fitur $x$.

**Perbedaan & Efektivitas pada Studi Kasus Berbeda:** Penerapan *Random Forest Regression* pada kasus peramalan penyebaran COVID-19 di Indonesia (Fadillah et al., 2022) maupun prediksi jumlah peserta kursus (Sari & Wijaya, 2022) menunjukkan bahwa metode *ensemble bagging* jauh lebih efektif dan stabil dalam menangkap pola fluktuatif dan non-linier dibandingkan model regresi tunggal. Dalam konteks prediksi harga rumah, *Random Forest* efektif karena mampu mengolah fitur campuran (numerik dan kategorikal hasil *encoding*) tanpa mudah terpengaruh oleh pencilan ekstrem.

### 2.4 Pembaruan Lanjutan ke Gradient Boosting: Extreme Gradient Boosting (XGBoost)
Pembaruan lanjutan dari arsitektur *ensemble* berbasis pohon adalah **Extreme Gradient Boosting (XGBoost)**. Berbeda dengan *Random Forest* yang membangun pohon secara paralel (*bagging*), XGBoost menerapkan mekanisme *Gradient Boosting* yang membangun pohon keputusan secara berurutan (*sequential*). Setiap pohon baru dilatih secara khusus untuk mengoreksi kesalahan (*residual error*) dari pohon-pohon sebelumnya, serta dilengkapi fungsi regularisasi ($L_1$ dan $L_2$) untuk mencegah *overfitting*.

Fungsi Objektif *XGBoost*:

$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$

Keterangan:
* $l(y_i, \hat{y}_i)$ = fungsi kerugian (*loss function*) yang mengukur selisih nilai aktual dan prediksi.
* $f_t(x_i)$ = fungsi pohon keputusan baru pada iterasi ke-$t$.
* $\Omega(f_t)$ = komponen regularisasi untuk mengontrol kompleksitas model.

**Perbedaan & Efektivitas pada Kasus Prediksi Harga:** Pada studi kasus prediksi harga komoditas bawang merah oleh Nugroho & Hidayat (2023), algoritma *XGBoost* terbukti jauh lebih efektif dan presisi dibandingkan metode regresi konvensional. Pembaruan melalui komputasi paralel dan optimasi gradien menjadikan XGBoost unggul untuk memprediksi data tabular berdimensi tinggi seperti dataset *House Prices*.

### 2.5 Pembaruan ke Pendekatan Deep Learning: Artificial Neural Network (ANN)
Selain pembaruan pada algoritma berbasis pohon (*tree-based*), perkembangan terbaru dalam permasalahan regresi juga mengadopsi **Artificial Neural Network (ANN)** yang meniru cara kerja jaringan saraf biologis melalui lapisan tersembunyi (*hidden layers*) dan fungsi aktivasi non-linier untuk memetakan pola hubungan kompleks antarvariabel.

Rumus Propagasi Maju (*Forward Propagation*) pada Neuron Regresi:

$$\hat{y} = f\left(\sum_{j=1}^{m} w_j h_j + b\right)$$

Keterangan:
* $\hat{y}$ = nilai output prediksi.
* $w_j$ = bobot (*weight*) koneksi dari neuron ke-$j$.
* $h_j$ = nilai aktivasi dari lapisan tersembunyi (*hidden layer*).
* $b$ = nilai bias.
* $f(\cdot)$ = fungsi aktivasi linier pada lapisan output.

**Perbedaan & Efektivitas pada Kasus Non-Linier:** Pada penelitian peramalan daya keluaran fotovoltaik berdasarkan parameter lingkungan oleh Silalahi et al. (2023), metode *Artificial Neural Network* terbukti efektif memodelkan hubungan non-linier yang sangat kompleks antarvariabel input, meskipun membutuhkan proses *feature scaling* (normalisasi/standardisasi) yang ketat agar proses pelatihan konvergen dengan optimal.

### 2.6 Ringkasan Komparasi & Pembaruan Algoritma

* **Multiple Linear Regression (Pendekatan Statistik Klasik)**
  * **Karakteristik:** Memodelkan hubungan linier langsung antara fitur properti dan harga jual.
  * **Efektivitas:** Efektif pada dataset dengan pola linier sederhana, tetapi kurang akurat jika terdapat interaksi fitur kompleks dan *outlier*.
  * **Sumber:** Pratama et al. (2023).
* **Random Forest Regression (Pembaruan Tahap 1 — Ensemble Bagging)**
  * **Karakteristik:** Menggabungkan ratusan pohon keputusan secara paralel untuk menekan varians dan *overfitting*.
  * **Efektivitas:** Sangat efektif menangani pola non-linier dan data campuran pada kasus peramalan pandemi maupun prediksi jumlah peserta.
  * **Sumber:** Fadillah et al. (2022); Sari & Wijaya (2022).
* **Extreme Gradient Boosting / XGBoost (Pembaruan Tahap 2 — Gradient Boosting)**
  * **Karakteristik:** Pembaruan dari *bagging* menjadi *boosting* sekuensial berbasis gradien dengan regularisasi $L_1/L_2$.
  * **Efektivitas:** Tingkat akurasi dan kecepatan komputasi paling tinggi untuk kasus prediksi harga dan data tabular berdimensi besar.
  * **Sumber:** Nugroho & Hidayat (2023).
* **Artificial Neural Network / ANN (Pembaruan Tahap 3 — Deep Learning Regresi)**
  * **Karakteristik:** Menggunakan lapisan saraf tiruan berlapis (*multi-layer*) untuk menangkap pola non-linier tingkat tinggi.
  * **Efektivitas:** Sangat efektif pada data dengan interaksi variabel yang kompleks setelah dilakukan *feature scaling*.
  * **Sumber:** Silalahi et al. (2023).

---

## Daftar Pustaka

1. Fadillah, R., & Kurniawan, A. (2022). Model Random Forest Regression untuk Peramalan Penyebaran COVID-19 di Indonesia. *Decode: Jurnal Pendidikan Teknologi Informasi*, 2(2), 45–52. https://doi.org/10.51454/decode.v2i2.48
2. Nugroho, A. S., & Hidayat, T. (2023). Implementation of Extreme Gradient Boosting Algorithm for Predicting the Red Onion Prices. *Moneter: Jurnal Keuangan dan Perbankan*, 11(1), 18–27. https://doi.org/10.32832/moneter.v11i1.55
3. Pratama, R. A., Saputra, D., & Wulandari, S. (2023). Penerapan Algoritma Multiple Linear Regression dalam Memprediksi Harga Jual Rumah. *Jurnal Media Informatika Budidarma*, 7(2), 612–620. https://doi.org/10.30865/mib.v7i2.5891
4. Sari, M. P., & Wijaya, B. (2022). Prediction of the Number of Course Participants Using Random Forest Regression Algorithm. *Jurnal Mantik*, 6(3), 3393–3399.
5. Silalahi, D. K., Rumapea, A. C. M., Priharti, W., & Aprillia, B. S. (2023). Forecasting Photovoltaic Output Power Based on Environmental Parameters Using Artificial Neural Network Methods. *Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi)*, 7(6), 1276–1284. https://doi.org/10.29207/resti.v7i6.5214