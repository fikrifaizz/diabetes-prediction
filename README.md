# End-to-End Diabetes Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet?style=for-the-badge&logo=mlflow)
![Kaggle API](https://img.shields.io/badge/Kaggle-API-20BEFF?style=for-the-badge&logo=kaggle)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Ensemble-orange?style=for-the-badge&logo=scikit-learn)

## Project Overview
Repository ini berisi solusi *Machine Learning Pipeline* lengkap untuk **Diabetes Prediction Challenge (Kaggle Playground S5E12)**. Sistem ini memprediksi probabilitas diagnosis diabetes berdasarkan data metabolik dan demografi pasien.

Fokus utama project ini adalah membangun **Fully Reproducible & Automated Pipeline** yang mengintegrasikan pengambilan data otomatis, pelacakan eksperimen (MLOps), dan pemodelan ensemble tingkat lanjut.

**Target Metric:** ROC AUC

---

## Key Technical Features

### 1. Automated Data Ingestion
Tidak perlu download manual. Project ini dilengkapi script otomatis yang menggunakan **Kaggle API** untuk mengunduh, mengekstrak, dan menyiapkan dataset dalam satu perintah terminal.

### 2. MLOps Integration (MLflow)
Setiap proses training dicatat secara otomatis menggunakan **MLflow**:
* **Hyperparameter Tracking:** Mencatat setting model (e.g., `xgb_learning_rate`).
* **Performance Metrics:** Mencatat skor validasi (AUC, Accuracy, F1, Log Loss) secara real-time.
* **Artifact Versioning:** Menyimpan model `.pkl` dan preprocessor untuk reproduksibilitas.

### 3. Medical Feature Engineering
Meningkatkan akurasi model dengan fitur berbasis domain medis:
* **Pulse Pressure & MAP:** Indikator kesehatan kardiovaskular.
* **Cholesterol Ratio:** Rasio LDL/HDL untuk deteksi risiko dini.
* **Lifestyle Risk Score:** Gabungan aktivitas fisik, diet, dan BMI.

### 4. Robust Ensemble Strategy
Menggunakan pendekatan *Soft Voting* dengan pembobotan dinamis dari 3 algoritma state-of-the-art:
* **XGBoost:** Bobot Tinggi (High Accuracy).
* **HistGradientBoosting:** Bobot Menengah (Speed & Stability).
* **Random Forest:** Bobot Rendah (Variance Reduction).

---

## 📂 Project Structure
Struktur direktori disusun modular menggunakan prinsip *Cookiecutter Data Science*:

```text
diabetes-prediction/
│
├── data/                    # Dataset (Train/Test csv) - Diabaikan oleh git
├── models/                  # Model .pkl yang sudah dilatih (disimpan otomatis)
├── notebooks/               # Jupyter Notebooks untuk eksperimen
│   ├── eda.ipynb            # Analisis Data Eksploratif (EDA)
│   └── modeling.ipynb       # Eksperimen Model & Optuna Tuning
│
├── outputs/                 # Hasil prediksi (submission.csv)
├── src/                     # Source Code untuk Produksi
│   ├── extract.py           # Script Extract Data dari Kaggle
│   ├── config.py            # Konfigurasi & Hyperparameters
│   ├── preprocessing.py     # Class Feature Engineering
│   ├── train.py             # Script Training Pipeline
│   └── inference.py         # Script Prediksi Pipeline
│
├── requirements.txt         # Daftar library
└── README.md                # Dokumentasi Project
```

## Installation
1. Clone repository ini:

```Bash
git clone [https://github.com/fikrifaizz/diabetes-prediction.git](https://github.com/fikrifaizz/diabetes-prediction.git)
cd diabetes-prediction
```

2. Install dependencies: Disarankan menggunakan Virtual Environment.

```Bash
pip install -r requirements.txt
```

3. Setup Kaggle API:
    - Buat API Token di akun Kaggle Anda (Settings -> Create New Token).
    - Simpan file `kaggle.json` di:
        - Linux/MacOS: `~/.kaggle/kaggle.json`
        - Windows: `C:\Users\<YourUsername>\kaggle.json`

4. Download Data Otomatis: Jalankan perintah ini untuk menarik data dari server Kaggle:

```Bash
python -m src.extract
```

## Usage (Cara Menjalankan)
Project ini dirancang modular. Anda tidak perlu membuka Notebook untuk melatih model ulang.

1. Training Model (Retrain) 

    Jalankan perintah ini untuk melatih model menggunakan parameter terbaik (yang tersimpan di src/config.py) pada 100% data training:

    ```Bash
    python -m src.train
    ```

    Output: Model .pkl akan disimpan di folder models/.

2. Monitoring Dashboard (MLflow)

    Melihat grafik performa dan perbandingan parameter:

    ```Bash
    mlflow ui
    ```
    Buka browser dan akses `http://127.0.0.1:5000`

3. Inference (Membuat Prediksi)

    Jalankan perintah ini untuk memprediksi data test dan membuat file submission:

    ```Bash
    python -m src.inference
    ```

    Output: File submission_final.csv akan muncul di folder outputs/.

## Results & Performance
| Model | CV Score (AUC) | Weight in Ensemble |
| --- | --- | --- |
| XGBoost | 0.726 | 0.50 |
| HistGradientBoosting | 0.724 | 0.35 |
| Random Forest | 0.701 | 0.15 |


## Visualizations
Berikut adalah hasil analisis korelasi fitur terhadap target Diabetes:

![Correlation Heatmap](outputs/distribution_correlation.png)

![Distribution of KDE](outputs/distribusi_kde.png)

![Distribution of Family History and Diet Score](outputs/family_history_diabetes_skor_diet.png)

![Distribution of Age and BMI](outputs/age_bmi.png)

![Distribution of Last Prediction](outputs/distribusi_prediksi_akhir.png)

![Feature Importance](outputs/feature_importance.png)


