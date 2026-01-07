# End-to-End Diabetes Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-2088FF?style=for-the-badge&logo=github-actions)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet?style=for-the-badge&logo=mlflow)
![Kaggle](https://img.shields.io/badge/Kaggle-API-20BEFF?style=for-the-badge&logo=kaggle)

## Project Overview
Repository ini berisi solusi *Machine Learning Pipeline* lengkap untuk **Diabetes Prediction Challenge (Kaggle Playground S5E12)**. Sistem ini memprediksi probabilitas diagnosis diabetes berdasarkan data metabolik dan demografi pasien.

Project ini dirancang untuk mendemonstrasikan kemampuan **End-to-End MLOps**, mulai dari analisis medis, pemodelan ensemble, hingga deployment API menggunakan Docker dan otomatisasi CI/CD.

**Target Metric:** ROC AUC

---

## Business Insights & Recommendations
Berdasarkan analisis *Feature Importance* dan hasil model, berikut rekomendasi strategis untuk implementasi klinis:

1.  **Prioritas Skrining Awal (Triage Tool):**
    * **Insight:** Fitur `Age`, `BMI`, dan `Pulse Pressure` (Tekanan Nadi) adalah prediktor terkuat.
    * **Rekomendasi:** Gunakan model ini sebagai alat *pre-screening* di klinik pratama untuk memprioritaskan pasien yang membutuhkan tes HbA1c (laboratorium).

2.  **Intervensi Gaya Hidup:**
    * **Insight:** Pasien dengan riwayat keluarga diabetes dan skor diet rendah memiliki risiko signifikan lebih tinggi.
    * **Rekomendasi:** Program preventif harus difokuskan pada kelompok usia muda (20-40 tahun) yang memiliki riwayat genetik untuk mencegah onset dini.

3.  **Indikator Medis Baru:**
    * **Insight:** Variabel turunan *Mean Arterial Pressure (MAP)* terbukti meningkatkan sensitivitas deteksi risiko kardiovaskular dibandingkan tekanan darah biasa.

---

## Key Engineering Features

### 1. Production-Ready Deployment (API & Docker)
Model tidak hanya berhenti di Notebook. Project ini menyertakan:
* **FastAPI:** REST API responsif untuk melakukan inferensi real-time.
* **Dockerized:** Seluruh aplikasi dibungkus dalam container, menjamin konsistensi environment (menghilangkan masalah *"it works on my machine"*).

### 2. CI/CD Pipeline (GitHub Actions)
Otomatisasi penuh menggunakan GitHub Actions:
* **Code Quality:** Otomatis menjalankan `flake8` linting setiap push.
* **Automated Retraining:** Pipeline otomatis menarik data baru dari Kaggle, melatih ulang model, dan menyimpan model `.pkl` sebagai **GitHub Artifacts**.

### 3. Automated Data Ingestion
Script otomatis (`src/extract.py`) yang menggunakan **Kaggle API** untuk mengunduh dan mengekstrak dataset secara aman.

### 4. MLOps Integration (MLflow)
Pelacakan eksperimen terpusat mencatat Hyperparameter, Metrics (AUC, Log Loss), dan Artifact Model untuk setiap run.

---
## Results & Performance
Evaluasi model menunjukkan stabilitas yang baik (*Robustness*) antara validasi internal dan data tes eksternal:

| Metric | Score | Keterangan |
| :--- | :--- | :--- |
| **CV Score (Mean)** | **0.7265** | Rata-rata 5-Fold Stratified Cross-Validation |
| **Public LB** | **0.6954** | Skor Kaggle pada 20% data test |
| **Private LB** | **0.6920** | Skor Kaggle pada 80% data test (Final Score) |

**Ensemble Strategy:** Soft Voting (XGBoost 50% + HistGradient 35% + Random Forest 15%).

---

## Project Structure
Struktur direktori modular (*Cookiecutter Data Science adapted*):

```text
diabetes-prediction/
│
├── .github/workflows/       # CI/CD Pipeline Config (YAML)
├── data/                    # Dataset (Train/Test)
├── models/                  # Artifacts Model & Preprocessor
├── mlruns/                  # MLflow Tracking Database
├── notebooks/               # EDA & Experiment Notebooks
├── outputs/                 # Visualisasi & Submission CSV
├── src/                     # Source Code Utama
│   ├── app.py               # FastAPI Server (Prediction API)
│   ├── extract.py           # Script Download Data
│   ├── preprocessing.py     # Feature Engineering Class
│   ├── train.py             # Training Pipeline
│   └── inference.py         # Batch Prediction Script
│
├── Dockerfile               # Konfigurasi Docker Image
├── requirements.txt         # Dependencies Python
└── README.md                # Dokumentasi Project
```

## Installation & Usage
### A. Cara Cepat (Menggunakan Docker)
Pastikan Docker Desktop sudah terinstall.
1. Build Image:
    ```Bash
    docker build -t diabetes-prediction .
    ```
2. Run Container:
    ```Bash
    docker run -d --name diabetes-prediction -p 8000:8000 diabetes-prediction
    ```
3. Akses API:
    - API dapat diakses di `http://localhost:8000`
    - Dokumentasi API dapat diakses di `http://localhost:8000/docs`

### B. Cara Manual (Local Environment)
1. Clone dan Install:

```Bash
git clone [https://github.com/fikrifaizz/diabetes-prediction.git](https://github.com/fikrifaizz/diabetes-prediction.git)
pip install -r requirements.txt
```

2. Setup Kaggle API:
    - Buat API Token di akun Kaggle Anda (Settings -> Create New Token).
    - Simpan file `kaggle.json` di:
        - Linux/MacOS: `~/.kaggle/kaggle.json`
        - Windows: `C:\Users\<YourUsername>\kaggle.json`

3. Download Data Otomatis: Jalankan perintah ini untuk menarik data dari server Kaggle:

```Bash
python -m src.extract
```

4. Training Model:
    ```Bash
    python -m src.train
    ```

5. Monitoring Dashboard (MLflow):
    ```Bash
    mlflow ui
    ```
    Buka browser dan akses `http://127.0.0.1:5000`

6. Jalankan API:
    ```Bash
    python -m src.app
    ```
    API dapat diakses di `http://localhost:8000`
    Dokumentasi API dapat diakses di `http://localhost:8000/docs`

## Visualizations
Berikut adalah hasil analisis korelasi fitur terhadap target Diabetes:

![Correlation Heatmap](outputs/distribution_correlation.png)

![Distribution of KDE](outputs/distribusi_kde.png)

![Distribution of Family History and Diet Score](outputs/family_history_diabetes_skor_diet.png)

![Distribution of Age and BMI](outputs/age_bmi.png)

![Distribution of Last Prediction](outputs/distribusi_prediksi_akhir.png)

![Feature Importance](outputs/feature_importance.png)


