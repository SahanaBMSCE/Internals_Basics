# 🌱 CropSense Irrigation Prediction — MLOps Pipeline

## 📌 Overview

This project implements a complete **MLOps pipeline** for predicting irrigation hours in smart agriculture systems.
It covers model training, experiment tracking, hyperparameter tuning, API deployment, and retraining.

The solution is built as part of the **MLOps Lab CIE (VII Semester)**.

---

## 🎯 Problem Statement

CropSense aims to optimize irrigation using machine learning.
Given environmental and field parameters, the system predicts the required **irrigation hours**.

### Input Features:

* Soil Moisture (%)
* Crop Type Index
* Field Size (hectares)
* Temperature (°C)

### Target:

* Irrigation Hours

---

## 🏗️ Project Structure

```
Internals_Basics/
 └── MLOPs_Lab_CIE/
     ├── data/
     │   ├── training_data.csv
     │   └── new_data.csv
     ├── src/
     │   ├── train.py
     │   ├── tune.py
     │   ├── api.py
     │   └── retrain.py
     ├── models/
     ├── results/
     │   ├── step1_s1.json
     │   ├── step2_s2.json
     │   ├── step3_s4.json
     │   └── step4_s8.json
     ├── requirements.txt
     └── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/Internals_Basics.git
cd Internals_Basics
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # Linux / WSL
venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```bash
pip install -r MLOPs_Lab_CIE/requirements.txt
```

---

## 🚀 Execution Steps

### 🔹 Task 1: Model Training & Experiment Tracking

```bash
python MLOPs_Lab_CIE/src/train.py
```

* Models: SVR, Gradient Boosting
* Metrics: MAE, RMSE, R²
* Logged using MLflow

---

### 🔹 Task 2: Hyperparameter Tuning

```bash
python MLOPs_Lab_CIE/src/tune.py
```

* Grid Search with 5-fold CV
* Best model saved in `models/`

---

### 🔹 Task 3: FastAPI Deployment

```bash
uvicorn MLOPs_Lab_CIE.src.api:app --port 8500
```

#### Endpoints:

* `GET /health`
* `POST /estimate`

#### Example Input:

```json
{
  "soil_moisture_pct": 25.3,
  "crop_type_index": 2,
  "field_size_hectares": 15.7,
  "temperature_c": 28.8
}
```

---

### 🔹 Task 4: Retraining Pipeline

```bash
python MLOPs_Lab_CIE/src/retrain.py
```

* Combines old + new data
* Compares RMSE
* Promotes model if improved

---

## 📊 Outputs

All results are stored as JSON files:

| Task                  | File          |
| --------------------- | ------------- |
| Experiment Tracking   | step1_s1.json |
| Hyperparameter Tuning | step2_s2.json |
| API Output            | step3_s4.json |
| Retraining Decision   | step4_s8.json |

---

## 📈 Key Features

* ✅ End-to-end MLOps pipeline
* ✅ MLflow experiment tracking
* ✅ Hyperparameter optimization
* ✅ FastAPI model serving
* ✅ Automated retraining pipeline
* ✅ JSON-based reproducible outputs

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* MLflow
* FastAPI
* Uvicorn
* Pandas & NumPy

---

## 📌 Notes

* All datasets are used as provided (no modifications)
* Random state fixed to ensure reproducibility
* Results are machine-readable JSON (as required)

---

## 👨‍💻 Author

**Name:** Sahana K Sonni
**USN:** 1BM23AI163
**Course:** MLOps Lab (24AM6AEMLO)

---

## 📜 License

This project is for academic purposes only.
