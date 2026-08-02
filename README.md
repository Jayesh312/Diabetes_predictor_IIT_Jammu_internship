# CriticalWatch-AI
# 🩺 CriticalWatch AI: ICU Early Warning System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://criticalwatch-ai-hackathon-project-healthcare.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ML Engine: XGBoost](https://img.shields.io/badge/ML%20Engine-XGBoost-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Team DeepThink AI | IIT Jammu**  
> *Predicting ICU patient clinical deterioration up to 12 hours in advance using ensemble machine learning.*

---

## 🔗 Quick Links

* **🚀 Live Interactive Application:** [CriticalWatch AI Dashboard](https://criticalwatch-ai-hackathon-project-healthcare.streamlit.app/)
* **📄 Research / Technical Report:** Available in the repo repository (`technical_report.pdf`)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Core Tech Stack](#-core-tech-stack)
- [AI & ML Architecture](#-ai--ml-architecture)
- [Dataset Details](#-dataset-details)
- [Repository Structure](#-repository-structure)
- [Local Setup & Installation](#-local-setup--installation)
- [Model Evaluation](#-model-evaluation)
- [Team Details](#-team-details)

---

## 💡 Overview

In Intensive Care Units (ICUs), delayed recognition of clinical deterioration often leads to severe, preventable adverse events. Traditional Early Warning Scores (EWS) rely on static, linear rules that fail to capture complex, non-linear interactions across physiological vitals.

**CriticalWatch AI** is a real-time clinical decision support system that continuously monitors hourly vitals and laboratory trends to generate a dynamic probabilistic risk score for deterioration over the next 12 hours ($>0$ and $\le12$ hours horizon).

---

## ✨ Key Features

* **Real-Time Patient Vitals Simulation:** Interactive bed-monitor interface to adjust parameters like Heart Rate, Blood Pressure, $\text{SpO}_2$, Respiratory Rate, and Temperature.
* **Dual Model Engine:** Evaluates predictions using an optimized **XGBoost Classifier** alongside a **Scaled Logistic Regression** baseline.
* **Dynamic Alerting:** Customizable risk threshold triggers immediate clinical warnings when patient risk exceeds safe operational boundaries.
* **Data Leakage Protection:** Patient-level grouped train-test splitting guarantees clean model validation without temporal leakage.

---

## 🛠️ Core Tech Stack

| Category | Technology | Usage |
| :--- | :--- | :--- |
| **Language** | Python 3.9+ | Core development & pipeline execution |
| **Machine Learning** | `XGBoost`, `Scikit-Learn` | Model training, scaling, evaluation |
| **Data Processing** | `Pandas`, `NumPy` | Time-series processing & data cleaning |
| **Visualization** | `Matplotlib`, `Seaborn` | ROC-AUC curve plotting & performance metrics |
| **Web Framework** | `Streamlit` | Interactive frontend & clinical simulator |
| **Deployment** | `GitHub` + `Streamlit Cloud` | Continuous Integration / Continuous Deployment |


                             ┌─────────────────────────┐
                             │   Hourly Vitals & Labs  │
                             └────────────┬────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │ Grouped Split (by ID)   │
                             └────────────┬────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
      ┌───────────────────────────┐                 ┌───────────────────────────┐
      │   Standard Scaling (LR)   │                 │     Feature Selection     │
      └────────────┬──────────────┘                 └────────────┬──────────────┘
                   │                                             │
                   ▼                                             ▼
      ┌───────────────────────────┐                 ┌───────────────────────────┐
      │    Logistic Regression    │                 │    XGBoost Classifier     │
      │         (Baseline)        │                 │     (Primary Engine)      │
      └────────────┬──────────────┘                 └────────────┬──────────────┘
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          │
                                          ▼
                             ┌─────────────────────────┐
                             │ 12-Hour Risk Score (%)  │
                             └─────────────────────────┘



                             ### 1. Grouped Train-Test Split (`GroupShuffleSplit`)
To prevent data leakage in healthcare time-series, records are grouped by `patient_id`. All 72 hourly snapshots for any given patient belong exclusively to either the training set or testing set.

### 2. Machine Learning Algorithms
* **XGBoost Classifier:** Primary prediction model. Efficiently handles non-linear interactions (e.g., high heart rate combined with low oxygen saturation) and implicit missing data logic.
* **Histogram-based Gradient Boosting (`HistGradientBoosting`):** Scalable option for continuous feature binning across large datasets.
* **Logistic Regression:** Linear baseline using standard feature normalization ($\mu = 0, \sigma = 1$).

---

## 📊 Dataset Details

We utilize the **Hospital Deterioration Dataset**, structured as an hourly time-series panel:

* **Scale:** ~1.68 Million hourly records across 10,000 distinct ICU patient admissions.
* **Time Horizon:** Max 72 hours per patient admission.
* **Features:** Heart Rate, Systolic/Diastolic BP, Respiratory Rate, Temperature, $\text{SpO}_2$, age, and lab metrics.
* **Target Label (`deterioration_next_12h`):** Binary label ($1 =$ deterioration occurs in the next 12 hours, $0 =$ stable).

---

## 📁 Repository Structure

```text
├── app.py                                   # Streamlit web application & inference script
├── hospital_deterioration_hourly_panel.csv # Dataset file
├── requirements.txt                         # Python dependencies for deployment
├── notebooks/
│   └── ICU_Deterioration_ML_Pipeline.ipynb  # Google Colab notebook for data cleaning & training
└── README.md                                # Project documentation

---

## 🧠 AI & ML Architecture
