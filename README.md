# 🎓 AI Placement Predictor & Employability Diagnostics Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-orange.svg)](https://shap.readthedocs.io/)

A complete, production-grade Machine Learning application for predicting student campus placement readiness, explaining predictions with SHAP, identifying target role skill gaps, generating personalized career roadmaps, and providing actionable institutional analytics for Training & Placement Officers (TPO).

---

## 🌟 Key Application Features

1. **🎓 Student Placement Predictor**
   - Evaluates a comprehensive **52-feature student employability profile** across 6 core pillars (Academics, Programming, Experience, Aptitude, Soft Skills, Extracurriculars).
   - Driven by a leak-free, hyperparameter-tuned **XGBoost Classification Engine**.
   - Auto-saves new student entries into dataset with STU1001+ student IDs while avoiding duplicate department registrations.

2. **🔍 SHAP Explainable AI**
   - Provides granular, feature-level explainability for every individual prediction using **SHAP TreeExplainer**.
   - Displays exact positive and negative contribution scores for full transparency.

3. **🎯 Target Role Skill Gap Analyzer**
   - Benchmarks candidate metrics against industry entry standards for target roles (Full Stack Developer, Data Analyst, DevOps Engineer, QA Engineer).
   - Generates priority-coded gap analysis tables (Critical, High, Medium, Low).

4. **🗺️ Personalized Career Roadmap Generator**
   - Constructs a tailored **4-Phase Learning Path** complete with estimated completion timelines, course recommendations, and actionable practice drills.

5. **📊 TPO Institutional Dashboard**
   - Multi-department analytics, placement distribution charts, skill heatmaps, and priority intervention lists for vulnerable students.

---

## 🏗️ Architecture

```
Student Input
     ↓
Streamlit UI
     ↓
XGBoost Placement Predictor (52 Features)
     ↓
SHAP TreeExplainer (Feature Importance & Attribution)
     ↓
Skill Gap Analyzer (Target Role Benchmarking)
     ↓
Personalized Career Roadmap (Actionable Phases)
     ↓
Institutional TPO Dashboard (Batch Analytics & Heatmaps)
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/harshahuppara/AI-PLACEMENT-PREDICTOR.git
cd AI-PLACEMENT-PREDICTOR
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
AI-PLACEMENT-PREDICTOR/
│
├── app.py                            # Streamlit Application Main Entrypoint
├── train_xgboost.py                  # Leak-Free XGBoost Pipeline & SHAP Training
├── AI_Placement_Predictor_Dataset_V3.csv # Baseline & Persistence Student Dataset
├── requirements.txt                  # Python Package Dependencies
├── README.md                         # Documentation
│
├── pages/                            # Streamlit Multi-Page Modules
│   ├── 1_🎓_Student_Predictor.py    # Profile Entry, Prediction, SHAP, Skill Gap & Roadmap
│   ├── 2_📊_TPO_Dashboard.py        # Institutional Analytics & Vulnerable Filter
│   ├── 3_🔍_Student_Analysis.py     # Individual Student Lookup & Deep Diagnostics
│   └── 4_🤖_Model_Explainability.py # Global Feature Importance & Ablation Reports
│
├── utils/                            # Core Modular Utility Functions
│   ├── model_loader.py               # Model & Preprocessor Loader
│   ├── prediction.py                 # Feature Engineering & Inference Engine
│   ├── shap_explainer.py             # SHAP Explanation & Visualization
│   ├── skill_gap.py                  # Industry Role Gap Analysis Logic
│   ├── roadmap.py                    # Career Roadmap Generation Engine
│   └── dashboard.py                  # TPO Statistics Computation Engine
│
└── models/                           # Serialized ML Model Artifacts & Metrics
    ├── xgboost_placement_model.pkl   # Trained XGBoost Binary Model
    ├── preprocessor.pkl              # ColumnTransformer Pipelines
    ├── feature_importance.csv        # Global Feature Importance Rankings
    └── metrics.json                  # Model Accuracy & Evaluation Metrics
```

---

## 🛡️ License & Acknowledgments
Built for **Smart India Hackathon (SIH)** - AI Placement Predictor Challenge.
