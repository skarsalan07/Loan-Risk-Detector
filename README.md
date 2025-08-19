# VeriScore AI – Loan Risk Intelligence System

🚀 **Live Demo**: https://loan-risk-detector.onrender.com/

An end-to-end deployed machine learning system that predicts **loan applicant risk** levels into:  
- ✅ No Risk  
- ⚠️ Less Risk  
- 🚨 High Risk  

It’s powered by an **XGBoost model**, served via **FastAPI**, styled in a **single-page UI** with dynamic glowing backgrounds, and supported by **MLOps tools (DVC + MLflow)** for experiment tracking and reproducible pipelines.  

---

## 📌 Problem Statement
Financial institutions face the dual challenge of:  
- Approving loans that later **default**, increasing losses 🔻.  
- Rejecting potentially safe borrowers, missing **growth opportunities** 🏦.  

📌 Traditional credit scoring methods are rigid and not real-time.  

---

## 💡 Solution
**VeriScore AI** provides:  
✔ Real-time **risk prediction** → borrower segmented into 3 levels.  
✔ **Single-page app frontend** with responsive form, animated glow themes:  
Green 🟢 for safe, Amber 🟠 for less risk, Red 🔴 for high risk.  
✔ Integrated **PDF dashboard viewer** for decision-makers.  
✔ **MLOps powered pipeline** ensuring reproducibility, versioning & experiment logging.  

---

## 🏗️ Pipeline Architecture

```text
Raw Dataset 
   ↓
[DVC] Data Cleaning & Feature Engineering (versioned pipelines)
   ↓
[XGBoost] Model Training + Hyperparameters
   ↓
[MLflow] Experiment Logging → AUC, Accuracy, Params
   ↓
Export: Model (xgb_model.json) + Encoders (encoders.pkl)
   ↓
FastAPI Backend serving predictions
   ↓
Frontend (HTML/CSS/JS Single Page App + Dashboard PDF)
   ↓
Deployment to Render (Cloud Hosting)
```
✨ Features
Risk Segmentation → 3-level classification: No Risk ✅, Less Risk ⚠️, High Risk 🚨
Dynamic Theming UI with glowing backgrounds depending on risk flag
PDF Dashboard Viewer → embedded analytics report (Bank loan insights)
MLOps-Ready → [DVC] for pipelines & data versioning, [MLflow] for experiment tracking
Encoders Saved (encoders.pkl) → ensures categorical encoding consistency
Render Deployed → accessible demo link for stakeholders
