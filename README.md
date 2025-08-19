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

- **Risk Segmentation** → 3-level classification:  
  - ✅ No Risk  
  - ⚠️ Less Risk  
  - 🚨 High Risk  

- **Dynamic Theming UI** → glowing backgrounds depending on risk flag  
- **PDF Dashboard Viewer** → embedded analytics report (Bank loan insights)  
- **MLOps-Ready** → [DVC] for pipelines & data versioning, [MLflow] for experiment tracking  
- **Encoders Saved (`encoders.pkl`)** → ensures categorical encoding consistency  
- **Render Deployed** → accessible demo link for stakeholders  

---

## 🛠️ Tech Stack

**Languages:** Python, HTML, CSS, JavaScript  
**Machine Learning:** Scikit-learn, XGBoost  

**MLOps:**  
- [DVC] → dataset & pipeline versioning  
- [MLflow] → experiment logging, model registry  

**Backend:** FastAPI, Uvicorn  
**Frontend:** HTML/CSS/JS (single-page UI with animations)  
**Deployment:** Render Cloud Hosting  

---

## 📸 Screenshots

<details>
<summary>🟢 No Risk Case</summary>  
<br>  
![No Risk Example](./Screenshots/no risk.png)
</details>

<details>
<summary>⚠️ Less Risk Case</summary>  
<br>  
Less Risk Example  
</details>

<details>
<summary>🚨 High Risk Case</summary>  
<br>  
High Risk Example  
</details>

<details>
<summary>📊 Dashboard PDF Viewer</summary>  
<br>  
Dashboard Example  
</details>  

---

## ⚡ Setup & Run Locally

<details>
<summary>📥 Installation & Setup Steps</summary>  

### 1️⃣ Clone repository
```bash
git clone https://github.com/your-username/veriscore-ai.git
cd veriscore-ai
```
### 2️⃣ Create & activate virtual environment
```Bash

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```
### 3️⃣ Install dependencies
```Bash

pip install -r requirements.txt
```
### 4️⃣ Run pipeline (DVC)
```Bash

dvc repro
```
### 5️⃣ Launch MLflow UI (optional, to see experiments)
```Bash

mlflow ui     # open: http://127.0.0.1:5000
```
### 6️⃣ Start FastAPI backend
```Bash

uvicorn deploy.serve:app --reload
```
### 7️⃣ Open App
```
➡ Head to: http://127.0.0.1:8000
```
### 📊 Results
- Loan applicants segmented into No Risk / Less Risk / High Risk categories with predicted probability of default
- DVC pipelines → new data triggers reproducible training & feature engineering
- MLflow tracking → experiments logged with parameters, AUC/Accuracy, and models
- Render deployment → real-time loan risk analysis demo available for end-users
  
### 🎯 Future Enhancements
- ✅ Add Hyperparameter Optimization (Optuna or Bayesian Optimization)
- ✅ Add more domain features (credit grade, employment length, delinquency history)
- ✅ Role-based dashboards (Applicant vs Analyst views)
- ✅ Auto-retraining CI/CD (GitHub Actions → Auto-deploy to Render)
- ✅ Containerization with Docker for scalable deployment

### 👨‍💻 Author
Developed by Arsalan Shaikh – Data Science & MLOps Enthusiast 👨‍💻

🔗 [LinkedIn](https://www.linkedin.com/in/arsalanshaikh123/) | 🌐 [Render](https://loan-risk-detector.onrender.com/) Demo | 📂 https://github.com/skarsalan07/Loan-Risk-Detector GitHub Repo


