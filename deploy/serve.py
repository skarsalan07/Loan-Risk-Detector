import os
import pandas as pd
import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="deploy/templates")

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "models", "xgb_model.json")
encoder_path = os.path.join(base_dir, "models", "encoders.pkl")

# Load trained model
model = XGBClassifier()
model.load_model(model_path)

# Load encoders
if os.path.exists(encoder_path):
    encoders = joblib.load(encoder_path)
else:
    encoders = {}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "probability": None,
        "theme": "neutral"
    })

@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    annual_inc: float = Form(...),
    loan_amnt: float = Form(...),
    term_num: float = Form(...),
    int_rate_num: float = Form(...),
    dti: float = Form(...),
    home_ownership: str = Form(...),
    purpose: str = Form(...),
    verification_status: str = Form(...)
):
    # Create DataFrame
    input_df = pd.DataFrame([{
        "annual_inc": annual_inc,
        "loan_amnt": loan_amnt,
        "term_num": term_num,
        "int_rate_num": int_rate_num,
        "dti": dti,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "verification_status": verification_status
    }])

    # Compute installment & drop unused
    input_df["installment"] = input_df["loan_amnt"] / (input_df["term_num"] + 1e-6)
    input_df = input_df.drop(columns=["loan_amnt", "annual_inc"])

    # Encode categoricals
    for col in ["home_ownership", "purpose", "verification_status"]:
        if col in input_df.columns:
            if col not in encoders:
                le = LabelEncoder()
                input_df[col] = le.fit_transform(input_df[col].astype(str))
                encoders[col] = le
            else:
                le = encoders[col]
                try:
                    input_df[col] = le.transform(input_df[col].astype(str))
                except ValueError:
                    input_df[col] = -1

    joblib.dump(encoders, encoder_path)

    # Fixed feature order
    expected_order = ['installment','dti','home_ownership','purpose','verification_status','term_num','int_rate_num']
    input_df = input_df[expected_order]

    # Prediction
    prob = model.predict_proba(input_df)[:, 1][0]

    # 3-level Risk classification
    if prob < 0.2:
        prediction = "✅ No Risk Detected"
        theme = "no-risk"
    elif prob < 0.3:
        prediction = "⚠️ Less Risk Detected"
        theme = "less-risk"
    else:
        prediction = "🚨 High Risk Detected"
        theme = "high-risk"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction,
        "probability": f"{prob:.2f}",
        "theme": theme
    })

if __name__ == "__main__":
    uvicorn.run("deploy.serve:app", host="0.0.0.0", port=8000, reload=True)