import pandas as pd
import argparse
import json
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder

def main(model_file, input_file, report_file):
    df = pd.read_csv(input_file)
    
    # Define features & target
    X = df.drop(columns=["loan_status", "term", "int_rate", "risk_flag"], errors="ignore")
    y = df["risk_flag"]

    # Encode categorical columns like in train.py
    cat_cols = ["home_ownership", "purpose", "verification_status"]
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Load trained model
    model = XGBClassifier()
    model.load_model(model_file)

    # Predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    # Metrics
    auc = roc_auc_score(y, y_prob)
    report = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred).tolist()

    results = {
        "AUC": auc,
        "ClassificationReport": report,
        "ConfusionMatrix": cm
    }

    with open(report_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Evaluation complete. Report saved to {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    main(args.model, args.data, args.report)