import pandas as pd
import argparse
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

def main(input_file, model_file):
    # Load dataset
    df = pd.read_csv(input_file)

    # Target variable
    if "risk_flag" not in df.columns:
        raise ValueError("Dataset must contain 'risk_flag' column created in feature_engineer.py")
    y = df["risk_flag"]

    # Drop non-usable columns
    drop_cols = ["loan_status", "term", "int_rate", "risk_flag"]  
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categorical variables (simple label encoding for now)
    cat_cols = ["home_ownership", "purpose", "verification_status"]
    for col in cat_cols:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Start MLflow run
    with mlflow.start_run():
        # Model hyperparameters
        params = {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "auc",
            "random_state": 42,
            "use_label_encoder": False
        }

        # Train model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)

        # Log params + metrics
        mlflow.log_params(params)
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("Accuracy", acc)

        # Log model to MLflow
        mlflow.xgboost.log_model(model, "model")

        # Save trained model as artifact
        model.save_model(model_file)

        print(f"✅ Training complete | AUC: {auc:.3f}, Accuracy: {acc:.3f}")
        print(f"Model saved at: {model_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)   # input = feature engineered dataset
    parser.add_argument("--model", required=True)   # output path for trained model
    args = parser.parse_args()

    main(args.input, args.model)