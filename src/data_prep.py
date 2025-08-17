import pandas as pd
import argparse

# Define required columns
KEEP_COLS = [
    "loan_amnt", 
    "term", 
    "int_rate", 
    "installment", 
    "annual_inc", 
    "dti", 
    "home_ownership", 
    "purpose", 
    "verification_status", 
    "loan_status"   # target
]

def main(input_file, output_file):
    df = pd.read_csv(input_file)

    # 1. Keep only selected columns (drop all others)
    available_cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[available_cols]

    # 2. Drop rows with missing target
    df = df.dropna(subset=["loan_status"])

    # 3. Simple cleaning (can extend later)
    if "annual_inc" in df.columns:
        df["annual_inc"].fillna(df["annual_inc"].median(), inplace=True)
    if "dti" in df.columns:
        df["dti"].fillna(df["dti"].median(), inplace=True)
    if "term" in df.columns:
        df["term"] = df["term"].astype(str)
    if "home_ownership" in df.columns:
        df["home_ownership"] = df["home_ownership"].astype(str)
    if "purpose" in df.columns:
        df["purpose"] = df["purpose"].astype(str)
    if "verification_status" in df.columns:
        df["verification_status"] = df["verification_status"].astype(str)

    # 4. Save cleaned dataset
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.input, args.output)