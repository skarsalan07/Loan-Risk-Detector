import pandas as pd
import argparse

def main(input_file, output_file):
    df = pd.read_csv(input_file)

    # --- Feature Engineering ---

    # 1. Payment-to-Income Ratio
    if "installment" in df.columns and "annual_inc" in df.columns:
        df["payment_to_income"] = df["installment"] / (df["annual_inc"]/12 + 1)

    # 2. Loan-to-Income Ratio
    if "loan_amnt" in df.columns and "annual_inc" in df.columns:
        df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)

    # 3. Convert Term to Numeric (36 months → 36, 60 months → 60)
    if "term" in df.columns:
        df["term_num"] = df["term"].str.extract("(\d+)").astype(float)

    # 4. Interest Rate Numeric (strip the % sign)
    if "int_rate" in df.columns:
        df["int_rate_num"] = df["int_rate"].astype(str).str.replace("%","").astype(float)

    # 5. Create target flag (Good loan=0, Default=1)
    if "loan_status" in df.columns:
        df["risk_flag"] = df["loan_status"].apply(lambda x: 1 if str(x).lower() in ["default", "charged off"] else 0)

    # Save engineered dataset
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.input, args.output)