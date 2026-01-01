"""Predict churn for new customers. Usage: python predict.py <model_name> [n]"""
import json
import sys

import joblib
import pandas as pd

from common import ROOT, clean_for_model, load_raw

NAME_MAP = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def score(model_name: str, raw: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    df = clean_for_model(raw)
    model = joblib.load(ROOT / "models" / f"{model_name.lower()}.joblib")
    if threshold is None:
        path = ROOT / "models" / "best_thresholds.json"
        thresholds = json.loads(path.read_text()) if path.exists() else {}
        threshold = thresholds.get(NAME_MAP.get(model_name.lower(), ""), 0.5)
    out = raw.copy()
    out["churn_probability"] = model.predict_proba(df)[:, 1].round(4)
    out["churn_prediction"] = (out["churn_probability"] >= threshold).astype(int)
    return out


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "logistic_regression"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    raw = load_raw().head(n)
    print(score(name, raw)[["customerID", "Churn", "churn_probability", "churn_prediction"]].to_string(index=False))
