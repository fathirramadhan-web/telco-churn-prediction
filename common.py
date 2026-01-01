"""Shared data loading + preprocessing for the churn project.

Single source of truth: every script (train/predict/eda/app) imports from here
so cleaning is applied identically at train and inference time.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
TARGET = "Churn"
DROP = ["customerID"]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]


def clean_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID + target, impute TotalCharges, add engineered features."""
    df = df.copy()
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    # Engineered features (applied identically at train + inference)
    df["tenure_bin"] = np.digitize(df["tenure"], [6, 12, 24, 48])
    df["avg_charge"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).round(2)
    df["n_addons"] = sum((df[s] == "Yes").astype(int) for s in ADDONS)
    df["fiber_month"] = ((df["InternetService"] == "Fiber optic")
                         & (df["Contract"] == "Month-to-month")).astype(int)
    df["auto_pay"] = (~df["PaymentMethod"].isin(["Electronic check", "Mailed check"])).astype(int)
    df["no_internet"] = (df["InternetService"] == "No").astype(int)
    return df.drop(columns=[TARGET, *DROP], errors="ignore")


def split_types(X: pd.DataFrame):
    cat = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X if pd.api.types.is_numeric_dtype(X[c])]
    return cat, num


def preprocessor(cat, num):
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ]
    )


def models(cat, num):
    pre = preprocessor(cat, num)
    return {
        "Logistic Regression": Pipeline(
            [("pre", pre), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
        ),
        "Random Forest": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("pre", pre),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        scale_pos_weight=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
