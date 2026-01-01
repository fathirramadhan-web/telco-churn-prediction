"""Train baseline models: Logistic Regression, Random Forest, XGBoost.

Usage: python train.py  -> models/comparison.csv
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from common import ROOT, TARGET, clean_for_model, load_raw, models, split_types


def main():
    df = load_raw()
    X = clean_for_model(df)
    y = df[TARGET]
    cat, num = split_types(X)
    print(f"Rows {len(X)} | categorical {len(cat)} | numeric {len(num)} | churn {y.mean():.1%}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train {len(X_train)} | Test {len(X_test)}\n")

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    Path(ROOT / "models").mkdir(exist_ok=True)
    results = []
    for name, model in models(cat, num).items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        results.append(
            {
                "model": name,
                "cv_auc": scores.mean(),
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, prob),
            }
        )
        joblib.dump(model, ROOT / "models" / f"{name.lower().replace(' ', '_')}.joblib")

        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            feats = model.named_steps["pre"].get_feature_names_out()
            imp = pd.Series(clf.feature_importances_, index=feats).nlargest(5)
            print(f"{name} top features:\n{imp.to_string()}\n")

    report = pd.DataFrame(results).set_index("model").sort_values("f1", ascending=False)
    report.to_csv(ROOT / "models" / "comparison.csv")
    print("=== Baseline results (sorted by F1) ===")
    print(report.round(4).to_string())
    print(f"\nBest by F1: {report.index[0]} | saved models/ + comparison.csv")


if __name__ == "__main__":
    main()
