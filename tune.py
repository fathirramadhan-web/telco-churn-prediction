"""Hyperparameter tuning with Optuna, maximizing F1 via out-of-fold threshold.

Usage: python tune.py [trials]   (default 30 trials per model)
Writes best params + per-model threshold to models/, retrains + evaluates all 3.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from common import ROOT, TARGET, clean_for_model, load_raw, preprocessor, split_types

optuna.logging.set_verbosity(optuna.logging.WARNING)
CV = StratifiedKFold(5, shuffle=True, random_state=42)
THRESHOLDS = np.arange(0.20, 0.71, 0.01)


def propose_clf(t, name):
    """Build classifier from Optuna trial (or FixedTrial for replay)."""
    if name == "Logistic Regression":
        return LogisticRegression(
            class_weight=t.suggest_categorical("class_weight", ["balanced", None]),
            max_iter=2000,
            C=t.suggest_float("C", 0.01, 20, log=True),
            solver="liblinear",
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=t.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=t.suggest_int("max_depth", 4, 28),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 12),
            max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            class_weight=t.suggest_categorical(
                "class_weight", ["balanced_subsample", "balanced", None]
            ),
            random_state=42,
            n_jobs=-1,
        )
    return XGBClassifier(
        n_estimators=t.suggest_int("n_estimators", 100, 700, step=50),
        max_depth=t.suggest_int("max_depth", 2, 8),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
        min_child_weight=t.suggest_int("min_child_weight", 1, 10),
        gamma=t.suggest_float("gamma", 0.0, 2.0),
        reg_lambda=t.suggest_float("reg_lambda", 0.1, 20, log=True),
        reg_alpha=t.suggest_float("reg_alpha", 0.0, 2.0),
        scale_pos_weight=t.suggest_int("scale_pos_weight", 1, 6),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def best_f1_with_threshold(y, prob):
    """Pick threshold maximizing F1; tie-break toward lower threshold (higher recall)."""
    scores = np.array([f1_score(y, (prob >= t).astype(int)) for t in THRESHOLDS])
    best = scores.max()
    # prefer lower threshold among near-equal scores
    idx = int(np.argmax(np.where(scores >= best - 1e-4, -THRESHOLDS, -np.inf)))
    return float(best), float(THRESHOLDS[idx])


def tune(name, X, y, pre, n_trials):
    def objective(t):
        model = Pipeline([("pre", pre), ("clf", propose_clf(t, name))])
        prob = cross_val_predict(model, X, y, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
        score, _ = best_f1_with_threshold(y, prob)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"{name}: best OOF F1 {study.best_value:.4f} | {study.best_params}")
    return study.best_params


def evaluate(models, thresholds, X_train, X_test, y_train, y_test):
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        thr = thresholds[name]
        pred = (prob >= thr).astype(int)
        rows.append(
            {
                "model": name,
                "threshold": thr,
                "cv_auc": np.nan,
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, prob),
            }
        )
        joblib.dump(model, ROOT / "models" / f"{name.lower().replace(' ', '_')}.joblib")
    return pd.DataFrame(rows).set_index("model").sort_values("f1", ascending=False)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    df = load_raw()
    X = clean_for_model(df)
    y = df[TARGET]
    cat, num = split_types(X)
    pre = preprocessor(cat, num)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Tuning {n} trials/model | train {len(X_train)} | test {len(X_test)} | F1 objective\n")

    best, thresholds = {}, {}
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        best[name] = tune(name, X_train, y_train, pre, n)
        # Recover the OOF threshold for the winning params (deterministic).
        m = Pipeline([("pre", pre), ("clf", propose_clf(optuna.trial.FixedTrial(best[name]), name))])
        prob = cross_val_predict(m, X_train, y_train, cv=CV, method="predict_proba", n_jobs=1)[:, 1]
        _, thresholds[name] = best_f1_with_threshold(y_train, prob)
        print(f"  -> threshold {thresholds[name]:.2f}")

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "tuned_params.json").write_text(json.dumps(best, indent=2))
    (ROOT / "models" / "best_thresholds.json").write_text(json.dumps(thresholds, indent=2))
    models = {
        name: Pipeline([("pre", pre), ("clf", propose_clf(optuna.trial.FixedTrial(p), name))])
        for name, p in best.items()
    }
    report = evaluate(models, thresholds, X_train, X_test, y_train, y_test)
    report.to_csv(ROOT / "models" / "tuned_comparison.csv")
    print("\n=== Tuned results (per-model threshold) ===")
    print(report.round(4).to_string())
    print(f"\nBest by F1: {report.index[0]}")


if __name__ == "__main__":
    main()
