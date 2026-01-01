"""Streamlit UI for churn prediction. Run: streamlit run app.py"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from common import ROOT, TARGET, clean_for_model, load_raw

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")

MODELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


@st.cache_resource
def get_model(key):
    return joblib.load(ROOT / "models" / f"{key}.joblib")


@st.cache_data
def get_data():
    return load_raw()


@st.cache_data
def get_thresholds():
    path = ROOT / "models" / "best_thresholds.json"
    return json.loads(path.read_text()) if path.exists() else {}


THRESHOLDS = get_thresholds()


def comp_table(fname):
    path = ROOT / "models" / fname
    return pd.read_csv(path).set_index("model") if path.exists() else None


df = get_data()
st.title("Telco Customer Churn — Prediction")
st.caption("Dataset: blastchar/telco-customer-churn (7,043 rows)")

tab_single, tab_batch, tab_eval, tab_eda = st.tabs(
    ["Single prediction", "Batch scoring", "Model evaluation", "EDA"]
)

# ---- Single prediction -----------------------------------------------------
with tab_single:
    st.subheader("Score one customer")
    model_key = st.selectbox(
        "Model", list(MODELS), key="single_model",
        on_change=lambda: st.session_state.pop("threshold", None),
    )
    auto_thr = THRESHOLDS.get(MODELS[model_key], 0.50)
    threshold = st.slider("Decision threshold", 0.10, 0.90, float(auto_thr), 0.01,
                          key="threshold",
                          help=f"Auto = tuned threshold {auto_thr:.2f} for {MODELS[model_key]}")
    st.caption(f"Tuned threshold for {MODELS[model_key]}: {auto_thr:.2f}")

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.number_input("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly charges", 0.0, 200.0, 70.0)
            total = st.number_input("Total charges", 0.0, 10000.0, 840.0)
        with c2:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        with c3:
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multiple = st.selectbox("Multiple lines", ["No phone service", "No", "Yes"])
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        c4, c5 = st.columns(2)
        with c4:
            for col, opts in [
                ("OnlineSecurity", ["No", "Yes", "No internet service"]),
                ("OnlineBackup", ["No", "Yes", "No internet service"]),
                ("DeviceProtection", ["No", "Yes", "No internet service"]),
                ("TechSupport", ["No", "Yes", "No internet service"]),
            ]:
                st.session_state.setdefault(f"s_{col}", opts[0])
                st.selectbox(col, opts, key=f"s_{col}")
        with c5:
            for col, opts in [
                ("StreamingTV", ["No", "Yes", "No internet service"]),
                ("StreamingMovies", ["No", "Yes", "No internet service"]),
                ("Contract", ["Month-to-month", "One year", "Two year"]),
                ("PaperlessBilling", ["Yes", "No"]),
                ("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
            ]:
                st.session_state.setdefault(f"s_{col}", opts[0])
                st.selectbox(col, opts, key=f"s_{col}")
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
            "MultipleLines": multiple, "InternetService": internet,
            "OnlineSecurity": st.session_state["s_OnlineSecurity"],
            "OnlineBackup": st.session_state["s_OnlineBackup"],
            "DeviceProtection": st.session_state["s_DeviceProtection"],
            "TechSupport": st.session_state["s_TechSupport"],
            "StreamingTV": st.session_state["s_StreamingTV"],
            "StreamingMovies": st.session_state["s_StreamingMovies"],
            "Contract": st.session_state["s_Contract"],
            "PaperlessBilling": st.session_state["s_PaperlessBilling"],
            "PaymentMethod": st.session_state["s_PaymentMethod"],
            "MonthlyCharges": monthly, "TotalCharges": total,
        }
        X = clean_for_model(pd.DataFrame([row]))
        prob = get_model(model_key).predict_proba(X)[0, 1]
        st.metric("Churn probability", f"{prob:.2%}")
        if prob >= threshold:
            st.error(f"Prediction: CHURN (threshold {threshold:.2f})")
        else:
            st.success(f"Prediction: STAY (threshold {threshold:.2f})")

# ---- Batch scoring ---------------------------------------------------------
with tab_batch:
    st.subheader("Score a CSV batch")
    st.caption("CSV must have same columns as dataset (Churn/customerID optional).")
    up = st.file_uploader("Upload CSV", type="csv")
    key_batch = st.selectbox("Model", list(MODELS), key="batch_model")
    if up:
        raw = pd.read_csv(up)
        X = clean_for_model(raw)
        prob = get_model(key_batch).predict_proba(X)[:, 1]
        out = raw.copy()
        batch_thr = THRESHOLDS.get(MODELS[key_batch], 0.50)
        out["churn_probability"] = prob.round(4)
        out["churn_prediction"] = (prob >= batch_thr).astype(int)
        st.caption(f"Using tuned threshold {batch_thr:.2f} for {MODELS[key_batch]}")
        st.dataframe(out.head(50), width="stretch")
        st.download_button("Download scored CSV", out.to_csv(index=False).encode(),
                           "scored_churn.csv", "text/csv")

# ---- Evaluation ------------------------------------------------------------
with tab_eval:
    st.subheader("Baseline vs tuned")
    base, tuned = comp_table("comparison.csv"), comp_table("tuned_comparison.csv")
    if base is None:
        st.warning("Run train.py first.")
    else:
        st.write("Baseline", base.round(4))
        st.bar_chart(base["f1"])
    if tuned is not None:
        st.write("Tuned", tuned.round(4))
        st.bar_chart(tuned["f1"])
        best_base = base.loc[base["f1"].idxmax(), "f1"]
        best_tuned = tuned.loc[tuned["f1"].idxmax(), "f1"]
        st.metric("Best F1 gain from tuning", f"{best_tuned - best_base:+.4f}")

# ---- EDA -------------------------------------------------------------------
with tab_eda:
    st.subheader("Data overview")
    st.write(f"Rows: {len(df)} | Churn rate: {df[TARGET].mean():.2%}")
    st.write(df.describe(include="all").T.head(25))
    for col in ["Contract", "InternetService", "PaymentMethod"]:
        rates = df.groupby(col)[TARGET].mean().sort_values()
        st.bar_chart(rates)
    for img in ["target_distribution.png", "numeric_distributions.png",
                "categorical_rates.png", "correlation.png"]:
        p = ROOT / "outputs" / "eda" / img
        if p.exists():
            st.image(str(p), width="stretch")
