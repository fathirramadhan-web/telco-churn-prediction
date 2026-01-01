"""Exploratory data analysis for Telco churn. Writes figs + summary to outputs/eda/."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import ROOT, TARGET, load_raw

OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    df = load_raw()
    raw = pd.read_csv(ROOT / "data" / "churn.csv")
    num_cols = df.select_dtypes("number").columns.drop(TARGET).tolist()

    lines = [
        "# EDA — Telco Customer Churn",
        "",
        f"- Rows: {len(df)} | Columns: {raw.shape[1]}",
        f"- Duplicated customerID: {raw['customerID'].duplicated().sum()}",
        f"- Missing cells (raw): {raw.isna().sum().sum()}",
        f"- TotalCharges blank strings: {(raw['TotalCharges'] == ' ').sum()} -> imputed median",
        f"- Churn rate: {df[TARGET].mean():.2%} (No {df[TARGET].eq(0).sum()}, Yes {df[TARGET].eq(1).sum()})",
        "",
        "## Numeric summary",
        "",
        df[num_cols].describe().T.round(2).to_markdown(),
        "",
        "## Categorical churn rates",
        "",
    ]
    for col in ["gender", "SeniorCitizen", "Partner", "Dependents", "Contract",
                "InternetService", "PaymentMethod", "PaperlessBilling", "TechSupport"]:
        rates = df.groupby(col)[TARGET].agg(["mean", "size"]).sort_values("mean", ascending=False)
        lines += [f"### {col}", "", rates.round(3).to_markdown(), ""]

    # Target distribution
    fig, ax = plt.subplots(figsize=(5, 3.5))
    plot_df = df.assign(ChurnLabel=df[TARGET].map({0: "No", 1: "Yes"}))
    sns.countplot(data=plot_df, x="ChurnLabel", hue="ChurnLabel", ax=ax,
                  palette="Set2", legend=False)
    ax.set(xlabel="Churn", ylabel="Count", title="Target distribution")
    fig.tight_layout()
    fig.savefig(OUT / "target_distribution.png", dpi=120)
    plt.close(fig)

    # Numeric distributions by churn
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.histplot(data=df, x=col, hue=TARGET, bins=30, ax=ax,
                     palette="Set1", element="step", stat="density", common_norm=False)
        ax.set_title(col)
    fig.suptitle("Numeric features by churn")
    fig.tight_layout()
    fig.savefig(OUT / "numeric_distributions.png", dpi=120)
    plt.close(fig)

    # Categorical churn rate heatmap-style bars
    cat_cols = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.ravel(), cat_cols):
        rates = df.groupby(col)[TARGET].mean().sort_values()
        rates.plot.barh(ax=ax, color="#d95f5f")
        ax.set(xlim=(0, 1), title=f"Churn rate by {col}", xlabel="Rate")
    fig.tight_layout()
    fig.savefig(OUT / "categorical_rates.png", dpi=120)
    plt.close(fig)

    # Correlation of numerics
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(df[num_cols + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation (numeric)")
    fig.tight_layout()
    fig.savefig(OUT / "correlation.png", dpi=120)
    plt.close(fig)

    (OUT / "summary.md").write_text("\n".join(lines))
    print(f"EDA written -> {OUT}")
    print(f"Churn rate {df[TARGET].mean():.2%} | blanks imputed {(raw['TotalCharges'] == ' ').sum()}")


if __name__ == "__main__":
    main()
