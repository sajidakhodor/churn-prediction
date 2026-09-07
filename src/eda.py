"""
Exploratory analysis of the telco churn dataset.
Generates plots into plots/ for use in the README and app.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import load_raw, clean

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def savefig(name):
    plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()


def run_eda():
    df = clean(load_raw())

    # Overall churn rate
    churn_rate = df["Churn"].mean()
    print(f"Overall churn rate: {churn_rate:.1%}")

    # Churn by contract type
    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=df, x="Contract", y="Churn", estimator=lambda x: sum(x) / len(x),
        palette="Set2", hue="Contract", legend=False,
    )
    plt.ylabel("Churn rate")
    plt.title("Churn rate by contract type")
    plt.tight_layout()
    savefig("churn_by_contract.png")

    # Churn by tenure (binned)
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
    )
    plt.figure(figsize=(7, 4))
    sns.barplot(
        data=df, x="tenure_bucket", y="Churn",
        estimator=lambda x: sum(x) / len(x), palette="Set2",
        hue="tenure_bucket", legend=False,
    )
    plt.ylabel("Churn rate")
    plt.title("Churn rate by tenure")
    plt.tight_layout()
    savefig("churn_by_tenure.png")

    # Monthly charges distribution by churn
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False)
    plt.title("Monthly charges distribution: churned vs retained")
    plt.tight_layout()
    savefig("monthly_charges_by_churn.png")

    print(f"Plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    run_eda()
