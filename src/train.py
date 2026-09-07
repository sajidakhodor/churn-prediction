"""
Train and evaluate churn prediction models.
Compares Logistic Regression (baseline, interpretable) vs XGBoost (stronger, non-linear).
Saves the best model + feature list to disk for use by the Streamlit app.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from preprocess import get_train_test_split

MODEL_DIR = "models"
PLOTS_DIR = "plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def savefig(name):
    plt.savefig(os.path.join(PLOTS_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()


def evaluate(name, y_true, y_pred, y_proba):
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v:.3f}")
    print(classification_report(y_true, y_pred, target_names=["Retained", "Churned"]))
    return metrics


def plot_roc_comparison(results):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in results.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve — model comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    savefig("roc_comparison.png")


def plot_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion matrix — {name}")
    plt.tight_layout()
    savefig(f"confusion_{name.lower().replace(' ', '_')}.png")


def main():
    X_train, X_test, y_train, y_test, feature_names = get_train_test_split()

    all_metrics = []
    roc_results = {}

    # --- Baseline: Logistic Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)
    y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

    all_metrics.append(evaluate("Logistic Regression", y_test, y_pred_lr, y_proba_lr))
    plot_confusion("Logistic Regression", y_test, y_pred_lr)
    fpr, tpr, _ = roc_curve(y_test, y_proba_lr)
    roc_results["Logistic Regression"] = (fpr, tpr, roc_auc_score(y_test, y_proba_lr))

    # --- XGBoost ---
    # scale_pos_weight handles class imbalance (roughly 3:1 retained:churned)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, random_state=42,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

    all_metrics.append(evaluate("XGBoost", y_test, y_pred_xgb, y_proba_xgb))
    plot_confusion("XGBoost", y_test, y_pred_xgb)
    fpr, tpr, _ = roc_curve(y_test, y_proba_xgb)
    roc_results["XGBoost"] = (fpr, tpr, roc_auc_score(y_test, y_proba_xgb))

    plot_roc_comparison(roc_results)

    # --- Summary table ---
    metrics_df = pd.DataFrame(all_metrics)
    print("\n=== Summary ===")
    print(metrics_df.round(3).to_string(index=False))

    plt.figure(figsize=(8, 5))
    metrics_melted = metrics_df.melt(id_vars="model", var_name="metric", value_name="score")
    sns.barplot(data=metrics_melted, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.title("Model comparison across metrics")
    plt.tight_layout()
    savefig("model_comparison.png")

    # --- Save the best model (XGBoost — better recall on the minority/churn class) ---
    joblib.dump(xgb, os.path.join(MODEL_DIR, "churn_model.joblib"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.joblib"))
    metrics_df.to_json(os.path.join(MODEL_DIR, "metrics.json"), orient="records", indent=2)

    print(f"\nModel and metadata saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
