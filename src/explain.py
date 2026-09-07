"""
SHAP explainability for the churn model.
Produces both global (which features matter overall) and local
(why this specific customer was flagged) explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from preprocess import get_train_test_split

MODEL_DIR = "models"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "churn_model.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    return model, feature_names


def get_explainer_and_values(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return explainer, shap_values


def global_summary_plot(shap_values, X, out_name="shap_summary.png"):
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, out_name), dpi=150, bbox_inches="tight")
    plt.close()


def top_global_drivers(shap_values, feature_names, top_n=10):
    """Mean |SHAP value| per feature — a clean, app-friendly ranking."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    ranking = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    ranking = ranking.sort_values("mean_abs_shap", ascending=False).head(top_n)
    return ranking


def explain_single_customer(explainer, model, customer_row, feature_names, top_n=5):
    """Return the top contributing features (with sign) for one customer's prediction."""
    shap_vals = explainer(customer_row)
    values = shap_vals.values[0]
    base_value = shap_vals.base_values[0]
    proba = model.predict_proba(customer_row)[0, 1]

    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": values,
    })
    contributions["abs_value"] = contributions["shap_value"].abs()
    contributions = contributions.sort_values("abs_value", ascending=False).head(top_n)
    contributions["direction"] = contributions["shap_value"].apply(
        lambda v: "increases risk" if v > 0 else "decreases risk"
    )

    return {
        "churn_probability": float(proba),
        "base_value": float(base_value),
        "top_factors": contributions[["feature", "shap_value", "direction"]].to_dict("records"),
    }


def main():
    model, feature_names = load_model()
    X_train, X_test, y_train, y_test, _ = get_train_test_split()

    explainer, shap_values = get_explainer_and_values(model, X_test)

    print("Generating global SHAP summary plot...")
    global_summary_plot(shap_values, X_test)

    print("\nTop 10 global churn drivers:")
    top_drivers = top_global_drivers(shap_values, feature_names)
    print(top_drivers.to_string(index=False))
    top_drivers.to_json(os.path.join(MODEL_DIR, "top_drivers.json"), orient="records", indent=2)

    # Example: explain one churned customer from the test set
    print("\n--- Example: explaining a single customer ---")
    example_idx = y_test[y_test == 1].index[0]
    example_row = X_test.loc[[example_idx]]
    result = explain_single_customer(explainer, model, example_row, feature_names)
    print(f"Predicted churn probability: {result['churn_probability']:.1%}")
    print("Top factors:")
    for factor in result["top_factors"]:
        print(f"  {factor['feature']}: {factor['shap_value']:+.3f} ({factor['direction']})")


if __name__ == "__main__":
    main()
