"""
Customer Churn Predictor — Streamlit app
Enter a customer profile, get a churn risk score, and see the top
SHAP-driven reasons behind the prediction with a suggested action.
"""

import sys
import os
import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from preprocess import build_feature_vector, readable_feature_name

MODEL_DIR = "models"

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "churn_model.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
    explainer = shap.TreeExplainer(model)
    return model, feature_names, explainer


model, feature_names, explainer = load_artifacts()

st.title("Customer churn predictor")
st.caption(
    "Enter a customer's profile to estimate churn risk, see what's driving it, "
    "and get a suggested retention action."
)

with st.form("customer_form"):
    st.subheader("Customer profile")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=4)
        monthly_charges = st.number_input("Monthly charges ($)", min_value=0.0, value=89.0, step=1.0)
        contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])

    with col2:
        tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])
        partner = st.selectbox("Has partner", ["No", "Yes"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])

    submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

if submitted:
    total_charges = monthly_charges * max(tenure, 1)

    raw_input = {
        "gender": "Female",  # not a meaningful churn driver; fixed for simplicity
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": "Yes",
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": tech_support,
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    X = build_feature_vector(raw_input, feature_names)
    proba = model.predict_proba(X)[0, 1]

    shap_vals = explainer(X)
    customer_row = X.iloc[0]
    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals.values[0],
    })
    contributions["readable"] = contributions["feature"].apply(
        lambda f: readable_feature_name(f, customer_row)
    )
    contributions["abs_value"] = contributions["shap_value"].abs()
    top_factors = contributions.sort_values("abs_value", ascending=False).head(4)

    st.divider()
    st.subheader("Result")

    risk_label = "High risk" if proba >= 0.5 else "Low risk"
    risk_color = "red" if proba >= 0.5 else "green"

    c1, c2 = st.columns([2, 1])
    with c1:
        st.metric("Churn probability", f"{proba:.0%}")
    with c2:
        st.markdown(f":{risk_color}[**{risk_label}**]")

    st.markdown("**Top factors**")
    for _, row in top_factors.iterrows():
        direction = "increases risk" if row["shap_value"] > 0 else "decreases risk"
        bar_color = "#d85a30" if row["shap_value"] > 0 else "#1d9e75"
        bar_width = min(abs(row["shap_value"]) * 100, 100)
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <div style="width:{bar_width}px; max-width:120px; height:10px; background:{bar_color}; border-radius:4px;"></div>
                <span style="font-size:13px;">{row['readable']} ({direction})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Suggested action based on the dominant risk factor
    top_feature = top_factors.iloc[0]["feature"]
    if proba < 0.5:
        suggestion = "This customer looks stable. No action needed right now."
    elif "Contract" in top_feature:
        suggestion = "Offer a discount for upgrading to a 1- or 2-year contract."
    elif "InternetService" in top_feature or "TechSupport" in top_feature:
        suggestion = "Offer a free tech support trial to reduce service friction."
    elif "PaymentMethod" in top_feature:
        suggestion = "Encourage a switch to automatic bank/credit payment (linked to lower churn)."
    elif "tenure" in top_feature.lower():
        suggestion = "New customer risk — consider a proactive check-in call or onboarding offer."
    else:
        suggestion = "Flag for the retention team to review manually."

    st.info(f"**Suggested action:** {suggestion}")

st.divider()
with st.expander("About this model"):
    st.write(
        "Trained on the IBM Telco Customer Churn dataset (7,043 customers). "
        "Uses XGBoost with class-imbalance handling (scale_pos_weight), achieving "
        "ROC AUC ≈ 0.84 and ~79% recall on churned customers. Explanations are "
        "generated with SHAP (TreeExplainer) at prediction time — not hardcoded rules."
    )
