"""
Data loading and preprocessing for the churn prediction pipeline.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/telco_churn.csv"
RANDOM_STATE = 42

# Columns that are pure identifiers, not predictive features
DROP_COLS = ["customerID"]

BINARY_YES_NO_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn",
]

CATEGORICAL_COLS = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaymentMethod",
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path)
    return df


def clean(df):
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # TotalCharges has some blank strings for brand-new customers (tenure=0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    for col in BINARY_YES_NO_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col])

    return df


def encode_features(df):
    """One-hot encode multi-category columns, keep binary/numeric as-is.
    Returns encoded dataframe and the fitted column list (for inference alignment).
    """
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df_encoded


def get_train_test_split(test_size=0.2):
    df = load_raw()
    df = clean(df)
    df_encoded = encode_features(df)

    y = df_encoded["Churn"].astype(int)
    X = df_encoded.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def build_feature_vector(raw_input: dict, feature_names: list) -> pd.DataFrame:
    """
    Take a dict of raw, human-readable inputs (as collected from a form) and
    produce a single-row, one-hot-encoded DataFrame aligned to the exact
    column order the model was trained on.

    raw_input keys expected (raw, pre-encoding values):
      gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
      MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
      DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
      Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges
    """
    df = pd.DataFrame([raw_input])
    df = clean(df.assign(Churn="No"))  # dummy Churn col so clean() doesn't error
    df = df.drop(columns=["Churn"])
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    # Align to training feature columns: add any missing dummy columns as 0,
    # drop any extras, and enforce the same column order.
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_names]

    return df_encoded


def readable_feature_name(feature: str, customer_row: pd.Series) -> str:
    """Turn a one-hot column name + this customer's value into a plain phrase."""
    if "_" in feature and feature not in NUMERIC_COLS + BINARY_YES_NO_COLS:
        base, _, category = feature.partition("_")
        is_present = customer_row.get(feature, 0) == 1
        if is_present:
            return f"{base}: {category}"
        else:
            return f"{base}: not {category}"
    return feature


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = get_train_test_split()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Churn rate (train): {y_train.mean():.3f}")
    print(f"Number of features after encoding: {len(feature_names)}")
