"""
prediction.py
---------------
Predicts each customer's FUTURE spend (next 90 days) using a Random
Forest Regressor trained on their historical purchasing behavior.

Methodology (time-based split to avoid label leakage):
  1. Pick a cutoff date 90 days before the last date in the cleaned data.
  2. Build customer-level features using ONLY transactions before the
     cutoff: Recency, Frequency, Tenure, AvgOrderValue, CategoryDiversity,
     and dominant Country (one-hot encoded).
  3. The prediction TARGET is total revenue each customer generates in
     the 90 days AFTER the cutoff (0 if they don't purchase at all).
  4. Train/test split, fit a Random Forest Regressor, evaluate with
     R^2, MAE, and RMSE — the standard regression metrics.

This mirrors a real "predicted customer value" model used for marketing
budget allocation and account prioritization.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_FEATURES = ["Recency", "Frequency", "Tenure", "AvgOrderValue", "CategoryDiversity"]


def build_prediction_dataset(df: pd.DataFrame, observation_window_days: int = 90):
    max_date = df["InvoiceDate"].max()
    cutoff_date = max_date - pd.Timedelta(days=observation_window_days)

    past = df[df["InvoiceDate"] <= cutoff_date]
    future = df[df["InvoiceDate"] > cutoff_date]

    if past.empty:
        raise ValueError("Not enough historical data before the cutoff date.")

    features = past.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (cutoff_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        CategoryDiversity=("ProductCategory", "nunique"),
        TopCountry=("Country", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    ).reset_index()

    features["Tenure"] = (cutoff_date - features["FirstPurchase"]).dt.days
    features["AvgOrderValue"] = features["Monetary"] / features["Frequency"]
    features.drop(columns=["FirstPurchase"], inplace=True)

    # One-hot encode top country (a handful of categories only)
    country_dummies = pd.get_dummies(features["TopCountry"], prefix="Country")
    features = pd.concat([features, country_dummies], axis=1)

    # --- Regression target: future revenue in the window AFTER cutoff ---
    future_revenue = future.groupby("CustomerID")["Revenue"].sum()
    features["FutureRevenue"] = features["CustomerID"].map(future_revenue).fillna(0.0)

    feature_columns = BASE_FEATURES + list(country_dummies.columns)
    return features, feature_columns


def train_prediction_model(features: pd.DataFrame, feature_columns, test_size: float = 0.25, random_state: int = 42):
    X = features[feature_columns]
    y = features["FutureRevenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "avg_actual_future_revenue": y_test.mean(),
    }

    feature_importance = pd.Series(
        model.feature_importances_, index=feature_columns
    ).sort_values(ascending=False)

    results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})

    # Predict for ALL customers for the dashboard's "predicted top spenders" table
    features = features.copy()
    features["PredictedFutureRevenue"] = model.predict(X).round(2)

    return model, metrics, feature_importance, results_df, features
