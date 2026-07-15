"""
eda_visualization.py
----------------------
Exploratory data analysis helpers: summary statistics and chart-ready
DataFrames used to visualize data quality (before cleaning) and business
patterns (after cleaning). Charts themselves are rendered in app.py with
Plotly for interactivity; this module just prepares the data.
"""

import pandas as pd
import numpy as np


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"MissingCount": missing, "MissingPct": pct})
    return summary[summary["MissingCount"] > 0].sort_values("MissingCount", ascending=False)


def numeric_summary(df: pd.DataFrame, columns) -> pd.DataFrame:
    return df[columns].describe().T.round(2)


def revenue_over_time(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("Month")["Revenue"].sum().reset_index()


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("ProductCategory")
        .agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique"), AvgPrice=("UnitPrice", "mean"))
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )


def country_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Country")["Revenue"]
        .sum()
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )


def correlation_matrix(df: pd.DataFrame, columns) -> pd.DataFrame:
    return df[columns].corr().round(2)


def detect_outlier_rows(raw_df: pd.DataFrame) -> dict:
    """Quick counts of the specific quality issues present in the raw data,
    used to visually justify why each cleaning step was necessary."""
    return {
        "Missing CustomerID": raw_df["CustomerID"].isna().sum(),
        "Missing Country": raw_df["Country"].isna().sum(),
        "Missing Quantity": raw_df["Quantity"].isna().sum(),
        "Missing UnitPrice": raw_df["UnitPrice"].isna().sum(),
        "Exact duplicate rows": raw_df.duplicated().sum(),
        "Non-positive Quantity": (raw_df["Quantity"] <= 0).sum(),
        "Non-positive UnitPrice": (raw_df["UnitPrice"] <= 0).sum(),
        "Inconsistent Country spellings": raw_df["Country"].nunique(dropna=True),
        "Inconsistent Category labels": raw_df["ProductCategory"].nunique(dropna=True),
    }
