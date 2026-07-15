"""
data_cleaning.py
------------------
The core data-cleaning pipeline for the raw retail sales dataset.
Every step is deliberate and logged so the before/after impact can be
shown on the dashboard — this is the main deliverable of the project.

Cleaning steps performed, in order:
  1. Parse inconsistent date formats into a single datetime type
  2. Standardize text fields (Country, ProductCategory): trim whitespace,
     fix casing, map spelling variants to one canonical value
  3. Handle missing values:
       - CustomerID missing  -> row dropped (can't attribute the sale)
       - Country missing     -> imputed as "Unknown"
       - Quantity missing    -> imputed with category median
       - UnitPrice missing   -> imputed with category median
  4. Remove exact duplicate rows
  5. Handle outliers / invalid values using the IQR method for Quantity
     and UnitPrice, plus hard business-rule bounds (e.g. Quantity must
     be positive, UnitPrice must be positive)
  6. Recompute the Revenue column from cleaned Quantity * UnitPrice

Returns both the cleaned DataFrame and a step-by-step "cleaning report"
(a list of dicts) describing exactly what changed at each step, so the
impact of cleaning is fully transparent and reproducible.
"""

import pandas as pd
import numpy as np

COUNTRY_MAP = {
    "usa": "USA", "u.s.a": "USA", "united states": "USA",
    "uk": "UK", "u.k.": "UK", "united kingdom": "UK",
    "germany": "Germany",
    "france": "France",
    "india": "India",
    "canada": "Canada",
    "australia": "Australia",
}

CATEGORY_MAP = {
    "electronics": "Electronics",
    "home & kitchen": "Home & Kitchen", "home and kitchen": "Home & Kitchen",
    "apparel": "Apparel",
    "beauty": "Beauty",
    "sports": "Sports",
    "books": "Books",
}

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%d/%m/%Y"]


def _parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Try multiple known date formats before falling back to pandas' inference."""
    def parse_one(value):
        if pd.isna(value):
            return pd.NaT
        for fmt in DATE_FORMATS:
            try:
                return pd.to_datetime(value, format=fmt)
            except (ValueError, TypeError):
                continue
        return pd.to_datetime(value, errors="coerce", dayfirst=False)

    return series.apply(parse_one)


def _standardize_text(series: pd.Series, mapping: dict, default=None) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    mapped = cleaned.map(mapping)
    if default is not None:
        mapped = mapped.fillna(default)
    return mapped


def _iqr_bounds(series: pd.Series, k: float = 1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def clean_data(raw_df: pd.DataFrame):
    report = []
    df = raw_df.copy()
    start_rows = len(df)

    # --- Step 1: Parse dates ---
    df["InvoiceDate"] = _parse_mixed_dates(df["InvoiceDate"])
    bad_dates = df["InvoiceDate"].isna().sum()
    report.append({
        "step": "Parse mixed date formats",
        "detail": f"Standardized 4 different date formats into datetime; {bad_dates} unparseable dates found.",
        "rows_before": len(df),
    })
    df = df.dropna(subset=["InvoiceDate"])

    # --- Step 2: Standardize text fields ---
    before_country_unique = raw_df["Country"].nunique(dropna=True)
    df["Country"] = _standardize_text(df["Country"], COUNTRY_MAP, default=np.nan)
    after_country_unique = df["Country"].nunique(dropna=True)
    report.append({
        "step": "Standardize Country spelling",
        "detail": f"Reduced {before_country_unique} inconsistent spellings down to {after_country_unique} canonical countries.",
        "rows_before": len(df),
    })

    before_cat_unique = raw_df["ProductCategory"].nunique(dropna=True)
    df["ProductCategory"] = _standardize_text(df["ProductCategory"], CATEGORY_MAP, default="Other")
    after_cat_unique = df["ProductCategory"].nunique(dropna=True)
    report.append({
        "step": "Standardize ProductCategory text",
        "detail": f"Reduced {before_cat_unique} inconsistent labels down to {after_cat_unique} canonical categories.",
        "rows_before": len(df),
    })

    # --- Step 3: Handle missing values ---
    rows_before = len(df)
    missing_customer = df["CustomerID"].isna().sum()
    df = df.dropna(subset=["CustomerID"])
    report.append({
        "step": "Drop rows with missing CustomerID",
        "detail": f"Dropped {missing_customer} rows ({missing_customer/rows_before*100:.1f}%) — sales can't be attributed to a customer.",
        "rows_before": rows_before,
    })

    missing_country = df["Country"].isna().sum()
    df["Country"] = df["Country"].fillna("Unknown")
    report.append({
        "step": "Impute missing Country",
        "detail": f"Filled {missing_country} missing country values with 'Unknown' rather than dropping the sale.",
        "rows_before": len(df),
    })

    for col in ["Quantity", "UnitPrice"]:
        missing_n = df[col].isna().sum()
        medians = df.groupby("ProductCategory")[col].transform("median")
        df[col] = df[col].fillna(medians)
        report.append({
            "step": f"Impute missing {col}",
            "detail": f"Filled {missing_n} missing values using the category-level median (preserves category price/quantity patterns).",
            "rows_before": len(df),
        })

    # --- Step 4: Remove duplicates ---
    rows_before = len(df)
    df = df.drop_duplicates()
    dup_removed = rows_before - len(df)
    report.append({
        "step": "Remove exact duplicate rows",
        "detail": f"Removed {dup_removed} exact duplicate transaction rows.",
        "rows_before": rows_before,
    })

    # --- Step 5: Handle outliers / invalid values ---
    rows_before = len(df)
    # Hard business rules first
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    invalid_removed = rows_before - len(df)

    # Statistical outliers via IQR (cap rather than drop, to preserve data volume)
    q_low, q_high = _iqr_bounds(df["Quantity"])
    p_low, p_high = _iqr_bounds(df["UnitPrice"])
    q_capped = ((df["Quantity"] < q_low) | (df["Quantity"] > q_high)).sum()
    p_capped = ((df["UnitPrice"] < p_low) | (df["UnitPrice"] > p_high)).sum()

    df["Quantity"] = df["Quantity"].clip(lower=max(1, q_low), upper=q_high)
    df["UnitPrice"] = df["UnitPrice"].clip(lower=max(0.01, p_low), upper=p_high)

    report.append({
        "step": "Handle outliers & invalid values",
        "detail": (
            f"Removed {invalid_removed} rows with non-positive Quantity/UnitPrice (data-entry errors). "
            f"Capped {q_capped} Quantity outliers and {p_capped} UnitPrice outliers to IQR bounds."
        ),
        "rows_before": rows_before,
    })

    # --- Step 6: Recompute derived fields ---
    df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    report.append({
        "step": "Recompute Revenue",
        "detail": "Revenue recalculated as Quantity x UnitPrice from the cleaned columns.",
        "rows_before": len(df),
    })

    df.reset_index(drop=True, inplace=True)

    summary = {
        "rows_raw": start_rows,
        "rows_cleaned": len(df),
        "rows_removed_total": start_rows - len(df),
        "pct_removed": round((start_rows - len(df)) / start_rows * 100, 2),
    }

    return df, report, summary
