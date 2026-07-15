"""
generate_raw_data.py
----------------------
Generates a REALISTICALLY MESSY raw retail sales dataset — on purpose.
This is the whole point of the project: real-world data is never clean,
so this simulates the kind of quality issues a data analyst actually
encounters, all of which get fixed in `src/data_cleaning.py`:

  - Missing values (CustomerID, UnitPrice, Quantity, Country)
  - Exact duplicate rows
  - Inconsistent country naming ("USA", "U.S.A", "United States", "usa")
  - Inconsistent / messy category text (extra whitespace, mixed case)
  - Mixed date formats (e.g. "2023-05-01", "05/01/2023", "1-May-2023")
  - Outliers / invalid values (negative quantity, zero/negative price,
    absurdly large quantities from data-entry errors)

Run directly to (re)generate the CSV:
    python data/generate_raw_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(7)

N_CUSTOMERS = 1200
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Intentionally inconsistent country spellings that all mean the same thing
COUNTRY_VARIANTS = {
    "USA": ["USA", "U.S.A", "United States", "usa", " USA "],
    "UK": ["UK", "U.K.", "United Kingdom", "uk"],
    "Germany": ["Germany", "germany", " Germany"],
    "France": ["France", "france ", "FRANCE"],
    "India": ["India", "india", "INDIA "],
    "Canada": ["Canada", "canada"],
    "Australia": ["Australia", "australia "],
}

CATEGORY_VARIANTS = {
    "Electronics": ["Electronics", "electronics", " Electronics", "ELECTRONICS"],
    "Home & Kitchen": ["Home & Kitchen", "home & kitchen", "Home and Kitchen "],
    "Apparel": ["Apparel", "apparel", " APPAREL"],
    "Beauty": ["Beauty", "beauty "],
    "Sports": ["Sports", "sports", "SPORTS "],
    "Books": ["Books", "books", " Books"],
}

BASE_PRICE = {
    "Electronics": (40, 600), "Home & Kitchen": (15, 200), "Apparel": (10, 120),
    "Beauty": (8, 80), "Sports": (15, 250), "Books": (5, 40),
}

DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d-%b-%Y", "%d/%m/%Y"]


def _messy_date_string(dt: datetime) -> str:
    fmt = np.random.choice(DATE_FORMATS)
    return dt.strftime(fmt)


def generate_raw_transactions(n_rows: int = 20000) -> pd.DataFrame:
    # --- Assign each customer a persistent archetype so future behavior is
    # actually PREDICTABLE from past behavior (loyal customers keep buying
    # a lot, occasional customers buy a little, at-risk customers taper
    # off) — without this, there's no genuine signal for the ML model. ---
    customer_archetype = {}
    customer_country = {}
    customer_category_pref = {}
    customer_spend_level = {}
    for cust_id in range(1, N_CUSTOMERS + 1):
        customer_archetype[cust_id] = np.random.choice(
            ["loyal", "occasional", "at_risk"], p=[0.25, 0.5, 0.25]
        )
        customer_country[cust_id] = np.random.choice(list(COUNTRY_VARIANTS.keys()))
        customer_category_pref[cust_id] = np.random.choice(list(CATEGORY_VARIANTS.keys()))
        # A persistent "how much this customer tends to spend" multiplier,
        # so past spending behavior consistently predicts future spending.
        customer_spend_level[cust_id] = np.random.lognormal(mean=0, sigma=0.4)

    rows = []
    invoice_counter = 5000
    total_days = (END_DATE - START_DATE).days

    while len(rows) < n_rows:
        cust_id = np.random.randint(1, N_CUSTOMERS + 1)
        archetype = customer_archetype[cust_id]

        # Archetype drives WHEN in the 2-year window the customer purchases
        # and how expensive/large their typical order is.
        if archetype == "loyal":
            offset_days = np.random.randint(0, total_days)  # active throughout
            qty_boost, price_boost = 1.4, 1.2
        elif archetype == "occasional":
            offset_days = np.random.randint(0, total_days)
            qty_boost, price_boost = 1.0, 1.0
        else:  # at_risk: activity concentrated in the FIRST half of the window
            offset_days = np.random.randint(0, int(total_days * 0.55))
            qty_boost, price_boost = 0.7, 0.85

        order_date = START_DATE + timedelta(days=int(offset_days))

        # Customers mostly buy from their preferred category, sometimes browse others
        if np.random.rand() < 0.7:
            clean_category = customer_category_pref[cust_id]
        else:
            clean_category = np.random.choice(list(CATEGORY_VARIANTS.keys()))
        category_str = np.random.choice(CATEGORY_VARIANTS[clean_category])

        clean_country = customer_country[cust_id]
        country_str = np.random.choice(COUNTRY_VARIANTS[clean_country])

        low, high = BASE_PRICE[clean_category]
        unit_price = round(np.random.uniform(low, high) * price_boost, 2)
        quantity = max(1, int(round(np.random.randint(1, 6) * qty_boost)))

        invoice_counter += 1

        rows.append({
            "InvoiceNo": invoice_counter,
            "InvoiceDate": _messy_date_string(order_date),
            "CustomerID": cust_id,
            "Country": country_str,
            "ProductCategory": category_str,
            "Quantity": quantity,
            "UnitPrice": unit_price,
        })

    df = pd.DataFrame(rows)

    # --- Inject missing values ---
    for col, frac in [("CustomerID", 0.03), ("UnitPrice", 0.02), ("Quantity", 0.015), ("Country", 0.02)]:
        idx = df.sample(frac=frac, random_state=np.random.randint(0, 10000)).index
        df.loc[idx, col] = np.nan

    # --- Inject outliers / invalid values ---
    outlier_idx = df.sample(frac=0.01, random_state=1).index
    df.loc[outlier_idx, "Quantity"] = np.random.choice([-5, 0, 500, 999], size=len(outlier_idx))

    price_outlier_idx = df.sample(frac=0.008, random_state=2).index
    df.loc[price_outlier_idx, "UnitPrice"] = np.random.choice([-10, 0, 9999.99], size=len(price_outlier_idx))

    # --- Inject exact duplicate rows ---
    dup_rows = df.sample(frac=0.02, random_state=3)
    df = pd.concat([df, dup_rows], ignore_index=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    df = generate_raw_transactions()
    out_path = "data/raw_sales_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} RAW (messy) rows.")
    print(f"Missing values per column:\n{df.isna().sum()}")
    print(f"Saved to {out_path}")
