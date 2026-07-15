# 🧹 Retail Sales Analytics: Data Cleaning, Visualization & Prediction

An end-to-end data analytics project built around the three core skills
most Data Analyst / Data Scientist roles ask for: **cleaning messy
real-world data, visualizing it to find patterns, and building a
predictive model** on top of it.

## Why this project stands out

Most portfolio projects use a dataset that's already clean. This one
**deliberately generates messy raw data** — missing values, duplicate
rows, inconsistent spelling ("USA" / "U.S.A" / "usa"), mixed date
formats, and outliers/invalid entries — and then builds a transparent,
auditable pipeline to fix it, visualize it, and predict from it.

| Stage | What it demonstrates | Module |
|---|---|---|
| **1. Data Cleaning** | Handling missing values, duplicates, inconsistent categorical text, mixed date formats, and outliers (IQR method) | `src/data_cleaning.py` |
| **2. Visualization / EDA** | Distributions, trends, category/country breakdowns, correlation heatmap | `src/eda_visualization.py` + Plotly charts in `app.py` |
| **3. Prediction** | Random Forest regression predicting each customer's future 90-day spend from historical behavior | `src/prediction.py` |

## Dataset

A synthetic e-commerce transaction dataset (~20,000 raw rows, 1,200
customers) generated with **intentional data quality issues** so the
cleaning step is genuine, not just a formality:

- Missing `CustomerID`, `Country`, `Quantity`, `UnitPrice`
- Duplicate transaction rows
- Inconsistent country spelling and product category casing/whitespace
- Mixed date formats (`YYYY-MM-DD`, `MM/DD/YYYY`, `DD-Mon-YYYY`, `DD/MM/YYYY`)
- Invalid values (negative/zero quantity or price) and statistical outliers

Customers also have persistent behavioral archetypes (loyal / occasional
/ at-risk) baked in, so there's genuine, learnable signal for the
prediction model — not just noise.

## Methodology highlights (good interview talking points)

- **Cleaning is fully transparent and reproducible**: `clean_data()`
  returns not just the cleaned DataFrame but a step-by-step report
  showing exactly what changed and why (e.g. "reduced 22 inconsistent
  country spellings down to 7 canonical countries").
- **Outliers are capped (IQR method), not blindly dropped**, to preserve
  data volume while limiting the influence of extreme values —a
  defensible, explainable choice over silently deleting rows.
- **Prediction avoids label leakage**: features come from a historical
  window *before* a cutoff date; the target (future revenue) comes from
  *after* it — mirroring how real "predicted customer value" models are
  validated in production.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The messy raw dataset is already included at `data/raw_sales_data.csv`.
To regenerate it (different seed/size):
```bash
python data/generate_raw_data.py
```

## Run the dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with three tabs:

1. **Data Cleaning** — raw data quality issues, step-by-step cleaning
   log, before/after row counts, before/after sample tables, and a
   button to download the cleaned CSV.
2. **Visualization / EDA** — revenue distribution, monthly trend,
   category/country breakdowns, correlation heatmap, summary statistics.
3. **Prediction** — Random Forest regressor predicting future customer
   spend, with R²/MAE/RMSE, actual-vs-predicted scatter plot, feature
   importance chart, and a ranked list of top predicted future spenders.

## Project structure

```
retail_analytics_project/
├── README.md
├── requirements.txt
├── app.py                        # Streamlit dashboard
├── data/
│   ├── generate_raw_data.py       # Messy raw data generator
│   └── raw_sales_data.csv
└── src/
    ├── data_cleaning.py            # Cleaning pipeline + step-by-step report
    ├── eda_visualization.py        # EDA helper functions
    └── prediction.py                # Future-spend prediction model
```

## Possible extensions (good "future work" resume bullets)

- Swap in a real-world messy dataset (e.g. Kaggle's "Messy Retail Data")
- Add automated data-quality tests (e.g. Great Expectations)
- Compare Random Forest against XGBoost / Linear Regression baselines
- Deploy the dashboard live on Streamlit Community Cloud
# retail-analytics-project
