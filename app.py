"""
app.py
-------
Retail Sales Analytics: Data Cleaning, Visualization & Prediction
=====================================================================
An interactive Streamlit dashboard structured around the three core
data-analyst skills this project demonstrates:

  1. DATA CLEANING   — transparent, step-by-step cleaning of a
                        realistically messy raw dataset
  2. VISUALIZATION    — exploratory charts revealing data quality issues
                        and business patterns
  3. PREDICTION       — a Random Forest model predicting each customer's
                        future spend from historical behavior

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from data_cleaning import clean_data
from eda_visualization import (
    missing_value_summary, numeric_summary, revenue_over_time,
    category_breakdown, country_breakdown, correlation_matrix, detect_outlier_rows,
)
from prediction import build_prediction_dataset, train_prediction_model

st.set_page_config(page_title="Retail Analytics: Cleaning, Viz & Prediction", page_icon="🧹", layout="wide")

st.title("🧹 Retail Sales Analytics: Data Cleaning, Visualization & Prediction")
st.caption(
    "A realistically messy raw dataset is cleaned end-to-end, explored "
    "visually, and used to train a predictive model of future customer spend."
)


@st.cache_data(show_spinner=False)
def load_raw(path="raw_sales_data.csv"):
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def run_cleaning(raw_df):
    return clean_data(raw_df)


raw_df = load_raw()
clean_df, cleaning_report, cleaning_summary = run_cleaning(raw_df)

tabs = st.tabs(["🧹 Data Cleaning", "📊 Visualization / EDA", "🔮 Prediction"])

# ============================================================
# TAB 1 — DATA CLEANING
# ============================================================
with tabs[0]:
    st.subheader("Raw Data: Quality Issues")
    st.caption("This is the raw dataset exactly as it would arrive from a real sales system — messy on purpose.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw Rows", f"{cleaning_summary['rows_raw']:,}")
    c2.metric("Cleaned Rows", f"{cleaning_summary['rows_cleaned']:,}")
    c3.metric("Rows Removed", f"{cleaning_summary['rows_removed_total']:,}")
    c4.metric("% Removed", f"{cleaning_summary['pct_removed']}%")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Data quality issues detected in raw data:**")
        issues = detect_outlier_rows(raw_df)
        issues_df = pd.DataFrame(issues.items(), columns=["Issue", "Count"])
        st.dataframe(issues_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Missing values by column (raw data):**")
        miss = missing_value_summary(raw_df)
        if not miss.empty:
            fig = px.bar(miss.reset_index(), x="index", y="MissingCount", labels={"index": "Column"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No missing values found.")

    st.markdown("---")
    st.subheader("Cleaning Pipeline — Step by Step")
    st.caption("Every transformation applied, in order, with the exact impact on the data.")

    for i, step in enumerate(cleaning_report, 1):
        st.markdown(f"**Step {i}: {step['step']}**")
        st.write(step["detail"])

    st.markdown("---")
    st.subheader("Before vs. After: Raw Sample vs. Cleaned Sample")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Raw (first 10 rows)**")
        st.dataframe(raw_df.head(10), use_container_width=True)
    with col2:
        st.markdown("**Cleaned (first 10 rows)**")
        st.dataframe(clean_df.head(10), use_container_width=True)

    st.download_button(
        "⬇️ Download cleaned dataset (CSV)",
        data=clean_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_sales_data.csv",
        mime="text/csv",
    )

# ============================================================
# TAB 2 — VISUALIZATION / EDA
# ============================================================
with tabs[1]:
    st.subheader("Exploratory Data Analysis (on cleaned data)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Revenue Distribution**")
        fig = px.histogram(clean_df, x="Revenue", nbins=50, color_discrete_sequence=["teal"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Unit Price Distribution (outliers capped)**")
        fig = px.box(clean_df, y="UnitPrice", points="outliers", color_discrete_sequence=["indianred"])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Monthly Revenue Trend**")
    trend = revenue_over_time(clean_df)
    fig = px.line(trend, x="Month", y="Revenue", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Revenue by Product Category**")
        cat = category_breakdown(clean_df)
        fig = px.bar(cat, x="ProductCategory", y="Revenue", color="Revenue", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Revenue by Country**")
        ctry = country_breakdown(clean_df)
        fig = px.bar(ctry, x="Country", y="Revenue", color="Revenue", text_auto=".2s")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Correlation Between Numeric Features**")
    corr = correlation_matrix(clean_df, ["Quantity", "UnitPrice", "Revenue"])
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Numeric summary statistics"):
        st.dataframe(numeric_summary(clean_df, ["Quantity", "UnitPrice", "Revenue"]), use_container_width=True)

# ============================================================
# TAB 3 — PREDICTION
# ============================================================
with tabs[2]:
    st.subheader("Predicting Future Customer Spend (Random Forest Regressor)")
    st.caption(
        "Trained on each customer's historical behavior (before a cutoff date) "
        "to predict how much revenue they'll generate in the following 90 days — "
        "the same kind of model used for marketing budget prioritization."
    )

    with st.spinner("Building features and training model..."):
        features, feature_cols = build_prediction_dataset(clean_df)
        model, metrics, importance, results_df, scored = train_prediction_model(features, feature_cols)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{metrics['r2']:.3f}")
    c2.metric("MAE", f"${metrics['mae']:,.2f}")
    c3.metric("RMSE", f"${metrics['rmse']:,.2f}")
    c4.metric("Avg Actual Future Revenue", f"${metrics['avg_actual_future_revenue']:,.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Actual vs. Predicted Future Revenue**")
        fig = px.scatter(
            results_df, x="Actual", y="Predicted",
            labels={"Actual": "Actual Future Revenue ($)", "Predicted": "Predicted Future Revenue ($)"},
            opacity=0.6,
        )
        max_val = max(results_df["Actual"].max(), results_df["Predicted"].max())
        fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Feature Importance**")
        imp_df = importance.reset_index()
        imp_df.columns = ["Feature", "Importance"]
        fig = px.bar(imp_df.head(10), x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Top 15 predicted future spenders (highest priority accounts):**")
    top_spenders = scored.sort_values("PredictedFutureRevenue", ascending=False).head(15)
    st.dataframe(
        top_spenders[["CustomerID", "Recency", "Frequency", "Monetary", "PredictedFutureRevenue"]],
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "**Interpreting R²:** predicting an individual customer's exact future spend "
        "is inherently noisy (future purchases depend on many factors outside the "
        "historical data). An R² in this range still provides meaningful lift over a "
        "naive 'predict the average for everyone' baseline, and the feature importances "
        "reveal which behaviors matter most for prioritization."
    )
