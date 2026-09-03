import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Canteen Analytics",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYTICS_DIR = PROJECT_ROOT / "outputs" / "analytics"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "smart_canteen_demand_model_final.pkl"
)

MEAL_INFO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "meal_info.csv"
)

CENTER_INFO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fulfilment_center_info.csv"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    opacity: 0.75;
    margin-bottom: 25px;
}

.section-title {
    font-size: 28px;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 15px;
}

div[data-testid="stMetric"] {
    padding: 18px 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.20);
}

.insight-card {
    padding: 18px 20px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.20);
    background: rgba(128,128,128,0.05);
    min-height: 115px;
}

.insight-title {
    font-size: 15px;
    font-weight: 650;
    margin-bottom: 8px;
}

.insight-value {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 5px;
}

.insight-description {
    font-size: 13px;
    opacity: 0.70;
}

.model-card {
    padding: 18px 12px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.20);
    background: rgba(128,128,128,0.05);
    text-align: center;
    min-height: 115px;
}

.model-card-title {
    font-size: 13px;
    opacity: 0.70;
    margin-bottom: 8px;
}

.model-card-value {
    font-size: 23px;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_csv(filename):

    path = ANALYTICS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(path)


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def format_number(value):

    if pd.isna(value):
        return "0"

    return f"{int(round(value)):,}"


def classify_demand(value):

    if value >= 500:
        return "High"

    elif value >= 150:
        return "Medium"

    return "Low"


def classify_waste_risk(value):

    if value >= 500:
        return "High Risk"

    elif value >= 150:
        return "Medium Risk"

    return "Low Risk"


# ============================================================
# NEW MONTH PREDICTION FUNCTION
# ============================================================

def predict_new_month(uploaded_df):

    required_columns = [
        "id",
        "week",
        "center_id",
        "meal_id",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured"
    ]

    # --------------------------------------------------------
    # Required column validation
    # --------------------------------------------------------

    missing_columns = [
        col
        for col in required_columns
        if col not in uploaded_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    df = uploaded_df.copy()

    # --------------------------------------------------------
    # Empty file check
    # --------------------------------------------------------

    if df.empty:

        raise ValueError(
            "Uploaded CSV is empty."
        )

    # --------------------------------------------------------
    # ID validation
    # --------------------------------------------------------

    if df["id"].isna().any():

        raise ValueError(
            "The 'id' column contains missing values."
        )

    duplicate_count = int(
        df["id"].duplicated().sum()
    )

    if duplicate_count > 0:

        df = df.drop_duplicates(
            subset=["id"],
            keep="first"
        )

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    numeric_columns = [
        "week",
        "center_id",
        "meal_id",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    invalid_columns = [
        col
        for col in numeric_columns
        if df[col].isna().any()
    ]

    if invalid_columns:

        raise ValueError(
            "Invalid or missing numeric values found in: "
            + ", ".join(invalid_columns)
        )

    # --------------------------------------------------------
    # Metadata files
    # --------------------------------------------------------

    if not MEAL_INFO_PATH.exists():

        raise FileNotFoundError(
            f"Meal information file not found: "
            f"{MEAL_INFO_PATH}"
        )

    if not CENTER_INFO_PATH.exists():

        raise FileNotFoundError(
            f"Fulfilment center information file not found: "
            f"{CENTER_INFO_PATH}"
        )

    meal_info = pd.read_csv(
        MEAL_INFO_PATH
    )

    center_info = pd.read_csv(
        CENTER_INFO_PATH
    )

    # --------------------------------------------------------
    # Merge meal metadata
    # --------------------------------------------------------

    df = df.merge(
        meal_info,
        on="meal_id",
        how="left"
    )

    # --------------------------------------------------------
    # Merge center metadata
    # --------------------------------------------------------

    df = df.merge(
        center_info,
        on="center_id",
        how="left"
    )

    # --------------------------------------------------------
    # Metadata validation
    # --------------------------------------------------------

    metadata_columns = [
        "category",
        "cuisine",
        "city_code",
        "region_code",
        "center_type",
        "op_area"
    ]

    missing_metadata = [
        col
        for col in metadata_columns
        if col not in df.columns
        or df[col].isna().any()
    ]

    if missing_metadata:

        raise ValueError(
            "Metadata could not be matched for: "
            + ", ".join(missing_metadata)
        )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df["discount_amount"] = (
        df["base_price"]
        - df["checkout_price"]
    )

    df["discount_percentage"] = (
        df["discount_amount"]
        / df["base_price"].replace(0, pd.NA)
    ) * 100

    df["discount_percentage"] = (
        df["discount_percentage"]
        .fillna(0)
    )

    df["price_ratio"] = (
        df["checkout_price"]
        / df["base_price"].replace(0, pd.NA)
    )

    df["price_ratio"] = (
        df["price_ratio"]
        .fillna(1)
    )

    df["month"] = (
        ((df["week"] - 1) // 4) % 12
    ) + 1

    df["quarter"] = (
        ((df["month"] - 1) // 3)
    ) + 1

    df["week_of_year"] = (
        ((df["week"] - 1) % 52) + 1
    )

    df["total_promotion"] = (
        df["emailer_for_promotion"]
        + df["homepage_featured"]
    )

    # IMPORTANT:
    #
    # orders_per_area is NOT created here.
    #
    # It depends on num_orders, which is the target.
    # Creating it during prediction would cause target leakage.

    # ========================================================
    # MODEL FEATURES
    # ========================================================

    feature_columns = [

        "week",
        "center_id",
        "meal_id",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured",

        "city_code",
        "region_code",
        "op_area",

        "discount_amount",
        "discount_percentage",
        "price_ratio",

        "month",
        "quarter",
        "week_of_year",

        "total_promotion",

        "category",
        "cuisine",
        "center_type"
    ]

    model_input = df[
        feature_columns
    ].copy()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_model()

    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = model.predict(
        model_input
    )

    df["predicted_num_orders"] = (
        pd.Series(
            predictions,
            index=df.index
        )
        .clip(lower=0)
        .round()
        .astype(int)
    )

    # ========================================================
    # DEMAND LEVEL
    # ========================================================

    df["demand_level"] = (
        df["predicted_num_orders"]
        .apply(classify_demand)
    )

    # ========================================================
    # POTENTIAL WASTE RISK
    # ========================================================

    df["potential_waste_risk"] = (
        df["predicted_num_orders"]
        .apply(classify_waste_risk)
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    result_columns = [

        "id",
        "week",
        "center_id",
        "meal_id",

        "category",
        "cuisine",
        "center_type",

        "predicted_num_orders",

        "demand_level",
        "potential_waste_risk"
    ]

    result = df[
        result_columns
    ].copy()

    return result, duplicate_count


# ============================================================
# LOAD HISTORICAL ANALYTICS
# ============================================================

try:

    category_df = load_csv(
        "category_demand_analysis.csv"
    )

    center_df = load_csv(
        "center_demand_analysis.csv"
    )

    center_waste_df = load_csv(
        "center_food_waste_risk.csv"
    )

    meal_df = load_csv(
        "meal_demand_analysis.csv"
    )

    meal_waste_df = load_csv(
        "potential_food_waste_risk.csv"
    )

    weekly_df = load_csv(
        "weekly_demand_analysis.csv"
    )

except Exception as e:

    st.error(
        "Unable to load dashboard data."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# DATA CLEANING
# ============================================================

category_df["category"] = (
    category_df["category"]
    .astype(str)
)

center_df["center_type"] = (
    center_df["center_type"]
    .astype(str)
)

center_waste_df["center_type"] = (
    center_waste_df["center_type"]
    .astype(str)
)

meal_df["category"] = (
    meal_df["category"]
    .astype(str)
)

meal_df["cuisine"] = (
    meal_df["cuisine"]
    .astype(str)
)

meal_waste_df["category"] = (
    meal_waste_df["category"]
    .astype(str)
)

meal_waste_df["cuisine"] = (
    meal_waste_df["cuisine"]
    .astype(str)
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🔎 Dashboard Filters"
)

st.sidebar.markdown("---")


# ============================================================
# CATEGORY FILTER
# ============================================================

categories = sorted(
    category_df[
        "category"
    ]
    .dropna()
    .unique()
    .tolist()
)

selected_categories = st.sidebar.multiselect(
    "Select Category",
    options=categories,
    default=categories
)


# ============================================================
# CENTER TYPE FILTER
# ============================================================

center_types = sorted(
    center_df[
        "center_type"
    ]
    .dropna()
    .unique()
    .tolist()
)

selected_center_types = st.sidebar.multiselect(
    "Select Center Type",
    options=center_types,
    default=center_types
)


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.info(
    """
**Dashboard Scope**

• ML-based demand predictions

• Category performance

• Meal-level demand

• Center-level demand

• Potential food-waste risk

• New-month prediction
"""
)


# ============================================================
# EMPTY FILTER CHECK
# ============================================================

if (
    not selected_categories
    or not selected_center_types
):

    st.warning(
        "Please select at least one category "
        "and one center type from the sidebar."
    )

    st.stop()


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_category = category_df[
    category_df["category"]
    .isin(selected_categories)
].copy()

filtered_meal = meal_df[
    meal_df["category"]
    .isin(selected_categories)
].copy()

filtered_meal_waste = meal_waste_df[
    meal_waste_df["category"]
    .isin(selected_categories)
].copy()

filtered_center = center_df[
    center_df["center_type"]
    .isin(selected_center_types)
].copy()

filtered_center_waste = center_waste_df[
    center_waste_df["center_type"]
    .isin(selected_center_types)
].copy()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🍱 Smart Canteen Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered demand prediction '
    'and potential food-waste risk analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_predicted_orders = (
    filtered_category[
        "total_predicted_orders"
    ].sum()
)

average_orders = (
    filtered_category[
        "average_predicted_orders"
    ].mean()
)

number_categories = (
    filtered_category[
        "category"
    ].nunique()
)

high_risk_count = (
    filtered_meal_waste[
        "potential_waste_risk"
    ]
    .eq("High Risk")
    .sum()
)


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "📦 Total Predicted Orders",
        format_number(
            total_predicted_orders
        )
    )

with col2:

    st.metric(
        "📈 Average Orders",
        format_number(
            average_orders
        )
    )

with col3:

    st.metric(
        "🍱 Categories",
        number_categories
    )

with col4:

    st.metric(
        "⚠️ High Waste-Risk Meals",
        int(high_risk_count)
    )


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">💡 Executive Summary</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# Highest-demand category
# ------------------------------------------------------------

top_category_row = (
    filtered_category
    .sort_values(
        "total_predicted_orders",
        ascending=False
    )
    .head(1)
)

if not top_category_row.empty:

    top_category_name = str(
        top_category_row.iloc[0]["category"]
    )

    top_category_orders = format_number(
        top_category_row.iloc[0]["total_predicted_orders"]
    )

else:

    top_category_name = "N/A"
    top_category_orders = "0"


# ------------------------------------------------------------
# Peak week
# ------------------------------------------------------------

peak_week_row = (
    weekly_df
    .sort_values(
        "total_predicted_orders",
        ascending=False
    )
    .head(1)
)

if not peak_week_row.empty:

    peak_week = int(
        peak_week_row.iloc[0]["week"]
    )

    peak_week_orders = format_number(
        peak_week_row.iloc[0]["total_predicted_orders"]
    )

else:

    peak_week = "N/A"
    peak_week_orders = "0"


# ============================================================
# EXECUTIVE SUMMARY CARDS
#
# IMPORTANT FIX:
# HTML starts immediately after the triple quote.
# No unwanted indentation before <div>.
# ============================================================

insight_col1, insight_col2, insight_col3 = st.columns(3)


with insight_col1:

    st.markdown(
        f"""<div class="insight-card">
<div class="insight-title">🍱 Highest-Demand Category</div>
<div class="insight-value">{top_category_name}</div>
<div class="insight-description">{top_category_orders} predicted orders</div>
</div>""",
        unsafe_allow_html=True
    )


with insight_col2:

    st.markdown(
        f"""<div class="insight-card">
<div class="insight-title">📈 Peak Demand Week</div>
<div class="insight-value">Week {peak_week}</div>
<div class="insight-description">{peak_week_orders} predicted orders</div>
</div>""",
        unsafe_allow_html=True
    )


with insight_col3:

    st.markdown(
        f"""<div class="insight-card">
<div class="insight-title">⚠️ High-Risk Meals</div>
<div class="insight-value">{int(high_risk_count)}</div>
<div class="insight-description">Meals classified as high potential risk</div>
</div>""",
        unsafe_allow_html=True
    )


st.caption(
    "Insights update automatically based on the selected dashboard filters."
)


# ============================================================
# WEEKLY DEMAND TREND
# ============================================================

st.markdown(
    '<div class="section-title">📊 Weekly Demand Trend</div>',
    unsafe_allow_html=True
)

st.caption(
    "Overall predicted demand across the forecast period."
)

weekly_plot = (
    weekly_df
    .sort_values("week")
    .copy()
)

fig_weekly = px.line(
    weekly_plot,
    x="week",
    y="total_predicted_orders",
    markers=True,
    labels={
        "week": "Week",
        "total_predicted_orders":
            "Total Predicted Orders"
    },
    title="Weekly Predicted Demand"
)

fig_weekly.update_layout(
    height=450,
    hovermode="x unified"
)

st.plotly_chart(
    fig_weekly,
    width="stretch"
)


# ============================================================
# CATEGORY-WISE DEMAND
# ============================================================

st.markdown(
    '<div class="section-title">🍱 Category-wise Demand</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


with col1:

    category_total = (
        filtered_category
        .sort_values(
            "total_predicted_orders",
            ascending=True
        )
    )

    fig_category_total = px.bar(
        category_total,
        x="total_predicted_orders",
        y="category",
        orientation="h",
        labels={
            "total_predicted_orders":
                "Predicted Orders",
            "category":
                "Category"
        },
        title="Total Predicted Orders"
    )

    fig_category_total.update_layout(
        height=520
    )

    st.plotly_chart(
        fig_category_total,
        width="stretch"
    )


with col2:

    category_average = (
        filtered_category
        .sort_values(
            "average_predicted_orders",
            ascending=True
        )
    )

    fig_category_avg = px.bar(
        category_average,
        x="average_predicted_orders",
        y="category",
        orientation="h",
        labels={
            "average_predicted_orders":
                "Average Predicted Orders",
            "category":
                "Category"
        },
        title="Average Predicted Orders"
    )

    fig_category_avg.update_layout(
        height=520
    )

    st.plotly_chart(
        fig_category_avg,
        width="stretch"
    )


# ============================================================
# TOP 10 MEALS
# ============================================================

st.markdown(
    '<div class="section-title">🍽️ Top 10 Meals</div>',
    unsafe_allow_html=True
)

top_meals = (
    filtered_meal
    .sort_values(
        "total_predicted_orders",
        ascending=False
    )
    .head(10)
    .copy()
)

top_meals["meal_label"] = (
    "Meal "
    + top_meals["meal_id"].astype(str)
    + " | "
    + top_meals["category"]
)

fig_meals = px.bar(
    top_meals.sort_values(
        "total_predicted_orders",
        ascending=True
    ),
    x="total_predicted_orders",
    y="meal_label",
    orientation="h",
    hover_data=[
        "meal_id",
        "category",
        "cuisine"
    ],
    labels={
        "total_predicted_orders":
            "Predicted Orders",
        "meal_label":
            "Meal"
    },
    title="Top 10 Meals by Predicted Demand"
)

fig_meals.update_layout(
    height=500
)

st.plotly_chart(
    fig_meals,
    width="stretch"
)


# ============================================================
# TOP 10 CENTERS
# ============================================================

st.markdown(
    '<div class="section-title">🏢 Top 10 Centers</div>',
    unsafe_allow_html=True
)

top_centers = (
    filtered_center
    .sort_values(
        "total_predicted_orders",
        ascending=False
    )
    .head(10)
    .copy()
)

top_centers["center_label"] = (
    "Center "
    + top_centers["center_id"].astype(str)
    + " | "
    + top_centers["center_type"]
)

fig_centers = px.bar(
    top_centers.sort_values(
        "total_predicted_orders",
        ascending=True
    ),
    x="total_predicted_orders",
    y="center_label",
    orientation="h",
    hover_data=[
        "center_id",
        "center_type"
    ],
    labels={
        "total_predicted_orders":
            "Predicted Orders",
        "center_label":
            "Center"
    },
    title="Top 10 Centers by Predicted Demand"
)

fig_centers.update_layout(
    height=500
)

st.plotly_chart(
    fig_centers,
    width="stretch"
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)

st.caption(
    "Performance of the trained Random Forest model "
    "on the held-out test set."
)

model_col1, model_col2, model_col3, model_col4 = (
    st.columns(4)
)


# ============================================================
# MODEL CARD 1
# ============================================================

with model_col1:

    st.markdown(
        """<div class="model-card">
<div class="model-card-title">🤖 Model</div>
<div class="model-card-value">Random Forest</div>
</div>""",
        unsafe_allow_html=True
    )


# ============================================================
# MODEL CARD 2
# ============================================================

with model_col2:

    st.markdown(
        """<div class="model-card">
<div class="model-card-title">📊 R² Score</div>
<div class="model-card-value">0.867</div>
</div>""",
        unsafe_allow_html=True
    )


# ============================================================
# MODEL CARD 3
# ============================================================

with model_col3:

    st.markdown(
        """<div class="model-card">
<div class="model-card-title">📉 MAE</div>
<div class="model-card-value">69.2</div>
</div>""",
        unsafe_allow_html=True
    )


# ============================================================
# MODEL CARD 4
# ============================================================

with model_col4:

    st.markdown(
        """<div class="model-card">
<div class="model-card-title">📐 RMSE</div>
<div class="model-card-value">142.55</div>
</div>""",
        unsafe_allow_html=True
    )


st.caption(
    "R², MAE and RMSE are calculated on the held-out test set "
    "and reflect model performance before deployment."
)


# ============================================================
# FOOD-WASTE RISK
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">♻️ Potential Food-Waste Risk Analytics</div>',
    unsafe_allow_html=True
)

st.info(
    """
**Important:** Food-waste risk is a demand-based proxy.

The current dataset does not contain prepared quantity
or actual waste quantity, so this dashboard does not
calculate actual food-waste percentage.
"""
)


# ============================================================
# TOP 10 CENTER WASTE RISK
# ============================================================

st.subheader(
    "Top 10 Centers — Potential Food-Waste Risk"
)

st.caption(
    "Centers with the highest predicted demand and corresponding "
    "potential food-waste risk."
)

top_risk_centers = (
    filtered_center_waste
    .sort_values(
        "total_predicted_orders",
        ascending=False
    )
    .head(10)
    .copy()
)

top_risk_centers["center_label"] = (
    "Center "
    + top_risk_centers["center_id"].astype(str)
    + " | "
    + top_risk_centers["center_type"]
)


fig_center_risk = px.bar(
    top_risk_centers,
    x="total_predicted_orders",
    y="center_label",
    orientation="h",
    color="potential_waste_risk",
    color_discrete_map={
        "High Risk": "#EF4444",
        "Medium Risk": "#F59E0B",
        "Low Risk": "#22C55E"
    },
    hover_data=[
        "center_id",
        "center_type",
        "potential_waste_risk"
    ],
    labels={
        "total_predicted_orders":
            "Predicted Orders",
        "center_label":
            "Center",
        "potential_waste_risk":
            "Risk Level"
    },
    title="Potential Food-Waste Risk by Center"
)

fig_center_risk.update_layout(
    height=500,
    legend_title_text="Risk Level",
    margin=dict(
        l=20,
        r=20,
        t=70,
        b=20
    ),
    yaxis=dict(
        categoryorder="total ascending"
    )
)

st.plotly_chart(
    fig_center_risk,
    width="stretch"
)


# ============================================================
# WASTE RISK BY MEAL
# ============================================================

st.subheader(
    "Potential Food-Waste Risk by Meal"
)

risk_counts = (
    filtered_meal_waste[
        "potential_waste_risk"
    ]
    .value_counts()
    .reset_index()
)

risk_counts.columns = [
    "potential_waste_risk",
    "meal_count"
]

fig_risk = px.pie(
    risk_counts,
    names="potential_waste_risk",
    values="meal_count",
    hole=0.45,
    title="Meal-wise Potential Food-Waste Risk",
    color="potential_waste_risk",
    color_discrete_map={
        "High Risk": "#EF4444",
        "Medium Risk": "#F59E0B",
        "Low Risk": "#22C55E"
    },
    labels={
        "potential_waste_risk":
            "Risk Level",
        "meal_count":
            "Number of Meals"
    }
)

fig_risk.update_layout(
    height=450,
    legend_title_text="Risk Level"
)

st.plotly_chart(
    fig_risk,
    width="stretch"
)


# ============================================================
# HIGH RISK MEALS
# ============================================================

st.subheader(
    "🚨 High Potential Food-Waste Risk Meals"
)

high_risk_meals = (
    filtered_meal_waste[
        filtered_meal_waste[
            "potential_waste_risk"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
        == "high risk"
    ]
    .copy()
)

high_risk_meals = (
    high_risk_meals
    .sort_values(
        "total_predicted_orders",
        ascending=False
    )
)

if high_risk_meals.empty:

    st.success(
        "No high-risk meals found "
        "for the selected filters."
    )

else:

    st.dataframe(
        high_risk_meals,
        width="stretch",
        hide_index=True
    )


# ============================================================
# NEW MONTH PREDICTION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📤 New Month Demand Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
Upload a new month's raw canteen demand CSV.

The dashboard validates the input and uses the
**existing trained Random Forest model** to predict demand.
"""
)

st.warning(
    """
**Required columns:**

`id`, `week`, `center_id`, `meal_id`,
`checkout_price`, `base_price`,
`emailer_for_promotion`, `homepage_featured`

`num_orders` is not required because it is the target
variable being predicted.
"""
)


uploaded_file = st.file_uploader(
    "Upload New Month CSV",
    type=["csv"],
    help="Upload the raw input CSV for a new month."
)


if uploaded_file is not None:

    st.subheader(
        "📋 Uploaded Data Preview"
    )

    try:

        uploaded_df = pd.read_csv(
            uploaded_file
        )

        st.write(
            f"Uploaded rows: **{len(uploaded_df):,}**"
        )

        st.dataframe(
            uploaded_df.head(10),
            width="stretch",
            hide_index=True
        )

        st.subheader(
            "🔎 Validation & Prediction"
        )

        with st.spinner(
            "Validating data and generating predictions..."
        ):

            prediction_result, duplicate_count = (
                predict_new_month(
                    uploaded_df
                )
            )

        if duplicate_count > 0:

            st.warning(
                f"{duplicate_count} duplicate ID(s) "
                "were found and removed before prediction."
            )

        st.success(
            "New month data validated successfully "
            "and predictions were generated."
        )

        # ====================================================
        # NEW MONTH KPIs
        # ====================================================

        new_total_orders = int(
            prediction_result[
                "predicted_num_orders"
            ].sum()
        )

        new_average_orders = round(
            prediction_result[
                "predicted_num_orders"
            ].mean(),
            2
        )

        new_high_demand = int(
            prediction_result[
                "demand_level"
            ]
            .eq("High")
            .sum()
        )

        new_high_risk = int(
            prediction_result[
                "potential_waste_risk"
            ]
            .eq("High Risk")
            .sum()
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Total Predicted Orders",
                f"{new_total_orders:,}"
            )

        with c2:

            st.metric(
                "Average Orders",
                f"{new_average_orders:,.2f}"
            )

        with c3:

            st.metric(
                "High Demand Rows",
                f"{new_high_demand:,}"
            )

        with c4:

            st.metric(
                "High Risk Rows",
                f"{new_high_risk:,}"
            )


        # ====================================================
        # NEW MONTH DEMAND DISTRIBUTION
        # ====================================================

        st.subheader(
            "📊 New Month Demand Distribution"
        )

        demand_counts = (
            prediction_result[
                "demand_level"
            ]
            .value_counts()
            .reset_index()
        )

        demand_counts.columns = [
            "demand_level",
            "row_count"
        ]

        fig_new_demand = px.bar(
            demand_counts,
            x="demand_level",
            y="row_count",
            labels={
                "demand_level":
                    "Demand Level",
                "row_count":
                    "Number of Rows"
            },
            color="demand_level",
            color_discrete_map={
                "High": "#EF4444",
                "Medium": "#F59E0B",
                "Low": "#22C55E"
            },
            title="Predicted Demand Level Distribution"
        )

        fig_new_demand.update_layout(
            height=400
        )

        st.plotly_chart(
            fig_new_demand,
            width="stretch"
        )


        # ====================================================
        # NEW MONTH RISK DISTRIBUTION
        # ====================================================

        st.subheader(
            "♻️ New Month Potential Food-Waste Risk"
        )

        new_risk_counts = (
            prediction_result[
                "potential_waste_risk"
            ]
            .value_counts()
            .reset_index()
        )

        new_risk_counts.columns = [
            "potential_waste_risk",
            "row_count"
        ]

        fig_new_risk = px.bar(
            new_risk_counts,
            x="potential_waste_risk",
            y="row_count",
            labels={
                "potential_waste_risk":
                    "Risk Level",
                "row_count":
                    "Number of Rows"
            },
            color="potential_waste_risk",
            color_discrete_map={
                "High Risk": "#EF4444",
                "Medium Risk": "#F59E0B",
                "Low Risk": "#22C55E"
            },
            title="Potential Food-Waste Risk Distribution"
        )

        fig_new_risk.update_layout(
            height=400
        )

        st.plotly_chart(
            fig_new_risk,
            width="stretch"
        )


        # ====================================================
        # TOP NEW MONTH MEALS
        # ====================================================

        st.subheader(
            "🍽️ Top Predicted Meals — New Month"
        )

        top_new_meals = (
            prediction_result
            .groupby(
                [
                    "meal_id",
                    "category",
                    "cuisine"
                ],
                as_index=False
            )[
                "predicted_num_orders"
            ]
            .sum()
            .sort_values(
                "predicted_num_orders",
                ascending=False
            )
            .head(10)
            .copy()
        )

        top_new_meals["meal_label"] = (
            "Meal "
            + top_new_meals[
                "meal_id"
            ].astype(str)
            + " | "
            + top_new_meals[
                "category"
            ]
        )

        fig_new_meals = px.bar(
            top_new_meals.sort_values(
                "predicted_num_orders",
                ascending=True
            ),
            x="predicted_num_orders",
            y="meal_label",
            orientation="h",
            hover_data=[
                "meal_id",
                "category",
                "cuisine"
            ],
            labels={
                "predicted_num_orders":
                    "Predicted Orders",
                "meal_label":
                    "Meal"
            },
            title="Top 10 Predicted Meals"
        )

        fig_new_meals.update_layout(
            height=500
        )

        st.plotly_chart(
            fig_new_meals,
            width="stretch"
        )


        # ====================================================
        # HIGH RISK NEW MONTH
        # ====================================================

        st.subheader(
            "🚨 High Potential Food-Waste Risk — New Month"
        )

        high_risk_new = (
            prediction_result[
                prediction_result[
                    "potential_waste_risk"
                ]
                == "High Risk"
            ]
            .sort_values(
                "predicted_num_orders",
                ascending=False
            )
        )

        if high_risk_new.empty:

            st.success(
                "No high potential "
                "waste-risk rows were detected."
            )

        else:

            st.dataframe(
                high_risk_new.head(20),
                width="stretch",
                hide_index=True
            )


        # ====================================================
        # COMPLETE RESULTS
        # ====================================================

        st.subheader(
            "📋 Complete New Month Predictions"
        )

        st.dataframe(
            prediction_result,
            width="stretch",
            hide_index=True
        )


        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        csv_data = (
            prediction_result
            .to_csv(index=False)
            .encode("utf-8")
        )

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_data,
            file_name="new_month_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            "Prediction could not be completed."
        )

        st.code(
            str(e)
        )


# ============================================================
# DATA SUMMARY
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📋 Dashboard Data Summary</div>',
    unsafe_allow_html=True
)

summary_col1, summary_col2, summary_col3 = (
    st.columns(3)
)


with summary_col1:

    st.metric(
        "Meals Analyzed",
        filtered_meal[
            "meal_id"
        ].nunique()
    )


with summary_col2:

    st.metric(
        "Centers Analyzed",
        filtered_center[
            "center_id"
        ].nunique()
    )


with summary_col3:

    st.metric(
        "Weeks Analyzed",
        weekly_df[
            "week"
        ].nunique()
    )


# ============================================================
# PROJECT NOTE
# ============================================================

st.markdown("---")

st.info(
    """
**Project Note:** Potential food-waste risk is a predictive
risk indicator derived from demand analytics.

It should not be interpreted as an actual percentage of
food wasted because prepared quantity and actual waste
quantity are not directly available in the current dataset.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Smart Canteen Demand Prediction & Food Waste Analytics "
    "| Machine Learning + Streamlit"
)