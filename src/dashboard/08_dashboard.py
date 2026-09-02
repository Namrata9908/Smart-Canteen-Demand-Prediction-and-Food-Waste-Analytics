import streamlit as st
import pandas as pd
import plotly.express as px
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


# ============================================================
# CUSTOM CSS
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
        color: #9aa0a6;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.15);
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
    """Load an analytics CSV safely."""
    path = ANALYTICS_DIR / filename

    if not path.exists():
        st.error(f"Required file not found: {path}")
        st.stop()

    return pd.read_csv(path)


def format_number(value):
    """Format numbers for dashboard display."""
    if pd.isna(value):
        return "0"

    return f"{int(round(value)):,}"


# ============================================================
# LOAD ANALYTICS DATA
# ============================================================

category_df = load_csv("category_demand_analysis.csv")
center_df = load_csv("center_demand_analysis.csv")
center_waste_df = load_csv("center_food_waste_risk.csv")
meal_df = load_csv("meal_demand_analysis.csv")
meal_waste_df = load_csv("potential_food_waste_risk.csv")
weekly_df = load_csv("weekly_demand_analysis.csv")


# ============================================================
# DATA CLEANING
# ============================================================

category_df["category"] = category_df["category"].astype(str)

center_df["center_type"] = center_df["center_type"].astype(str)

meal_df["category"] = meal_df["category"].astype(str)
meal_df["cuisine"] = meal_df["cuisine"].astype(str)

meal_waste_df["category"] = meal_waste_df["category"].astype(str)
meal_waste_df["cuisine"] = meal_waste_df["cuisine"].astype(str)

center_waste_df["center_type"] = center_waste_df["center_type"].astype(str)


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🔎 Dashboard Filters")

st.sidebar.markdown("---")

# Category filter
categories = sorted(category_df["category"].unique().tolist())

selected_categories = st.sidebar.multiselect(
    "Select Category",
    options=categories,
    default=categories
)

# Center type filter
center_types = sorted(center_df["center_type"].unique().tolist())

selected_center_types = st.sidebar.multiselect(
    "Select Center Type",
    options=center_types,
    default=center_types
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Dashboard Scope**

    This dashboard presents:

    • ML-based demand predictions  
    • Category performance  
    • Meal-level demand  
    • Center-level demand  
    • Potential food-waste risk
    """
)


# ============================================================
# APPLY FILTERS
# ============================================================

# Category filtering
filtered_category = category_df[
    category_df["category"].isin(selected_categories)
].copy()


filtered_meal = meal_df[
    meal_df["category"].isin(selected_categories)
].copy()


filtered_meal_waste = meal_waste_df[
    meal_waste_df["category"].isin(selected_categories)
].copy()


# Center filtering
filtered_center = center_df[
    center_df["center_type"].isin(selected_center_types)
].copy()


filtered_center_waste = center_waste_df[
    center_waste_df["center_type"].isin(selected_center_types)
].copy()


# ============================================================
# FILTER EMPTY STATE
# ============================================================

if len(selected_categories) == 0 or len(selected_center_types) == 0:

    st.warning(
        "Please select at least one category and one center type "
        "from the sidebar."
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🍱 Smart Canteen Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning powered demand prediction and '
    'potential food-waste risk analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_predicted_orders = filtered_category["total_predicted_orders"].sum()

average_orders = filtered_category["average_predicted_orders"].mean()

number_categories = filtered_category["category"].nunique()

high_risk_count = (
    filtered_meal_waste["potential_waste_risk"]
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
        format_number(total_predicted_orders)
    )

with col2:
    st.metric(
        "📊 Average Orders",
        format_number(average_orders)
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
# WEEKLY DEMAND TREND
# ============================================================

st.markdown(
    '<div class="section-title">📈 Weekly Demand Trend</div>',
    unsafe_allow_html=True
)

st.caption("Overall predicted demand across the forecast period.")

weekly_plot = weekly_df.copy()

weekly_plot = weekly_plot.sort_values("week")

fig_weekly = px.line(
    weekly_plot,
    x="week",
    y="total_predicted_orders",
    markers=True,
    labels={
        "week": "Week",
        "total_predicted_orders": "Total Predicted Orders"
    }
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

    category_total = filtered_category.sort_values(
        "total_predicted_orders",
        ascending=False
    )

    fig_category_total = px.bar(
        category_total,
        x="total_predicted_orders",
        y="category",
        orientation="h",
        text="total_predicted_orders",
        labels={
            "total_predicted_orders": "Predicted Orders",
            "category": "Category"
        }
    )

    fig_category_total.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="outside"
    )

    fig_category_total.update_layout(
        height=520,
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig_category_total,
        width="stretch"
    )


with col2:

    category_average = filtered_category.sort_values(
        "average_predicted_orders",
        ascending=False
    )

    fig_category_avg = px.bar(
        category_average,
        x="average_predicted_orders",
        y="category",
        orientation="h",
        text="average_predicted_orders",
        labels={
            "average_predicted_orders": "Average Predicted Orders",
            "category": "Category"
        }
    )

    fig_category_avg.update_traces(
        texttemplate="%{text:.0f}",
        textposition="outside"
    )

    fig_category_avg.update_layout(
        height=520,
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig_category_avg,
        width="stretch"
    )


# ============================================================
# TOP 10 MEALS
# ============================================================

st.markdown(
    '<div class="section-title">🍽️ Top 10 Meals by Predicted Demand</div>',
    unsafe_allow_html=True
)

top_meals = (
    filtered_meal
    .sort_values("total_predicted_orders", ascending=False)
    .head(10)
    .copy()
)

top_meals["meal"] = (
    top_meals["meal_id"].astype(str)
    + " - "
    + top_meals["category"]
    + " - "
    + top_meals["cuisine"]
)

fig_meals = px.bar(
    top_meals.sort_values("total_predicted_orders"),
    x="total_predicted_orders",
    y="meal",
    orientation="h",
    text="total_predicted_orders",
    labels={
        "total_predicted_orders": "Predicted Orders",
        "meal": "Meal"
    }
)

fig_meals.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)

fig_meals.update_layout(
    height=520
)

st.plotly_chart(
    fig_meals,
    width="stretch"
)


# ============================================================
# TOP 10 CENTERS
# ============================================================

st.markdown(
    '<div class="section-title">🏢 Top 10 Centers by Predicted Demand</div>',
    unsafe_allow_html=True
)

top_centers = (
    filtered_center
    .sort_values("total_predicted_orders", ascending=False)
    .head(10)
    .copy()
)

top_centers["center"] = (
    "Center "
    + top_centers["center_id"].astype(str)
    + " - "
    + top_centers["center_type"]
)

fig_centers = px.bar(
    top_centers.sort_values("total_predicted_orders"),
    x="total_predicted_orders",
    y="center",
    orientation="h",
    text="total_predicted_orders",
    labels={
        "total_predicted_orders": "Predicted Orders",
        "center": "Center"
    }
)

fig_centers.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside"
)

fig_centers.update_layout(
    height=520
)

st.plotly_chart(
    fig_centers,
    width="stretch"
)


# ============================================================
# CENTER-WISE FOOD WASTE RISK
# ============================================================

st.markdown(
    '<div class="section-title">⚠️ Center-wise Potential Food-Waste Risk</div>',
    unsafe_allow_html=True
)

center_risk_plot = (
    filtered_center_waste
    .sort_values("total_predicted_orders", ascending=False)
    .copy()
)

fig_center_risk = px.bar(
    center_risk_plot,
    x="center_id",
    y="total_predicted_orders",
    color="potential_waste_risk",
    labels={
        "center_id": "Center ID",
        "total_predicted_orders": "Total Predicted Orders",
        "potential_waste_risk": "Waste Risk"
    }
)

fig_center_risk.update_layout(
    height=500
)

st.plotly_chart(
    fig_center_risk,
    width="stretch"
)


# ============================================================
# WASTE RISK BY MEAL
# ============================================================

st.markdown(
    '<div class="section-title">⚠️ Potential Food-Waste Risk by Meal</div>',
    unsafe_allow_html=True
)

risk_counts = (
    filtered_meal_waste["potential_waste_risk"]
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
    labels={
        "potential_waste_risk": "Risk Level",
        "meal_count": "Number of Meals"
    }
)

fig_risk.update_layout(
    height=450
)

st.plotly_chart(
    fig_risk,
    width="stretch"
)


# ============================================================
# HIGH RISK MEALS TABLE
# ============================================================

st.markdown(
    '<div class="section-title">🚨 High Potential Food-Waste Risk Meals</div>',
    unsafe_allow_html=True
)

high_risk_meals = filtered_meal_waste[
    filtered_meal_waste["potential_waste_risk"] == "High Risk"
].copy()

high_risk_meals = high_risk_meals.sort_values(
    "total_predicted_orders",
    ascending=False
)

if high_risk_meals.empty:

    st.success(
        "No high-risk meals found for the selected filters."
    )

else:

    st.dataframe(
        high_risk_meals,
        width="stretch",
        hide_index=True
    )


# ============================================================
# DASHBOARD DATA SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📋 Dashboard Data Summary</div>',
    unsafe_allow_html=True
)

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.metric(
        "🍽️ Meals Analyzed",
        filtered_meal["meal_id"].nunique()
    )

with summary_col2:
    st.metric(
        "🏢 Centers Analyzed",
        filtered_center["center_id"].nunique()
    )

with summary_col3:
    st.metric(
        "📅 Weeks Analyzed",
        weekly_df["week"].nunique()
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Smart Canteen Demand Prediction & Food Waste Analytics "
    "| Machine Learning + Streamlit"
)