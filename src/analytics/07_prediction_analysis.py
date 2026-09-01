import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# SMART CANTEEN - PREDICTION & FOOD WASTE ANALYTICS
# ============================================================

print("=" * 70)
print("SMART CANTEEN - PREDICTION & FOOD WASTE ANALYTICS")
print("=" * 70)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

INPUT_FILE = Path(
    "data/processed/food_demand_predictions.csv"
)

MEAL_FILE = Path(
    "data/raw/meal_info.csv"
)

CENTER_FILE = Path(
    "data/raw/fulfilment_center_info.csv"
)

OUTPUT_DIR = Path(
    "outputs/analytics"
)

CHART_DIR = Path(
    "outputs/charts"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHART_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Load prediction data
# ------------------------------------------------------------

print("\nLoading prediction data...")

df = pd.read_csv(INPUT_FILE)

print("Prediction dataset shape:", df.shape)


# ------------------------------------------------------------
# Load supporting datasets
# ------------------------------------------------------------

print("\nLoading meal information...")

meal_df = pd.read_csv(MEAL_FILE)

print("Meal information shape:", meal_df.shape)


print("\nLoading fulfilment center information...")

center_df = pd.read_csv(CENTER_FILE)

print("Center information shape:", center_df.shape)


# ------------------------------------------------------------
# Merge meal information
# ------------------------------------------------------------

print("\nMerging meal information...")

df = df.merge(
    meal_df,
    on="meal_id",
    how="left"
)

print("After meal merge:", df.shape)


# ------------------------------------------------------------
# Merge center information
# ------------------------------------------------------------

print("\nMerging center information...")

df = df.merge(
    center_df,
    on="center_id",
    how="left"
)

print("After center merge:", df.shape)


# ------------------------------------------------------------
# Validate data
# ------------------------------------------------------------

print("\nChecking missing values...")

missing = df.isnull().sum()

missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)


# ------------------------------------------------------------
# 1. Overall demand summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL DEMAND SUMMARY")
print("=" * 70)

total_predicted_orders = (
    df["predicted_num_orders"].sum()
)

average_predicted_orders = (
    df["predicted_num_orders"].mean()
)

median_predicted_orders = (
    df["predicted_num_orders"].median()
)

minimum_predicted_orders = (
    df["predicted_num_orders"].min()
)

maximum_predicted_orders = (
    df["predicted_num_orders"].max()
)

print(
    "\nTotal predicted orders:",
    int(total_predicted_orders)
)

print(
    "Average predicted orders:",
    round(average_predicted_orders, 2)
)

print(
    "Median predicted orders:",
    round(median_predicted_orders, 2)
)

print(
    "Minimum predicted orders:",
    int(minimum_predicted_orders)
)

print(
    "Maximum predicted orders:",
    int(maximum_predicted_orders)
)


# ------------------------------------------------------------
# 2. Demand classification
# ------------------------------------------------------------

# Use quartiles to create relative demand groups.

low_threshold = df[
    "predicted_num_orders"
].quantile(0.25)

high_threshold = df[
    "predicted_num_orders"
].quantile(0.75)


def classify_demand(value):
    if value <= low_threshold:
        return "Low Demand"
    elif value >= high_threshold:
        return "High Demand"
    else:
        return "Medium Demand"


df["demand_level"] = (
    df["predicted_num_orders"]
    .apply(classify_demand)
)


print("\n" + "=" * 70)
print("DEMAND LEVEL DISTRIBUTION")
print("=" * 70)

print(
    df["demand_level"]
    .value_counts()
)


# ------------------------------------------------------------
# 3. Meal-wise demand analysis
# ------------------------------------------------------------

meal_demand = (
    df.groupby(
        ["meal_id", "category", "cuisine"],
        as_index=False
    )["predicted_num_orders"]
    .agg(
        total_predicted_orders="sum",
        average_predicted_orders="mean",
        prediction_count="count"
    )
    .sort_values(
        "total_predicted_orders",
        ascending=False
    )
)


print("\n" + "=" * 70)
print("TOP 10 MEALS BY PREDICTED DEMAND")
print("=" * 70)

print(
    meal_demand.head(10).to_string(
        index=False
    )
)


meal_demand.to_csv(
    OUTPUT_DIR / "meal_demand_analysis.csv",
    index=False
)


# ------------------------------------------------------------
# 4. Category-wise demand analysis
# ------------------------------------------------------------

category_demand = (
    df.groupby(
        "category",
        as_index=False
    )["predicted_num_orders"]
    .agg(
        total_predicted_orders="sum",
        average_predicted_orders="mean"
    )
    .sort_values(
        "total_predicted_orders",
        ascending=False
    )
)


print("\n" + "=" * 70)
print("CATEGORY-WISE DEMAND")
print("=" * 70)

print(
    category_demand.to_string(
        index=False
    )
)


category_demand.to_csv(
    OUTPUT_DIR / "category_demand_analysis.csv",
    index=False
)


# ------------------------------------------------------------
# 5. Center-wise demand analysis
# ------------------------------------------------------------

center_demand = (
    df.groupby(
        [
            "center_id",
            "center_type",
            "city_code",
            "region_code"
        ],
        as_index=False
    )["predicted_num_orders"]
    .agg(
        total_predicted_orders="sum",
        average_predicted_orders="mean",
        prediction_count="count"
    )
    .sort_values(
        "total_predicted_orders",
        ascending=False
    )
)


print("\n" + "=" * 70)
print("TOP 10 CENTERS BY PREDICTED DEMAND")
print("=" * 70)

print(
    center_demand.head(10).to_string(
        index=False
    )
)


center_demand.to_csv(
    OUTPUT_DIR / "center_demand_analysis.csv",
    index=False
)


# ------------------------------------------------------------
# 6. Weekly demand analysis
# ------------------------------------------------------------

weekly_demand = (
    df.groupby(
        "week",
        as_index=False
    )["predicted_num_orders"]
    .agg(
        total_predicted_orders="sum",
        average_predicted_orders="mean"
    )
    .sort_values("week")
)


print("\n" + "=" * 70)
print("WEEKLY DEMAND")
print("=" * 70)

print(
    weekly_demand.head(10).to_string(
        index=False
    )
)


weekly_demand.to_csv(
    OUTPUT_DIR / "weekly_demand_analysis.csv",
    index=False
)


# ------------------------------------------------------------
# 7. Potential food-waste risk analysis
# ------------------------------------------------------------

# IMPORTANT:
#
# The dataset does NOT contain actual prepared quantity,
# leftover quantity, or discarded food quantity.
#
# Therefore, we cannot calculate actual food waste.
#
# Instead, we create a "Potential Waste Risk" indicator
# based on predicted demand.
#
# Low-demand items can have comparatively higher risk of
# over-preparation if a fixed quantity is prepared.


def classify_waste_risk(value):
    if value <= low_threshold:
        return "High Risk"
    elif value >= high_threshold:
        return "Low Risk"
    else:
        return "Medium Risk"


df["potential_waste_risk"] = (
    df["predicted_num_orders"]
    .apply(classify_waste_risk)
)


print("\n" + "=" * 70)
print("POTENTIAL FOOD-WASTE RISK")
print("=" * 70)

print(
    df["potential_waste_risk"]
    .value_counts()
)


# ------------------------------------------------------------
# 8. Potential waste-risk meal analysis
# ------------------------------------------------------------

waste_risk_meals = (
    df.groupby(
        [
            "meal_id",
            "category",
            "cuisine"
        ],
        as_index=False
    )
    .agg(
        total_predicted_orders=(
            "predicted_num_orders",
            "sum"
        ),
        average_predicted_orders=(
            "predicted_num_orders",
            "mean"
        ),
        observations=(
            "id",
            "count"
        )
    )
    .sort_values(
        "average_predicted_orders",
        ascending=True
    )
)


waste_risk_meals[
    "potential_waste_risk"
] = waste_risk_meals[
    "average_predicted_orders"
].apply(
    classify_waste_risk
)


print("\n" + "=" * 70)
print("TOP 10 MEALS WITH POTENTIAL WASTE RISK")
print("=" * 70)

print(
    waste_risk_meals.head(10).to_string(
        index=False
    )
)


waste_risk_meals.to_csv(
    OUTPUT_DIR / "potential_food_waste_risk.csv",
    index=False
)


# ------------------------------------------------------------
# 9. Center waste-risk analysis
# ------------------------------------------------------------

center_waste_risk = (
    df.groupby(
        [
            "center_id",
            "center_type"
        ],
        as_index=False
    )["predicted_num_orders"]
    .agg(
        average_predicted_orders="mean",
        total_predicted_orders="sum"
    )
    .sort_values(
        "average_predicted_orders"
    )
)


center_waste_risk[
    "potential_waste_risk"
] = center_waste_risk[
    "average_predicted_orders"
].apply(
    classify_waste_risk
)


center_waste_risk.to_csv(
    OUTPUT_DIR / "center_food_waste_risk.csv",
    index=False
)


# ------------------------------------------------------------
# 10. Save enriched prediction dataset
# ------------------------------------------------------------

enriched_output = (
    OUTPUT_DIR /
    "prediction_analytics_dataset.csv"
)

df.to_csv(
    enriched_output,
    index=False
)


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION & FOOD-WASTE ANALYTICS COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(
    "1.",
    OUTPUT_DIR / "meal_demand_analysis.csv"
)

print(
    "2.",
    OUTPUT_DIR / "category_demand_analysis.csv"
)

print(
    "3.",
    OUTPUT_DIR / "center_demand_analysis.csv"
)

print(
    "4.",
    OUTPUT_DIR / "weekly_demand_analysis.csv"
)

print(
    "5.",
    OUTPUT_DIR / "potential_food_waste_risk.csv"
)

print(
    "6.",
    OUTPUT_DIR / "center_food_waste_risk.csv"
)

print(
    "7.",
    enriched_output
)

print("\n" + "=" * 70)