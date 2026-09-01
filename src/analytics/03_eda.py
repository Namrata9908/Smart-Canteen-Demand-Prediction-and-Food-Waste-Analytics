import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_FILE = Path("data/processed/food_demand_merged.csv")
OUTPUT_DIR = Path("outputs/charts")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("=" * 70)
print("SMART CANTEEN - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)

# --------------------------------------------------
# Basic statistics
# --------------------------------------------------

print("\n" + "=" * 70)
print("BASIC STATISTICS")
print("=" * 70)

print(df.describe().to_string())

# --------------------------------------------------
# 1. Weekly demand
# --------------------------------------------------

weekly_demand = (
    df.groupby("week")["num_orders"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(12, 6))
plt.plot(
    weekly_demand["week"],
    weekly_demand["num_orders"]
)
plt.xlabel("Week")
plt.ylabel("Total Orders")
plt.title("Weekly Food Demand Trend")
plt.tight_layout()

weekly_chart = OUTPUT_DIR / "01_weekly_demand.png"
plt.savefig(weekly_chart)
plt.close()

print("\nCreated:", weekly_chart)

# --------------------------------------------------
# 2. Category-wise demand
# --------------------------------------------------

category_demand = (
    df.groupby("category")["num_orders"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))
category_demand.plot(kind="bar")
plt.xlabel("Meal Category")
plt.ylabel("Total Orders")
plt.title("Demand by Meal Category")
plt.xticks(rotation=45)
plt.tight_layout()

category_chart = OUTPUT_DIR / "02_category_demand.png"
plt.savefig(category_chart)
plt.close()

print("Created:", category_chart)

# --------------------------------------------------
# 3. Cuisine-wise demand
# --------------------------------------------------

cuisine_demand = (
    df.groupby("cuisine")["num_orders"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))
cuisine_demand.plot(kind="bar")
plt.xlabel("Cuisine")
plt.ylabel("Total Orders")
plt.title("Demand by Cuisine")
plt.xticks(rotation=45)
plt.tight_layout()

cuisine_chart = OUTPUT_DIR / "03_cuisine_demand.png"
plt.savefig(cuisine_chart)
plt.close()

print("Created:", cuisine_chart)

# --------------------------------------------------
# 4. Center type demand
# --------------------------------------------------

center_type_demand = (
    df.groupby("center_type")["num_orders"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 6))
center_type_demand.plot(kind="bar")
plt.xlabel("Center Type")
plt.ylabel("Total Orders")
plt.title("Demand by Center Type")
plt.tight_layout()

center_chart = OUTPUT_DIR / "04_center_type_demand.png"
plt.savefig(center_chart)
plt.close()

print("Created:", center_chart)

# --------------------------------------------------
# 5. Promotion vs demand
# --------------------------------------------------

promotion_demand = (
    df.groupby("emailer_for_promotion")["num_orders"]
    .mean()
)

print("\nAverage orders by email promotion:")
print(promotion_demand)

plt.figure(figsize=(8, 6))
promotion_demand.plot(kind="bar")
plt.xlabel("Emailer for Promotion (0 = No, 1 = Yes)")
plt.ylabel("Average Orders")
plt.title("Promotion vs Average Demand")
plt.tight_layout()

promotion_chart = OUTPUT_DIR / "05_promotion_demand.png"
plt.savefig(promotion_chart)
plt.close()

print("Created:", promotion_chart)

# --------------------------------------------------
# 6. Homepage featured vs demand
# --------------------------------------------------

homepage_demand = (
    df.groupby("homepage_featured")["num_orders"]
    .mean()
)

print("\nAverage orders by homepage feature:")
print(homepage_demand)

plt.figure(figsize=(8, 6))
homepage_demand.plot(kind="bar")
plt.xlabel("Homepage Featured (0 = No, 1 = Yes)")
plt.ylabel("Average Orders")
plt.title("Homepage Feature vs Average Demand")
plt.tight_layout()

homepage_chart = OUTPUT_DIR / "06_homepage_demand.png"
plt.savefig(homepage_chart)
plt.close()

print("Created:", homepage_chart)

# --------------------------------------------------
# 7. Top 10 meals
# --------------------------------------------------

top_meals = (
    df.groupby("meal_id")["num_orders"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 meals by total orders:")
print(top_meals)

plt.figure(figsize=(10, 6))
top_meals.sort_values().plot(kind="barh")
plt.xlabel("Total Orders")
plt.ylabel("Meal ID")
plt.title("Top 10 Meals by Demand")
plt.tight_layout()

meal_chart = OUTPUT_DIR / "07_top_10_meals.png"
plt.savefig(meal_chart)
plt.close()

print("Created:", meal_chart)

# --------------------------------------------------
# 8. Top 10 centers
# --------------------------------------------------

top_centers = (
    df.groupby("center_id")["num_orders"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 centers by total orders:")
print(top_centers)

plt.figure(figsize=(10, 6))
top_centers.sort_values().plot(kind="barh")
plt.xlabel("Total Orders")
plt.ylabel("Center ID")
plt.title("Top 10 Centers by Demand")
plt.tight_layout()

center_top_chart = OUTPUT_DIR / "08_top_10_centers.png"
plt.savefig(center_top_chart)
plt.close()

print("Created:", center_top_chart)

# --------------------------------------------------
# 9. Price vs demand
# --------------------------------------------------

price_summary = (
    df.groupby("checkout_price")["num_orders"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))
plt.scatter(
    price_summary["checkout_price"],
    price_summary["num_orders"],
    alpha=0.4
)
plt.xlabel("Checkout Price")
plt.ylabel("Average Orders")
plt.title("Checkout Price vs Average Demand")
plt.tight_layout()

price_chart = OUTPUT_DIR / "09_price_vs_demand.png"
plt.savefig(price_chart)
plt.close()

print("Created:", price_chart)

# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCharts saved inside:")
print(OUTPUT_DIR)