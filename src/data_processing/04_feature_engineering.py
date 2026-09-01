import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_FILE = Path("data/processed/food_demand_merged.csv")
OUTPUT_DIR = Path("data/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("=" * 70)
print("SMART CANTEEN - FEATURE ENGINEERING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal shape:", df.shape)

# --------------------------------------------------
# 1. Price-based features
# --------------------------------------------------

df["discount_amount"] = (
    df["base_price"] - df["checkout_price"]
)

df["discount_percentage"] = (
    df["discount_amount"] / df["base_price"]
) * 100

df["price_ratio"] = (
    df["checkout_price"] / df["base_price"]
)

# --------------------------------------------------
# 2. Time-based features
# --------------------------------------------------

# Dataset contains 145 weeks.
# Approximate month, quarter and week-of-year
# features are created from sequential week number.

df["month"] = (
    (df["week"] - 1) // 4
) + 1

df["quarter"] = (
    (df["week"] - 1) // 13
) + 1

df["week_of_year"] = (
    (df["week"] - 1) % 52
) + 1

# --------------------------------------------------
# 3. Promotion interaction
# --------------------------------------------------

df["total_promotion"] = (
    df["emailer_for_promotion"]
    + df["homepage_featured"]
)

# --------------------------------------------------
# 4. Operational feature
# --------------------------------------------------

df["orders_per_area"] = (
    df["num_orders"] / df["op_area"]
)

# --------------------------------------------------
# Check for invalid values
# --------------------------------------------------

print("\nChecking engineered features...")

# Missing values
print("\nMissing values:")

missing_counts = df.isnull().sum()

print(
    missing_counts[
        missing_counts > 0
    ]
)

# Infinite values
print("\nInfinite values:")

numeric_df = df.select_dtypes(
    include="number"
)

infinite_counts = np.isinf(
    numeric_df
).sum()

print(
    infinite_counts[
        infinite_counts > 0
    ]
)

# --------------------------------------------------
# Basic feature summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("ENGINEERED FEATURES")
print("=" * 70)

new_features = [
    "discount_amount",
    "discount_percentage",
    "price_ratio",
    "month",
    "quarter",
    "week_of_year",
    "total_promotion",
    "orders_per_area"
]

for feature in new_features:

    print(f"\n{feature}:")
    print(df[feature].describe())

# --------------------------------------------------
# Save engineered dataset
# --------------------------------------------------

output_file = (
    OUTPUT_DIR / "food_demand_features.csv"
)

df.to_csv(
    output_file,
    index=False
)

# --------------------------------------------------
# Completion message
# --------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print("\nOriginal shape :", (456548, 15))
print("New shape      :", df.shape)

print("\nSaved file:")
print(output_file)