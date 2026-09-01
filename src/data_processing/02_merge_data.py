import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load datasets
# --------------------------------------------------

print("=" * 70)
print("SMART CANTEEN - DATA MERGING")
print("=" * 70)

print("\nLoading datasets...")

train = pd.read_csv(RAW_DIR / "train.csv")
meal_info = pd.read_csv(RAW_DIR / "meal_info.csv")
center_info = pd.read_csv(RAW_DIR / "fulfilment_center_info.csv")

print(f"Train data        : {train.shape}")
print(f"Meal information  : {meal_info.shape}")
print(f"Center information: {center_info.shape}")

# --------------------------------------------------
# Merge train + meal information
# --------------------------------------------------

print("\nMerging train with meal information...")

merged = train.merge(
    meal_info,
    on="meal_id",
    how="left"
)

print(f"After meal merge: {merged.shape}")

# --------------------------------------------------
# Merge with fulfilment center information
# --------------------------------------------------

print("\nMerging with fulfilment center information...")

merged = merged.merge(
    center_info,
    on="center_id",
    how="left"
)

print(f"After center merge: {merged.shape}")

# --------------------------------------------------
# Check missing values after merging
# --------------------------------------------------

print("\nMissing values after merge:")

missing = merged.isnull().sum()

print(missing[missing > 0])

# --------------------------------------------------
# Check duplicate rows
# --------------------------------------------------

print("\nDuplicate rows:", merged.duplicated().sum())

# --------------------------------------------------
# Display final columns
# --------------------------------------------------

print("\nFinal columns:")

for column in merged.columns:
    print("-", column)

# --------------------------------------------------
# Save processed dataset
# --------------------------------------------------

output_file = PROCESSED_DIR / "food_demand_merged.csv"

merged.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("MERGE COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nSaved file:")
print(output_file)

print(f"\nFinal dataset shape: {merged.shape}")
