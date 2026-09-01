import pandas as pd
from pathlib import Path

# Project data location
DATA_DIR = Path("data/raw")

files = [
    "train.csv",
    "food_Demand_test.csv",
    "meal_info.csv",
    "fulfilment_center_info.csv"
]

print("=" * 70)
print("SMART CANTEEN - DATASET EXPLORATION")
print("=" * 70)

for file_name in files:
    file_path = DATA_DIR / file_name

    print("\n" + "=" * 70)
    print(f"FILE: {file_name}")
    print("=" * 70)

    if not file_path.exists():
        print("FILE NOT FOUND!")
        continue

    df = pd.read_csv(file_path)

    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\nColumn Names:")
    print(list(df.columns))

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nFirst 5 Rows:")
    print(df.head().to_string())

print("\n" + "=" * 70)
print("EXPLORATION COMPLETED")
print("=" * 70)