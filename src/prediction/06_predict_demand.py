import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ============================================================
# SMART CANTEEN - DEMAND PREDICTION
# ============================================================

print("=" * 70)
print("SMART CANTEEN - DEMAND PREDICTION")
print("=" * 70)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

MODEL_FILE = Path(
    "models/smart_canteen_demand_model_final.pkl"
)

TEST_FILE = Path(
    "data/raw/food_Demand_test.csv"
)

MEAL_FILE = Path(
    "data/raw/meal_info.csv"
)

CENTER_FILE = Path(
    "data/raw/fulfilment_center_info.csv"
)

OUTPUT_DIR = Path(
    "data/processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "food_demand_predictions.csv"


# ------------------------------------------------------------
# 1. Load trained model
# ------------------------------------------------------------

print("\nLoading final trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")


# ------------------------------------------------------------
# 2. Load test dataset
# ------------------------------------------------------------

print("\nLoading test dataset...")

test_df = pd.read_csv(TEST_FILE)

print("Test dataset shape:", test_df.shape)


# ------------------------------------------------------------
# 3. Load supporting datasets
# ------------------------------------------------------------

print("\nLoading meal information...")

meal_df = pd.read_csv(MEAL_FILE)

print("Meal information shape:", meal_df.shape)


print("\nLoading fulfilment center information...")

center_df = pd.read_csv(CENTER_FILE)

print("Center information shape:", center_df.shape)


# ------------------------------------------------------------
# 4. Merge test data with meal information
# ------------------------------------------------------------

print("\nMerging test data with meal information...")

test_df = test_df.merge(
    meal_df,
    on="meal_id",
    how="left"
)

print("After meal merge:", test_df.shape)


# ------------------------------------------------------------
# 5. Merge with fulfilment center information
# ------------------------------------------------------------

print("\nMerging test data with fulfilment center information...")

test_df = test_df.merge(
    center_df,
    on="center_id",
    how="left"
)

print("After center merge:", test_df.shape)


# ------------------------------------------------------------
# 6. Check missing values after merge
# ------------------------------------------------------------

print("\nChecking missing values...")

missing_values = test_df.isnull().sum()

missing_values = missing_values[
    missing_values > 0
]

if len(missing_values) == 0:
    print("No missing values found.")
else:
    print(missing_values)


# ------------------------------------------------------------
# 7. Feature engineering
# ------------------------------------------------------------

print("\nPreparing prediction features...")


# Price-based features

test_df["discount_amount"] = (
    test_df["base_price"]
    - test_df["checkout_price"]
)

test_df["discount_percentage"] = (
    test_df["discount_amount"]
    / test_df["base_price"]
) * 100

test_df["price_ratio"] = (
    test_df["checkout_price"]
    / test_df["base_price"]
)


# Time-based features

test_df["month"] = (
    (test_df["week"] - 1) // 4
) + 1

test_df["quarter"] = (
    (test_df["week"] - 1) // 13
) + 1

test_df["week_of_year"] = (
    (test_df["week"] - 1) % 52
) + 1


# Promotion feature

test_df["total_promotion"] = (
    test_df["emailer_for_promotion"]
    + test_df["homepage_featured"]
)


# IMPORTANT:
# Do NOT create orders_per_area here.
# It uses num_orders, which is unknown for test data
# and was removed from the final model because of target leakage.


# ------------------------------------------------------------
# 8. Define model features
# ------------------------------------------------------------

categorical_features = [
    "category",
    "cuisine",
    "center_type"
]

numeric_features = [
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
    "total_promotion"
]

feature_columns = (
    numeric_features
    + categorical_features
)


# ------------------------------------------------------------
# 9. Verify required columns
# ------------------------------------------------------------

print("\nChecking required model features...")

missing_features = [
    col
    for col in feature_columns
    if col not in test_df.columns
]

if missing_features:
    print("\nERROR: Missing features:")
    print(missing_features)
    raise ValueError(
        "Required model features are missing."
    )

print("All required features are available.")


# ------------------------------------------------------------
# 10. Prepare X_test
# ------------------------------------------------------------

X_test = test_df[
    feature_columns
].copy()

print("\nPrediction feature shape:", X_test.shape)


# ------------------------------------------------------------
# 11. Generate predictions
# ------------------------------------------------------------

print("\nGenerating demand predictions...")

predictions = model.predict(X_test)


# ------------------------------------------------------------
# 12. Clean predictions
# ------------------------------------------------------------

# Demand cannot be negative

predictions = np.clip(
    predictions,
    a_min=0,
    a_max=None
)


# ------------------------------------------------------------
# 13. Create output dataframe
# ------------------------------------------------------------

prediction_df = pd.DataFrame({
    "id": test_df["id"],
    "week": test_df["week"],
    "center_id": test_df["center_id"],
    "meal_id": test_df["meal_id"],
    "predicted_num_orders": predictions
})


# ------------------------------------------------------------
# 14. Round predictions
# ------------------------------------------------------------

prediction_df["predicted_num_orders"] = (
    prediction_df["predicted_num_orders"]
    .round()
    .astype(int)
)


# ------------------------------------------------------------
# 15. Save predictions
# ------------------------------------------------------------

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 16. Prediction summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PREDICTION SUMMARY")
print("=" * 70)

print(
    "\nNumber of predictions:",
    len(prediction_df)
)

print(
    "Minimum predicted orders:",
    prediction_df["predicted_num_orders"].min()
)

print(
    "Maximum predicted orders:",
    prediction_df["predicted_num_orders"].max()
)

print(
    "Average predicted orders:",
    round(
        prediction_df["predicted_num_orders"].mean(),
        2
    )
)


print("\nFirst 10 predictions:")

print(
    prediction_df.head(10).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Completed
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DEMAND PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nPredictions saved at:")
print(OUTPUT_FILE)