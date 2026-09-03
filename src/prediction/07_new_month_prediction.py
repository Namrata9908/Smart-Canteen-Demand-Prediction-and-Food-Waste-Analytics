import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# SMART CANTEEN - NEW MONTH DEMAND PREDICTION
# ============================================================

print("=" * 70)
print("SMART CANTEEN - NEW MONTH DEMAND PREDICTION")
print("=" * 70)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "smart_canteen_demand_model_final.pkl"
)

MEAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "meal_info.csv"
)

CENTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "fulfilment_center_info.csv"
)

NEW_MONTH_FILE = (
    PROJECT_ROOT
    / "data"
    / "new_month"
    / "new_month_data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "new_month"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "new_month_predictions.csv"
)


# ============================================================
# 1. CHECK INPUT FILE
# ============================================================

print("\nChecking new-month input file...")

if not NEW_MONTH_FILE.exists():

    raise FileNotFoundError(
        f"""
New-month data file not found:

{NEW_MONTH_FILE}

Please place your new monthly CSV file at:

data/new_month/new_month_data.csv
"""
    )

print("New-month input file found!")


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\nLoading trained model...")

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Trained model not found:\n{MODEL_FILE}"
    )

model = joblib.load(MODEL_FILE)

print("Model loaded successfully!")


# ============================================================
# 3. LOAD NEW MONTH DATA
# ============================================================

print("\nLoading new-month dataset...")

new_df = pd.read_csv(
    NEW_MONTH_FILE
)

print(
    "New-month dataset shape:",
    new_df.shape
)


# ============================================================
# 4. VALIDATE REQUIRED RAW COLUMNS
# ============================================================

required_raw_columns = [
    "id",
    "week",
    "center_id",
    "meal_id",
    "checkout_price",
    "base_price",
    "emailer_for_promotion",
    "homepage_featured"
]

missing_raw_columns = [
    column
    for column in required_raw_columns
    if column not in new_df.columns
]

if missing_raw_columns:

    print("\nERROR: Missing required columns:")
    print(missing_raw_columns)

    raise ValueError(
        "New-month CSV does not contain the required columns."
    )

print("All required raw columns are available.")


# ============================================================
# 5. LOAD SUPPORTING DATA
# ============================================================

print("\nLoading meal information...")

meal_df = pd.read_csv(
    MEAL_FILE
)

print(
    "Meal information shape:",
    meal_df.shape
)


print("\nLoading fulfilment center information...")

center_df = pd.read_csv(
    CENTER_FILE
)

print(
    "Center information shape:",
    center_df.shape
)


# ============================================================
# 6. CHECK DUPLICATES IN INPUT
# ============================================================

print("\nChecking duplicate IDs...")

duplicate_count = new_df["id"].duplicated().sum()

if duplicate_count > 0:

    print(
        f"WARNING: {duplicate_count} duplicate IDs found."
    )

    new_df = new_df.drop_duplicates(
        subset=["id"]
    )

else:

    print("No duplicate IDs found.")


# ============================================================
# 7. MERGE MEAL INFORMATION
# ============================================================

print("\nMerging meal information...")

new_df = new_df.merge(
    meal_df,
    on="meal_id",
    how="left"
)

print(
    "After meal merge:",
    new_df.shape
)


# ============================================================
# 8. MERGE CENTER INFORMATION
# ============================================================

print("\nMerging fulfilment center information...")

new_df = new_df.merge(
    center_df,
    on="center_id",
    how="left"
)

print(
    "After center merge:",
    new_df.shape
)


# ============================================================
# 9. CHECK MERGE QUALITY
# ============================================================

print("\nChecking merged data...")

merge_columns = [
    "category",
    "cuisine",
    "city_code",
    "region_code",
    "center_type",
    "op_area"
]

missing_after_merge = (
    new_df[merge_columns]
    .isnull()
    .sum()
)

missing_after_merge = (
    missing_after_merge[
        missing_after_merge > 0
    ]
)

if len(missing_after_merge) > 0:

    print(
        "ERROR: Missing metadata after merge:"
    )

    print(
        missing_after_merge
    )

    raise ValueError(
        "Some meal IDs or center IDs were not found "
        "in the supporting metadata."
    )

else:

    print(
        "No missing metadata after merge."
    )


# ============================================================
# 10. FEATURE ENGINEERING
# ============================================================

print("\nCreating prediction features...")


# ------------------------------------------------------------
# Price Features
# ------------------------------------------------------------

new_df["discount_amount"] = (
    new_df["base_price"]
    - new_df["checkout_price"]
)

new_df["discount_percentage"] = (
    new_df["discount_amount"]
    / new_df["base_price"]
) * 100

new_df["price_ratio"] = (
    new_df["checkout_price"]
    / new_df["base_price"]
)


# ------------------------------------------------------------
# Time Features
# ------------------------------------------------------------

new_df["month"] = (
    (new_df["week"] - 1) // 4
) + 1

new_df["quarter"] = (
    (new_df["week"] - 1) // 13
) + 1

new_df["week_of_year"] = (
    (new_df["week"] - 1) % 52
) + 1


# ------------------------------------------------------------
# Promotion Feature
# ------------------------------------------------------------

new_df["total_promotion"] = (
    new_df["emailer_for_promotion"]
    + new_df["homepage_featured"]
)


# IMPORTANT:
# Do NOT create orders_per_area.
#
# It depends on actual num_orders, which is unknown
# when predicting a new month.


# ============================================================
# 11. DEFINE MODEL FEATURES
# ============================================================

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


# ============================================================
# 12. VERIFY MODEL FEATURES
# ============================================================

print("\nChecking model features...")

missing_features = [
    column
    for column in feature_columns
    if column not in new_df.columns
]

if missing_features:

    print(
        "\nERROR: Missing model features:"
    )

    print(
        missing_features
    )

    raise ValueError(
        "Required model features are missing."
    )

print(
    "All model features are available."
)


# ============================================================
# 13. PREPARE INPUT FOR MODEL
# ============================================================

X_new = new_df[
    feature_columns
].copy()

print(
    "\nPrediction feature shape:",
    X_new.shape
)


# ============================================================
# 14. GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_new
)


# ============================================================
# 15. CLEAN PREDICTIONS
# ============================================================

predictions = np.clip(
    predictions,
    a_min=0,
    a_max=None
)

predictions = (
    predictions
    .round()
    .astype(int)
)


# ============================================================
# 16. CREATE OUTPUT
# ============================================================

prediction_df = pd.DataFrame({

    "id":
        new_df["id"],

    "week":
        new_df["week"],

    "center_id":
        new_df["center_id"],

    "meal_id":
        new_df["meal_id"],

    "category":
        new_df["category"],

    "cuisine":
        new_df["cuisine"],

    "center_type":
        new_df["center_type"],

    "predicted_num_orders":
        predictions

})


# ============================================================
# 17. ADD DEMAND LEVEL
# ============================================================

def classify_demand(value):

    if value >= 500:
        return "High"

    elif value >= 150:
        return "Medium"

    else:
        return "Low"


prediction_df["demand_level"] = (
    prediction_df[
        "predicted_num_orders"
    ]
    .apply(classify_demand)
)


# ============================================================
# 18. ADD POTENTIAL WASTE-RISK PROXY
# ============================================================

def classify_waste_risk(value):

    if value < 150:
        return "Low Risk"

    elif value < 500:
        return "Medium Risk"

    else:
        return "High Risk"


prediction_df["potential_waste_risk"] = (
    prediction_df[
        "predicted_num_orders"
    ]
    .apply(classify_waste_risk)
)


# ============================================================
# 19. SAVE PREDICTIONS
# ============================================================

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 20. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NEW MONTH PREDICTION SUMMARY")
print("=" * 70)

print(
    "\nNumber of predictions:",
    len(prediction_df)
)

print(
    "Minimum predicted orders:",
    prediction_df[
        "predicted_num_orders"
    ].min()
)

print(
    "Maximum predicted orders:",
    prediction_df[
        "predicted_num_orders"
    ].max()
)

print(
    "Average predicted orders:",
    round(
        prediction_df[
            "predicted_num_orders"
        ].mean(),
        2
    )
)


print("\nDemand level distribution:")

print(
    prediction_df[
        "demand_level"
    ].value_counts()
)


print("\nPotential waste-risk distribution:")

print(
    prediction_df[
        "potential_waste_risk"
    ].value_counts()
)


print("\nFirst 10 predictions:")

print(
    prediction_df
    .head(10)
    .to_string(index=False)
)


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("NEW MONTH DEMAND PREDICTION COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nPredictions saved at:"
)

print(
    OUTPUT_FILE
)