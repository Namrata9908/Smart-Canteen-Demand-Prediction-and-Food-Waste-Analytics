import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "data/processed/food_demand_features.csv"
)

MODEL_FILE = Path(
    "models/smart_canteen_demand_model_final.pkl"
)

OUTPUT_DIR = Path(
    "outputs/model_analysis"
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


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SMART CANTEEN - MODEL ANALYSIS & FEATURE IMPORTANCE")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}"
    )

print("Dataset found :", DATA_FILE)
print("Model found   :", MODEL_FILE)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset shape:", df.shape)


# ============================================================
# TARGET
# ============================================================

TARGET = "num_orders"

print("\nTarget variable:", TARGET)


# ============================================================
# REMOVE TARGET LEAKAGE
# ============================================================

print("\nRemoving target leakage...")

if "orders_per_area" in df.columns:

    df = df.drop(
        columns=["orders_per_area"]
    )

    print("Removed: orders_per_area")

else:

    print("orders_per_area not found")


# ============================================================
# CREATE X AND y
# ============================================================

X = df.drop(
    columns=[TARGET, "id"]
)

y = df[TARGET]


print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("RECREATING TEST SET")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# LOAD FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING FINAL MODEL")
print("=" * 70)

pipeline = joblib.load(
    MODEL_FILE
)

print("\nModel loaded successfully!")


# ============================================================
# MODEL PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = pipeline.predict(
    X_test
)

print("Predictions generated:", len(y_pred))


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\nMean Absolute Error (MAE):")
print(round(mae, 2))

print("\nRoot Mean Squared Error (RMSE):")
print(round(rmse, 2))

print("\nR2 Score:")
print(round(r2, 4))


# ============================================================
# GET PREPROCESSOR
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING FEATURE IMPORTANCE")
print("=" * 70)

preprocessor = pipeline.named_steps[
    "preprocessor"
]

model = pipeline.named_steps[
    "model"
]


# ============================================================
# GET TRANSFORMED FEATURE NAMES
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)

feature_importances = (
    model.feature_importances_
)


print("\nNumber of transformed features:",
      len(feature_names))

print("Number of importance values:",
      len(feature_importances))


# ============================================================
# FEATURE IMPORTANCE DATAFRAME
# ============================================================

importance_df = pd.DataFrame(
    {
        "transformed_feature": feature_names,
        "importance": feature_importances
    }
)

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
).reset_index(drop=True)


# ============================================================
# CLEAN FEATURE NAMES
# ============================================================

def get_original_feature_name(feature):

    if feature.startswith("categorical__"):

        feature = feature.replace(
            "categorical__",
            ""
        )

        # category__Beverages
        # cuisine__Indian
        # center_type__TYPE_A

        if "__" in feature:

            return feature.split("__")[0]

        return feature

    if feature.startswith("numeric__"):

        return feature.replace(
            "numeric__",
            ""
        )

    return feature


importance_df[
    "original_feature"
] = importance_df[
    "transformed_feature"
].apply(
    get_original_feature_name
)


# ============================================================
# AGGREGATE IMPORTANCE
# ============================================================

aggregated_importance = (
    importance_df
    .groupby(
        "original_feature",
        as_index=False
    )["importance"]
    .sum()
)

aggregated_importance = (
    aggregated_importance
    .sort_values(
        by="importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE FULL FEATURE IMPORTANCE
# ============================================================

full_output_file = (
    OUTPUT_DIR /
    "feature_importance_all.csv"
)

importance_df.to_csv(
    full_output_file,
    index=False
)

print("\nFull feature importance saved:")
print(full_output_file)


# ============================================================
# SAVE AGGREGATED FEATURE IMPORTANCE
# ============================================================

aggregated_output_file = (
    OUTPUT_DIR /
    "feature_importance.csv"
)

aggregated_importance.to_csv(
    aggregated_output_file,
    index=False
)

print("\nAggregated feature importance saved:")
print(aggregated_output_file)


# ============================================================
# TOP 10 FEATURES
# ============================================================

top_10 = (
    aggregated_importance
    .head(10)
    .copy()
)

print("\n" + "=" * 70)
print("TOP 10 IMPORTANT FEATURES")
print("=" * 70)

for index, row in top_10.iterrows():

    print(
        f"{index + 1:2}. "
        f"{row['original_feature']:<25} "
        f"{row['importance']:.4f}"
    )


# ============================================================
# SAVE TOP 10
# ============================================================

top10_output_file = (
    OUTPUT_DIR /
    "top_10_features.csv"
)

top_10.to_csv(
    top10_output_file,
    index=False
)

print("\nTop 10 features saved:")
print(top10_output_file)


# ============================================================
# FEATURE IMPORTANCE CHART
# ============================================================

print("\nCreating feature importance chart...")

plot_df = (
    top_10
    .sort_values(
        by="importance",
        ascending=True
    )
)

plt.figure(
    figsize=(10, 6)
)

plt.barh(
    plot_df["original_feature"],
    plot_df["importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 Features Influencing Demand Prediction"
)

plt.tight_layout()


chart_file = (
    CHART_DIR /
    "10_feature_importance.png"
)

plt.savefig(
    chart_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nChart saved:")
print(chart_file)


# ============================================================
# MODEL PERFORMANCE SUMMARY
# ============================================================

performance = pd.DataFrame(
    {
        "metric": [
            "MAE",
            "RMSE",
            "R2"
        ],
        "value": [
            mae,
            rmse,
            r2
        ]
    }
)

performance_file = (
    OUTPUT_DIR /
    "model_performance.csv"
)

performance.to_csv(
    performance_file,
    index=False
)

print("\nModel performance saved:")
print(performance_file)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nModel:")
print("Random Forest Regressor")

print("\nMAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))

print("\nGenerated files:")

print(
    "1.",
    full_output_file
)

print(
    "2.",
    aggregated_output_file
)

print(
    "3.",
    top10_output_file
)

print(
    "4.",
    performance_file
)

print(
    "5.",
    chart_file
)

print("\n" + "=" * 70)