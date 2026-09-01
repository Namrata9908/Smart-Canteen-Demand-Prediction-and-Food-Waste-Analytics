import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_FILE = Path(
    "data/processed/food_demand_features.csv"
)

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

print("=" * 70)
print("SMART CANTEEN - FINAL MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# --------------------------------------------------
# Target variable
# --------------------------------------------------

TARGET = "num_orders"

print("\nTarget variable:", TARGET)


# --------------------------------------------------
# Remove target leakage
# --------------------------------------------------

print("\nRemoving target leakage feature...")

if "orders_per_area" in df.columns:
    df = df.drop(
        columns=["orders_per_area"]
    )

    print("Removed: orders_per_area")

else:
    print("orders_per_area not found")


# --------------------------------------------------
# Create X and y
# --------------------------------------------------

X = df.drop(
    columns=[TARGET, "id"]
)

y = df[TARGET]


# --------------------------------------------------
# Feature types
# --------------------------------------------------

categorical_features = [
    "category",
    "cuisine",
    "center_type"
]

numeric_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


print("\nCategorical features:")
print(categorical_features)

print("\nNumeric features:")
print(numeric_features)


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows :", len(X_train))
print("Testing rows  :", len(X_test))


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# --------------------------------------------------
# Random Forest
# --------------------------------------------------

print("\nCreating Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# --------------------------------------------------
# Model Training
# --------------------------------------------------

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

print("\nTraining Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("\nModel training completed successfully!")


# --------------------------------------------------
# Predictions
# --------------------------------------------------

print("\nGenerating predictions...")

y_pred = pipeline.predict(
    X_test
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION")
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

print("\nR² Score:")
print(round(r2, 4))


# --------------------------------------------------
# Save final model
# --------------------------------------------------

model_file = (
    MODEL_DIR /
    "smart_canteen_demand_model_final.pkl"
)

joblib.dump(
    pipeline,
    model_file
)


# --------------------------------------------------
# Completion
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL MODEL TRAINING COMPLETED")
print("=" * 70)

print("\nTarget leakage removed:")
print("orders_per_area")

print("\nModel type:")
print("Random Forest Regressor")

print("\nModel saved at:")
print(model_file)

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))

print("\nFinal evaluation:")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))