"""
Feature prep: encode a categorical, prep a train/test split, export a clean CSV.

Why this matters: this is the exact shape of the last step before any model
(scikit-learn or TensorFlow) can consume the data -- numeric features, a clean
target column, and a defined train/test boundary.
"""

import os
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# --- Create a categorical feature to demonstrate encoding (income tier) ---
df["income_tier"] = pd.cut(
    df["MedInc"],
    bins=[0, 2.5, 4.5, 6.5, df["MedInc"].max()],
    labels=["low", "mid", "high", "very_high"],
)
print("income_tier value counts:")
print(df["income_tier"].value_counts())

# --- One-hot encode the categorical column ---
df_encoded = pd.get_dummies(df, columns=["income_tier"], prefix="tier")
print("\nColumns after one-hot encoding:")
print([c for c in df_encoded.columns if c.startswith("tier_")])

# --- Train/test split prep ---
target_col = "MedHouseVal"
feature_cols = [c for c in df_encoded.columns if c != target_col]

X = df_encoded[feature_cols]
y = df_encoded[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain set: {X_train.shape[0]} rows")
print(f"Test set:  {X_test.shape[0]} rows")

# --- Export the cleaned, encoded, full dataset (features + target back together) ---
clean_df = df_encoded.copy()
out_path = os.path.join(HERE, "california_housing_clean.csv")
clean_df.to_csv(out_path, index=False)
print(f"\nSaved cleaned dataset ({clean_df.shape[0]} rows, {clean_df.shape[1]} columns) to {out_path}")
