"""
Cleaning: missing values, dtype conversion, renaming.

California Housing has no missing values out of the box, so we deliberately
inject some (clearly marked as synthetic corruption) to practice the real
cleaning workflow you'd need on messier field data.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("Missing values before any corruption:")
print(df.isna().sum())

# --- Deliberately inject missing values to practice handling them ---
# This is clearly synthetic corruption of real data, not fabricated data.
rng = np.random.default_rng(seed=0)
corrupt_idx = rng.choice(df.index, size=50, replace=False)
df.loc[corrupt_idx, "AveBedrms"] = np.nan

print(f"\nAfter injecting {len(corrupt_idx)} synthetic NaNs into AveBedrms:")
print(df.isna().sum()[df.isna().sum() > 0])

# --- Handling missing values: fill with the median (robust to outliers) ---
median_bedrms = df["AveBedrms"].median()
df["AveBedrms_filled"] = df["AveBedrms"].fillna(median_bedrms)
print(f"\nFilled {df['AveBedrms'].isna().sum()} missing AveBedrms with median={median_bedrms:.3f}")

# Drop the original column with NaNs now that we have a cleaned version.
df = df.drop(columns=["AveBedrms"]).rename(columns={"AveBedrms_filled": "AveBedrms"})

# --- dtype conversion: HouseAge is really a whole number of years ---
print("\nHouseAge dtype before:", df["HouseAge"].dtype)
df["HouseAge"] = df["HouseAge"].astype(int)
print("HouseAge dtype after:", df["HouseAge"].dtype)

# --- Renaming columns to consistent snake_case ---
df = df.rename(columns={
    "MedInc": "median_income",
    "HouseAge": "house_age",
    "AveRooms": "avg_rooms",
    "AveBedrms": "avg_bedrooms",
    "Population": "population",
    "AveOccup": "avg_occupancy",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "MedHouseVal": "median_house_value",
})

print("\nFinal columns:", list(df.columns))
print("\nFinal isna check (should be all zero):")
print(df.isna().sum().sum(), "total missing values remaining")
print("\ndf.head() after cleaning:")
print(df.head())
