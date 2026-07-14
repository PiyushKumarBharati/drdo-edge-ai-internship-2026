"""
groupby, agg, sorting on the California Housing dataset.

Why this matters: understanding your data distribution (e.g. "does house
value vary a lot by age bracket?") is the same skill as understanding class
balance and per-class performance later when we look at model accuracy.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# --- Bin a continuous column into groups -- like bucketing house age ---
df["age_bracket"] = pd.cut(
    df["HouseAge"],
    bins=[0, 15, 30, 45, 60],
    labels=["0-15", "16-30", "31-45", "46-60"],
)

print("--- Mean house value by age bracket ---")
by_age = df.groupby("age_bracket", observed=True)["MedHouseVal"].mean().sort_values(ascending=False)
print(by_age)

# --- Multiple aggregations at once with .agg() ---
print("\n--- Multiple stats per age bracket ---")
stats = df.groupby("age_bracket", observed=True)["MedHouseVal"].agg(["mean", "median", "std", "count"])
print(stats)

# --- Bin location into a rough grid to see regional value patterns ---
df["lat_bin"] = (df["Latitude"] // 1).astype(int)
df["lon_bin"] = (df["Longitude"] // 1).astype(int)

print("\n--- Top 5 lat/lon grid cells by average house value ---")
by_region = (
    df.groupby(["lat_bin", "lon_bin"])["MedHouseVal"]
    .agg(["mean", "count"])
    .query("count >= 20")               # ignore sparsely populated grid cells
    .sort_values("mean", ascending=False)
    .head(5)
)
print(by_region)

# --- groupby with multiple aggregated columns at once ---
print("\n--- Income and rooms by age bracket ---")
combined = df.groupby("age_bracket", observed=True).agg(
    avg_income=("MedInc", "mean"),
    avg_rooms=("AveRooms", "mean"),
    n_samples=("MedHouseVal", "size"),
)
print(combined)
