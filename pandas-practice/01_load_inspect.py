"""
Load and inspect a real dataset: sklearn's California Housing.

Why this dataset: it's a real, well-known regression dataset (1990 census
block groups in California), bundled with scikit-learn so it loads offline
and instantly -- no fabricated rows, no flaky download.
"""

import pandas as pd
from sklearn.datasets import fetch_california_housing

# as_frame=True gives us a ready-made pandas DataFrame instead of raw numpy arrays.
housing = fetch_california_housing(as_frame=True)
df = housing.frame  # features + target ("MedHouseVal") in one DataFrame

print("Dataset description (first few lines):")
print(housing.DESCR.strip().split("\n\n")[0])

print("\n--- df.head() ---")
print(df.head())

print("\n--- df.shape ---")
print(df.shape, "-> (rows, columns)")

print("\n--- df.info() ---")
df.info()

print("\n--- df.describe() ---")
print(df.describe())

print("\n--- df.dtypes ---")
print(df.dtypes)

print("\nColumn meanings:")
print("  MedInc      : median income in block group (10k USD)")
print("  HouseAge    : median house age in block group (years)")
print("  AveRooms    : average rooms per household")
print("  AveBedrms   : average bedrooms per household")
print("  Population  : block group population")
print("  AveOccup    : average household occupancy")
print("  Latitude/Longitude : block group location")
print("  MedHouseVal : target -- median house value (100k USD)")
