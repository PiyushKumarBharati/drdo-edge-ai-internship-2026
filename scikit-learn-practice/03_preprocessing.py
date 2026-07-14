"""
StandardScaler: why scaling matters, with a measured accuracy difference.

Why this matters: KNN and logistic regression (and gradient descent generally)
are distance/gradient based -- features on wildly different scales (e.g.
"population" in the thousands vs "median income" in single digits) distort
the model unless everything is scaled first.
"""

import os
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Turn California Housing into a classification problem: "above-median value or not"
# -- this keeps the feature scale problem (income ~0-15, population ~3-35000) intact.
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()
median_value = df["MedHouseVal"].median()
y = (df["MedHouseVal"] > median_value).astype(int)
X = df.drop(columns=["MedHouseVal"])

print("Feature ranges BEFORE scaling:")
print(X.describe().loc[["min", "max"]].T)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Without scaling ---
model_unscaled = LogisticRegression(max_iter=1000)
model_unscaled.fit(X_train, y_train)
acc_unscaled = accuracy_score(y_test, model_unscaled.predict(X_test))

# --- With scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use train-fitted scaler, never refit on test data

print("\nFeature ranges AFTER scaling (train set):")
print(f"  mean per feature (~0): {np.round(X_train_scaled.mean(axis=0), 3)}")
print(f"  std per feature (~1):  {np.round(X_train_scaled.std(axis=0), 3)}")

model_scaled = LogisticRegression(max_iter=1000)
model_scaled.fit(X_train_scaled, y_train)
acc_scaled = accuracy_score(y_test, model_scaled.predict(X_test_scaled))

print(f"\nAccuracy WITHOUT scaling: {acc_unscaled:.4f}")
print(f"Accuracy WITH scaling:    {acc_scaled:.4f}")
print(f"Difference: {(acc_scaled - acc_unscaled):+.4f}")

if acc_scaled > acc_unscaled:
    print("\nScaling helped here -- consistent with gradient-based solvers converging")
    print("better when features share a comparable scale.")
else:
    print("\nScaling did not improve accuracy on this split/model -- reported honestly,")
    print("not every model/dataset combination shows a big gap, but scaling is still")
    print("best practice for distance-based and gradient-based methods.")
