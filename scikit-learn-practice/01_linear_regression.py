"""
Linear regression on California Housing: fit, predict, MSE/R^2, plot predicted vs actual.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))

housing = fetch_california_housing(as_frame=True)
X = housing.frame.drop(columns=["MedHouseVal"])
y = housing.frame["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression on California Housing")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f}  (in units of 100k USD, same as target)")
print(f"  R^2  : {r2:.4f}")

print("\nLearned coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name:12s}: {coef:+.4f}")
print(f"  intercept   : {model.intercept_:+.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred, alpha=0.3, s=10)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", linewidth=1, label="perfect prediction")
ax.set_xlabel("Actual median house value (100k USD)")
ax.set_ylabel("Predicted median house value (100k USD)")
ax.set_title(f"Linear Regression: Predicted vs Actual (R^2={r2:.3f})")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = os.path.join(HERE, "01_predicted_vs_actual.png")
fig.savefig(out_path, dpi=120)
print(f"\nSaved {out_path}")
