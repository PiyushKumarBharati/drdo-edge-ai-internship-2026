"""
Classification on Iris: LogisticRegression + KNN, accuracy, confusion matrix.
"""

import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))

iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

results = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\naccuracy={acc:.4f}")

    print(f"{name}: accuracy = {acc:.4f}")
    print(f"  confusion matrix:\n{cm}")

fig.tight_layout()
out_path = os.path.join(HERE, "02_confusion_matrices.png")
fig.savefig(out_path, dpi=120)
print(f"\nSaved {out_path}")

print("\nSummary:")
for name, acc in results.items():
    print(f"  {name}: {acc:.4f}")
