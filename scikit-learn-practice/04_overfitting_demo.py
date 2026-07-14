"""
Deliberately overfit a DecisionTree with increasing depth; plot train vs test
accuracy to show the gap widening.

Why this matters: an overfit model that looks great on training data but
degrades on unseen input is exactly the failure mode to check for before
trusting a model enough to freeze its weights and ship it to an edge device.
"""

import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))

iris = load_iris()
X, y = iris.data, iris.target

# A relatively small train split so the tree has room to overfit a small sample.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42, stratify=y
)

depths = list(range(1, 16))
train_accs, test_accs = [], []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    print(f"max_depth={depth:2d}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  gap={train_acc - test_acc:+.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(depths, train_accs, marker="o", label="train accuracy")
ax.plot(depths, test_accs, marker="o", label="test accuracy")
ax.fill_between(depths, train_accs, test_accs, alpha=0.15, color="red", label="overfitting gap")
ax.set_xlabel("max_depth")
ax.set_ylabel("accuracy")
ax.set_title("DecisionTree overfitting on Iris as depth increases")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = os.path.join(HERE, "04_overfitting_curve.png")
fig.savefig(out_path, dpi=120)
print(f"\nSaved {out_path}")

max_gap_idx = max(range(len(depths)), key=lambda i: train_accs[i] - test_accs[i])
print(f"\nLargest train/test gap at max_depth={depths[max_gap_idx]}: "
      f"train={train_accs[max_gap_idx]:.4f}, test={test_accs[max_gap_idx]:.4f}, "
      f"gap={train_accs[max_gap_idx] - test_accs[max_gap_idx]:+.4f}")
