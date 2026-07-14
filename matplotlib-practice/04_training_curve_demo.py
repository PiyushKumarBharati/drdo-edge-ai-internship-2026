"""
A reusable plot_training_curves() function, demoed on a REAL dict of values.

Why this matters: this exact function gets reused in tensorflow-basics/ and
final-project/ to plot the real loss/accuracy history returned by
model.fit() (a Keras History object's .history dict has this same shape).

The numbers plotted here are not invented: this script trains a tiny logistic
regression by hand (plain NumPy gradient descent) on the real Iris dataset,
and records genuine loss/accuracy per epoch from that run.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))


def plot_training_curves(history_dict, out_path, title="Training curves"):
    """Plot loss and accuracy (train + val) side by side from a Keras-style history dict.

    history_dict is expected to have some subset of these keys:
      'loss', 'val_loss', 'accuracy', 'val_accuracy'
    (exactly what model.fit(...).history looks like).
    """
    has_acc = "accuracy" in history_dict

    n_panels = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    epochs_range = range(1, len(history_dict["loss"]) + 1)

    axes[0].plot(epochs_range, history_dict["loss"], marker="o", label="train loss")
    if "val_loss" in history_dict:
        axes[0].plot(epochs_range, history_dict["val_loss"], marker="o", label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if has_acc:
        axes[1].plot(epochs_range, history_dict["accuracy"], marker="o", label="train accuracy")
        if "val_accuracy" in history_dict:
            axes[1].plot(epochs_range, history_dict["val_accuracy"], marker="o", label="val accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_logreg_by_hand(X_train, y_train, X_val, y_val, epochs=15, lr=0.1):
    """Binary logistic regression trained with plain gradient descent.

    This is deliberately not sklearn's LogisticRegression -- writing the
    gradient descent loop by hand is what makes the per-epoch loss/accuracy
    numbers real intermediate values instead of a black box's final score.
    """
    n_features = X_train.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for _epoch in range(epochs):
        # Forward pass
        z = X_train @ weights + bias
        preds = sigmoid(z)

        # Binary cross-entropy loss (clip to avoid log(0))
        eps = 1e-9
        loss = -np.mean(y_train * np.log(preds + eps) + (1 - y_train) * np.log(1 - preds + eps))
        acc = np.mean((preds >= 0.5).astype(int) == y_train)

        # Gradients
        error = preds - y_train
        grad_w = X_train.T @ error / len(y_train)
        grad_b = np.mean(error)

        # Update
        weights -= lr * grad_w
        bias -= lr * grad_b

        # Validation metrics with the just-updated weights
        val_preds = sigmoid(X_val @ weights + bias)
        val_loss = -np.mean(y_val * np.log(val_preds + eps) + (1 - y_val) * np.log(1 - val_preds + eps))
        val_acc = np.mean((val_preds >= 0.5).astype(int) == y_val)

        history["loss"].append(loss)
        history["accuracy"].append(acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

    return history


if __name__ == "__main__":
    # Real dataset, reduced to a binary problem: "is this Iris setosa or not?"
    iris = load_iris()
    X = iris.data
    y = (iris.target == 0).astype(int)  # setosa vs rest

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalize features (mean 0, std 1) -- gradient descent converges far more reliably this way.
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    history = train_logreg_by_hand(X_train, y_train, X_val, y_val, epochs=15, lr=0.5)

    print("Per-epoch history from this real run:")
    for k, v in history.items():
        print(f"  {k}: {[round(x, 4) for x in v]}")

    out_path = os.path.join(HERE, "04_training_curve_demo.png")
    plot_training_curves(history, out_path, title="Hand-written logistic regression on Iris (setosa vs rest)")
    print(f"\nSaved {out_path}")
    print(f"Final train accuracy: {history['accuracy'][-1]:.4f}, final val accuracy: {history['val_accuracy'][-1]:.4f}")
