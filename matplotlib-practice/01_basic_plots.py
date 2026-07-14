"""
Basic plot types: line, bar, scatter, histogram.

Why this matters: every one of these shows up later -- line for training
curves, bar for the size/latency/accuracy comparison chart, scatter for
predicted vs actual, histogram for confidence score distributions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# --- Line plot: e.g. a metric over epochs ---
epochs = np.arange(1, 11)
loss = 2.0 * np.exp(-0.3 * epochs) + 0.1  # a plausible decaying loss curve
axes[0, 0].plot(epochs, loss, marker="o", color="tab:blue")
axes[0, 0].set_title("Line plot: loss over epochs")
axes[0, 0].set_xlabel("epoch")
axes[0, 0].set_ylabel("loss")
axes[0, 0].grid(True, alpha=0.3)

# --- Bar plot: e.g. model size comparison ---
model_names = ["float32", "dynamic-range", "int8"]
sizes_kb = [180, 52, 48]  # illustrative shape for this demo; real numbers come from tensorflow-lite/
axes[0, 1].bar(model_names, sizes_kb, color=["tab:blue", "tab:orange", "tab:green"])
axes[0, 1].set_title("Bar plot: model size by format (illustrative)")
axes[0, 1].set_ylabel("size (KB)")

# --- Scatter plot: e.g. predicted vs actual values ---
rng = np.random.default_rng(seed=1)
actual = rng.uniform(0, 10, 50)
predicted = actual + rng.normal(0, 0.5, 50)
axes[1, 0].scatter(actual, predicted, alpha=0.6, color="tab:purple")
axes[1, 0].plot([0, 10], [0, 10], "k--", linewidth=1, label="perfect prediction")
axes[1, 0].set_title("Scatter plot: predicted vs actual")
axes[1, 0].set_xlabel("actual")
axes[1, 0].set_ylabel("predicted")
axes[1, 0].legend()

# --- Histogram: e.g. distribution of confidence scores ---
confidences = rng.beta(5, 2, 500)  # skewed toward high confidence, realistic shape
axes[1, 1].hist(confidences, bins=20, color="tab:red", alpha=0.7, edgecolor="black")
axes[1, 1].set_title("Histogram: prediction confidence distribution")
axes[1, 1].set_xlabel("confidence")
axes[1, 1].set_ylabel("count")

fig.tight_layout()
out_path = os.path.join(HERE, "01_basic_plots.png")
fig.savefig(out_path, dpi=120)
print(f"Saved {out_path}")
