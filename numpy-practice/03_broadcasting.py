"""
Broadcasting rules with concrete examples.

Why this matters: image normalization ((img - mean) / std), adding a bias
vector to every row of a batch, and scaling a whole batch by one scalar are
all broadcasting. Understanding it is the difference between writing one
vectorized line and writing a slow, buggy nested loop.
"""

import numpy as np

# --- Rule 1: scalar broadcasts against anything ---
a = np.array([1, 2, 3, 4])
print("a:", a)
print("a * 10:", a * 10)  # scalar "stretched" to match a's shape

# --- Rule 2: a smaller array broadcasts against a larger one if trailing dims match ---
batch = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])  # shape (3, 3) -- e.g. 3 samples, 3 features each
per_feature_mean = np.array([2, 4, 6])  # shape (3,)

print("\nbatch shape:", batch.shape, "per_feature_mean shape:", per_feature_mean.shape)
print("batch:\n", batch)
print("batch - per_feature_mean (broadcast across rows):\n", batch - per_feature_mean)
# What actually happens: per_feature_mean is treated as if it were repeated
# for every row, WITHOUT actually copying it in memory. That's the point.

# --- A column vector broadcasting across columns instead ---
per_sample_scale = np.array([[1], [10], [100]])  # shape (3, 1)
print("\nper_sample_scale shape:", per_sample_scale.shape)
print("batch * per_sample_scale (broadcast across columns):\n", batch * per_sample_scale)

# --- Realistic case: normalize an image with per-channel mean/std (like ImageNet stats) ---
# Shape (H, W, C) = (2, 2, 3) -- a tiny 2x2 RGB "image"
image = np.array([
    [[100, 150, 200], [50, 75, 90]],
    [[10, 20, 30], [220, 210, 200]],
], dtype=np.float32)

channel_mean = np.array([100.0, 100.0, 100.0])  # shape (3,) -- one value per channel
channel_std = np.array([50.0, 50.0, 50.0])       # shape (3,)

normalized = (image - channel_mean) / channel_std
print("\nimage shape:", image.shape)
print("channel_mean shape:", channel_mean.shape, "-> broadcasts against last axis (C)")
print("normalized:\n", normalized)

# --- What does NOT broadcast: mismatched shapes with no dimension of size 1 or equal ---
try:
    bad = np.array([1, 2, 3]) + np.array([1, 2])
except ValueError as e:
    print("\nExpected failure -- shapes (3,) and (2,) don't broadcast:")
    print(" ", e)
