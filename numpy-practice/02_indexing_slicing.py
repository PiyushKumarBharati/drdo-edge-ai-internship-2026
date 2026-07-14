"""
Indexing, slicing, boolean masks, fancy indexing.

Why this matters: cropping a region of an image, dropping low-confidence
predictions, and selecting specific channels are all just indexing operations
on arrays -- this is the vocabulary the rest of the repo assumes.
"""

import numpy as np

# --- Basic slicing on a 1D array ---
a = np.arange(10)
print("a:", a)
print("a[2:5]:", a[2:5])
print("a[::2] (every other element):", a[::2])
print("a[::-1] (reversed):", a[::-1])

# --- 2D slicing, e.g. a tiny "grayscale image" ---
img = np.arange(25).reshape(5, 5)
print("\nimg:\n", img)
print("top-left 2x2 corner (img[:2, :2]):\n", img[:2, :2])
print("middle row (img[2, :]):", img[2, :])
print("middle column (img[:, 2]):", img[:, 2])
print("center crop (img[1:4, 1:4]):\n", img[1:4, 1:4])

# --- Boolean masks: the mechanism behind "keep only confident predictions" ---
confidences = np.array([0.95, 0.42, 0.88, 0.15, 0.67, 0.99])
mask = confidences > 0.6
print("\nconfidences:", confidences)
print("mask (confidences > 0.6):", mask)
print("confident predictions only:", confidences[mask])
print("count above threshold:", np.sum(mask))  # True counts as 1

# Boolean masking on a 2D array: zero out low-value pixels (simple thresholding).
noisy_img = np.array([
    [10, 200, 30],
    [180, 20, 210],
    [40, 190, 15],
])
thresholded = noisy_img.copy()
thresholded[thresholded < 100] = 0
print("\noriginal:\n", noisy_img)
print("thresholded (<100 -> 0):\n", thresholded)

# --- Fancy indexing: select specific rows/indices with a list/array of indices ---
class_ids = np.array([0, 1, 2, 3, 4])
class_names = np.array(["cat", "dog", "bird", "fish", "horse"])
selected = [3, 0, 4]  # e.g. the top-3 predicted class indices for one sample
print("\nselecting classes at indices", selected, "->", class_names[selected])

# Fancy indexing also works for picking specific rows out of a batch.
batch = np.arange(20).reshape(4, 5)  # 4 samples, 5 features each
rows_wanted = [0, 2]
print("\nbatch:\n", batch)
print(f"rows {rows_wanted} only:\n", batch[rows_wanted])
