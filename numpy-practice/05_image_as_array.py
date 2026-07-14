"""
An image IS a NumPy array. Load one, inspect it, normalize it, batch it.

Why this matters: this is the exact shape every model input takes: HWC layout
(Height, Width, Channels), float values in [0, 1], with an extra leading
batch dimension. Every preprocessing function in opencv-experiments/ and
final-project/ builds on this.
"""

import os
import numpy as np
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
# Shared sample image committed under opencv-experiments/, reused across folders.
SAMPLE_IMAGE_PATH = os.path.join(HERE, "..", "opencv-experiments", "sample.jpg")

if not os.path.exists(SAMPLE_IMAGE_PATH):
    raise FileNotFoundError(
        f"Expected sample image at {SAMPLE_IMAGE_PATH}. "
        "Run opencv-experiments/generate_sample_image.py first."
    )

# --- Load the image. matplotlib reads it straight into a NumPy array. ---
img = mpimg.imread(SAMPLE_IMAGE_PATH)
print("Loaded:", SAMPLE_IMAGE_PATH)
print("shape:", img.shape, "  (Height, Width, Channels)")
print("dtype:", img.dtype)
print("min/max pixel values:", img.min(), img.max())

# JPEG reads as uint8 in [0, 255]. Confirm that assumption before trusting it.
assert img.dtype == np.uint8, "expected uint8 image straight from JPEG decode"

# --- Normalize to [0, 1] float32 -- the range every Keras/TFLite model expects ---
normalized = img.astype(np.float32) / 255.0
print("\nAfter normalization:")
print("dtype:", normalized.dtype)
print("min/max pixel values:", normalized.min(), normalized.max())

# --- HWC layout explained concretely ---
height, width, channels = img.shape
print(f"\nHWC layout: height={height}, width={width}, channels={channels}")
print("img[0, 0] (top-left pixel, RGB):", img[0, 0])
print("img[:, :, 0] is the full RED channel, shape:", img[:, :, 0].shape)

# --- Add a batch dimension: models expect (batch, H, W, C), not just (H, W, C) ---
batched = np.expand_dims(normalized, axis=0)
print(f"\nSingle image shape:        {normalized.shape}")
print(f"Batched (batch size 1) shape: {batched.shape}")

# A "batch" of more than one image is just stacking several of these along axis 0.
fake_batch = np.stack([normalized, normalized, normalized], axis=0)
print(f"Fake batch of 3 identical images, shape: {fake_batch.shape}")

print("\nThis is exactly the tensor shape a tf.lite.Interpreter expects as input")
print("-- see opencv-experiments/05_preprocess_for_model.py and final-project/src/infer.py")
