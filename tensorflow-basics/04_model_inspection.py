"""
model.summary(), parameter counting, and estimating model size in MB.

Why this matters: parameter count is THE number that predicts whether a model
will fit and run acceptably on a Raspberry Pi. This script makes the
parameter-count -> disk-size -> RAM-footprint chain concrete and measured, not
just a summary table nobody reads.
"""

import os
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "fashion_mnist_cnn.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Expected {MODEL_PATH}. Run 03_cnn_fashion_mnist.py first.")

model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# --- Count parameters per layer, by hand, to show where they actually come from ---
print("\nPer-layer parameter breakdown:")
total_params = 0
for layer in model.layers:
    n_params = layer.count_params()
    total_params += n_params
    print(f"  {layer.name:15s} ({layer.__class__.__name__:15s}): {n_params:>8,} params")

print(f"\nTotal parameters (sum of layers): {total_params:,}")
print(f"Total parameters (model.count_params()): {model.count_params():,}")
assert total_params == model.count_params()

# --- Estimate size in MB assuming float32 storage (4 bytes per parameter) ---
bytes_per_param_f32 = 4
estimated_mb_f32 = (total_params * bytes_per_param_f32) / (1024 ** 2)
bytes_per_param_int8 = 1
estimated_mb_int8 = (total_params * bytes_per_param_int8) / (1024 ** 2)

print(f"\nEstimated size at float32 (4 bytes/param): {estimated_mb_f32:.3f} MB")
print(f"Estimated size at int8 (1 byte/param):      {estimated_mb_int8:.3f} MB")

# --- Compare estimate to the ACTUAL file size on disk ---
actual_size_mb = os.path.getsize(MODEL_PATH) / (1024 ** 2)
print(f"\nActual .keras file size on disk: {actual_size_mb:.3f} MB")
print("(The .keras file is larger than the raw float32 weight estimate because")
print(" it also stores architecture, optimizer state, and metadata -- the raw")
print(" weight count is what actually matters for the .tflite conversion in")
print(" tensorflow-lite/, which strips all of that away.)")

# --- Where do the parameters concentrate? ---
layer_param_pairs = [(layer.name, layer.count_params()) for layer in model.layers]
layer_param_pairs.sort(key=lambda p: p[1], reverse=True)
biggest_layer, biggest_count = layer_param_pairs[0]
print(f"\nLargest layer by parameter count: '{biggest_layer}' with {biggest_count:,} "
      f"params ({100 * biggest_count / total_params:.1f}% of the model)")
