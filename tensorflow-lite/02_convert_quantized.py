"""
Produce three TFLite models from the same trained Fashion-MNIST CNN:
  (a) float32 baseline        -- no quantization
  (b) dynamic range quantized -- weights quantized to int8, activations stay float32
  (c) full integer INT8       -- weights AND activations quantized to int8

Why this matters: this is the actual size/speed/accuracy lever for edge
deployment. (b) is a one-line change with no data needed. (c) needs a
representative dataset -- real samples the converter uses to measure the
actual range of activations at each layer, so it can pick a good int8 scale
instead of guessing.
"""

import os
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "tensorflow-basics", "fashion_mnist_cnn.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Expected {MODEL_PATH}. Run tensorflow-basics/03_cnn_fashion_mnist.py first.")

model = tf.keras.models.load_model(MODEL_PATH)

# --- Load real training images to build a representative dataset for INT8 calibration ---
(x_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32")[..., None] / 255.0


def representative_dataset():
    """Yields real training samples (not random noise) so the converter can
    measure the true activation range at each layer before picking int8
    scale/zero-point values. Using only 100 samples keeps calibration fast
    while still covering the input distribution reasonably well."""
    rng = np.random.default_rng(seed=0)
    indices = rng.choice(len(x_train), size=100, replace=False)
    for i in indices:
        # Converter expects a list of input tensors, batch size 1, per call.
        yield [x_train[i:i + 1]]


# --- (a) float32 baseline (same as 01_convert_basic.py, saved again here for a self-contained script) ---
converter_f32 = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_f32 = converter_f32.convert()
path_f32 = os.path.join(HERE, "model_float32.tflite")
with open(path_f32, "wb") as f:
    f.write(tflite_f32)

# --- (b) dynamic range quantization: just set the default optimization flag ---
converter_dynamic = tf.lite.TFLiteConverter.from_keras_model(model)
converter_dynamic.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_dynamic = converter_dynamic.convert()
path_dynamic = os.path.join(HERE, "model_dynamic_range.tflite")
with open(path_dynamic, "wb") as f:
    f.write(tflite_dynamic)

# --- (c) full integer INT8: needs the representative dataset to calibrate activation ranges ---
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_dataset
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8
tflite_int8 = converter_int8.convert()
path_int8 = os.path.join(HERE, "model_int8.tflite")
with open(path_int8, "wb") as f:
    f.write(tflite_int8)

# --- Report real file sizes ---
print("Model file sizes (real, measured from disk):")
for name, path in [("float32", path_f32), ("dynamic range", path_dynamic), ("int8", path_int8)]:
    size_kb = os.path.getsize(path) / 1024
    print(f"  {name:15s}: {size_kb:8.2f} KB  ({path})")

size_f32 = os.path.getsize(path_f32)
size_dynamic = os.path.getsize(path_dynamic)
size_int8 = os.path.getsize(path_int8)
print(f"\nDynamic range is {size_f32 / size_dynamic:.2f}x smaller than float32")
print(f"INT8 is {size_f32 / size_int8:.2f}x smaller than float32")
