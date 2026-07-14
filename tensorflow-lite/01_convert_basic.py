"""
Load the Fashion-MNIST CNN and convert it to a plain float32 .tflite model.

Why this matters: this is the baseline every later quantized model gets
compared against -- same architecture, same weights, just repackaged into
TFLite's flatbuffer format with no precision loss.
"""

import os
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "..", "tensorflow-basics", "fashion_mnist_cnn.keras")
OUT_PATH = os.path.join(HERE, "model_float32.tflite")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Expected {MODEL_PATH}. Run tensorflow-basics/03_cnn_fashion_mnist.py first.")

model = tf.keras.models.load_model(MODEL_PATH)

# --- Convert to TFLite with no optimizations: keeps every weight as float32 ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(OUT_PATH, "wb") as f:
    f.write(tflite_model)

size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"Saved float32 TFLite model to {OUT_PATH}")
print(f"File size: {size_kb:.2f} KB")

# Compare against the original .keras file for context.
original_size_kb = os.path.getsize(MODEL_PATH) / 1024
print(f"Original .keras file size: {original_size_kb:.2f} KB")
print(f"TFLite is {'smaller' if size_kb < original_size_kb else 'larger'} than the .keras file")
print("(TFLite drops optimizer state and Keras-specific metadata that .keras keeps -- ")
print(" even at the same float32 precision, the flatbuffer format alone shrinks the file.)")
