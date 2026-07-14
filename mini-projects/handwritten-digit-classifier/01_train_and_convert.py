"""
Handwritten digit classifier: train a small CNN on MNIST, convert to TFLite.

This is a complete, small, standalone project -- it doesn't import from
tensorflow-basics/, it re-derives the same idea end to end so this folder
works if copied out of the repo on its own.
"""

import os
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))

print("TensorFlow version:", tf.__version__)

# --- Load MNIST ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32")[..., None] / 255.0
x_test = x_test.astype("float32")[..., None] / 255.0
print(f"train: {x_train.shape}, test: {x_test.shape}")

# --- A small CNN (fewer filters than tensorflow-basics' Fashion-MNIST CNN --
# MNIST digits are an easier task and don't need as much capacity) ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(8, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, validation_split=0.1, epochs=4, batch_size=128, verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal test accuracy: {test_acc:.4f}")

model_path = os.path.join(HERE, "digit_classifier.keras")
model.save(model_path)
print(f"Saved Keras model to {model_path}")

# --- Convert to TFLite (dynamic-range quantized -- small and simple for a mini-project) ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = os.path.join(HERE, "digit_classifier.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

keras_size_kb = os.path.getsize(model_path) / 1024
tflite_size_kb = os.path.getsize(tflite_path) / 1024
print(f"\nSaved TFLite model to {tflite_path}")
print(f".keras size: {keras_size_kb:.2f} KB, .tflite size: {tflite_size_kb:.2f} KB "
      f"({keras_size_kb / tflite_size_kb:.2f}x smaller)")

# --- Save a handful of real test images to disk, to use in 02_predict.py ---
sample_dir = os.path.join(HERE, "sample_digits")
os.makedirs(sample_dir, exist_ok=True)
import cv2
rng = np.random.default_rng(seed=1)
sample_indices = rng.choice(len(x_test), size=5, replace=False)
for idx in sample_indices:
    img_uint8 = (x_test[idx, :, :, 0] * 255).astype(np.uint8)
    true_label = int(y_test[idx])
    out_path = os.path.join(sample_dir, f"digit_{idx}_true{true_label}.png")
    cv2.imwrite(out_path, img_uint8)
print(f"\nSaved {len(sample_indices)} real MNIST test images to {sample_dir}/ for 02_predict.py")
