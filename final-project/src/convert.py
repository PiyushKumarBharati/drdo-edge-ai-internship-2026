"""
Convert the trained CNN to three TFLite variants: float32, dynamic-range,
and full-integer INT8 -- same approach as tensorflow-lite/02_convert_quantized.py,
rebuilt here as part of this project's own self-contained pipeline.
"""

import os
import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(HERE, "..")

sys.path.insert(0, HERE)
from data import load_data

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "cnn_classifier.keras")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def convert_all():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Expected {MODEL_PATH}. Run train.py first.")

    model = tf.keras.models.load_model(MODEL_PATH)
    data = load_data()
    x_train = data["x_train"]

    def representative_dataset():
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(len(x_train), size=100, replace=False)
        for i in indices:
            yield [x_train[i:i + 1]]

    os.makedirs(MODELS_DIR, exist_ok=True)
    paths = {}

    # (a) float32 baseline
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    paths["float32"] = os.path.join(MODELS_DIR, "model_float32.tflite")
    with open(paths["float32"], "wb") as f:
        f.write(tflite_model)

    # (b) dynamic range quantized
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    paths["dynamic_range"] = os.path.join(MODELS_DIR, "model_dynamic_range.tflite")
    with open(paths["dynamic_range"], "wb") as f:
        f.write(tflite_model)

    # (c) full integer INT8, calibrated on real training samples
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    paths["int8"] = os.path.join(MODELS_DIR, "model_int8.tflite")
    with open(paths["int8"], "wb") as f:
        f.write(tflite_model)

    print("Converted models (real file sizes):")
    for name, path in paths.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"  {name:15s}: {size_kb:8.2f} KB  ({path})")

    return paths


if __name__ == "__main__":
    convert_all()
