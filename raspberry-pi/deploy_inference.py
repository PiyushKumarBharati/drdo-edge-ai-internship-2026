"""
Single-image inference using tflite_runtime (falls back to tf.lite if
tflite_runtime isn't installed) -- the deployment-target version of
tensorflow-lite/03_tflite_inference.py.

Why tflite_runtime: it's a small, Pi-friendly package with none of full
TensorFlow's weight -- exactly what you'd actually pip-install on a Pi (see
setup_pi.md). This script runs on either tflite_runtime (Pi) or full
TensorFlow's tf.lite (this development laptop) without code changes, so it
was actually tested here before ever touching a Pi.
"""

import argparse
import os
import sys
import time

import numpy as np
import cv2

# --- Import the Interpreter from whichever package is available ---
# On a Raspberry Pi: `pip install tflite-runtime` (see setup_pi.md), a lightweight
# package with just the interpreter, no full TensorFlow install needed.
# On this development machine: full `tensorflow` is already installed, so we fall
# back to tf.lite.Interpreter, which has an identical API for our purposes.
try:
    from tflite_runtime.interpreter import Interpreter
    RUNTIME = "tflite_runtime"
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    RUNTIME = "tensorflow.lite (tflite_runtime not installed on this machine)"

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(HERE, "..", "tensorflow-lite", "model_int8.tflite")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def preprocess_image(image_path, target_size=(28, 28)):
    """Same preprocessing steps as opencv-experiments/05_preprocess_for_model.py:
    read -> grayscale -> resize -> normalize -> add channel + batch dims."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # (H, W) -> (H, W, 1)
    img = np.expand_dims(img, axis=0)   # (H, W, 1) -> (1, H, W, 1)
    return img


def run_inference(model_path, image_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    image_tensor = preprocess_image(image_path)

    is_int8_io = input_details["dtype"] == np.int8
    if is_int8_io:
        scale, zero_point = input_details["quantization"]
        image_tensor = (image_tensor / scale + zero_point).astype(np.int8)

    start = time.perf_counter()
    interpreter.set_tensor(input_details["index"], image_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    latency_ms = (time.perf_counter() - start) * 1000

    if is_int8_io:
        out_scale, out_zero_point = output_details["quantization"]
        output = (output.astype(np.float32) - out_zero_point) * out_scale

    predicted_index = int(np.argmax(output))
    confidence = float(output[predicted_index])

    return predicted_index, confidence, latency_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TFLite inference on a single image.")
    parser.add_argument("image_path", help="Path to an image file to classify.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to a .tflite model.")
    args = parser.parse_args()

    print(f"Interpreter backend: {RUNTIME}")
    print(f"Model: {args.model}")

    if not os.path.exists(args.model):
        print(f"ERROR: model not found at {args.model}. Run tensorflow-lite/02_convert_quantized.py first.")
        sys.exit(1)

    predicted_index, confidence, latency_ms = run_inference(args.model, args.image_path)
    predicted_label = CLASS_NAMES[predicted_index] if predicted_index < len(CLASS_NAMES) else str(predicted_index)

    print(f"\nImage: {args.image_path}")
    print(f"Predicted class: {predicted_label} (index {predicted_index})")
    print(f"Confidence: {confidence:.4f}")
    print(f"Inference latency: {latency_ms:.4f} ms")
