"""
Load a .tflite model with tf.lite.Interpreter and run inference on the real
Fashion-MNIST test set, reporting real accuracy.

Why this matters: a .tflite file isn't useful until something can actually
run it and get correct answers back. This script is also the input/output
handling later reused (in spirit) by raspberry-pi/deploy_inference.py --
including the int8 quantized model's extra scale/zero-point step, which is
the part people most often get wrong.
"""

import os
import sys
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_test_data():
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = x_test.astype("float32")[..., None] / 255.0
    return x_test, y_test


def run_tflite_inference(tflite_path, x_test, y_test, n_samples=None):
    """Runs every test sample through the given .tflite model and returns accuracy.

    Handles both float32/float32 models (dynamic-range, plain float32) and
    fully int8-quantized models (which need input quantized to int8 before
    the interpreter will accept it, and output dequantized back to compare).
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    is_int8_io = input_details["dtype"] == np.int8

    if n_samples is None:
        n_samples = len(x_test)

    correct = 0
    for i in range(n_samples):
        sample = x_test[i:i + 1]  # keep batch dim, shape (1, 28, 28, 1)

        if is_int8_io:
            # The converter picked a (scale, zero_point) during calibration in 02_convert_quantized.py.
            # Real int8 value = round(real_float_value / scale) + zero_point.
            scale, zero_point = input_details["quantization"]
            sample = (sample / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        if is_int8_io:
            # Dequantize output back to a comparable scale before taking argmax
            # (argmax is the same either way here, but this is the correct general pattern).
            out_scale, out_zero_point = output_details["quantization"]
            output = (output.astype(np.float32) - out_zero_point) * out_scale

        predicted_class = int(np.argmax(output))
        if predicted_class == y_test[i]:
            correct += 1

    accuracy = correct / n_samples
    return accuracy


if __name__ == "__main__":
    x_test, y_test = load_test_data()
    print(f"Test set: {len(x_test)} samples")

    # Use the full test set -- Fashion-MNIST's 10,000 test images run quickly enough on CPU.
    models = {
        "float32": os.path.join(HERE, "model_float32.tflite"),
        "dynamic_range": os.path.join(HERE, "model_dynamic_range.tflite"),
        "int8": os.path.join(HERE, "model_int8.tflite"),
    }

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found. Run 02_convert_quantized.py first.")
            continue
        acc = run_tflite_inference(path, x_test, y_test)
        print(f"{name:15s}: test accuracy = {acc:.4f}  ({os.path.basename(path)})")
