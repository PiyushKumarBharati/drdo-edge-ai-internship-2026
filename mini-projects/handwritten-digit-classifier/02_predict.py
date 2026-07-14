"""
Predict on a single digit image using the trained TFLite model.

Usage: python 02_predict.py path/to/digit_image.png
Defaults to one of the real MNIST sample images saved by 01_train_and_convert.py.
"""

import argparse
import glob
import os

import cv2
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "digit_classifier.tflite")


def predict_digit(image_path, model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expected {model_path}. Run 01_train_and_convert.py first.")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    tensor = img.astype("float32")[None, ..., None] / 255.0  # (1, 28, 28, 1)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predicted_digit = int(np.argmax(output))
    confidence = float(output[predicted_digit])
    return predicted_digit, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", nargs="?", default=None,
                         help="Path to a digit image. Defaults to a sample from sample_digits/.")
    args = parser.parse_args()

    if args.image_path is None:
        sample_dir = os.path.join(HERE, "sample_digits")
        candidates = sorted(glob.glob(os.path.join(sample_dir, "*.png")))
        if not candidates:
            raise FileNotFoundError(f"No sample images in {sample_dir}. Run 01_train_and_convert.py first.")
        image_path = candidates[0]
    else:
        image_path = args.image_path

    predicted_digit, confidence = predict_digit(image_path)

    basename = os.path.basename(image_path)
    print(f"Image: {basename}")
    print(f"Predicted digit: {predicted_digit}")
    print(f"Confidence: {confidence:.4f}")

    # If the filename encodes the true label (as 01_train_and_convert.py's samples do), check it.
    if "true" in basename:
        true_label = int(basename.split("true")[1].split(".")[0])
        correct = "CORRECT" if predicted_digit == true_label else "WRONG"
        print(f"True label (from filename): {true_label} -- {correct}")
