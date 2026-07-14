"""
End-to-end inference: image path in -> prediction out, via a TFLite interpreter.

Same preprocessing contract as opencv-experiments/05_preprocess_for_model.py
and raspberry-pi/deploy_inference.py -- read -> grayscale -> resize -> normalize
-> batch -> (quantize if the model is int8) -> invoke -> (dequantize) -> argmax.
"""

import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(HERE, "..")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_int8.tflite")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def preprocess_image(image_path, target_size=(28, 28)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img


def predict(image_path, model_path=DEFAULT_MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Expected {model_path}. Run convert.py first.")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    tensor = preprocess_image(image_path)

    is_int8_io = input_details["dtype"] == np.int8
    if is_int8_io:
        scale, zero_point = input_details["quantization"]
        tensor = (tensor / scale + zero_point).astype(np.int8)

    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    if is_int8_io:
        out_scale, out_zero_point = output_details["quantization"]
        output = (output.astype(np.float32) - out_zero_point) * out_scale

    predicted_index = int(np.argmax(output))
    confidence = float(output[predicted_index])
    return CLASS_NAMES[predicted_index], confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a clothing image with the final TFLite model.")
    parser.add_argument("image_path", help="Path to an image to classify.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    label, confidence = predict(args.image_path, args.model)
    print(f"Image: {args.image_path}")
    print(f"Model: {args.model}")
    print(f"Predicted class: {label}")
    print(f"Confidence: {confidence:.4f}")
