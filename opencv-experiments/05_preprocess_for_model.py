"""
THE important one: image -> model-ready tensor.

This function (preprocess_for_model) is reused as-is in mini-projects/ and
final-project/src/infer.py. It's the bridge between "a file on disk" and
"something a tf.lite.Interpreter will accept."
"""

import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))


def preprocess_for_model(image_path, target_size=(28, 28), grayscale=True):
    """Load an image from disk and turn it into a model-ready float32 tensor.

    Steps, in order (each one matters):
      1. Read with OpenCV (BGR).
      2. Convert to grayscale or RGB depending on what the model expects.
      3. Resize to the model's expected input size.
      4. Normalize pixel values from [0, 255] to [0, 1].
      5. Add a batch dimension so shape becomes (1, H, W, C).

    Returns a numpy array ready to hand to a tf.lite.Interpreter.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV loads BGR; models expect RGB

    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    img = img.astype("float32") / 255.0

    if grayscale:
        img = np.expand_dims(img, axis=-1)  # (H, W) -> (H, W, 1), models expect an explicit channel dim

    img = np.expand_dims(img, axis=0)  # (H, W, C) -> (1, H, W, C), models expect a batch dim

    return img


if __name__ == "__main__":
    sample_path = os.path.join(HERE, "sample.jpg")

    # Grayscale, 28x28 -- matches the Fashion-MNIST CNN from tensorflow-basics/.
    tensor_gray = preprocess_for_model(sample_path, target_size=(28, 28), grayscale=True)
    print("Grayscale 28x28 tensor:")
    print("  shape:", tensor_gray.shape, " dtype:", tensor_gray.dtype)
    print("  min/max:", tensor_gray.min(), tensor_gray.max())

    # RGB, 224x224 -- a more typical size for RGB image classifiers.
    tensor_rgb = preprocess_for_model(sample_path, target_size=(224, 224), grayscale=False)
    print("\nRGB 224x224 tensor:")
    print("  shape:", tensor_rgb.shape, " dtype:", tensor_rgb.dtype)
    print("  min/max:", tensor_rgb.min(), tensor_rgb.max())

    print("\nThis exact function (preprocess_for_model) is imported directly by")
    print("raspberry-pi/deploy_inference.py and final-project/src/infer.py.")
