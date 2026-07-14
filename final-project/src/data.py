"""
Data loading and preprocessing pipeline for the final project.

Fashion-MNIST again, deliberately -- it's the same dataset used in
tensorflow-basics/ and tensorflow-lite/, which lets this final project be a
clean, complete rebuild of the full pipeline (train -> convert -> benchmark
-> infer) rather than re-explaining a new dataset from scratch.
"""

import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_data(validation_split=0.1, seed=42):
    """Loads Fashion-MNIST, normalizes to [0,1] float32, adds a channel dim,
    and splits off a validation set from the training data.

    Returns a dict of numpy arrays: x_train, y_train, x_val, y_val, x_test, y_test.
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train_full = x_train_full.astype("float32")[..., None] / 255.0
    x_test = x_test.astype("float32")[..., None] / 255.0

    # Shuffle before splitting so the validation set isn't just "the last N samples"
    # (Fashion-MNIST's classes are stored in contiguous blocks in some versions of this dataset).
    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(len(x_train_full))
    x_train_full = x_train_full[permutation]
    y_train_full = y_train_full[permutation]

    n_val = int(len(x_train_full) * validation_split)
    x_val, y_val = x_train_full[:n_val], y_train_full[:n_val]
    x_train, y_train = x_train_full[n_val:], y_train_full[n_val:]

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
    }


if __name__ == "__main__":
    data = load_data()
    for key, arr in data.items():
        print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
    print("\nClass names:", CLASS_NAMES)
    print("Pixel value range in x_train:", data["x_train"].min(), "-", data["x_train"].max())
