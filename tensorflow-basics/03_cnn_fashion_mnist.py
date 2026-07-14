"""
A small CNN on Fashion-MNIST. This exact trained model is what gets converted
to TFLite in tensorflow-lite/ and reused in mini-projects/handwritten-digit-classifier/
style pipelines.

Why a CNN instead of the dense net from 02: convolution layers share weights
across spatial positions, so they need far fewer parameters to recognize a
shape (a sleeve, a shoe outline) regardless of where it appears in the image
-- a big deal when "fewer parameters" directly means "smaller model file."
"""

import os
import importlib.util
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_plot_training_curves():
    path = os.path.join(HERE, "..", "matplotlib-practice", "04_training_curve_demo.py")
    spec = importlib.util.spec_from_file_location("training_curve_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_training_curves


plot_training_curves = _load_plot_training_curves()

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

print("TensorFlow version:", tf.__version__)

# --- Load Fashion-MNIST: 60,000 train / 10,000 test, 28x28 grayscale clothing images ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(f"train: {x_train.shape}, test: {x_test.shape}")

# Normalize to [0,1] and add a channel dimension: CNNs expect (H, W, C) even for grayscale.
x_train = x_train.astype("float32")[..., None] / 255.0
x_test = x_test.astype("float32")[..., None] / 255.0
print(f"after adding channel dim: train={x_train.shape}")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),      # 16 learned 3x3 filters
    tf.keras.layers.MaxPooling2D(2),                        # downsample 2x -- keeps strongest signal, halves resolution
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

EPOCHS = 5
history = model.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=128,
    verbose=2,
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal test accuracy: {test_acc:.4f}")
print(f"Final test loss: {test_loss:.4f}")

# --- Save the trained model -- this exact file gets converted to TFLite next ---
model_path = os.path.join(HERE, "fashion_mnist_cnn.keras")
model.save(model_path)
print(f"Saved model to {model_path}")

# --- Save training curve ---
curve_path = os.path.join(HERE, "03_fashion_mnist_training_curve.png")
plot_training_curves(history.history, curve_path, title=f"CNN on Fashion-MNIST (test acc={test_acc:.4f})")
print(f"Saved {curve_path}")

print("\nFull history (real values from this run):")
for key, values in history.history.items():
    print(f"  {key}: {[round(v, 4) for v in values]}")
