"""
A small Sequential dense network on MNIST -- first real neural net training run.

Why this matters: this is the "hello world" of neural nets, but the shape of
the code (load -> normalize -> build -> compile -> fit -> evaluate) is
identical to every training script later in this repo, including
final-project/src/train.py.
"""

import os
import importlib.util
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_plot_training_curves():
    """Import plot_training_curves() from matplotlib-practice/04_training_curve_demo.py
    by file path, since folder names with hyphens aren't valid Python package names."""
    path = os.path.join(HERE, "..", "matplotlib-practice", "04_training_curve_demo.py")
    spec = importlib.util.spec_from_file_location("training_curve_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_training_curves


plot_training_curves = _load_plot_training_curves()

print("TensorFlow version:", tf.__version__)

# --- Load MNIST (60,000 train / 10,000 test handwritten digit images, 28x28 grayscale) ---
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(f"train: {x_train.shape}, test: {x_test.shape}")

# --- Normalize pixel values from [0, 255] uint8 to [0, 1] float32 ---
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# --- Build a small dense network ---
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),                          # 28x28 image -> 784-length vector
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),                        # randomly zero 20% of activations during training, reduces overfitting
    tf.keras.layers.Dense(10, activation="softmax"),      # 10 digit classes, probabilities sum to 1
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",  # labels are integers (0-9), not one-hot
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

# --- Save the training curve using the reusable plotting function from Phase 4 ---
out_path = os.path.join(HERE, "02_mnist_training_curve.png")
plot_training_curves(
    history.history, out_path, title=f"Dense net on MNIST (test acc={test_acc:.4f})"
)
print(f"Saved {out_path}")

print("\nFull history (real values from this run):")
for key, values in history.history.items():
    print(f"  {key}: {[round(v, 4) for v in values]}")
