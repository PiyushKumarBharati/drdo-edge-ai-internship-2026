"""
Training loop: loads data, builds the model, trains, saves the model and a
training curve plot into results/.
"""

import os
import sys
import importlib.util

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(HERE, "..")
REPO_ROOT = os.path.join(HERE, "..", "..")

sys.path.insert(0, HERE)
from data import load_data, CLASS_NAMES
from model import build_model


def _load_plot_training_curves():
    path = os.path.join(REPO_ROOT, "matplotlib-practice", "04_training_curve_demo.py")
    spec = importlib.util.spec_from_file_location("training_curve_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_training_curves


plot_training_curves = _load_plot_training_curves()

EPOCHS = 8
BATCH_SIZE = 128


def train():
    print("TensorFlow version:", tf.__version__)

    data = load_data()
    print(f"train: {data['x_train'].shape}, val: {data['x_val'].shape}, test: {data['x_test'].shape}")

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # EarlyStopping with restore_best_weights was added after a first real run of this
    # script showed validation accuracy was noisy epoch to epoch (it dropped to 0.35 on
    # epoch 1, then bounced between 0.84-0.90, and the LAST epoch was not the best one --
    # without this callback, training just keeps whatever the final epoch happened to
    # produce, which is not necessarily the best model seen during training).
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=4, restore_best_weights=True
    )

    history = model.fit(
        data["x_train"], data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(data["x_test"], data["y_test"], verbose=0)
    print(f"\nFinal test accuracy: {test_acc:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

    model_path = os.path.join(PROJECT_ROOT, "models", "cnn_classifier.keras")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Saved model to {model_path}")

    curve_path = os.path.join(PROJECT_ROOT, "results", "training_curve.png")
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    plot_training_curves(history.history, curve_path, title=f"Final CNN on Fashion-MNIST (test acc={test_acc:.4f})")
    print(f"Saved {curve_path}")

    print("\nFull history (real values from this run):")
    for key, values in history.history.items():
        print(f"  {key}: {[round(v, 4) for v in values]}")

    return model, history, test_acc, test_loss


if __name__ == "__main__":
    train()
