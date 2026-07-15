"""
Quantization-aware training (QAT) of the SAME final CNN architecture
(model.py's build_model()), using tensorflow-model-optimization (tfmot).

Goal: measure whether QAT recovers the INT8 accuracy drop seen with plain
post-training quantization (convert.py + benchmark.py), using the exact same
benchmark methodology (benchmark.py's benchmark_variant()/run_benchmark()).

Why this needs the legacy Keras compat layer: tfmot 0.8.1's
quantize_model() imports tf.keras in a way that requires the standalone
tf_keras package under TensorFlow 2.21 (which defaults to Keras 3) --
verified directly on this machine: without TF_USE_LEGACY_KERAS=1 and
tf_keras installed, `import tensorflow_model_optimization` raises
`ImportError: Keras cannot be imported. Check that it is installed.`
Setting the env var here (before any tensorflow import) routes tf.keras to
tf_keras for this process only; train.py/convert.py/benchmark.py are
unaffected since they run as separate processes.

Also verified directly: tfmot's quantize_model() fails on this architecture
with `RuntimeError: Layer batch_normalization:<...BatchNormalization> is not
supported` when Conv2D's activation is fused (Conv2D(activation="relu")).
model.py already builds Conv2D -> BatchNormalization -> Activation as
separate layers specifically so quantize_model() recognizes the standard
Conv-BN-ReLU fusion pattern -- this is the same architecture used by
train.py, not a different one built for QAT.

Because QAT training happens in this separate legacy-Keras process, its
float32 starting point is its own training run (same seed, same
architecture, same hyperparameters as train.py) rather than a re-load of
models/cnn_classifier.keras -- tf_keras cannot load a .keras file saved by
Keras 3. Its baseline accuracy is expected to be close to, but not
identical to, train.py's, for the same reason two runs of train.py with
different seeds would differ.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import sys

import numpy as np
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(HERE, "..")

sys.path.insert(0, HERE)
from data import load_data
from model import build_model
from train import set_seeds, BATCH_SIZE
from benchmark import run_benchmark

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
QAT_FINE_TUNE_EPOCHS = 5
BASELINE_EPOCHS = 40
BASELINE_PATIENCE = 8


def try_import_tfmot():
    try:
        import tensorflow_model_optimization as tfmot
        return tfmot, None
    except Exception as e:
        return None, e


def train_baseline(data):
    """Trains the same architecture/hyperparameters as train.py, inside this
    process's legacy-Keras environment (needed so tfmot can wrap it)."""
    set_seeds()
    model = build_model(batch_norm=True)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=BASELINE_PATIENCE, restore_best_weights=True
    )
    model.fit(
        data["x_train"], data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=BASELINE_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=2,
    )
    return model


def run_qat():
    tfmot, import_error = try_import_tfmot()
    if tfmot is None:
        print("tensorflow-model-optimization is not usable in this environment.")
        print(f"Import error: {type(import_error).__name__}: {import_error}")
        print("QAT skipped -- see reports/final-report.md Future Work for details.")
        return None

    print("TensorFlow version:", tf.__version__)
    print("Keras backend:", tf.keras.__name__ if hasattr(tf.keras, "__name__") else type(tf.keras))

    data = load_data()

    print("\n=== Training float32 baseline (same architecture/seed/hyperparameters as train.py) ===")
    baseline = train_baseline(data)
    base_loss, base_acc = baseline.evaluate(data["x_test"], data["y_test"], verbose=0)
    print(f"Baseline float32 test accuracy (this QAT process's own run): {base_acc:.4f}")

    print("\n=== Applying tfmot.quantization.keras.quantize_model() ===")
    q_aware_model = tfmot.quantization.keras.quantize_model(baseline)
    q_aware_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n=== Fine-tuning quantization-aware model for {QAT_FINE_TUNE_EPOCHS} epochs ===")
    q_aware_model.fit(
        data["x_train"], data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=QAT_FINE_TUNE_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    qat_loss, qat_acc = q_aware_model.evaluate(data["x_test"], data["y_test"], verbose=0)
    print(f"\nQAT-aware model test accuracy (still float32 graph, fake-quant nodes active): {qat_acc:.4f}")

    print("\n=== Converting QAT model to real INT8 TFLite ===")
    x_train = data["x_train"]

    def representative_dataset():
        rng = np.random.default_rng(seed=0)
        indices = rng.choice(len(x_train), size=100, replace=False)
        for i in indices:
            yield [x_train[i:i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    os.makedirs(MODELS_DIR, exist_ok=True)
    qat_path = os.path.join(MODELS_DIR, "model_qat_int8.tflite")
    with open(qat_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {qat_path} ({os.path.getsize(qat_path) / 1024:.2f} KB)")

    print("\n=== Benchmarking qat_int8 alongside the main pipeline's variants (same methodology) ===")
    results = run_benchmark(extra_models={"qat_int8": qat_path})

    return {
        "baseline_float32_acc": base_acc,
        "qat_aware_float_acc": qat_acc,
        "results": results,
    }


if __name__ == "__main__":
    run_qat()
