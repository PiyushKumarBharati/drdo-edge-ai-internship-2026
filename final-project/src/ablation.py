"""
Tests the hypothesis stated in README.md/final-report.md: that
BatchNormalization is the likely cause of the final CNN's INT8 accuracy
drop. Instead of just asserting this, this script trains the SAME
architecture with BatchNormalization layers removed (model.py's
build_model(batch_norm=False)), converts it to INT8 the same way
(convert.py), benchmarks it the same way (benchmark.py), and reports
whether the INT8 accuracy drop actually shrinks.

Also generates a per-class comparison (float32 vs int8, precision/recall/F1
and confusion matrices) for both the with-BatchNorm and without-BatchNorm
models, to show where accuracy is actually lost, rather than only reporting
one aggregate accuracy number.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(HERE, "..")

sys.path.insert(0, HERE)
from data import load_data, CLASS_NAMES
from train import train
from convert import convert_all
from benchmark import benchmark_variant, save_benchmark_outputs, save_classification_reports, save_confusion_matrix

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "ablation_no_batchnorm")


def run_ablation():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Training the same architecture with BatchNormalization removed ===")
    model, history, test_acc, test_loss = train(
        batch_norm=False,
        model_filename="cnn_classifier_no_bn.keras",
        curve_filename="training_curve_no_bn.png",
        title_prefix="No-BatchNorm CNN",
    )
    print(f"No-BatchNorm float32 test accuracy: {test_acc:.4f}")

    print("\n=== Converting to TFLite (float32 / dynamic_range / int8) ===")
    model_path = os.path.join(MODELS_DIR, "cnn_classifier_no_bn.keras")
    paths = convert_all(model_path=model_path, output_prefix="model_no_bn")

    print("\n=== Benchmarking (same methodology as benchmark.py) ===")
    data = load_data()
    x_test, y_test = data["x_test"], data["y_test"]

    results = {}
    for name, path in paths.items():
        results[name] = benchmark_variant(f"no_bn_{name}", path, x_test, y_test)

    save_benchmark_outputs(
        {f"no_bn_{k}": v for k, v in results.items()},
        results_dir=RESULTS_DIR,
        chart_title="No-BatchNorm CNN: TFLite variant comparison (real, measured values)",
    )
    save_classification_reports({f"no_bn_{k}": v for k, v in results.items()}, y_test, results_dir=RESULTS_DIR)
    save_confusion_matrix("no_bn_float32", {f"no_bn_{k}": v for k, v in results.items()}, y_test,
                           results_dir=RESULTS_DIR, filename="confusion_matrix_float32.png")
    save_confusion_matrix("no_bn_int8", {f"no_bn_{k}": v for k, v in results.items()}, y_test,
                           results_dir=RESULTS_DIR, filename="confusion_matrix_int8.png")

    drop = results["float32"]["accuracy"] - results["int8"]["accuracy"]
    print(f"\nNo-BatchNorm INT8 accuracy drop: {drop * 100:.2f} percentage points "
          f"({results['float32']['accuracy']:.4f} -> {results['int8']['accuracy']:.4f})")

    # --- Per-class accuracy: float32 vs int8, to see where accuracy is lost ---
    f32_preds = results["float32"]["predictions"]
    int8_preds = results["int8"]["predictions"]
    per_class_path = os.path.join(RESULTS_DIR, "per_class_float32_vs_int8.csv")
    with open(per_class_path, "w") as f:
        f.write("class,float32_accuracy,int8_accuracy,delta\n")
        print("\nPer-class accuracy, no-BatchNorm model (float32 vs int8):")
        for class_idx, class_name in enumerate(CLASS_NAMES):
            mask = y_test == class_idx
            f32_acc = float(np.mean(f32_preds[mask] == y_test[mask]))
            int8_acc = float(np.mean(int8_preds[mask] == y_test[mask]))
            delta = int8_acc - f32_acc
            f.write(f"{class_name},{f32_acc:.4f},{int8_acc:.4f},{delta:+.4f}\n")
            print(f"  {class_name:15s}: float32={f32_acc:.4f}  int8={int8_acc:.4f}  delta={delta:+.4f}")
    print(f"Saved {per_class_path}")

    return {"drop_pp": drop * 100, "results": results}


if __name__ == "__main__":
    run_ablation()
