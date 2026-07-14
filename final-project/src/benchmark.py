"""
Benchmark all three TFLite variants: real file size, real mean latency
(warmup discarded), real test accuracy. Saves a markdown table, a CSV, and
a comparison chart into results/.
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(HERE, "..")

sys.path.insert(0, HERE)
from data import load_data, CLASS_NAMES

MODEL_PATHS = {
    "float32": os.path.join(PROJECT_ROOT, "models", "model_float32.tflite"),
    "dynamic_range": os.path.join(PROJECT_ROOT, "models", "model_dynamic_range.tflite"),
    "int8": os.path.join(PROJECT_ROOT, "models", "model_int8.tflite"),
}

N_LATENCY_RUNS = 200
N_WARMUP_RUNS = 20


def get_predictions(interpreter, input_details, output_details, x_test):
    """Same forward pass as measure_accuracy, but returns every prediction
    (not just a correct/incorrect count) for building a confusion matrix."""
    is_int8_io = input_details["dtype"] == np.int8
    predictions = np.zeros(len(x_test), dtype=int)
    for i in range(len(x_test)):
        sample = x_test[i:i + 1]
        if is_int8_io:
            scale, zero_point = input_details["quantization"]
            sample = (sample / scale + zero_point).astype(np.int8)
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        if is_int8_io:
            out_scale, out_zero_point = output_details["quantization"]
            output = (output.astype(np.float32) - out_zero_point) * out_scale
        predictions[i] = int(np.argmax(output))
    return predictions


def measure_latency(interpreter, input_details, x_test):
    is_int8_io = input_details["dtype"] == np.int8
    sample = x_test[0:1]
    if is_int8_io:
        scale, zero_point = input_details["quantization"]
        sample = (sample / scale + zero_point).astype(np.int8)

    for _ in range(N_WARMUP_RUNS):
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()

    times_ms = []
    for _ in range(N_LATENCY_RUNS):
        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        times_ms.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times_ms))


def run_benchmark():
    data = load_data()
    x_test, y_test = data["x_test"], data["y_test"]
    print(f"Test set: {len(x_test)} samples. Latency: {N_LATENCY_RUNS} timed runs "
          f"after {N_WARMUP_RUNS} discarded warmup runs.\n")

    results = {}
    predictions_by_model = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: not found. Run convert.py first.")
            continue

        size_kb = os.path.getsize(path) / 1024
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        predictions = get_predictions(interpreter, input_details, output_details, x_test)
        accuracy = float(np.mean(predictions == y_test))
        latency_ms = measure_latency(interpreter, input_details, x_test)

        results[name] = {"size_kb": size_kb, "accuracy": accuracy, "latency_ms": latency_ms}
        predictions_by_model[name] = predictions
        print(f"{name:15s}: size={size_kb:8.2f} KB  accuracy={accuracy:.4f}  latency={latency_ms:.4f} ms/image")

    # --- Save markdown table ---
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    md_lines = ["| Model | Size (KB) | Accuracy | Mean latency (ms/image) |", "|---|---|---|---|"]
    for name, r in results.items():
        md_lines.append(f"| {name} | {r['size_kb']:.2f} | {r['accuracy']:.4f} | {r['latency_ms']:.4f} |")
    md_path = os.path.join(results_dir, "benchmark_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nSaved {md_path}")

    # --- Save CSV ---
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    with open(csv_path, "w") as f:
        f.write("model,size_kb,accuracy,latency_ms\n")
        for name, r in results.items():
            f.write(f"{name},{r['size_kb']:.4f},{r['accuracy']:.4f},{r['latency_ms']:.4f}\n")
    print(f"Saved {csv_path}")

    # --- Save comparison chart ---
    names = list(results.keys())
    sizes = [results[n]["size_kb"] for n in names]
    latencies = [results[n]["latency_ms"] for n in names]
    accuracies = [results[n]["accuracy"] * 100 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    axes[0].bar(names, sizes, color=colors)
    axes[0].set_title("Model size (KB)")
    axes[0].set_ylabel("KB")

    axes[1].bar(names, latencies, color=colors)
    axes[1].set_title("Mean inference latency (ms/image)")
    axes[1].set_ylabel("ms")

    axes[2].bar(names, accuracies, color=colors)
    axes[2].set_title("Test accuracy (%)")
    axes[2].set_ylabel("%")
    axes[2].set_ylim(min(accuracies) - 2, 100)

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Final CNN: TFLite variant comparison (real, measured values)")
    fig.tight_layout()
    chart_path = os.path.join(results_dir, "comparison_chart.png")
    fig.savefig(chart_path, dpi=120)
    print(f"Saved {chart_path}")

    # --- Confusion matrix for the float32 model (the reference/full-precision result) ---
    if "float32" in predictions_by_model:
        cm = confusion_matrix(y_test, predictions_by_model["float32"])
        fig_cm, ax_cm = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax_cm, colorbar=False, cmap="Blues", xticks_rotation=45)
        ax_cm.set_title(f"Float32 model confusion matrix (accuracy={results['float32']['accuracy']:.4f})")
        fig_cm.tight_layout()
        cm_path = os.path.join(results_dir, "confusion_matrix.png")
        fig_cm.savefig(cm_path, dpi=120)
        print(f"Saved {cm_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
