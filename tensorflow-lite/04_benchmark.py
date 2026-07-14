"""
THE benchmark: for all three .tflite models, measure real file size, real
mean inference latency (averaged over many runs, warmup discarded), and real
test-set accuracy. Print a markdown table and save a grouped bar chart.

This is the single most important script in the repo -- every claim in
reports/final-report.md about the size/latency/accuracy tradeoff traces back
to the numbers printed here.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "float32": os.path.join(HERE, "model_float32.tflite"),
    "dynamic_range": os.path.join(HERE, "model_dynamic_range.tflite"),
    "int8": os.path.join(HERE, "model_int8.tflite"),
}

N_LATENCY_RUNS = 200   # inference calls timed per model, after warmup
N_WARMUP_RUNS = 20     # discarded -- first few calls include one-time setup cost


def load_test_data():
    (_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_test = x_test.astype("float32")[..., None] / 255.0
    return x_test, y_test


def measure_accuracy(interpreter, input_details, output_details, x_test, y_test):
    is_int8_io = input_details["dtype"] == np.int8
    correct = 0
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
        if int(np.argmax(output)) == y_test[i]:
            correct += 1
    return correct / len(x_test)


def measure_latency(interpreter, input_details, x_test):
    is_int8_io = input_details["dtype"] == np.int8
    sample = x_test[0:1]
    if is_int8_io:
        scale, zero_point = input_details["quantization"]
        sample = (sample / scale + zero_point).astype(np.int8)

    # Warmup: first calls pay for lazy memory allocation / delegate setup, not representative.
    for _ in range(N_WARMUP_RUNS):
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()

    times_ms = []
    for _ in range(N_LATENCY_RUNS):
        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        times_ms.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times_ms)), float(np.std(times_ms))


if __name__ == "__main__":
    x_test, y_test = load_test_data()
    print(f"Test set: {len(x_test)} samples. Latency measured over {N_LATENCY_RUNS} runs "
          f"(after {N_WARMUP_RUNS} discarded warmup runs), single-image batches.\n")

    results = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found.")
            continue

        size_kb = os.path.getsize(path) / 1024

        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        accuracy = measure_accuracy(interpreter, input_details, output_details, x_test, y_test)
        mean_latency_ms, std_latency_ms = measure_latency(interpreter, input_details, x_test)

        results[name] = {
            "size_kb": size_kb,
            "accuracy": accuracy,
            "latency_ms": mean_latency_ms,
            "latency_std_ms": std_latency_ms,
        }
        print(f"{name:15s}: size={size_kb:8.2f} KB  accuracy={accuracy:.4f}  "
              f"latency={mean_latency_ms:.4f} +/- {std_latency_ms:.4f} ms/image")

    # --- Markdown table ---
    print("\n--- Markdown table ---")
    header = "| Model | Size (KB) | Accuracy | Mean latency (ms/image) |"
    sep = "|---|---|---|---|"
    print(header)
    print(sep)
    md_lines = [header, sep]
    for name, r in results.items():
        line = f"| {name} | {r['size_kb']:.2f} | {r['accuracy']:.4f} | {r['latency_ms']:.4f} |"
        print(line)
        md_lines.append(line)

    md_table_path = os.path.join(HERE, "benchmark_results.md")
    with open(md_table_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\nSaved {md_table_path}")

    # --- Grouped bar chart: size, latency, accuracy side by side ---
    names = list(results.keys())
    sizes = [results[n]["size_kb"] for n in names]
    latencies = [results[n]["latency_ms"] for n in names]
    accuracies = [results[n]["accuracy"] * 100 for n in names]  # as percentage for readability

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].bar(names, sizes, color=["tab:blue", "tab:orange", "tab:green"])
    axes[0].set_title("Model size (KB)")
    axes[0].set_ylabel("KB")

    axes[1].bar(names, latencies, color=["tab:blue", "tab:orange", "tab:green"])
    axes[1].set_title("Mean inference latency (ms/image)")
    axes[1].set_ylabel("ms")

    axes[2].bar(names, accuracies, color=["tab:blue", "tab:orange", "tab:green"])
    axes[2].set_title("Test accuracy (%)")
    axes[2].set_ylabel("%")
    axes[2].set_ylim(min(accuracies) - 2, 100)  # zoom in -- accuracy differences here are small

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("TFLite model comparison: Fashion-MNIST CNN (real, measured values)")
    fig.tight_layout()
    chart_path = os.path.join(HERE, "04_benchmark_comparison.png")
    fig.savefig(chart_path, dpi=120)
    print(f"Saved {chart_path}")
