"""
File I/O without pandas: plain CSV and JSON.

Why this matters: on a Pi or any edge box you sometimes don't want the weight
of pandas just to log a few inference results. Knowing the stdlib way matters.
"""

import csv
import json
import os

# Keep all output inside this folder regardless of where the script is run from.
HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "inference_log.csv")
JSON_PATH = os.path.join(HERE, "run_config.json")

# --- Writing CSV: simulate logging predictions from an inference run. ---
predictions = [
    {"image": "cat_001.jpg", "predicted_class": "cat", "confidence": 0.94},
    {"image": "dog_001.jpg", "predicted_class": "dog", "confidence": 0.88},
    {"image": "cat_002.jpg", "predicted_class": "dog", "confidence": 0.51},  # wrong / low-confidence
]

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "predicted_class", "confidence"])
    writer.writeheader()
    writer.writerows(predictions)
print(f"Wrote {len(predictions)} rows to {CSV_PATH}")

# --- Reading CSV back ---
with open(CSV_PATH, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("Read back from CSV:")
for row in rows:
    # Note: everything read from csv.DictReader is a string -- confidence needs casting.
    confidence = float(row["confidence"])
    flag = "LOW CONFIDENCE" if confidence < 0.6 else "ok"
    print(f"  {row['image']}: {row['predicted_class']} ({confidence:.2f}) [{flag}]")

# --- Writing JSON: a run config, the kind of thing you'd save next to a model. ---
run_config = {
    "model_name": "fashion_mnist_cnn",
    "input_shape": [28, 28, 1],
    "num_classes": 10,
    "quantized": False,
    "class_names": ["tshirt", "trouser", "pullover", "dress", "coat"],
}
with open(JSON_PATH, "w") as f:
    json.dump(run_config, f, indent=2)
print(f"\nWrote run config to {JSON_PATH}")

# --- Reading JSON back ---
with open(JSON_PATH, "r") as f:
    loaded_config = json.load(f)
print("Read back from JSON:", loaded_config)
assert loaded_config == run_config, "round-trip through JSON changed the data!"
print("Round-trip check passed: JSON in == JSON out.")
