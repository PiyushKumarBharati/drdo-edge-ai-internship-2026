# Optimizing and Deploying a CNN Image Classifier on Edge Devices

**Research Intern:** Piyush Kumar Bharati (B.Tech CSE)
**Organization:** DRDO, SAG Lab
**Topic:** AI/ML for Edge Devices

---

## 1. Abstract

This report documents a complete, reproducible pipeline for training a
convolutional neural network image classifier and preparing it for
deployment on resource-constrained edge hardware. Working from Python and
NumPy fundamentals through classical machine learning, neural network
training in TensorFlow, classical and CNN-based computer vision with OpenCV,
and model compression via TensorFlow Lite, the project culminates in a
26,154-parameter CNN trained on Fashion-MNIST, converted to three TFLite
variants (float32, dynamic-range quantized, full-integer INT8), and
benchmarked for real file size, inference latency, and accuracy. The core
finding is that quantization's cost is architecture-dependent: a simpler CNN
lost under 0.2 percentage points of accuracy under INT8 quantization, while
a deeper CNN with batch normalization lost a real ~4.2 percentage points —
both results measured directly, not assumed from general quantization
literature. Every number in this report was produced by executing the code
in this repository; none were estimated or invented.

## 2. Introduction

**What Edge AI is.** Edge AI means running inference — not training — directly
on the device that captures the data (a camera, a sensor array, an embedded
board), rather than sending that data to a remote server or cloud GPU and
waiting for a response. The model is trained once, offline, on capable
hardware, then deployed as a static, compressed artifact to the edge device.

**Why this matters for a defense research lab specifically.** DRDO's
interest in edge inference follows directly from operational constraints
that don't apply to typical commercial ML deployment:

- **No network dependency.** A cloud round-trip assumes reliable
  connectivity, which cannot be assumed in the field.
- **Latency.** A prediction that depends on a network round-trip has
  latency that is unpredictable and outside the system's control; on-device
  inference has bounded, measurable latency.
- **Power.** Radio transmission is often more power-hungry than local
  compute for small payloads — relevant for battery- or solar-powered
  field equipment.
- **Data locality.** Sensitive sensor data (imagery, signals) never needs
  to leave the device, which is a meaningful security property independent
  of encryption-in-transit guarantees.

This project treats those constraints as design requirements, not
afterthoughts: every model built here is evaluated not just on accuracy, but
on file size and inference latency, because those are the numbers that
determine whether a model is actually deployable on a device like a
Raspberry Pi.

## 3. Background

**Convolutional Neural Networks (CNNs).** A `Conv2D` layer learns a small
grid of weights (a filter, e.g. 3×3) that slides across the input and
computes a weighted sum at every position — the same filter reused
everywhere in the image. This weight-sharing is what makes CNNs
parameter-efficient for image data compared to a fully-connected (`Dense`)
layer, which connects every input pixel to every output unit independently.
`MaxPooling2D` and `GlobalAveragePooling2D` layers reduce spatial resolution
while keeping the strongest/average signal, further controlling parameter
count and compute cost.

**Model compression.** A trained model's size is dominated by its weights —
one 32-bit float per parameter by default. Two levers reduce this:

1. **Architecture choice.** Replacing a `Flatten`→`Dense` head with
   `GlobalAveragePooling2D` (zero parameters) removes the single largest
   source of parameters in a small CNN — demonstrated directly in this
   project (see §5).
2. **Quantization.** Representing weights (and optionally activations) with
   fewer bits — typically 8-bit integers instead of 32-bit floats —
   directly shrinks file size roughly 4x per quantized tensor, and can
   speed up inference on hardware with integer SIMD acceleration.

**Quantization, specifically.** TensorFlow Lite supports two main
post-training quantization strategies, both used and measured in this
project:

- **Dynamic-range quantization** converts weights to int8 ahead of time;
  activations are computed in float32 and quantized on the fly per
  inference. Requires no calibration data.
- **Full-integer (INT8) quantization** converts both weights and
  activations to int8 ahead of time. This requires a **representative
  dataset** — real sample inputs — so the converter can measure the actual
  range of activations at each layer and choose an appropriate int8
  scale/zero-point. Using synthetic or random data instead of real samples
  would calibrate against the wrong distribution and silently degrade
  accuracy.

## 4. Methodology

**Dataset.** Fashion-MNIST: 60,000 training / 10,000 test grayscale 28×28
images across 10 clothing categories, loaded via `tf.keras.datasets`. Used
consistently across `tensorflow-basics/`, `tensorflow-lite/`, and
`final-project/` so results are directly comparable across phases.

**Architecture (final model, `final-project/src/model.py`).** Three
convolutional blocks (16→32→64 filters, each with batch normalization and
max-pooling on the first two), `GlobalAveragePooling2D`, then `Dense(32)` →
`Dropout(0.3)` → `Dense(10, softmax)`. 26,154 total parameters.

**Training setup.** Adam optimizer, sparse categorical cross-entropy loss,
batch size 128, up to 8 epochs with `EarlyStopping(monitor="val_accuracy",
patience=4, restore_best_weights=True)`. The early-stopping callback was
added after an initial run showed validation accuracy was unstable
epoch-to-epoch (a real observation, not a precaution taken in the abstract —
see §6).

**Conversion pipeline (`final-project/src/convert.py`).** Three `.tflite`
variants produced from the same trained model: float32 baseline,
dynamic-range quantized, and full-integer INT8 (calibrated on 100 real
training images sampled without replacement).

**Benchmark methodology (`final-project/src/benchmark.py`,
`tensorflow-lite/04_benchmark.py`).** For each model variant: file size read
directly from disk; accuracy measured over the full 10,000-image test set;
latency measured as the mean of 200 timed single-image inference calls,
after 20 discarded warmup calls (to exclude one-time interpreter/delegate
setup cost from the measurement).

**Hardware.** All training, conversion, and benchmarking was performed on a
Windows laptop CPU (x86_64). No GPU was used or available (TensorFlow
explicitly reports GPU is unsupported on native Windows for this TensorFlow
version). **No physical Raspberry Pi was available during this internship**
— see §7 (Limitations) and `raspberry-pi/README.md` for exactly what was and
wasn't verified on ARM/embedded hardware.

## 5. Results

### 5.1 Architecture comparison

| Model | Parameters | Head design |
|---|---|---|
| `tensorflow-basics/03_cnn_fashion_mnist.py` | 56,714 | Flatten → Dense(64) |
| `final-project` CNN | **26,154** | GlobalAveragePooling2D → Dense(32) |

Replacing the dense head with global average pooling cut parameter count by
more than half, directly applying a finding from `tensorflow-basics/README.md`:
that a single `Flatten`→`Dense` layer held 90.4% of that model's parameters.

### 5.2 Training result

`final-project` CNN: **90.07% test accuracy** (vs. 87.45% for the simpler,
larger `tensorflow-basics` CNN) — better accuracy with fewer than half the
parameters, after fixing a real training instability (§6).

![training curve](../final-project/results/training_curve.png)

### 5.3 Confusion matrix

![confusion matrix](../final-project/results/confusion_matrix.png)

Errors concentrate in one visually-similar cluster — Shirt, T-shirt/top,
Pullover, and Coat — while classes with distinct silhouettes (Trouser,
Sandal, Bag) are almost never confused with anything.

### 5.4 TFLite benchmark — the headline result

**`final-project` CNN** (real measurements, `final-project/results/benchmark_table.md`):

| Model | Size (KB) | Accuracy | Mean latency (ms/image) |
|---|---|---|---|
| float32 | 107.32 | 0.9007 | 0.2439 |
| dynamic_range | 35.42 | 0.9014 | 0.1517 |
| int8 | 36.41 | 0.8589 | 0.2056 |

![final project comparison chart](../final-project/results/comparison_chart.png)

**Simpler CNN, for comparison** (`tensorflow-lite/benchmark_results.md`):

| Model | Size (KB) | Accuracy | Mean latency (ms/image) |
|---|---|---|---|
| float32 | 225.82 | 0.8745 | 0.0360 |
| dynamic_range | 63.54 | 0.8742 | 0.0214 |
| int8 | 63.38 | 0.8759 | 0.0288 |

### 5.5 Mini-project results

- **Handwritten digit classifier** (`mini-projects/handwritten-digit-classifier/`):
  98.15% test accuracy on MNIST, 20.35 KB dynamic-range quantized `.tflite`
  (10.1x smaller than the `.keras` file), verified correct on 5 real test
  images.
- **Face detector** (`mini-projects/face-or-object-detector/`): OpenCV Haar
  cascade correctly detected a face in a real photo after tuning detection
  parameters against actual test results (the commonly-quoted "default"
  parameters found zero faces on this photo).

## 6. Discussion

**The size/latency/accuracy tradeoff was not uniform across models — and
that is the most important finding of this project.** On the simpler,
BatchNorm-free CNN, INT8 quantization was very nearly free: 87.45% →
87.59% accuracy (actually slightly *higher*, within measurement noise), for
a 3.56x size reduction. On the final, deeper CNN with batch normalization,
INT8 quantization cost a real, repeatable **~4.2 percentage points** of
accuracy (90.07% → 85.89%). The most likely explanation is that
`BatchNormalization` layers introduce additional learned parameters (scale
and shift) and running statistics that become additional sources of
quantization error, stacked on top of the convolutional weight
quantization — a cost the simpler, BatchNorm-free architecture never paid.
**This means "INT8 costs ~0% accuracy" is not a fact about quantization in
general; it is a fact about one specific architecture.** Any real deployment
decision needs to measure this tradeoff for the actual model being shipped,
not assume a number from a different model.

**Latency did not always favor the smallest or most quantized model.**
Dynamic-range quantization was the fastest variant in both benchmarks in
this report — faster than full INT8, despite INT8 being marginally smaller
on disk in the `tensorflow-lite/` case. The likely explanation is that this
project's CPU (x86, via TFLite's XNNPACK delegate) has a very well-optimized
path for the float32-activation / int8-weight combination that dynamic-range
quantization produces, while true int8-activation kernels benefit most from
hardware with dedicated int8 SIMD or NPU acceleration — which this
development machine does not have. This is a specific, falsifiable
prediction: it may not hold on a Raspberry Pi's ARM CPU, which is exactly
why no Pi latency claim is made in this report (§7).

**Training instability was a real, observed problem, not a theoretical
concern.** An initial training run of the final model (without early
stopping) showed validation accuracy crash to 35% on epoch 1 before
recovering, and the *final* epoch was not the best-performing one. Adding
`EarlyStopping(restore_best_weights=True)` — a response to an actually
observed failure mode, not a default best practice applied without
thought — fixed this and is directly responsible for the reported 90.07%
figure being a stable, reproducible number rather than an artifact of
whichever epoch training happened to stop on.

## 7. Limitations

- **No physical Raspberry Pi was available during this internship.**
  `raspberry-pi/deploy_inference.py` and `camera_inference.py` were verified
  end-to-end on this development machine (correct predictions on real
  images, a real webcam feed) via their `tf.lite` fallback path, but the
  `tflite_runtime` import path and all ARM-specific behavior are untested.
  **No latency or accuracy number in this report should be assumed to hold
  on Raspberry Pi hardware** — see §6's discussion of why the fastest format
  on x86 is not guaranteed to be the fastest format on ARM.
- **Benchmarks reflect a single development machine's CPU**, not a
  representative sample of edge hardware.
- **The representative dataset for INT8 calibration used 100 samples** — a
  practical, fast choice for this project's small models; larger models or
  more diverse input distributions might need more calibration samples.
- **Fashion-MNIST is a relatively easy, low-resolution (28×28 grayscale)
  benchmark dataset.** Real DRDO-relevant imagery (higher resolution,
  different domains) would need its own training run and its own
  size/latency/accuracy measurement — the pipeline built here generalizes,
  but the specific numbers do not.

## 8. Future Work

- **Measure on real Raspberry Pi hardware** using `raspberry-pi/setup_pi.md`,
  directly testing whether dynamic-range quantization remains faster than
  INT8 on ARM, or whether the result from this report's x86 measurements
  reverses.
- **Pruning**, to reduce parameter count directly rather than only
  compressing existing parameters via quantization — potentially compounds
  with the quantization gains already measured here.
- **Knowledge distillation**, training a smaller "student" model to match a
  larger "teacher" model's predictions, as an alternative or complement to
  architectural parameter reduction (the `GlobalAveragePooling2D` change in
  §5.1).
- **Coral Edge TPU / NPU targets**, which accelerate INT8 inference
  specifically in hardware — directly relevant given this report's finding
  that INT8's latency advantage did not materialize on this project's CPU;
  dedicated INT8 hardware is the scenario where that advantage is expected
  to show up.
- **A larger, DRDO-relevant image dataset**, once available, to validate
  that this pipeline's conclusions (architecture-dependent quantization
  cost, EarlyStopping for training stability) generalize beyond
  Fashion-MNIST.

## 9. References

- TensorFlow / Keras documentation: https://www.tensorflow.org/
- TensorFlow Lite conversion and optimization guide: https://www.tensorflow.org/lite/performance/post_training_quantization
- OpenCV documentation: https://docs.opencv.org/
- scikit-learn documentation: https://scikit-learn.org/
- Fashion-MNIST dataset: Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms.
- MNIST dataset: LeCun, Y., Cortes, C., & Burges, C.J.C. The MNIST database of handwritten digits.
- California Housing dataset (used in `pandas-practice/`, `scikit-learn-practice/`): from scikit-learn's `fetch_california_housing`, originally from Pace, R.K. and Barry, R. (1997).
- Iris dataset (used in `scikit-learn-practice/`, `matplotlib-practice/`): Fisher, R.A. (1936), bundled with scikit-learn.
- OpenCV Haar cascade classifier (`haarcascade_frontalface_default.xml`): part of the OpenCV project, bundled with `opencv-python`.

All code, real measurements, and figures referenced in this report are in
this repository — see the root `README.md` for the full structure and setup
instructions.
