# Optimizing and Deploying a CNN Image Classifier on Edge Devices

**Research Intern:** Piyush Kumar Bharati (B.Tech CSE)
**Organization:** DRDO, SAG Lab
**Topic:** AI/ML for Edge Devices

---

## 1. Abstract

This report documents a complete pipeline for training a convolutional
neural network image classifier and preparing it for deployment on
resource-constrained edge hardware: Python and NumPy fundamentals, classical
machine learning, neural network training in TensorFlow, computer vision
with OpenCV, and model compression via TensorFlow Lite, culminating in a
26,154-parameter CNN trained on Fashion-MNIST and converted to four variants
(float32, dynamic-range quantized, full-integer INT8, and
quantization-aware-trained INT8). Every number in this report was produced
by running the code in this repository, including the ones that overturned
an earlier draft of this same report's headline finding. That finding was
that INT8 quantization cost the final CNN a real ~4.2 percentage points of
accuracy, caused by batch normalization. Adding reproducibility (fixed random
seeds) forced a retrain, and the retrained model's INT8 drop turned out to be
under 0.1 percentage points. Actually testing the batch-normalization claim
(§6.2) rather than asserting it showed the claim doesn't hold either: removing
batch normalization didn't change the (already negligible) drop. What
correlates with the drop is a smaller architectural detail than
BatchNormalization itself, discussed in §6.2. Quantization-aware training
(§5.6) reached 90.91% accuracy, the best of any variant tested.

## 2. Introduction

**What Edge AI is.** Edge AI means running inference, not training, directly
on the device that captures the data (a camera, a sensor array, an embedded
board), rather than sending that data to a remote server or cloud GPU and
waiting for a response. The model is trained once, offline, on capable
hardware, then deployed as a static, compressed artifact to the edge device.

**Why this matters for a defense research lab specifically.** DRDO's
interest in edge inference follows from operational constraints that don't
apply to typical commercial ML deployment:

- **No network dependency.** A cloud round-trip assumes reliable
  connectivity, which cannot be assumed in the field.
- **Latency.** A prediction that depends on a network round-trip has
  latency that is unpredictable and outside the system's control; on-device
  inference has bounded, measurable latency.
- **Power.** Radio transmission is often more power-hungry than local
  compute for small payloads, relevant for battery- or solar-powered field
  equipment.
- **Data locality.** Sensitive sensor data (imagery, signals) never needs
  to leave the device, a meaningful security property independent of
  encryption-in-transit guarantees.

This project treats those constraints as design requirements: every model
built here is evaluated not just on accuracy, but on file size and inference
latency, because those numbers determine whether a model is actually
deployable on a device like a Raspberry Pi.

## 3. Background

**Convolutional Neural Networks (CNNs).** A `Conv2D` layer learns a small
grid of weights (a filter, e.g. 3×3) that slides across the input and
computes a weighted sum at every position, the same filter reused everywhere
in the image. This weight-sharing is what makes CNNs parameter-efficient for
image data compared to a fully-connected (`Dense`) layer, which connects
every input pixel to every output unit independently. `MaxPooling2D` and
`GlobalAveragePooling2D` layers reduce spatial resolution while keeping the
strongest/average signal, further controlling parameter count and compute
cost.

**Model compression.** A trained model's size is dominated by its weights,
one 32-bit float per parameter by default. Two levers reduce this:

1. **Architecture choice.** Replacing a `Flatten`→`Dense` head with
   `GlobalAveragePooling2D` (zero parameters) removes the single largest
   source of parameters in a small CNN (§5.1).
2. **Quantization.** Representing weights (and optionally activations) with
   fewer bits, typically 8-bit integers instead of 32-bit floats, shrinks
   file size roughly 4x per quantized tensor and can speed up inference on
   hardware with integer SIMD acceleration.

**Quantization strategies used here.**

- **Dynamic-range quantization** converts weights to int8 ahead of time;
  activations are computed in float32 and quantized on the fly per
  inference. Requires no calibration data.
- **Full-integer (INT8) quantization** converts both weights and
  activations to int8 ahead of time, using a **representative dataset**
  (real sample inputs) so the converter can measure the actual range of
  activations at each layer and choose an appropriate scale/zero-point.
  Random data would calibrate against the wrong distribution and degrade
  accuracy.
- **Quantization-aware training (QAT)** simulates int8 rounding during
  training itself (via fake-quantization nodes), so the model's weights
  adapt to quantization noise before conversion, rather than being quantized
  post-hoc. Implemented here in `final-project/src/qat.py` using
  `tensorflow-model-optimization` (§5.6).

## 4. Methodology

**Dataset.** Fashion-MNIST: 60,000 training / 10,000 test grayscale 28×28
images across 10 clothing categories, loaded via `tf.keras.datasets`. Used
consistently across `tensorflow-basics/`, `tensorflow-lite/`, and
`final-project/` so results are comparable across phases.

**Architecture (`final-project/src/model.py`).** Three convolutional blocks
(16→32→64 filters), each `Conv2D → BatchNormalization → Activation("relu")`,
the first two followed by `MaxPooling2D`, then `GlobalAveragePooling2D →
Dense(32) → Dropout(0.3) → Dense(10, softmax)`. 26,154 total parameters.
`build_model()` takes a `batch_norm` flag so the same function builds the
ablation variant in §6.2.

**Reproducibility.** `train.py` seeds Python's `random`, NumPy, and
`tf.random` (seed 42) before building or training the model. Reruns on this
machine reproduce the same epoch-by-epoch history and final test accuracy.
Adding this seed is what surfaced the retraining discussed in §5.2 and §6.1:
the model's originally-reported 90.07% accuracy came from an unseeded run
that could not be reproduced or audited against, and the seeded rerun landed
meaningfully lower until the early-stopping patience (tuned against that one
unseeded run) was also corrected.

**Training setup.** Adam optimizer, sparse categorical cross-entropy loss,
batch size 128, up to 40 epochs with `EarlyStopping(monitor="val_accuracy",
patience=8, restore_best_weights=True)`.

**Conversion pipeline (`final-project/src/convert.py`).** Float32 baseline,
dynamic-range quantized, and full-integer INT8 (calibrated on 100 real
training images sampled without replacement), all from the same trained
model.

**Benchmark methodology (`final-project/src/benchmark.py`,
`tensorflow-lite/04_benchmark.py`).** File size read directly from disk;
accuracy measured over the full 10,000-image test set; latency measured as
the mean, standard deviation, and 95th percentile of 200 timed single-image
inference calls, after 20 discarded warmup calls. Per-class precision,
recall, and F1 are computed with `sklearn.metrics.classification_report`
(`final-project/results/classification_reports.txt`).

**Hardware.** All training, conversion, and benchmarking ran on a Windows
laptop CPU (x86_64). No GPU was used or available (TensorFlow reports GPU is
unsupported on native Windows for this TensorFlow version). No physical
Raspberry Pi was available during this internship (§7).

## 5. Results

### 5.1 Architecture comparison

| Model | Parameters | Head design |
|---|---|---|
| `tensorflow-basics/03_cnn_fashion_mnist.py` | 56,714 | Flatten → Dense(64) |
| `final-project` CNN | **26,154** | GlobalAveragePooling2D → Dense(32) |

Replacing the dense head with global average pooling cut parameter count by
more than half, applying a finding from `tensorflow-basics/README.md`: a
single `Flatten`→`Dense` layer held 90.4% of that model's parameters.

### 5.2 Training result

`final-project` CNN: **87.21% test accuracy** with the seeded, corrected
training setup described in §4, close to, and not better than, the simpler
`tensorflow-basics` CNN's 87.45%, despite fewer than half the parameters. An
earlier, unseeded run of this same architecture reached 90.07%; that number
could not be reproduced once training was made auditable and is no longer
reported as this model's accuracy (§6.1).

![training curve](../final-project/results/training_curve.png)

### 5.3 Confusion matrix

![confusion matrix](../final-project/results/confusion_matrix.png)

Errors concentrate in one visually-similar cluster: Shirt, T-shirt/top,
Pullover, and Coat, while classes with distinct silhouettes (Trouser,
Sandal, Bag) are almost never confused with anything. This holds at every
quantization level (§5.4's classification reports).

### 5.4 TFLite benchmark

**`final-project` CNN** (`final-project/results/benchmark_table.md`; latency
is mean of 200 timed runs after 20 warmup runs, ± one standard deviation,
with the 95th percentile also reported):

| Model | Size (KB) | Accuracy | Mean latency (ms) | Std (ms) | P95 (ms) |
|---|---|---|---|---|---|
| float32 | 105.86 | 0.8721 | 0.0694 | 0.0014 | 0.0725 |
| dynamic_range | 33.96 | 0.8723 | 0.0353 | 0.0011 | 0.0357 |
| int8 | 35.18 | 0.8722 | 0.0461 | 0.0013 | 0.0495 |
| qat_int8 | 33.81 | 0.9091 | 0.0471 | 0.0012 | 0.0479 |

![final project comparison chart](../final-project/results/comparison_chart.png)

**Simpler CNN, for comparison** (`tensorflow-lite/benchmark_results.md`):

| Model | Size (KB) | Accuracy | Mean latency (ms/image) |
|---|---|---|---|
| float32 | 225.82 | 0.8745 | 0.0360 |
| dynamic_range | 63.54 | 0.8742 | 0.0214 |
| int8 | 63.38 | 0.8759 | 0.0288 |

Full per-class precision/recall/F1 for every `final-project` variant is in
`final-project/results/classification_reports.txt`. The weakest class
throughout is Shirt (recall ~74% in the float32/dynamic_range/int8 variants,
~74% in qat_int8 too), consistent with the confusion cluster in §5.3 at
every quantization level, not something quantization introduces.

### 5.5 Mini-project results

- **Handwritten digit classifier** (`mini-projects/handwritten-digit-classifier/`):
  98.15% test accuracy on MNIST, 20.35 KB dynamic-range quantized `.tflite`
  (10.1x smaller than the `.keras` file), verified correct on 5 real test
  images. Note: this script has no random seed, so a rerun will not
  reproduce these exact figures (unlike `final-project/`, seeding wasn't
  added here since this mini-project was out of this audit's scope).
- **Face detector** (`mini-projects/face-or-object-detector/`): OpenCV Haar
  cascade correctly detected a face in a real photo after tuning detection
  parameters against actual test results (the commonly-quoted "default"
  parameters found zero faces on this photo).

### 5.6 Quantization-aware training

`final-project/src/qat.py` runs real QAT using `tensorflow-model-optimization`
0.8.1. Getting it running required two things, both found by testing rather
than assumed from the package's documentation:

1. `tensorflow-model-optimization` needs the standalone `tf_keras` package
   and the `TF_USE_LEGACY_KERAS=1` environment variable to import at all
   under TensorFlow 2.21 (which defaults to Keras 3). Without both, `import
   tensorflow_model_optimization` raises `ImportError: Keras cannot be
   imported. Check that it is installed.` Both are now real dependencies in
   `requirements.txt`.
2. `tfmot.quantization.keras.quantize_model()` raises `RuntimeError: Layer
   batch_normalization:<...> is not supported` when a `Conv2D`'s activation
   is fused (`Conv2D(activation="relu")`). It only recognizes the
   Conv→BatchNorm→Activation fusion pattern when activation is its own
   layer. `model.py` was restructured this way for every model in this
   folder, not just the QAT one, so there is one architecture, not two.

Because `qat.py` runs in a separate process with the legacy Keras compat
layer active, it cannot load `models/cnn_classifier.keras` (a Keras-3
serialization format `tf_keras` doesn't read). It trains its own float32
baseline first, same architecture/seed/hyperparameters as `train.py`, then
applies `quantize_model()`, fine-tunes for 5 more epochs, and converts to
INT8. That baseline reached 89.06% test accuracy, not identical to
`train.py`'s 87.21% figure, despite the same seed: Keras 3 and the legacy
`tf_keras` package don't consume randomness identically for the same model,
so a fixed seed reproduces results within a Keras backend, not across two
different ones.

**Result: 90.91% INT8 accuracy**, the highest of any variant in §5.4's
table, and higher than its own float32 starting point. This is not a clean
"QAT recovered the drop" result, because the plain post-training INT8 model
already had almost no drop to recover (§5.4). Some of the QAT number is
plausibly just 5 extra epochs of training rather than quantization-awareness
specifically, a confound not separated out here since no float32-only
control was trained for those same 5 extra epochs. What's established is
that QAT ran end to end and produced a real, benchmarked, high-accuracy INT8
model.

## 6. Discussion

### 6.1 A number that didn't survive reproducibility

This report originally stated the final CNN reached 90.07% test accuracy.
That run had no random seed, so it could not be rerun or checked. Adding
`tf.random.set_seed()` (and NumPy/Python seeds) to `train.py` for
reproducibility forced a retrain, and the retrained, now-reproducible model
reached 87.21%, a materially different number. Investigating why (a
diagnostic run with a longer patience budget, `epochs=30, patience=10`)
showed validation accuracy was still climbing well past the original
`patience=4` cutoff: the original hyperparameters had only looked adequate
against one specific lucky unseeded run. `patience` was widened to 8 and
`epochs` to 40 so `EarlyStopping` stops on genuine convergence rather than
cutting off a still-improving run; that is the number reported in §5.2. The
lesson generalizes past this one model: an unseeded result is not just
unreproducible, it can also hide that a hyperparameter choice was never
actually validated, only lucky.

### 6.2 Testing the batch-normalization hypothesis, instead of asserting it

The original version of this report attributed the final CNN's ~4.2
percentage point INT8 accuracy drop to `BatchNormalization`, without
training a version without it to check. `final-project/src/ablation.py` now
does that: same architecture, same seed, same training budget,
`batch_norm=False`.

| Model | Float32 → INT8 | Delta |
|---|---|---|
| With BatchNorm (main model, §5.4) | 87.21% → 87.22% | +0.01pp |
| Without BatchNorm | 89.37% → 89.48% | +0.11pp |

Removing BatchNorm changed nothing, because the with-BatchNorm model already
had no drop to shrink under the current (seeded, retrained) pipeline. This
is evidence against the original hypothesis, not for it.

So where did the originally-reported ~4.2pp number come from? To find out, I
retrained the *original* model code, the one with `Conv2D(activation="relu")`
fused rather than split into a separate `Activation` layer (the form used
before this repository added QAT support, which requires the split), with
the same seed used everywhere else in this report. It reproduced a real
drop: 90.60% → 87.89% (2.71pp), concentrated almost entirely in two classes:
Coat (−13.3pp) and Shirt (−13.9pp), the same visually-similar cluster from
§5.3.

That result points at something narrower than "BatchNorm causes
quantization error": whether `Conv2D`'s activation is fused into the layer
or built as a separate `Activation` layer appears to change how TFLite's
converter fuses and quantizes the Conv+BatchNorm+ReLU sequence, and that,
not BatchNorm's presence by itself, tracks with whether the drop appears.
Both the with- and without-BatchNorm split-activation models in the table
above show no drop; the fused-activation model does. I have not confirmed
the mechanism inside the converter's graph optimization passes, so this is
reported as an observed correlation across three controlled training runs,
not a proven cause. It would be worth a closer investigation, ideally by
inspecting the converted graphs directly (e.g. with the TFLite model
visualizer) rather than only comparing accuracy numbers.

### 6.3 Latency did not favor the smallest or most quantized model

Dynamic-range quantization was the fastest variant in both benchmarks in
this report, faster than full INT8, despite INT8 being marginally smaller
on disk in the `tensorflow-lite/` case. The likely explanation is that this
project's CPU (x86, via TFLite's XNNPACK delegate) has a well-optimized path
for the float32-activation / int8-weight combination that dynamic-range
quantization produces, while true int8-activation kernels benefit most from
hardware with dedicated int8 SIMD or NPU acceleration, which this
development machine does not have. This is a falsifiable prediction: it may
not hold on a Raspberry Pi's ARM CPU, which is why no Pi latency claim is
made in this report (§7).

## 7. Limitations

- **No physical Raspberry Pi was available during this internship.**
  `raspberry-pi/deploy_inference.py` and `camera_inference.py` were verified
  end-to-end on this development machine (correct predictions on real
  images, a real webcam feed) via their `tf.lite` fallback path, but the
  `tflite_runtime` import path and all ARM-specific behavior are untested.
  No latency or accuracy number in this report should be assumed to hold on
  Raspberry Pi hardware (§6.3).
- **Benchmarks reflect a single development machine's CPU**, not a
  representative sample of edge hardware.
- **The representative dataset for INT8 calibration used 100 samples**, a
  practical, fast choice for this project's small models; larger models or
  more diverse input distributions might need more calibration samples.
- **Fashion-MNIST is a relatively easy, low-resolution (28×28 grayscale)
  benchmark dataset.** Real DRDO-relevant imagery would need its own
  training run and its own size/latency/accuracy measurement; the pipeline
  generalizes, the specific numbers do not.
- **The fused-vs-split-activation finding in §6.2 rests on one seed per
  configuration**, not a distribution of seeds. It is reported as a
  reproducible observation under this specific seed, not a statistically
  validated effect size.
- **QAT's baseline (§5.6) was trained under a different Keras backend
  (`tf_keras`) than the main pipeline**, so its accuracy isn't directly
  comparable to `train.py`'s figure on equal footing, only to its own
  quantized result.

## 8. Future Work

- **Measure on real Raspberry Pi hardware** using `raspberry-pi/setup_pi.md`,
  testing directly whether dynamic-range quantization remains faster than
  INT8 on ARM.
- **Confirm the fused-vs-split-activation finding (§6.2) across multiple
  seeds**, and inspect the converted TFLite graphs directly to identify
  which operator fusion difference actually causes the accuracy drop,
  rather than inferring it from accuracy numbers alone.
- **Isolate QAT's real contribution** by training a float32-only control for
  the same 5 extra epochs used in `qat.py`'s fine-tuning step, to separate
  "more training" from "quantization-aware training" in the 90.91% result.
- **Pruning**, to reduce parameter count directly rather than only
  compressing existing parameters via quantization, potentially compounding
  with the quantization gains already measured here.
- **Knowledge distillation**, training a smaller "student" model to match a
  larger "teacher" model's predictions, as an alternative or complement to
  the architectural parameter reduction in §5.1.
- **Coral Edge TPU / NPU targets**, which accelerate INT8 inference
  specifically in hardware, directly relevant given §6.3's finding that
  INT8's latency advantage did not materialize on this project's CPU.
- **A larger, DRDO-relevant image dataset**, once available, to check
  whether this pipeline's conclusions generalize beyond Fashion-MNIST.

## 9. References

- TensorFlow / Keras documentation: https://www.tensorflow.org/
- TensorFlow Lite conversion and optimization guide: https://www.tensorflow.org/lite/performance/post_training_quantization
- TensorFlow Model Optimization Toolkit (quantization-aware training): https://www.tensorflow.org/model_optimization/guide/quantization/training
- OpenCV documentation: https://docs.opencv.org/
- scikit-learn documentation: https://scikit-learn.org/
- Fashion-MNIST dataset: Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms.
- MNIST dataset: LeCun, Y., Cortes, C., & Burges, C.J.C. The MNIST database of handwritten digits.
- California Housing dataset (used in `pandas-practice/`, `scikit-learn-practice/`): from scikit-learn's `fetch_california_housing`, originally from Pace, R.K. and Barry, R. (1997).
- Iris dataset (used in `scikit-learn-practice/`, `matplotlib-practice/`): Fisher, R.A. (1936), bundled with scikit-learn.
- OpenCV Haar cascade classifier (`haarcascade_frontalface_default.xml`): part of the OpenCV project, bundled with `opencv-python`.

All code, measurements, and figures referenced in this report are in this
repository; see the root `README.md` for the full structure and setup
instructions.
