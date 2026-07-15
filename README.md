# DRDO SAG Lab — AI/ML for Edge Devices (Research Internship)

**Author:** Piyush Kumar Bharati (B.Tech CSE)
**Organization:** DRDO, SAG Lab
**Research topic:** AI/ML for Edge Devices

## Objective

Build a working, reproducible pipeline for training a CNN image classifier
and preparing it for deployment on resource-constrained edge hardware,
covering the full stack from Python/NumPy fundamentals through classical
ML, neural network training, computer vision preprocessing, and TensorFlow
Lite model compression (quantization).

Every number in this repository comes from running the code in it. Where
that turned out to matter (see the headline benchmark below), I say so.

## Repository structure

```
drdo-edge-ai-internship/
├── README.md
├── requirements.txt
├── .gitignore
├── python-basics/             - core Python patterns (data structures, file I/O, OOP, exceptions, comprehensions)
├── numpy-practice/             - arrays, dtype/memory, indexing, broadcasting, vectorization, images as arrays
├── pandas-practice/             - load/clean/aggregate/feature-prep on the real California Housing dataset
├── matplotlib-practice/         - plotting fundamentals + a reusable training-curve plotter
├── scikit-learn-practice/       - regression, classification, scaling, overfitting demo
├── tensorflow-basics/           - tensors, a dense net on MNIST, a CNN on Fashion-MNIST, model inspection
├── opencv-experiments/          - image I/O, resize/crop, color/edges, model preprocessing, webcam capture
├── tensorflow-lite/              - the core topic: float32/dynamic-range/INT8 conversion + real benchmark
├── raspberry-pi/                 - deployment scripts + setup guide (explicit about what's Pi-verified vs not)
├── mini-projects/
│   ├── handwritten-digit-classifier/  - train -> TFLite -> predict, complete and standalone
│   └── face-or-object-detector/        - OpenCV Haar cascade face detection
├── reports/
│   ├── final-report.md          - full technical report
│   └── weekly-summary.md        - topic-wise summary
└── final-project/                - capstone: complete data-to-deployment pipeline (src/, models/, results/, notebooks/)
```

Every folder has its own `README.md`: purpose, file list, key concepts, how
to run, why it matters for edge AI, and common mistakes/gotchas actually hit
while building it.

## Setup / how to run

```bash
py -3.13 -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Verified installed versions (printed by running the code in this repo, not
copied from documentation):

```
Python:        3.13.2
numpy:         2.5.1
pandas:        3.0.3
matplotlib:    3.11.0
scikit-learn:  1.9.0
tensorflow:    2.21.0
opencv-python: 4.13.0.92 (cv2.__version__ reports "4.13.0")
seaborn:       0.13.2
```

`final-project/src/qat.py` additionally needs `tensorflow-model-optimization`
and `tf-keras` (both in `requirements.txt`); see that folder's README for
why.

Each folder is self-contained; run any script directly, e.g.:

```bash
python python-basics/01_data_structures.py
python final-project/src/train.py
```

For the full capstone pipeline:

```bash
python final-project/src/train.py       # trains the CNN, saves model + training curve
python final-project/src/convert.py     # converts to float32/dynamic-range/INT8 TFLite
python final-project/src/benchmark.py   # real size/latency/accuracy benchmark + confusion matrix
python final-project/src/qat.py          # quantization-aware training, adds a row to the benchmark
python final-project/src/ablation.py     # batch-norm-removed variant, tests what the INT8 gap depends on
python final-project/src/infer.py <image_path> --model final-project/models/model_int8.tflite
```

## Key results: headline benchmark

Two CNNs, same dataset (Fashion-MNIST), same benchmark methodology (200
timed inference runs after 20 discarded warmup runs, full 10,000-image test
set, CPU only, latency reported as mean/std/p95):

**Simpler CNN** (`tensorflow-basics/` → `tensorflow-lite/`, 56,714 params,
no batch normalization):

| Model | Size (KB) | Accuracy | Mean latency (ms) |
|---|---|---|---|
| float32 | 225.82 | 87.45% | 0.0360 |
| dynamic_range | 63.54 | 87.42% | 0.0214 |
| int8 | 63.38 | 87.59% | 0.0288 |

**Final project CNN** (`final-project/`, 26,154 params, batch normalization,
seeded training):

| Model | Size (KB) | Accuracy | Mean latency (ms) | Std (ms) | P95 (ms) |
|---|---|---|---|---|---|
| float32 | 105.86 | 87.21% | 0.0694 | 0.0014 | 0.0725 |
| dynamic_range | 33.96 | 87.23% | 0.0353 | 0.0011 | 0.0357 |
| int8 | 35.18 | 87.22% | 0.0461 | 0.0013 | 0.0495 |
| qat_int8 | 33.81 | 90.91% | 0.0471 | 0.0012 | 0.0479 |

Both models lose well under a percentage point to INT8 quantization. An
earlier version of this project reported the final CNN losing ~4.2 points to
INT8 and attributed it to batch normalization; that number came from an
unseeded training run and turned out not to replicate. Once training was
made reproducible (`final-project/src/train.py` now seeds Python, NumPy, and
TensorFlow) and the batch-norm claim was actually tested rather than assumed
(`final-project/src/ablation.py`), the drop mostly disappeared, and removing
batch normalization didn't change that. What did reproduce a ~2.7pp drop was
retraining the *original* model code (before this repo added
quantization-aware training support, which requires a small architectural
change, see `final-project/README.md`'s Discussion section) with the same
seed, concentrated almost entirely in the "Coat" and "Shirt" classes. So the
accuracy cost of INT8 quantization is real and architecture-sensitive, worth
measuring on the exact model being shipped, but it isn't explained by batch
normalization the way this README used to claim. Quantization-aware
training (`final-project/src/qat.py`, using `tensorflow-model-optimization`)
reached the highest accuracy of any variant, 90.91%.

**Dynamic-range quantization was the fastest variant in both benchmarks**,
faster than full INT8, despite INT8 being marginally smaller on disk. This
is very likely an artifact of this development machine's x86 CPU (TFLite's
XNNPACK delegate has a well-optimized float32/dynamic-range path); it may
not hold on a Raspberry Pi's ARM CPU. See `raspberry-pi/README.md` for what
was and wasn't verified on real edge hardware, and
`reports/final-report.md` §6 for the full discussion.

## Key learnings

- **Parameter count is the number that matters for edge deployment**, more
  than accuracy alone. A single `Flatten`→`Dense` layer held 90.4% of one
  model's parameters (`tensorflow-basics/README.md`); replacing it with
  `GlobalAveragePooling2D` in `final-project/` cut total parameters by more
  than half.
- **Quantization needs real calibration data.** INT8 conversion requires a
  representative dataset of real samples, not random noise, to measure
  activation ranges per layer correctly (`tensorflow-lite/README.md`).
- **A plausible explanation isn't the same as a tested one.** This repo
  shipped a wrong causal story for a while: attributing the final CNN's INT8
  accuracy drop to batch normalization without actually removing batch
  normalization and checking. It took retraining with a controlled ablation
  to find that out (`final-project/README.md`). Elsewhere, measured numbers
  overturned other reasonable-sounding assumptions too: blurring before edge
  detection increased edge pixel count on a clean synthetic image instead of
  reducing it (`opencv-experiments/README.md`); dynamic-range quantization
  beat full INT8 on latency (`tensorflow-lite/README.md`); an unscaled
  logistic regression model matched a scaled one on accuracy but threw a
  `ConvergenceWarning` the scaled version didn't (`scikit-learn-practice/README.md`).
- **A fixed random seed can expose bad hyperparameters, not just add
  reproducibility.** Seeding `final-project/src/train.py` revealed that its
  early-stopping patience had only looked adequate against one lucky
  unseeded run.
- **No physical Raspberry Pi was available for this internship.**
  `raspberry-pi/README.md` states plainly which parts were verified on this
  development machine (real webcam, real image predictions) versus written
  correctly per documentation but never run on ARM hardware.

## Tech stack

Python 3.13 · NumPy · Pandas · Matplotlib · scikit-learn · TensorFlow /
Keras · OpenCV · TensorFlow Lite · Jupyter

## References

See `reports/final-report.md` §9 for the full reference list (TensorFlow,
TensorFlow Lite, OpenCV, and scikit-learn documentation; Fashion-MNIST,
MNIST, California Housing, and Iris dataset citations).
