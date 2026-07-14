# DRDO SAG Lab — AI/ML for Edge Devices (Research Internship)

**Author:** Piyush Kumar Bharati (B.Tech CSE)
**Organization:** DRDO, SAG Lab
**Research topic:** AI/ML for Edge Devices

## Objective

Build a working, reproducible pipeline for training a CNN image classifier
and preparing it for deployment on resource-constrained edge hardware —
covering the full stack from Python/NumPy fundamentals through classical
ML, neural network training, computer vision preprocessing, and TensorFlow
Lite model compression (quantization), with every result backed by code
that actually runs.

Every result reported here was reproduced from the code in this repository.

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
├── tensorflow-lite/              - THE core topic: float32/dynamic-range/INT8 conversion + real benchmark
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
python final-project/src/infer.py <image_path> --model final-project/models/model_int8.tflite
```

## Key results — headline benchmark

The central technical finding of this project: **TFLite quantization's
accuracy cost is architecture-dependent, not fixed.** Two CNNs, same
dataset (Fashion-MNIST), same benchmark methodology (200 timed inference
runs after 20 discarded warmup runs, full 10,000-image test set, CPU only):

**Simpler CNN** (`tensorflow-basics/` → `tensorflow-lite/`, 56,714 params,
no batch normalization):

| Model | Size (KB) | Accuracy | Latency (ms/image) |
|---|---|---|---|
| float32 | 225.82 | 87.45% | 0.0360 |
| dynamic_range | 63.54 | 87.42% | 0.0214 |
| int8 | 63.38 | 87.59% | 0.0288 |

INT8 accuracy cost: **≤0.17 percentage points** — effectively free.

**Final project CNN** (`final-project/`, 26,154 params, with batch
normalization):

| Model | Size (KB) | Accuracy | Latency (ms/image) |
|---|---|---|---|
| float32 | 107.32 | 90.07% | 0.2439 |
| dynamic_range | 35.42 | 90.14% | 0.1517 |
| int8 | 36.41 | 85.89% | 0.2056 |

INT8 accuracy cost: **~4.2 percentage points** — a real, measurable
tradeoff, most likely from batch normalization adding extra quantization
sensitivity on top of the convolutional weights.

**Both benchmarks agree on one more counter-intuitive point**: dynamic-range
quantization was the *fastest* variant in both cases — faster than full
INT8, despite INT8 being marginally smaller on disk. This is very likely an
artifact of this development machine's x86 CPU (TFLite's XNNPACK delegate
has a well-optimized float32/dynamic-range path); it may not hold on a
Raspberry Pi's ARM CPU — see `raspberry-pi/README.md` for exactly what was
and wasn't verified on real edge hardware, and `reports/final-report.md`
§6 for the full discussion.

## Key learnings

- **Parameter count is the number that matters for edge deployment**, more
  than accuracy alone — a single `Flatten`→`Dense` layer held 90.4% of one
  model's parameters (`tensorflow-basics/README.md`); replacing it with
  `GlobalAveragePooling2D` in `final-project/` cut total parameters by more
  than half while *improving* accuracy.
- **Quantization needs real calibration data.** INT8 conversion requires a
  representative dataset of real samples, not random noise, to correctly
  measure activation ranges per layer (`tensorflow-lite/README.md`).
- **Measured numbers beat assumed numbers, repeatedly.** Several results in
  this repo contradicted a reasonable-sounding prior assumption: blurring
  before edge detection *increased* edge pixel count on a clean synthetic
  image (`opencv-experiments/README.md`); dynamic-range quantization beat
  full INT8 on latency (`tensorflow-lite/README.md`); an unscaled logistic
  regression model converged to nearly the same accuracy as a scaled one
  but threw a real `ConvergenceWarning` the scaled version didn't
  (`scikit-learn-practice/README.md`). Each was investigated and reported
  honestly rather than adjusted to match the expected story.
- **Honesty about what wasn't tested matters as much as what was.** No
  physical Raspberry Pi was available; `raspberry-pi/README.md` states
  plainly which parts were verified on this development machine (real
  webcam, real image predictions) versus written correctly per
  documentation but never run on ARM hardware.

## Tech stack

Python 3.13 · NumPy · Pandas · Matplotlib · scikit-learn · TensorFlow /
Keras · OpenCV · TensorFlow Lite · Jupyter

## References

See `reports/final-report.md` §9 for the full reference list (TensorFlow,
TensorFlow Lite, OpenCV, and scikit-learn documentation; Fashion-MNIST,
MNIST, California Housing, and Iris dataset citations).
