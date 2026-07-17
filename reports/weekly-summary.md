# Topic-wise Summary

A topic-by-topic summary of everything covered in this internship, organized
by subject area (matching the repository's folder structure) rather than by
calendar date.

## Python fundamentals (`python-basics/`)

Core language patterns reused throughout the rest of the repo: data
structures for label maps and configs, file I/O without pandas, a `Dataset`
class with `__len__`/`__getitem__` (the same shape as `tf.data`/PyTorch
`Dataset`), exception handling and logging for bad input, and comprehensions
vs. generators as a memory/eagerness tradeoff.

## NumPy (`numpy-practice/`)

Array creation, dtype, and shape; indexing/slicing/masking; broadcasting
rules; vectorization vs. pure-Python loops (measured speedup); treating an
image as a NumPy array (HWC layout, normalization, batching). The
float64→float32→int8 memory measurement here is the direct seed of the
quantization work later.

## Pandas (`pandas-practice/`)

Load/inspect/clean/aggregate/feature-prep workflow on the real California
Housing dataset (20,640 rows): missing value handling, dtype conversion,
`groupby`/`agg`, one-hot encoding, and train/test split discipline.

## Matplotlib (`matplotlib-practice/`)

Line/bar/scatter/histogram, multi-panel figures, image display with
colorbars, and a reusable `plot_training_curves()` function (demoed on a
real hand-written logistic regression run) that gets reused for every neural
network training run later in the repo.

## Scikit-learn (`scikit-learn-practice/`)

Linear regression (California Housing, R²=0.576), classification (Iris,
94.7–97.4% accuracy), why feature scaling matters (a real convergence
warning, not just an accuracy number), and a measured overfitting
demonstration (DecisionTree train/test accuracy gap vs. `max_depth`).

## TensorFlow fundamentals (`tensorflow-basics/`)

Tensors vs. NumPy arrays, a dense network on MNIST (97.16% test accuracy), a
small CNN on Fashion-MNIST (87.45% test accuracy), and model inspection,
finding that a single `Flatten`→`Dense` layer held 90.4% of that model's
parameters, which directly informed the final project's architecture choice.

## OpenCV (`opencv-experiments/`)

Image read/write and the BGR-vs-RGB bug; resize/crop with different
interpolation methods; grayscale/thresholding/color masking; edge detection
(Canny) and why blur-before-edges matters; the `preprocess_for_model()`
function reused verbatim in `raspberry-pi/` and `final-project/`; webcam
capture with graceful no-camera handling. Found and fixed a real bug during
this phase: an early version of the synthetic test image had color contrast
but almost no *luminance* contrast, which nearly defeated edge detection.

## TensorFlow Lite (`tensorflow-lite/`), the core topic

Converting a trained model to float32/dynamic-range/INT8 TFLite variants;
what a representative dataset is and why INT8 needs one; running inference
with `tf.lite.Interpreter`, including the int8 quantize/dequantize step;
building a real benchmark (size, latency, accuracy) rather than relying on
theoretical numbers. Core finding: ~3.55x size reduction for both quantized
formats, with a genuinely surprising latency result (dynamic-range quantization
was faster than full INT8 on this x86 CPU).

## Raspberry Pi deployment (`raspberry-pi/`)

Building an inference script against `tflite_runtime` with a `tf.lite`
fallback so the same code runs on both a development machine and (in
principle) a Pi; a live camera inference loop; exact Pi setup steps. No
physical Pi was available, so this topic's output is explicit about the
verified/unverified boundary rather than claiming hardware results that
don't exist.

## Mini-projects (`mini-projects/`)

Two complete small pipelines: a handwritten digit classifier (98.15%
accuracy, 20.35 KB TFLite model) and a Haar-cascade face detector (with a
real parameter-tuning story: default settings found zero faces on the test
photo, tuned settings found one). Building the face detector also surfaced
and fixed two real environment bugs: a newer `opencv-python` version missing
`cv2.CascadeClassifier` entirely, and an unreliable default camera backend
on Windows.

## Final project (`final-project/`)

The complete pipeline rebuilt as a standalone package: a smaller CNN
(26,154 params, using `GlobalAveragePooling2D` instead of a large dense
head) reaching 87.21% test accuracy with seeded, reproducible training,
about level with the earlier, larger `tensorflow-basics` CNN (87.45%) at
under half the parameter count. An earlier unseeded run of this same
architecture had reported 90.07%; that couldn't be reproduced once training
was made auditable, and isn't this model's reported accuracy anymore. The
INT8 accuracy drop this report used to attribute to batch normalization
(~4.2 points) also didn't hold up once actually tested: training the same
architecture with BatchNorm removed showed no change, because the
with-BatchNorm model's own drop had already shrunk to near zero after
retraining. What does reproduce a real ~2.7-point drop is the *original*
model code, before this repository's structure changed to support
quantization-aware training. See `final-project/README.md`'s Discussion
for the full investigation. Quantization-aware training
(`final-project/src/qat.py`, via `tensorflow-model-optimization`) reached
90.91% accuracy, the best of any variant benchmarked.
