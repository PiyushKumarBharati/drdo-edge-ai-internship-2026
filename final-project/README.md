# final-project

## Optimizing and Deploying a CNN Image Classifier on Edge Devices

The capstone of this repository: a complete pipeline from data to
deployment (training, TFLite conversion, benchmarking, single-image
inference), built as a standalone `src/` package rather than the
step-by-step scripts in `tensorflow-basics/` and `tensorflow-lite/`.

## Structure

```
final-project/
├── src/
│   ├── data.py        - load Fashion-MNIST, normalize, train/val/test split
│   ├── model.py        - CNN architecture (26,154 params, batch_norm on/off)
│   ├── train.py        - training loop, seeded, saves model + training curve
│   ├── convert.py      - float32 / dynamic-range / INT8 TFLite conversion
│   ├── benchmark.py    - size, latency (mean/std/p95), accuracy, per-class report
│   ├── qat.py           - quantization-aware training via tensorflow-model-optimization
│   ├── ablation.py      - batch-norm-removed variant, to test what actually causes the INT8 gap
│   └── infer.py         - image path in -> prediction out, via TFLite
├── models/              - cnn_classifier.keras + all TFLite variants (main, QAT, no-BN)
├── results/             - training curves, confusion matrices, benchmark tables/CSVs, demo images
│   └── ablation_no_batchnorm/  - the no-BN variant's own results, same layout
└── notebooks/
    └── walkthrough.ipynb - executed walkthrough, outputs saved
```

## Running it

```bash
python final-project/src/train.py       # trains, saves models/cnn_classifier.keras + results/training_curve.png
python final-project/src/convert.py     # produces models/model_{float32,dynamic_range,int8}.tflite
python final-project/src/benchmark.py   # produces results/ tables, chart, confusion matrix, per-class report
python final-project/src/qat.py          # quantization-aware training, adds a qat_int8 row to the benchmark
python final-project/src/ablation.py     # trains a no-BatchNorm variant, tests what the INT8 gap depends on
python final-project/src/infer.py final-project/results/test_sample_correct.png --model final-project/models/model_int8.tflite
```

Or open `notebooks/walkthrough.ipynb`, which loads the already-trained
artifacts and walks through each stage.

## Architecture

Three convolutional blocks (16→32→64 filters), each `Conv2D →
BatchNormalization → Activation("relu")`, the first two followed by
`MaxPooling2D`. Then `GlobalAveragePooling2D → Dense(32) → Dropout(0.3) →
Dense(10, softmax)`. 26,154 parameters total.

`GlobalAveragePooling2D` replaces the `Flatten → Dense` head that
`tensorflow-basics/README.md` found holding 90% of that model's parameters.
GAP has zero parameters, so the conv layers carry the representational load
instead.

`model.py`'s `build_model(batch_norm=True)` keeps the activation as its own
layer rather than fusing it into `Conv2D(activation="relu")`. That's not
cosmetic: `qat.py`'s quantization-aware training only works with activation
as a separate layer (see Discussion below), so the same builder function is
used everywhere in this folder, keeping one architecture instead of two.

## Training

Adam optimizer, sparse categorical cross-entropy, batch size 128, up to 40
epochs with `EarlyStopping(monitor="val_accuracy", patience=8,
restore_best_weights=True)`. `train.py` seeds Python's `random`, NumPy, and
`tf.random` (seed 42) before building or training the model. Reruns of
`train.py` on this machine reproduce the same epoch-by-epoch history and the
same final test accuracy.

The patience/epoch budget was widened from an earlier 4/8 after adding the
seed exposed a real problem. With a fixed seed, validation accuracy was
still climbing at epoch 25 in a diagnostic run, so the old patience=4 was
cutting training off before it converged: not a hypothetical risk, an
observed one. Validation accuracy is still noisy epoch to epoch (a
BatchNorm running-statistics effect noted below), which is why
`restore_best_weights=True` matters here.

**Test accuracy: 87.21%** (10,000-image Fashion-MNIST test set).

![training curve](results/training_curve.png)

## Benchmark

Same methodology throughout: accuracy over the full 10,000-image test set;
latency as the mean, std, and p95 of 200 timed single-image calls after 20
discarded warmup calls.

| Model | Size (KB) | Accuracy | Mean latency (ms) | Std (ms) | P95 (ms) |
|---|---|---|---|---|---|
| float32 | 105.86 | 0.8721 | 0.0694 | 0.0014 | 0.0725 |
| dynamic_range | 33.96 | 0.8723 | 0.0353 | 0.0011 | 0.0357 |
| int8 | 35.18 | 0.8722 | 0.0461 | 0.0013 | 0.0495 |
| qat_int8 | 33.81 | 0.9091 | 0.0471 | 0.0012 | 0.0479 |

![comparison chart](results/comparison_chart.png)

Post-training INT8 quantization costs almost nothing here: 87.21% to 87.22%.
That's a different result from what this README used to say (a ~4.2
percentage point drop). See Discussion below for why, and what changed.

`results/classification_reports.txt` has full per-class precision/recall/F1
for every variant (`sklearn.metrics.classification_report`). The weak class
throughout is **Shirt** (recall around 74%, most often confused with
T-shirt/top, Pullover, and Coat), visually the hardest category at 28×28
grayscale. This holds across every quantization level; it isn't something
quantization introduces.

## Confusion matrix (float32)

![confusion matrix](results/confusion_matrix.png)

Shirt, T-shirt/top, Pullover, and Coat cluster together. Trouser, Sandal,
and Bag are almost never confused with anything else.

## Quantization-aware training (`qat.py`)

Real QAT, using `tensorflow-model-optimization` 0.8.1. Two things had to be
worked out first, both found by actually trying it rather than assumed from
the package's docs:

1. `tensorflow-model-optimization` needs the standalone `tf_keras` package
   and `TF_USE_LEGACY_KERAS=1` to import at all under TensorFlow 2.21 (which
   defaults to Keras 3). Without it, `import tensorflow_model_optimization`
   raises `ImportError: Keras cannot be imported`. Both are now real
   dependencies in `requirements.txt`.
2. `tfmot.quantization.keras.quantize_model()` fails with `RuntimeError:
   Layer batch_normalization:<...> is not supported` when a Conv2D's
   activation is fused (`Conv2D(activation="relu")`). It only recognizes the
   Conv → BatchNorm → Activation fusion pattern when activation is its own
   layer, which is why `model.py` builds it that way everywhere, not just
   for QAT.

Because `qat.py` runs in a separate process with the legacy Keras compat
layer active, it can't load `models/cnn_classifier.keras` (saved by Keras 3,
a different serialization format). It trains its own float32 baseline
first, same architecture/seed/hyperparameters as `train.py`, then applies
`quantize_model()`, fine-tunes for 5 more epochs, and converts to INT8. Its
own baseline reached 89.06%, not identical to `train.py`'s 87.21% despite
the same seed: Keras 3 and legacy `tf_keras` don't consume randomness
identically for the same model, so the same seed doesn't mean the same run
across the two backends.

**Result: 90.91% INT8 accuracy**, the best of any variant in the table
above, and higher than its own float32 starting point. Read that carefully:
this isn't a clean "QAT recovered the drop" story, because there was barely
a drop to recover in the plain post-training INT8 model already (87.21% to
87.22%). Part of the QAT number is likely just 5 extra epochs of training
rather than quantization-awareness specifically, a confound I can't fully
separate out without training a float32 model for 5 more epochs under the
same legacy-Keras process for comparison, which wasn't done here. What's
solid is that QAT ran successfully end to end and produced a real,
verified, high-accuracy INT8 model.

## Testing the BatchNorm hypothesis (`ablation.py`)

This README used to assert that BatchNormalization caused the INT8 accuracy
drop, without testing it. `ablation.py` actually tests it: same
architecture, same seed, same training budget, `batch_norm=False`.

| Model | Accuracy | INT8 delta |
|---|---|---|
| with BatchNorm (main model) | 87.21% float32 → 87.22% int8 | +0.01pp |
| without BatchNorm | 89.37% float32 → 89.48% int8 | +0.11pp |

Removing BatchNorm didn't reveal anything, because the with-BatchNorm model
already had no drop to shrink. That's evidence against the original
hypothesis, not for it. Full results and per-class numbers are in
`results/ablation_no_batchnorm/`.

So where did the originally-reported ~4.2pp drop actually come from? To
check, I retrained the *original* architecture (`Conv2D(activation="relu")`
fused, not split) with the same seed. It reproduced a real gap: 90.60%
float32 to 87.89% int8 (2.71pp), concentrated almost entirely in **Coat**
(-13.3pp) and **Shirt** (-13.9pp), the same visually-confused cluster called
out above. The split-activation architecture (`model.py`'s current form,
needed for QAT) doesn't show this gap, with or without BatchNorm.

That points at something more specific than "BatchNorm causes quantization
error." It looks like whether Conv2D's activation is fused or a separate
layer changes how TFLite's converter fuses and quantizes the
Conv+BatchNorm+ReLU sequence, and that, not BatchNorm's presence by itself,
is what the accuracy drop actually tracked. I haven't confirmed the exact
mechanism inside the converter, so this is a reported correlation, not a
proven cause. It would be worth a real investigation rather than another
assumption.

## Single-image inference (`infer.py`)

| Image | True label | float32 | dynamic_range | int8 |
|---|---|---|---|---|
| ![clean](results/test_sample_correct.png) | Ankle boot | Ankle boot (0.9989) | Ankle boot (0.9990) | Ankle boot (0.9961) |
| ![ambiguous](results/test_sample_ambiguous.png) | Dress | Coat (0.4023) | Coat (0.4081) | Coat (0.3789) |

All three variants agree on both images: correct on the first, wrong on the
second, all under 50% confidence on the wrong one. The retrained model's
predictions no longer split by quantization format on the ambiguous example
the way an earlier version of this model did. Genuinely uncertain, not
staged either way.

## Why this matters for Edge AI

The real lesson here isn't "BatchNorm is dangerous for quantization"; the
ablation shows that isn't well supported. It's that a small, easy-to-miss
architectural detail (fused vs. separate activation layers) can determine
whether INT8 quantization is free or costs several accuracy points, and the
only way to know which one you're getting is to measure it on the actual
model you're shipping, the way `benchmark.py` and `ablation.py` do here.
`raspberry-pi/README.md` covers the separate question of what carries over
to real ARM hardware versus what's specific to this x86 laptop.

## Common mistakes / gotchas

- `EarlyStopping(restore_best_weights=True)` matters more with
  BatchNormalization in the architecture. Its running statistics make early
  epochs noisier, so the last epoch is less likely to be the best one.
- A fixed random seed doesn't just make results reproducible, it can also
  expose that your hyperparameters were tuned against a different, lucky
  random run. This repo's `patience=4/EPOCHS=8` only looked adequate before
  seeding was added.
- Two Keras backends (Keras 3 vs. the legacy `tf_keras` package used by
  `qat.py`) don't consume the same random seed identically for the same
  model. A fixed seed gives reproducibility *within* a backend, not
  identical results *across* backends.
- `tf.keras.models.load_model()` can't load a Keras-3-saved `.keras` file
  from a process running `TF_USE_LEGACY_KERAS=1`, which is why `qat.py`
  trains its own baseline instead of loading `cnn_classifier.keras`.
- The confusion matrix and classification report are generated from the
  **float32** model specifically. `benchmark.py` keeps every variant's
  predictions, but float32 is the reference for what the model actually
  gets confused about, independent of quantization.
- `notebooks/walkthrough.ipynb` does not retrain anything. It loads the
  already-trained artifacts from `models/` and `results/`, so every number
  it displays comes from `src/`'s real execution, not retyped by hand.
