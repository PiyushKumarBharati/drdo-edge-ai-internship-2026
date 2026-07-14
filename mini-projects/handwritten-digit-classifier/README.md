# mini-projects / handwritten-digit-classifier

## Purpose

A complete, small, standalone project: train a digit classifier on MNIST,
convert it to TFLite, and predict on single images end to end. Deliberately
self-contained (doesn't import from `tensorflow-basics/`) so this folder
would still work if copied out of the repo on its own.

## Files

| File | Description |
|---|---|
| `01_train_and_convert.py` | Trains a small CNN on MNIST (4 epochs), saves `.keras` and a dynamic-range quantized `.tflite`, and exports 5 real MNIST test images to `sample_digits/` for the next script to use. |
| `02_predict.py` | Loads the `.tflite` model, predicts on a single image (defaults to a `sample_digits/` image if none given), prints the predicted digit and confidence. |

## How to run

```bash
python mini-projects/handwritten-digit-classifier/01_train_and_convert.py
python mini-projects/handwritten-digit-classifier/02_predict.py                                    # uses a default sample
python mini-projects/handwritten-digit-classifier/02_predict.py path/to/any_digit_image.png         # or your own image
```

## Real results

Training (4 epochs, batch size 128, 10% validation split):

| Epoch | train acc | val acc |
|---|---|---|
| 1 | 0.8601 | 0.9648 |
| 2 | 0.9631 | 0.9752 |
| 3 | 0.9727 | 0.9792 |
| 4 | 0.9764 | 0.9817 |

**Final test accuracy: 0.9815** (9,815 / 10,000 correct on the real MNIST
test set). Model size: `.keras` = 205.59 KB, `.tflite` (dynamic-range
quantized) = 20.35 KB — **10.1x smaller**.

Prediction on 5 real MNIST test images, run through the actual `.tflite`
model via `02_predict.py`:

| Image | True label | Predicted | Confidence | Result |
|---|---|---|---|---|
| digit_348_true1.png | 1 | 1 | 0.9986 | CORRECT |
| digit_4729_true1.png | 1 | 1 | 0.9981 | CORRECT |
| digit_5116_true6.png | 6 | 6 | 1.0000 | CORRECT |
| digit_7550_true4.png | 4 | 4 | 0.9999 | CORRECT |
| digit_9503_true5.png | 5 | 5 | 0.9996 | CORRECT |

Sample digit (index 5116, true label 6, correctly predicted):

![sample digit](sample_digits/digit_5116_true6.png)

## Why this matters for Edge AI

This is the complete pipeline in miniature — train once on a desktop/laptop
with plenty of compute, convert once, then deploy a 20 KB file that runs
inference in milliseconds with no retraining ever happening on the target
device. That asymmetry (expensive training, cheap repeated inference) is the
whole reason edge deployment of a *pre-trained* model is practical on
hardware as modest as a Raspberry Pi.

## Common mistakes / gotchas

- `02_predict.py` reads images with `cv2.IMREAD_GRAYSCALE` directly instead
  of reading color and converting — simpler when you know in advance the
  target is always grayscale, but it would silently misbehave (three
  identical channels collapsed to one) if handed a color photo — worth
  checking `img.ndim` if this script is ever pointed at unknown input.
- The filename convention `digit_<index>_true<label>.png` is a
  debugging/verification convenience specific to `sample_digits/` — real
  deployment images obviously won't encode their own ground truth in the
  filename, which is why `02_predict.py` treats that parsing as optional
  (`if "true" in basename`).
