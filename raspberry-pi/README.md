# raspberry-pi

## Purpose

Deployment-target scripts for running this repo's trained TFLite models on a
Raspberry Pi, plus exact setup steps.

## What was and wasn't verified on real hardware

No physical Raspberry Pi was available during this internship. Here's
exactly what that does and doesn't mean.

**Verified, on this development machine (Windows laptop, CPU only):**
- `deploy_inference.py` end to end: run against a real Fashion-MNIST test
  image (`sample_fashion_item.png`, true label "Ankle boot"), correctly
  predicted "Ankle boot" with 0.9883 confidence in ~0.14 ms (single-shot,
  no warmup, see below).
- `camera_inference.py`'s full loop: opened this machine's own webcam,
  classified 20 live frames in real time through the actual
  `model_int8.tflite` model, and released the camera cleanly.
- The `tflite_runtime` to `tensorflow.lite` import fallback. This machine
  doesn't have `tflite_runtime` installed, so every run above exercised the
  fallback path, not the primary one.
- All preprocessing logic (`preprocess_image`/`preprocess_frame`) is the
  same code path as `opencv-experiments/05_preprocess_for_model.py`, which
  is independently tested there.

**Not verified, written correctly per documentation but not run on physical
Pi hardware:**
- Anything in `setup_pi.md` (OS flash, `apt`/`pip` installs, `raspi-config`
  camera enable). These are the standard, documented steps, but they were
  never executed against a real device.
- The `tflite_runtime` import path itself. Never actually imported
  `tflite_runtime` on this machine, since it's a Pi-specific package this
  laptop doesn't need.
- Any latency or accuracy number on ARM hardware. Every millisecond figure
  in this repo, including `tensorflow-lite/04_benchmark.py`'s numbers, was
  measured on an x86 laptop CPU. `tensorflow-lite/README.md` found that
  dynamic-range quantization was faster than full INT8 on this x86 CPU;
  that may not hold on a Pi's ARM CPU, which has different int8/SIMD
  characteristics. No claim is made here about what the Pi's actual
  numbers would be.

## Files

| File | Description | Status |
|---|---|---|
| `deploy_inference.py` | Single-image inference: `tflite_runtime` with a `tf.lite` fallback, OpenCV preprocessing, prints predicted class + latency. | Verified on this machine (fallback path) |
| `camera_inference.py` | Same, looping over a live camera feed with on-frame overlay of prediction + latency. | Verified on this machine (fallback path, real webcam) |
| `setup_pi.md` | Exact OS flash, SSH, dependencies, deploy, run steps. | Written per documentation, not executed on hardware |

## How to run

```bash
# Single image (uses tensorflow-lite/model_int8.tflite by default)
python raspberry-pi/deploy_inference.py raspberry-pi/sample_fashion_item.png

# Live camera loop (press 'q' to quit, or set --max-frames for a headless run)
python raspberry-pi/camera_inference.py --max-frames 100
```

## Output from this machine

```
Interpreter backend: tensorflow.lite (tflite_runtime not installed on this machine)
Model: tensorflow-lite/model_int8.tflite

Image: raspberry-pi/sample_fashion_item.png
Predicted class: Ankle boot (index 9)
Confidence: 0.9883
Inference latency: 0.1372 ms
```

This is a single-shot measurement (no warmup runs, no averaging), so the
latency figure is noisier than `tensorflow-lite/04_benchmark.py`'s 200-run
average. Rerunning this exact command gives values in the 0.13-0.16 ms
range; the prediction and confidence are deterministic and reproduce
exactly.

(`sample_fashion_item.png` is a real Fashion-MNIST test-set image, true
label "Ankle boot," and the model got it right.)

## Why this matters for Edge AI

The whole premise of this internship's topic, on-device inference, no
network dependency, data locality, only pays off if the model actually runs
acceptably on the target hardware, not just on a development laptop. Being
explicit about the laptop/Pi boundary here is the correct scope of what a
research internship without physical hardware access can and cannot claim.
`setup_pi.md` gives the exact path to closing that gap once hardware is
available.

## Common mistakes / gotchas

- `tflite_runtime`'s `Interpreter` API is intentionally a subset of full
  TensorFlow's `tf.lite.Interpreter`. The two scripts here only use the
  overlapping subset (`allocate_tensors`, `get_input_details`,
  `get_output_details`, `set_tensor`, `invoke`, `get_tensor`) so the same
  code runs unmodified on both.
- `cv2.VideoCapture(0)` behaves differently across platforms. Index `0` is
  not guaranteed to be the same physical camera on a Pi with a CSI camera
  module vs. a USB webcam vs. this laptop's built-in camera. `setup_pi.md`
  covers enabling the CSI camera specifically.
- It would be easy to produce a plausible-looking Pi performance number by
  running the same laptop benchmark code and relabeling the output. Every
  number in this folder's README is either a real laptop measurement,
  clearly labeled as such, or explicitly marked unmeasured, precisely to
  avoid that.
