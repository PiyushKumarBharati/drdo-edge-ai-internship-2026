# Deploying to a Raspberry Pi

**Status: these are the exact steps to deploy this repo's models to a
Raspberry Pi, written from documentation and prior general Pi experience.
They have not been executed against physical Pi hardware during this
internship — no Pi was available. See `README.md` in this folder for exactly
what was and wasn't verified.**

## 1. Flash the OS

- Use Raspberry Pi Imager (https://www.raspberrypi.com/software/) on a
  desktop machine.
- Choose **Raspberry Pi OS Lite (64-bit)** — no desktop environment needed
  for a headless inference deployment, which keeps RAM free for the model.
- In the Imager's advanced options (gear icon / Ctrl+Shift+X), set:
  - hostname
  - enable SSH, set a password (or add an SSH public key)
  - configure Wi-Fi SSID/password if not using Ethernet
- Flash to a microSD card (16 GB+ recommended).

## 2. First boot and SSH

```bash
ssh <username>@<pi-hostname>.local
# or ssh <username>@<pi-ip-address> if .local resolution isn't working
```

If SSH wasn't enabled during flashing, it can be enabled by placing an empty
file named `ssh` in the boot partition of the SD card before first boot.

## 3. System dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv libatlas-base-dev libopenjp2-7 libtiff6
```

`libatlas-base-dev` provides BLAS routines NumPy/TFLite rely on for fast
linear algebra; `libopenjp2-7` and `libtiff6` are common OpenCV image-codec
dependencies on Raspberry Pi OS.

## 4. Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

## 5. Install tflite-runtime and OpenCV

```bash
pip install tflite-runtime
pip install opencv-python-headless   # "headless" build: no GUI deps needed on a headless Pi
pip install numpy
```

`tflite-runtime` is a small package containing just the TFLite interpreter —
a fraction of full TensorFlow's size and install time, which matters on a
Pi's SD card and modest CPU. `deploy_inference.py` and `camera_inference.py`
in this folder both import it with a fallback to full `tensorflow.lite`, so
the exact same script also runs on a development machine that has full
TensorFlow instead.

## 6. Copy the model and scripts to the Pi

From the development machine, in this repo's root:

```bash
scp tensorflow-lite/model_int8.tflite <username>@<pi-hostname>.local:~/
scp raspberry-pi/deploy_inference.py <username>@<pi-hostname>.local:~/
scp raspberry-pi/camera_inference.py <username>@<pi-hostname>.local:~/
```

(Or `scp -r` the whole repo if disk space allows — it's small.)

## 7. Run inference on a single image

```bash
scp path/to/test_image.jpg <username>@<pi-hostname>.local:~/
ssh <username>@<pi-hostname>.local
source venv/bin/activate
python3 deploy_inference.py test_image.jpg --model model_int8.tflite
```

## 8. Run live camera inference

For the Raspberry Pi Camera Module (CSI ribbon cable), enable the camera
interface first:

```bash
sudo raspi-config
# Interface Options -> Camera -> Enable, then reboot
```

Raspberry Pi OS's current camera stack (libcamera) exposes the CSI camera as
a standard V4L2 device, so `cv2.VideoCapture(0)` in `camera_inference.py`
should work without code changes. If it doesn't detect the camera, the
Picamera2 library (`sudo apt install python3-picamera2`) is the documented
fallback and would need a small adapter to feed frames into the same
`classify_frame()` function this script already has.

```bash
python3 camera_inference.py --model model_int8.tflite
```

## 9. What to actually measure once on real hardware

Repeat `tensorflow-lite/04_benchmark.py`'s latency methodology (warmup runs
discarded, mean over many timed calls) directly on the Pi, since CPU
architecture (ARM vs. x86) and available SIMD instructions materially change
which quantization format is actually fastest — see
`tensorflow-lite/README.md`'s discussion of why INT8 wasn't the fastest
format on this development laptop's x86 CPU. That result may not hold on
ARM.
