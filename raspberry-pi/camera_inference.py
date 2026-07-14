"""
Loop over a live camera feed, running TFLite inference on each frame.

Written against cv2.VideoCapture (works on a laptop webcam AND, with
libcamera's V4L2 compatibility layer, a Raspberry Pi camera module) rather
than Picamera2 directly, so the exact same script is what would actually run
in both places -- one fewer thing to port later.

Honesty note: the inference call and preprocessing here have been run for
real on this development machine's webcam (see raspberry-pi/README.md for
which parts were actually verified). This script has NOT been run against a
physical Raspberry Pi camera, since no Pi was available during this
internship -- see setup_pi.md for how it would be deployed there.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# On Windows, OpenCV's default MSMF backend was unreliable in testing here
# (opened successfully but then failed to grab frames). CAP_DSHOW is the
# more reliable Windows backend; other platforms (e.g. the Pi's Linux/V4L2)
# use OpenCV's normal default by passing CAP_ANY.
_CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY

try:
    from tflite_runtime.interpreter import Interpreter
    RUNTIME = "tflite_runtime"
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    RUNTIME = "tensorflow.lite (tflite_runtime not installed on this machine)"

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(HERE, "..", "tensorflow-lite", "model_int8.tflite")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def preprocess_frame(frame, target_size=(28, 28)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    tensor = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)
    return tensor


def classify_frame(interpreter, input_details, output_details, frame):
    tensor = preprocess_frame(frame)

    is_int8_io = input_details["dtype"] == np.int8
    if is_int8_io:
        scale, zero_point = input_details["quantization"]
        tensor = (tensor / scale + zero_point).astype(np.int8)

    start = time.perf_counter()
    interpreter.set_tensor(input_details["index"], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    latency_ms = (time.perf_counter() - start) * 1000

    if is_int8_io:
        out_scale, out_zero_point = output_details["quantization"]
        output = (output.astype(np.float32) - out_zero_point) * out_scale

    predicted_index = int(np.argmax(output))
    confidence = float(output[predicted_index])
    return predicted_index, confidence, latency_ms


def run_camera_loop(model_path, camera_index=0, max_frames=None):
    """Runs live classification on camera frames until 'q' is pressed.
    Returns True if a camera was available and used, False if not (mirrors
    opencv-experiments/06_webcam_capture.py's graceful no-camera handling)."""
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    cap = cv2.VideoCapture(camera_index, _CAMERA_BACKEND)
    if not cap.isOpened():
        print(f"No camera available at index {camera_index}. Exiting cleanly instead of crashing.")
        return False

    print(f"Interpreter backend: {RUNTIME}")
    print("Camera opened. Press 'q' to quit.")
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read a frame -- stopping.")
                break

            predicted_index, confidence, latency_ms = classify_frame(
                interpreter, input_details, output_details, frame
            )
            label = CLASS_NAMES[predicted_index]
            text = f"{label} ({confidence:.2f}) {latency_ms:.1f}ms"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Live TFLite classification (press q to quit)", frame)

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                print(f"Reached max_frames={max_frames}, stopping automatically.")
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Released camera after {frame_count} frames.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live TFLite classification from a camera feed.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to a .tflite model.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=100,
                         help="Cap frames processed so this can run unattended/headless for verification.")
    args = parser.parse_args()

    camera_was_available = run_camera_loop(args.model, args.camera_index, args.max_frames)
    print(f"camera_was_available: {camera_was_available}")

    print("\nNOTE: this script's inference and preprocessing logic is identical to")
    print("deploy_inference.py, verified against real images on this development")
    print("machine. The camera loop itself has been run against this machine's own")
    print("webcam (see raspberry-pi/README.md) -- it has NOT been run against a")
    print("physical Raspberry Pi + camera module, since none was available.")
