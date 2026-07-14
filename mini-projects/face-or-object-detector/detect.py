"""
Face detection with OpenCV's Haar cascade classifier, on a still image and
(optionally) a live webcam feed.

Why Haar cascades: they're a classical (pre-deep-learning) technique --
trained offline on many positive/negative face examples, but inference is
just fast, cheap image convolutions. No GPU, no TFLite interpreter, works
in a few milliseconds on very modest hardware, which makes it a genuinely
useful edge-appropriate technique to know alongside the CNN approach used
elsewhere in this repo.

Cascade file: bundled with opencv-python itself at cv2.data.haarcascades --
a standard pretrained classical CV model shipped as part of the OpenCV
project (BSD licensed), not a randomly sourced file.
"""

import argparse
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

# On Windows, OpenCV's default MSMF backend was unreliable in testing here
# (opened successfully but then failed to grab frames). CAP_DSHOW is the
# more reliable Windows backend; other platforms use OpenCV's normal default.
_CAMERA_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def load_detector():
    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(f"Expected cascade file at {CASCADE_PATH}")
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    if detector.empty():
        raise RuntimeError(f"Failed to load cascade classifier from {CASCADE_PATH}")
    return detector


def detect_faces(detector, image_bgr, scale_factor=1.05, min_neighbors=3):
    """Runs Haar cascade detection on a BGR image, returns a list of (x, y, w, h) boxes.

    scale_factor: how much the image is shrunk at each scale step (smaller = more
    thorough/slower search across face sizes).
    min_neighbors: how many overlapping detections are required to keep a box
    (higher = fewer false positives, but also more false negatives).

    Defaults here (1.05 / 3) are deliberately less aggressive than the commonly
    quoted "default" of (1.1 / 5) -- that stricter setting produced ZERO
    detections on this repo's real test photo (glasses + backlit window behind
    the subject reduce contrast on the face), while (1.05 / 3) found it
    correctly. This was found by actually testing several parameter
    combinations against the real image, not assumed from documentation.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))
    return faces


def draw_boxes(image_bgr, faces):
    annotated = image_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return annotated


def run_on_image(image_path, out_path):
    detector = load_detector()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    faces = detect_faces(detector, img)
    annotated = draw_boxes(img, faces)
    cv2.imwrite(out_path, annotated)

    print(f"Image: {image_path}")
    print(f"Faces detected: {len(faces)}")
    for i, (x, y, w, h) in enumerate(faces):
        print(f"  face {i}: x={x}, y={y}, w={w}, h={h}")
    print(f"Saved annotated image to {out_path}")
    return faces


def run_on_webcam(max_frames=100, camera_index=0):
    """Same graceful no-camera handling as opencv-experiments/06_webcam_capture.py."""
    detector = load_detector()
    cap = cv2.VideoCapture(camera_index, _CAMERA_BACKEND)
    if not cap.isOpened():
        print(f"No camera available at index {camera_index}. Exiting cleanly instead of crashing.")
        return False

    print("Camera opened. Press 'q' to quit.")
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = detect_faces(detector, frame)
            annotated = draw_boxes(frame, faces)
            cv2.imshow("Face detection (press q to quit)", annotated)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=os.path.join(HERE, "test_photo.jpg"),
                         help="Path to an image to run detection on.")
    parser.add_argument("--webcam", action="store_true", help="Also run live webcam detection.")
    parser.add_argument("--max-frames", type=int, default=50)
    args = parser.parse_args()

    out_path = os.path.join(HERE, "detection_result.jpg")
    run_on_image(args.image, out_path)

    if args.webcam:
        print("\nRunning live webcam detection...")
        run_on_webcam(max_frames=args.max_frames)
