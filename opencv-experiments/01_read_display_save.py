"""
imread/imwrite, and the classic BGR vs RGB bug.

Why this matters: OpenCV loads images as BGR, not RGB. Every other library in
this repo (matplotlib, TensorFlow, PIL) expects RGB. Forgetting to convert is
the single most common bug when mixing OpenCV with anything else -- colors
look subtly (or wildly) wrong, not crash-wrong, which makes it sneaky.
"""

import os
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "sample.jpg")

if not os.path.exists(SAMPLE_PATH):
    raise FileNotFoundError(f"Run generate_sample_image.py first to create {SAMPLE_PATH}")

# --- imread: loads as BGR by default (OpenCV's historical quirk) ---
img_bgr = cv2.imread(SAMPLE_PATH)
print("Loaded with cv2.imread():")
print("  shape:", img_bgr.shape, " dtype:", img_bgr.dtype)
print("  pixel at (10,10) in BGR order:", img_bgr[10, 10])

# --- The bug: saving/displaying a BGR array as if it were RGB swaps red and blue ---
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
print("\nAfter cv2.cvtColor(BGR2RGB):")
print("  pixel at (10,10) in RGB order:", img_rgb[10, 10])
print("  (red and blue channels are swapped compared to the BGR read above)")

# --- imwrite: also expects BGR -- writing an RGB array with imwrite silently swaps colors ---
wrong_path = os.path.join(HERE, "01_wrong_colors_demo.jpg")
correct_path = os.path.join(HERE, "01_correct_colors_demo.jpg")

cv2.imwrite(wrong_path, img_rgb)     # WRONG: img_rgb is RGB, imwrite expects BGR -- colors will be off
cv2.imwrite(correct_path, img_bgr)   # correct: img_bgr is already BGR, as imwrite expects

print(f"\nSaved (deliberately wrong colors, RGB array passed to a BGR-expecting function): {wrong_path}")
print(f"Saved (correct colors): {correct_path}")
print("\nOpen both files side by side -- the orange circle and navy background shift noticeably in the 'wrong' one.")
