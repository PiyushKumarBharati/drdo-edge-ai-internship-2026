"""
cvtColor and thresholding.

Why this matters: many edge models (and almost all classical CV techniques
like Haar cascades, used in mini-projects/face-or-object-detector/) run on
grayscale input -- 1 channel instead of 3 is a direct 3x reduction in the
data a model or filter has to process.
"""

import os
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(HERE, "sample.jpg"))

# --- Grayscale conversion ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Color image shape:", img.shape)
print("Grayscale shape:", gray.shape, "(single channel)")
cv2.imwrite(os.path.join(HERE, "03_grayscale.jpg"), gray)

# --- Simple binary thresholding: every pixel becomes either 0 or 255 ---
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(HERE, "03_threshold_binary.jpg"), binary)
print("Saved binary threshold (fixed cutoff=127): 03_threshold_binary.jpg")

# --- Otsu's method: automatically picks the threshold from the image's histogram ---
otsu_value, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite(os.path.join(HERE, "03_threshold_otsu.jpg"), otsu_binary)
print(f"Saved Otsu threshold (auto-selected cutoff={otsu_value}): 03_threshold_otsu.jpg")

# --- Adaptive thresholding: different cutoff per local region, robust to uneven lighting ---
adaptive = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2
)
cv2.imwrite(os.path.join(HERE, "03_threshold_adaptive.jpg"), adaptive)
print("Saved adaptive threshold: 03_threshold_adaptive.jpg")

# --- Also demonstrate HSV, useful for color-based segmentation (e.g. isolate the orange circle) ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Our synthetic circle is bright orange, which sits around hue~15-25 on OpenCV's 0-179 H scale.
lower_orange = (10, 100, 100)
upper_orange = (25, 255, 255)
orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
cv2.imwrite(os.path.join(HERE, "03_orange_mask.jpg"), orange_mask)
orange_pixel_count = int((orange_mask > 0).sum())
print(f"Saved HSV-based orange color mask: 03_orange_mask.jpg ({orange_pixel_count} pixels matched)")
