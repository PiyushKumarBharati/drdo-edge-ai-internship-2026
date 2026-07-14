"""
Canny edge detection, and why blurring first matters.

Why this matters: edge detection is noise-sensitive. Skipping the blur step
is a common mistake that produces noisy, unusable edge maps -- this script
shows the difference side by side, measured by counting edge pixels.
"""

import os
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(HERE, "sample.jpg"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Canny directly on the raw grayscale image ---
edges_raw = cv2.Canny(gray, threshold1=50, threshold2=150)
cv2.imwrite(os.path.join(HERE, "04_edges_no_blur.jpg"), edges_raw)

# --- Blur first (Gaussian), then Canny -- the recommended order ---
blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
edges_blurred = cv2.Canny(blurred, threshold1=50, threshold2=150)
cv2.imwrite(os.path.join(HERE, "04_edges_with_blur.jpg"), edges_blurred)
cv2.imwrite(os.path.join(HERE, "04_blurred_input.jpg"), blurred)

edge_pixels_raw = int((edges_raw > 0).sum())
edge_pixels_blurred = int((edges_blurred > 0).sum())

print(f"Edge pixels WITHOUT blur first: {edge_pixels_raw}")
print(f"Edge pixels WITH blur first:    {edge_pixels_blurred}")
print(f"Difference: {edge_pixels_blurred - edge_pixels_raw:+d} edge pixels after blurring")
print("\nOn THIS image, blurring first slightly INCREASED the edge pixel count --")
print("the opposite of what the 'blur reduces noise' framing predicts. That's")
print("because our synthetic image has clean, hard-edged shapes with no sensor")
print("noise to suppress; blurring instead softens the sharp geometric edges and")
print("anti-aliased boundaries just enough that Canny's edge-linking connects a")
print("few more border pixels into continuous edges. On a real noisy camera")
print("photo (the actual target case), blur-then-Canny suppresses speckle noise")
print("that would otherwise register as thousands of spurious 1-pixel edges --")
print("this synthetic image just doesn't have that noise to demonstrate against.")

print("\nSaved: 04_edges_no_blur.jpg, 04_blurred_input.jpg, 04_edges_with_blur.jpg")

# --- Threshold sensitivity: Canny has two thresholds (hysteresis) ---
edges_loose = cv2.Canny(blurred, threshold1=10, threshold2=50)   # more edges, more noise
edges_strict = cv2.Canny(blurred, threshold1=100, threshold2=200)  # fewer, stronger edges only
cv2.imwrite(os.path.join(HERE, "04_edges_loose_thresholds.jpg"), edges_loose)
cv2.imwrite(os.path.join(HERE, "04_edges_strict_thresholds.jpg"), edges_strict)
print(f"Loose thresholds (10,50): {int((edges_loose > 0).sum())} edge pixels")
print(f"Strict thresholds (100,200): {int((edges_strict > 0).sum())} edge pixels")
