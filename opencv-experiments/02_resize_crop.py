"""
Resize with different interpolation methods, and aspect-ratio-aware cropping.

Why this matters: every model has a fixed input size (e.g. 224x224). Getting
from an arbitrary camera frame to that fixed size, without distorting the
image or picking a bad interpolation method, is the first step of every
inference pipeline in this repo.
"""

import os
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(HERE, "sample.jpg")
img = cv2.imread(SAMPLE_PATH)
print("Original shape:", img.shape)

# --- Naive resize: stretches the image if target aspect ratio differs ---
target_size = (128, 128)  # (width, height) -- note cv2.resize takes (W, H), NOT (H, W)
stretched = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
print(f"Naive resize to {target_size}: shape={stretched.shape} (distorted, since 320x240 != square)")

# --- Different interpolation methods, same target size ---
nearest = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)  # fast, blocky
linear = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)    # smooth, good default
area = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)        # best for shrinking

cv2.imwrite(os.path.join(HERE, "02_resize_nearest.jpg"), nearest)
cv2.imwrite(os.path.join(HERE, "02_resize_linear.jpg"), linear)
cv2.imwrite(os.path.join(HERE, "02_resize_area.jpg"), area)
print("\nSaved resize comparisons: 02_resize_nearest.jpg, 02_resize_linear.jpg, 02_resize_area.jpg")

# --- Aspect-ratio-preserving resize: pad instead of stretch ---
def resize_with_padding(image, target_w, target_h):
    """Resize keeping aspect ratio, then pad with black to hit the exact target size.
    This is what you actually want before feeding a model -- stretching distorts
    the shapes a CNN learned to recognize."""
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to center the resized image within the target canvas.
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded


padded = resize_with_padding(img, 128, 128)
cv2.imwrite(os.path.join(HERE, "02_resize_padded.jpg"), padded)
print(f"Saved aspect-ratio-preserving padded resize: 02_resize_padded.jpg, shape={padded.shape}")

# --- Cropping: a center crop, the other common way to reach a square input ---
h, w = img.shape[:2]
crop_size = min(h, w)
top = (h - crop_size) // 2
left = (w - crop_size) // 2
center_crop = img[top:top + crop_size, left:left + crop_size]
cv2.imwrite(os.path.join(HERE, "02_center_crop.jpg"), center_crop)
print(f"Saved center crop: 02_center_crop.jpg, shape={center_crop.shape}")
