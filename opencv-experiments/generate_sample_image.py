"""
Generate a small synthetic sample image and commit it to the repo.

Why synthetic and not downloaded: it keeps the repo self-contained (no
internet dependency to reproduce results) and license-free. It's built from
simple shapes on purpose -- enough structure (edges, color blocks, gradient)
to make resize/crop/color/edge-detection demos meaningful, without needing an
external dataset.

Run this once; sample.jpg is committed so later scripts don't need to
regenerate it.
"""

import os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "sample.jpg")

HEIGHT, WIDTH = 240, 320

# Start with a horizontal gradient background (dark blue -> light blue), BGR order for OpenCV.
img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
for x in range(WIDTH):
    t = x / (WIDTH - 1)
    img[:, x] = [int(150 + 100 * t), int(80 + 60 * t), int(30 + 20 * t)]  # B, G, R

# A filled red circle (BGR: red = (0, 0, 255)) -- gives edge detection something round to find.
cv2.circle(img, center=(90, 120), radius=55, color=(30, 30, 220), thickness=-1)

# A filled green rectangle -- gives it something with straight edges/corners.
cv2.rectangle(img, pt1=(180, 60), pt2=(280, 160), color=(40, 160, 40), thickness=-1)

# A white filled triangle -- one more distinct shape / edge case for contour-ish demos.
triangle_pts = np.array([[230, 200], [300, 200], [265, 130]], dtype=np.int32)
cv2.fillPoly(img, [triangle_pts], color=(230, 230, 230))

# A bit of text -- useful for eyeballing that BGR/RGB conversion didn't corrupt anything.
cv2.putText(img, "EDGE-AI SAMPLE", (55, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imwrite(OUT_PATH, img)
print(f"Wrote synthetic sample image to {OUT_PATH}, shape={img.shape}, dtype={img.dtype}")
