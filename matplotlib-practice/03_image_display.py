"""
imshow: displaying images, grayscale vs RGB, colorbars.

Why this matters: sanity-checking preprocessing visually (did the resize/crop/
normalize actually do what I think it did?) catches bugs that print statements
alone miss.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE_IMAGE_PATH = os.path.join(HERE, "..", "opencv-experiments", "sample.jpg")

if not os.path.exists(SAMPLE_IMAGE_PATH):
    raise FileNotFoundError(
        f"Expected sample image at {SAMPLE_IMAGE_PATH}. "
        "Run opencv-experiments/generate_sample_image.py first."
    )

img_rgb = mpimg.imread(SAMPLE_IMAGE_PATH)  # matplotlib reads as RGB

# Convert to grayscale using the standard luminance-weighted formula
# (matches how human eyes perceive brightness -- green contributes most).
gray = np.dot(img_rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].imshow(img_rgb)
axes[0].set_title(f"RGB image\nshape={img_rgb.shape}")
axes[0].axis("off")

im = axes[1].imshow(gray, cmap="gray")
axes[1].set_title(f"Grayscale\nshape={gray.shape}")
axes[1].axis("off")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="intensity")

# Show just the red channel with a colorbar, to make single-channel data concrete.
im2 = axes[2].imshow(img_rgb[:, :, 0], cmap="Reds")
axes[2].set_title("Red channel only")
axes[2].axis("off")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="red intensity")

fig.tight_layout()
out_path = os.path.join(HERE, "03_image_display.png")
fig.savefig(out_path, dpi=120)
print(f"Saved {out_path}")
print(f"RGB shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
print(f"Grayscale shape: {gray.shape}, dtype: {gray.dtype}")
