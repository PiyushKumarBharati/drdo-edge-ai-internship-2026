"""
Multi-panel figures with plt.subplots, sharing axes, and a shared legend.

Why this matters: benchmark reports need side-by-side comparisons (size vs
latency vs accuracy) in one figure, not three separate images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

rng = np.random.default_rng(seed=2)

# --- 1x3 row of subplots sharing the y-axis, e.g. train/val/test comparisons ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

epochs = np.arange(1, 16)
for ax, name, start_acc, noise in zip(
    axes, ["Train accuracy", "Validation accuracy", "Test accuracy (final only)"], [0.55, 0.5, None], [0.02, 0.03, None]
):
    if name.startswith("Test"):
        # Test accuracy is usually a single final number, not a curve -- show it as a flat line for contrast.
        ax.axhline(0.87, color="tab:green", linestyle="--", label="final test accuracy")
        ax.set_title(name)
        ax.legend()
    else:
        curve = 1 - (1 - start_acc) * np.exp(-0.2 * epochs) + rng.normal(0, noise, len(epochs))
        curve = np.clip(curve, 0, 1)
        ax.plot(epochs, curve, color="tab:blue")
        ax.set_title(name)
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("accuracy")
fig.suptitle("Example subplot layout (illustrative curves, not a real training run)")
fig.tight_layout()

out_path = os.path.join(HERE, "02_subplots.png")
fig.savefig(out_path, dpi=120)
print(f"Saved {out_path}")

# --- A 2x1 layout with different plot types stacked, sharing the x-axis ---
fig2, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

x = np.linspace(0, 4 * np.pi, 200)
ax_top.plot(x, np.sin(x), label="sin(x)", color="tab:blue")
ax_top.plot(x, np.cos(x), label="cos(x)", color="tab:orange")
ax_top.legend()
ax_top.set_title("Two curves sharing the same x-axis as the plot below")
ax_top.grid(True, alpha=0.3)

ax_bottom.fill_between(x, np.sin(x) * np.cos(x), color="tab:green", alpha=0.5)
ax_bottom.set_title("sin(x) * cos(x)")
ax_bottom.set_xlabel("x")
ax_bottom.grid(True, alpha=0.3)

fig2.tight_layout()
out_path2 = os.path.join(HERE, "02_subplots_shared_x.png")
fig2.savefig(out_path2, dpi=120)
print(f"Saved {out_path2}")
