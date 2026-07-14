"""
Array creation, dtype, shape -- and the float32 vs float64 memory story.

Why this matters: this script is the seed of the whole quantization story in
tensorflow-lite/. A model's weights are just a big NumPy array under the hood.
Halving the bytes per weight (float64 -> float32 -> int8) is literally what
quantization does, and the memory numbers here show why it matters.
"""

import numpy as np

# --- Creation ---
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3, 4))
c = np.ones((2, 2, 3))
d = np.arange(0, 10, 2)          # start, stop, step
e = np.linspace(0, 1, 5)         # 5 evenly spaced points between 0 and 1

print("a:", a, "shape:", a.shape, "dtype:", a.dtype)
print("b shape:", b.shape, "dtype:", b.dtype)
print("c shape:", c.shape)
print("d:", d)
print("e:", e)

# --- dtype is chosen automatically, but can be forced ---
int_array = np.array([1, 2, 3])
float_array = np.array([1.0, 2.0, 3.0])
forced_float32 = np.array([1, 2, 3], dtype=np.float32)

print("\nint_array dtype:", int_array.dtype)
print("float_array dtype:", float_array.dtype)
print("forced_float32 dtype:", forced_float32.dtype)

# --- The memory story: same values, different dtypes ---
n_weights = 1_000_000  # roughly the size of a small dense layer's weights
rng = np.random.default_rng(seed=42)
weights_float64 = rng.standard_normal(n_weights)              # numpy default
weights_float32 = weights_float64.astype(np.float32)          # what TF trains with by default
weights_int8 = (weights_float64 * 127).astype(np.int8)        # what INT8 quantization stores

print(f"\n{n_weights:,} weights stored as:")
for name, arr in [
    ("float64", weights_float64),
    ("float32", weights_float32),
    ("int8", weights_int8),
]:
    mb = arr.nbytes / (1024 ** 2)
    print(f"  {name:8s}: dtype={str(arr.dtype):10s} nbytes={arr.nbytes:>10,} ({mb:.2f} MB)")

ratio_64_to_32 = weights_float64.nbytes / weights_float32.nbytes
ratio_32_to_8 = weights_float32.nbytes / weights_int8.nbytes
print(f"\nfloat64 -> float32 shrinks storage by {ratio_64_to_32:.1f}x")
print(f"float32 -> int8 shrinks storage by {ratio_32_to_8:.1f}x")
print("This exact shrink (per weight, times millions of weights) is why a")
print("quantized .tflite model file is smaller -- see tensorflow-lite/04_benchmark.py")
