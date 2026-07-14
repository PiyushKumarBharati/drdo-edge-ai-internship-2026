"""
Time a pure-Python loop vs a vectorized NumPy op on the same computation.

Why this matters: this is the single most important habit for anyone doing ML
on constrained hardware. A Python for-loop over pixels is a real bottleneck on
a Pi; NumPy pushes the loop into compiled C. The speedup number printed here
is measured, not claimed.
"""

import time
import numpy as np

N = 2_000_000  # array size -- big enough that the timing difference is stable

# Same computation both ways: elementwise (x * 2.5 + 1) then sum.
x_list = list(range(N))
x_array = np.arange(N, dtype=np.float64)


def pure_python_version(values):
    total = 0.0
    for v in values:
        total += v * 2.5 + 1
    return total


def numpy_version(values):
    return np.sum(values * 2.5 + 1)


# --- Time the pure Python loop ---
start = time.perf_counter()
result_python = pure_python_version(x_list)
python_time = time.perf_counter() - start

# --- Time the vectorized NumPy version ---
start = time.perf_counter()
result_numpy = numpy_version(x_array)
numpy_time = time.perf_counter() - start

print(f"N = {N:,} elements")
print(f"Pure Python loop : {python_time:.4f} s   (result={result_python:,.1f})")
print(f"NumPy vectorized : {numpy_time:.4f} s   (result={result_numpy:,.1f})")

# Sanity check: both methods should agree closely (float rounding aside).
assert abs(result_python - result_numpy) < 1.0, "results diverged too much!"

speedup = python_time / numpy_time
print(f"\nMeasured speedup: {speedup:.1f}x faster with NumPy")
print("This is the real, measured number on this machine -- not a claimed one.")
print("On a Raspberry Pi (weaker CPU, no SIMD-heavy pipelines), the gap is")
print("typically even more important because every millisecond of latency budget counts.")
