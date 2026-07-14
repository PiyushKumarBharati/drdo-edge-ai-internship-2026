"""
List comprehensions and generators: readability + memory efficiency.

Why this matters on constrained devices: a list comprehension builds the WHOLE
result in memory at once. A generator produces one item at a time. On a Pi
with limited RAM, looping over thousands of images as a generator instead of
a list can be the difference between running and crashing (OOM).
"""

import sys

# --- Basic comprehension: square every number ---
numbers = list(range(10))
squares = [n ** 2 for n in numbers]
print("numbers:", numbers)
print("squares:", squares)

# --- Comprehension with a filter condition ---
even_squares = [n ** 2 for n in numbers if n % 2 == 0]
print("even_squares:", even_squares)

# --- Dict comprehension: build a filename -> label map from two lists ---
filenames = ["img_0.jpg", "img_1.jpg", "img_2.jpg"]
labels = ["cat", "dog", "cat"]
file_label_map = {fname: label for fname, label in zip(filenames, labels)}
print("file_label_map:", file_label_map)

# --- Nested comprehension: flatten a batch of (mini-batch) label lists ---
batches = [["cat", "dog"], ["cat", "cat"], ["dog"]]
flat_labels = [label for batch in batches for label in batch]
print("flat_labels:", flat_labels)

# --- Memory comparison: list comprehension vs generator expression ---
N = 1_000_000

list_version = [n * 2 for n in range(N)]          # fully materialized in memory
generator_version = (n * 2 for n in range(N))      # produces values lazily, one at a time

list_bytes = sys.getsizeof(list_version)
gen_bytes = sys.getsizeof(generator_version)

print(f"\nN = {N:,} elements")
print(f"list comprehension size in memory: {list_bytes:,} bytes")
print(f"generator expression size in memory: {gen_bytes:,} bytes")
print(f"list uses ~{list_bytes / gen_bytes:,.0f}x more memory than the generator object")

# The generator still produces every value, just one at a time on demand.
total_from_generator = sum(n * 2 for n in range(N))
total_from_list = sum(list_version)
print(f"\nsum via generator: {total_from_generator:,}")
print(f"sum via list:      {total_from_list:,}")
assert total_from_generator == total_from_list
print("Both approaches produce the same result -- generator just doesn't pay the memory cost upfront.")
