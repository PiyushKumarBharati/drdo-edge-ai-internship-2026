"""
tf.constant, tf.Variable, shapes, dtype, and NumPy interop.

Why this matters: a Keras model is a graph of tensor operations. Before any
training, you need to be fluent in the difference between a constant (fixed,
e.g. input data) and a Variable (mutable, e.g. trainable weights).
"""

import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# --- tf.constant: immutable, e.g. a fixed input tensor ---
x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
print("\nx (tf.constant):\n", x)
print("x.shape:", x.shape, " x.dtype:", x.dtype)

# --- tf.Variable: mutable, e.g. a trainable weight ---
w = tf.Variable(tf.random.normal((3, 2), seed=42))
print("\nw (tf.Variable), shape:", w.shape)
print(w.numpy())

# Variables can be updated in place -- this is literally what an optimizer does each step.
print("\nBefore assign_add:", w[0, 0].numpy())
w.assign_add(tf.ones((3, 2)) * 0.1)
print("After assign_add(+0.1):", w[0, 0].numpy())

# --- Basic tensor ops ---
y = x @ w  # matrix multiply: (2,3) @ (3,2) -> (2,2)
print("\nx @ w, shape:", y.shape)
print(y.numpy())

# --- dtype matters: mixing float32 and float64 raises, must cast explicitly ---
a = tf.constant([1.0, 2.0], dtype=tf.float32)
b = tf.constant([1.0, 2.0], dtype=tf.float64)
try:
    a + b
except tf.errors.InvalidArgumentError as e:
    print("\nExpected failure mixing dtypes without casting:")
    print(" ", str(e).splitlines()[0])

b_cast = tf.cast(b, tf.float32)
print("\nAfter casting b to float32, a + b_cast:", (a + b_cast).numpy())

# --- NumPy interop: tensors convert to/from numpy arrays freely ---
np_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
tensor_from_numpy = tf.constant(np_array)
back_to_numpy = tensor_from_numpy.numpy()

print("\nnumpy -> tensor -> numpy round-trip matches:", np.array_equal(np_array, back_to_numpy))
print("type of tensor_from_numpy:", type(tensor_from_numpy))
print("type of .numpy() result:", type(back_to_numpy))

# --- Reshaping, a very common operation before feeding data into a model ---
flat = tf.constant(tf.range(12))
reshaped = tf.reshape(flat, (3, 4))
print("\nflat shape:", flat.shape, "-> reshaped to:", reshaped.shape)
print(reshaped.numpy())
