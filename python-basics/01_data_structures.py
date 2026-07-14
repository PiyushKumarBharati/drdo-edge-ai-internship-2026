"""
Data structures: lists, dicts, sets.

Why this matters later: an ML pipeline is full of exactly these three things --
a label map (dict), a batch of samples (list), and a set of classes seen so far
(set) -- long before any tensor shows up.
"""

# --- Lists: ordered, mutable. Used for a batch of file paths / samples. ---
image_paths = [
    "data/cat_001.jpg",
    "data/dog_001.jpg",
    "data/cat_002.jpg",
]
image_paths.append("data/dog_002.jpg")
print("Batch of image paths:", image_paths)
print("Number of samples:", len(image_paths))

# --- Dicts: the standard way to store a label map (class name -> integer id). ---
label_map = {
    "cat": 0,
    "dog": 1,
}
# Models output integers; humans want names. We need both directions.
inverse_label_map = {v: k for k, v in label_map.items()}
print("label_map:", label_map)
print("inverse_label_map:", inverse_label_map)

# A config dict is how most training scripts pass hyperparameters around.
config = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 5,
    "input_shape": (28, 28, 1),
}
print("config:", config)
# .get() with a default avoids a KeyError if an optional key is missing.
print("dropout (not set, default 0.0):", config.get("dropout", 0.0))

# --- Sets: unordered, unique. Used to find distinct classes in a dataset fast. ---
labels_seen_in_batch = ["cat", "dog", "cat", "cat", "dog"]
unique_classes = set(labels_seen_in_batch)
print("labels_seen_in_batch:", labels_seen_in_batch)
print("unique_classes:", unique_classes)
print("number of classes:", len(unique_classes))

# Set operations matter when comparing "classes in training set" vs "classes in test set".
train_classes = {"cat", "dog", "bird"}
test_classes = {"cat", "dog", "fish"}
print("classes only in test, missing from train:", test_classes - train_classes)
print("classes common to both:", train_classes & test_classes)
