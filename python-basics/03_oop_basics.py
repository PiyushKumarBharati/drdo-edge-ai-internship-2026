"""
OOP basics: a small Dataset wrapper class.

Why this matters: every ML framework (PyTorch's Dataset, tf.data) is built on
exactly this pattern -- something with __len__ and __getitem__ so it can be
looped over and indexed like a list, regardless of what's actually behind it.
"""


class Dataset:
    """Wraps a list of (feature, label) pairs and behaves like a sequence."""

    def __init__(self, samples, labels):
        if len(samples) != len(labels):
            raise ValueError("samples and labels must be the same length")
        self.samples = samples
        self.labels = labels

    def __len__(self):
        # Lets us write len(dataset) instead of len(dataset.samples).
        return len(self.samples)

    def __getitem__(self, index):
        # Lets us write dataset[i] and also enables iteration (for x in dataset).
        return self.samples[index], self.labels[index]

    def class_counts(self):
        """How many samples belong to each class -- the first thing to check
        for class imbalance before training anything."""
        counts = {}
        for label in self.labels:
            counts[label] = counts.get(label, 0) + 1
        return counts


class BatchedDataset(Dataset):
    """Subclass demonstrating inheritance: adds batching on top of Dataset."""

    def __init__(self, samples, labels, batch_size):
        super().__init__(samples, labels)
        self.batch_size = batch_size

    def batches(self):
        """Yield (feature_batch, label_batch) tuples of size <= batch_size."""
        for start in range(0, len(self), self.batch_size):
            end = start + self.batch_size
            yield self.samples[start:end], self.labels[start:end]


if __name__ == "__main__":
    # Fake tiny "feature vectors" (normally these would be image arrays).
    features = [[0.1, 0.2], [0.4, 0.1], [0.9, 0.8], [0.3, 0.3], [0.7, 0.6]]
    labels = ["cat", "dog", "dog", "cat", "dog"]

    ds = Dataset(features, labels)
    print("Dataset length:", len(ds))
    print("First item (ds[0]):", ds[0])
    print("Class counts:", ds.class_counts())

    print("\nIterating with a for loop (uses __getitem__ under the hood):")
    for i, (feature, label) in enumerate(ds):
        print(f"  sample {i}: feature={feature}, label={label}")

    print("\nBatchedDataset with batch_size=2:")
    batched = BatchedDataset(features, labels, batch_size=2)
    for batch_idx, (feature_batch, label_batch) in enumerate(batched.batches()):
        print(f"  batch {batch_idx}: features={feature_batch}, labels={label_batch}")
