"""
CNN architecture for the final project.

Slightly deeper than tensorflow-basics/03_cnn_fashion_mnist.py's CNN (one
extra conv block, batch normalization) -- this is the "final" version after
having already seen that a smaller CNN gets ~87% test accuracy, this
architecture trades a modest parameter increase for better accuracy while
staying well within a size budget that converts to a small TFLite file.
"""

import tensorflow as tf


def build_model(input_shape=(28, 28, 1), num_classes=10, batch_norm=True):
    """Builds the final CNN. Conv2D's activation is applied as a separate
    Activation layer (not fused via Conv2D(activation=...)) so the exact same
    architecture builder works for both normal training (train.py) and
    quantization-aware training (qat.py) -- tfmot's quantize_model() only
    recognizes the Conv2D -> BatchNorm -> Activation fusion pattern when
    activation is its own layer, not when it's fused into Conv2D.

    batch_norm=False builds the same architecture with the BatchNormalization
    layers removed, used by ablation.py to test whether they're the source of
    the INT8 accuracy drop.
    """
    layers = [tf.keras.layers.Input(shape=input_shape)]

    for filters in (16, 32, 64):
        layers.append(tf.keras.layers.Conv2D(filters, 3, padding="same"))
        if batch_norm:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation("relu"))
        if filters != 64:
            layers.append(tf.keras.layers.MaxPooling2D(2))

    # GlobalAveragePooling2D instead of Flatten+Dense here on purpose --
    # tensorflow-basics/README.md found that a Flatten->Dense layer held
    # 90% of that model's parameters. GAP has ZERO parameters and forces
    # the conv layers to do the representational work instead, which
    # keeps this deeper model's size increase modest.
    layers.append(tf.keras.layers.GlobalAveragePooling2D())
    layers.append(tf.keras.layers.Dense(32, activation="relu"))
    layers.append(tf.keras.layers.Dropout(0.3))
    layers.append(tf.keras.layers.Dense(num_classes, activation="softmax"))

    return tf.keras.Sequential(layers)


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
