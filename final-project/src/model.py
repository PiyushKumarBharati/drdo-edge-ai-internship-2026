"""
CNN architecture for the final project.

Slightly deeper than tensorflow-basics/03_cnn_fashion_mnist.py's CNN (one
extra conv block, batch normalization) -- this is the "final" version after
having already seen that a smaller CNN gets ~87% test accuracy, this
architecture trades a modest parameter increase for better accuracy while
staying well within a size budget that converts to a small TFLite file.
"""

import tensorflow as tf


def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2),

        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        # GlobalAveragePooling2D instead of Flatten+Dense here on purpose --
        # tensorflow-basics/README.md found that a Flatten->Dense layer held
        # 90% of that model's parameters. GAP has ZERO parameters and forces
        # the conv layers to do the representational work instead, which
        # keeps this deeper model's size increase modest.

        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
