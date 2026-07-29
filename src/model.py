"""Model architectures: baseline CNN and MobileNetV2 transfer-learning model."""
from __future__ import annotations

from src.config import settings


def build_baseline_cnn(num_classes: int, input_shape: tuple[int, int, int] | None = None):
    """A small custom CNN used as the baseline classifier."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    input_shape = input_shape or settings.image_shape

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="baseline_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_v2(num_classes: int, input_shape: tuple[int, int, int] | None = None, fine_tune_layers: int = 30):
    """MobileNetV2 with an ImageNet-pretrained frozen base and a new classification head.

    Call `unfreeze_for_fine_tuning` after the head has converged to fine-tune the top layers.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    input_shape = input_shape or settings.image_shape

    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="mobilenet_v2")
    model._base_model = base_model  # stashed for unfreeze_for_fine_tuning
    model._fine_tune_layers = fine_tune_layers

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_fine_tuning(model, learning_rate: float = 1e-5):
    """Stage 2 of MobileNetV2 training: unfreeze the top N layers and recompile at a low LR."""
    import tensorflow as tf

    base_model = model._base_model
    base_model.trainable = True

    fine_tune_at = max(len(base_model.layers) - model._fine_tune_layers, 0)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


MODEL_BUILDERS = {
    "baseline_cnn": build_baseline_cnn,
    "mobilenet_v2": build_mobilenet_v2,
}
