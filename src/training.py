"""Training entrypoint: loads the prepared image splits and trains a model.

Usage:
    python -m src.training --model baseline_cnn --epochs 10
    python -m src.training --model mobilenet_v2 --epochs 10 --fine-tune-epochs 5
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.config import settings
from src.model import MODEL_BUILDERS, build_mobilenet_v2, unfreeze_for_fine_tuning
from src.preprocessing import get_augmentation_layer, load_class_names

logger = logging.getLogger(__name__)


def load_datasets(
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    image_size: int | None = None,
    batch_size: int = 32,
):
    import tensorflow as tf

    image_size = image_size or settings.image_size

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(image_size, image_size), batch_size=batch_size, label_mode="categorical"
    )
    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(image_size, image_size), batch_size=batch_size, label_mode="categorical"
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=(image_size, image_size), batch_size=batch_size, label_mode="categorical"
    )

    normalization = tf.keras.layers.Rescaling(1.0 / 255)
    augmentation = get_augmentation_layer()

    train_ds = train_ds.map(lambda x, y: (augmentation(normalization(x), training=True), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization(x), y))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names


def train_model(
    model_name: str,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    epochs: int = 10,
    fine_tune_epochs: int = 5,
    batch_size: int = 32,
):
    import tensorflow as tf

    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_BUILDERS)}.")

    train_ds, val_ds, test_ds, class_names = load_datasets(train_dir, val_dir, test_dir, batch_size=batch_size)
    num_classes = len(class_names)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ]

    if model_name == "mobilenet_v2":
        model = build_mobilenet_v2(num_classes)
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
        if fine_tune_epochs > 0:
            model = unfreeze_for_fine_tuning(model)
            history = model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs, callbacks=callbacks)
    else:
        model = MODEL_BUILDERS[model_name](num_classes)
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    return model, history, class_names, test_ds


def save_trained_model(model, class_names: list[str], output_dir: Path, model_name: str) -> Path:
    """Saves to `output_dir/model.h5` — a fixed filename so the registry/predictor
    don't need to know which architecture (baseline_cnn vs mobilenet_v2) produced it.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.h5"
    model.save(model_path)
    with open(output_dir / "class_names.json", "w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)
    return model_path


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train a RurangaSort AI classification model.")
    parser.add_argument("--model", choices=list(MODEL_BUILDERS), default="baseline_cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-dir", default=settings.data_train_dir)
    parser.add_argument("--val-dir", default=settings.data_validation_dir)
    parser.add_argument("--test-dir", default=settings.data_test_dir)
    parser.add_argument("--output-dir", default="models/candidate")
    args = parser.parse_args()

    model, history, class_names, test_ds = train_model(
        model_name=args.model,
        train_dir=settings.resolve(args.train_dir),
        val_dir=settings.resolve(args.val_dir),
        test_dir=settings.resolve(args.test_dir),
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
    )

    model_path = save_trained_model(model, class_names, settings.resolve(args.output_dir), args.model)
    logger.info("Saved trained model to %s", model_path)

    from src.evaluation import evaluate_model

    metrics = evaluate_model(model, test_ds, class_names)
    metrics_path = settings.resolve(args.output_dir) / f"{args.model}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Test macro F1: %.4f", metrics["macro_f1"])


if __name__ == "__main__":
    main()
