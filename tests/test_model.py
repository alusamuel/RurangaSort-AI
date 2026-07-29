"""Regression coverage for model architectures: they must round-trip through
save/load, since a model that only fails on *reload* (not on save) will pass a
quick smoke test and then 503 in production the moment the API process restarts."""
from __future__ import annotations

from src.model import build_baseline_cnn, build_mobilenet_v2


def test_baseline_cnn_round_trips_through_save_and_load(tmp_path):
    import tensorflow as tf

    model = build_baseline_cnn(num_classes=3, input_shape=(32, 32, 3))
    model_path = tmp_path / "model.h5"
    model.save(model_path)

    reloaded = tf.keras.models.load_model(model_path)
    assert reloaded.output_shape == (None, 3)


def test_mobilenet_v2_round_trips_through_save_and_load(tmp_path):
    """MobileNetV2's preprocessing must be built from proper Keras layers (e.g. Rescaling),
    not raw tensor arithmetic -- a raw `inputs * 255.0` op saves fine but fails to reload
    from .h5 with `Unknown layer: 'TrueDivide'`, which only surfaces the next time the
    process starts and tries to load the already-promoted model."""
    import tensorflow as tf

    model = build_mobilenet_v2(num_classes=3, input_shape=(96, 96, 3), fine_tune_layers=5)
    model_path = tmp_path / "model.h5"
    model.save(model_path)

    reloaded = tf.keras.models.load_model(model_path)
    assert reloaded.output_shape == (None, 3)
