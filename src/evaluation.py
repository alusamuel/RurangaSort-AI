"""Classification metrics, confusion matrix, ROC/PR-AUC, and latency benchmarking."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _gather_predictions(model, dataset):
    y_true, y_pred_proba = [], []
    for batch_x, batch_y in dataset:
        probs = model.predict(batch_x, verbose=0)
        y_true.append(batch_y.numpy())
        y_pred_proba.append(probs)
    return np.concatenate(y_true, axis=0), np.concatenate(y_pred_proba, axis=0)


def compute_classification_metrics(y_true_onehot: np.ndarray, y_pred_proba: np.ndarray, class_names: list[str]) -> dict:
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = float(np.mean(y_true == y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    per_class = {
        class_names[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names))).tolist()

    try:
        roc_auc = float(roc_auc_score(y_true_onehot, y_pred_proba, multi_class="ovr", average="macro"))
    except ValueError:
        roc_auc = None

    try:
        pr_auc = float(average_precision_score(y_true_onehot, y_pred_proba, average="macro"))
    except ValueError:
        pr_auc = None

    try:
        loss = float(log_loss(y_true, y_pred_proba, labels=list(range(len(class_names)))))
    except ValueError:
        loss = None

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm,
        "class_names": class_names,
        "roc_auc_ovr_macro": roc_auc,
        "pr_auc_macro": pr_auc,
        "log_loss": loss,
    }


def measure_inference_latency(model, sample_batch: np.ndarray, num_runs: int = 30) -> dict:
    """Warm up then time single-image inference; returns average and P95 latency in ms."""
    single = sample_batch[:1]
    model.predict(single, verbose=0)  # warm-up

    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        model.predict(single, verbose=0)
        timings.append((time.perf_counter() - start) * 1000.0)

    timings.sort()
    p95_index = min(int(0.95 * len(timings)), len(timings) - 1)
    return {
        "avg_latency_ms": float(np.mean(timings)),
        "p95_latency_ms": float(timings[p95_index]),
        "num_runs": num_runs,
    }


def get_model_file_size_mb(model_path: Path) -> float:
    return round(Path(model_path).stat().st_size / (1024 * 1024), 3)


def evaluate_model(model, test_dataset, class_names: list[str], model_path: Path | None = None) -> dict:
    y_true_onehot, y_pred_proba = _gather_predictions(model, test_dataset)
    metrics = compute_classification_metrics(y_true_onehot, y_pred_proba, class_names)

    sample_batch, _ = next(iter(test_dataset))
    latency = measure_inference_latency(model, sample_batch.numpy())
    metrics.update(latency)

    if model_path is not None and Path(model_path).exists():
        metrics["model_size_mb"] = get_model_file_size_mb(model_path)

    return metrics


def compare_candidate_to_active(candidate_metrics: dict, active_metrics: dict | None) -> dict:
    """Compute the deltas used by the model-promotion decision in src.model_registry."""
    if active_metrics is None:
        return {"macro_f1_delta": None, "max_per_class_recall_drop": 0.0, "is_first_model": True}

    macro_f1_delta = candidate_metrics["macro_f1"] - active_metrics["macro_f1"]

    recall_drops = []
    for class_name, active_class_metrics in active_metrics.get("per_class", {}).items():
        candidate_class_metrics = candidate_metrics.get("per_class", {}).get(class_name)
        if candidate_class_metrics is None:
            continue
        drop = active_class_metrics["recall"] - candidate_class_metrics["recall"]
        recall_drops.append(drop)

    return {
        "macro_f1_delta": macro_f1_delta,
        "max_per_class_recall_drop": max(recall_drops) if recall_drops else 0.0,
        "is_first_model": False,
    }
