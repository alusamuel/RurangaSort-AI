"""Celery tasks: background model retraining, kept out of the API process."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from worker.celery_app import celery_app

logger = logging.getLogger(__name__)


def _merge_uploaded_into_raw(upload_dir: Path, raw_dir: Path) -> int:
    """Move any newly-uploaded, already-validated images into data/raw/<class>/ and
    return how many were merged."""
    from src.config import DEFAULT_CLASS_NAMES

    merged = 0
    if not upload_dir.exists():
        return merged

    for class_name in DEFAULT_CLASS_NAMES:
        class_upload_dir = upload_dir / class_name
        if not class_upload_dir.exists():
            continue
        target_dir = raw_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in list(class_upload_dir.iterdir()):
            destination = target_dir / image_path.name
            if destination.exists():
                destination = target_dir / f"uploaded_{image_path.name}"
            shutil.move(str(image_path), str(destination))
            merged += 1
    return merged


@celery_app.task(bind=True, name="worker.tasks.retrain_model")
def retrain_model(self, model_name: str = "baseline_cnn", epochs: int = 10, fine_tune_epochs: int = 5) -> dict:
    """Full retraining pipeline: merge new data -> re-split -> train -> evaluate -> promote/reject."""
    from src.config import settings
    from src.evaluation import evaluate_model
    from src.model_registry import (
        evaluate_promotion,
        get_active_metrics,
        promote_candidate,
        register_candidate,
        reject_candidate,
    )
    from src.monitoring import record_retraining_event
    from src.preprocessing import (
        DEFAULT_CLASS_NAMES,
        build_dataset_index,
        deduplicate_records,
        materialize_split,
        save_class_names,
        save_manifest,
        split_dataset,
    )
    from src.training import save_trained_model, train_model

    root = settings.resolve(".")
    raw_dir = settings.resolve(settings.data_raw_dir)
    upload_dir = settings.resolve(settings.upload_directory)

    try:
        self.update_state(state="PROGRESS", meta={"stage": "merging_uploaded_data"})
        merged_count = _merge_uploaded_into_raw(upload_dir, raw_dir)
        logger.info("Merged %d newly uploaded images into data/raw", merged_count)

        self.update_state(state="PROGRESS", meta={"stage": "preprocessing"})
        records = build_dataset_index(raw_dir, DEFAULT_CLASS_NAMES)
        records, duplicates = deduplicate_records(records)
        split = split_dataset(records)
        materialize_split(split, root / "data")
        save_class_names(DEFAULT_CLASS_NAMES, settings.resolve(settings.class_names_path))
        save_manifest(
            split,
            settings.resolve(settings.data_processed_dir) / "manifest.json",
            extra={"duplicates_removed": len(duplicates), "total_images": len(records), "triggered_by": "retrain"},
        )

        self.update_state(state="PROGRESS", meta={"stage": "training", "model_name": model_name})
        model, history, class_names, test_ds = train_model(
            model_name=model_name,
            train_dir=settings.resolve(settings.data_train_dir),
            val_dir=settings.resolve(settings.data_validation_dir),
            test_dir=settings.resolve(settings.data_test_dir),
            epochs=epochs,
            fine_tune_epochs=fine_tune_epochs,
        )

        self.update_state(state="PROGRESS", meta={"stage": "evaluating"})
        staging_dir = root / "models" / "_staging"
        model_path = save_trained_model(model, class_names, staging_dir, model_name)
        candidate_metrics = evaluate_model(model, test_ds, class_names, model_path=model_path)

        self.update_state(state="PROGRESS", meta={"stage": "comparing_with_active"})
        active_metrics = get_active_metrics()
        decision = evaluate_promotion(candidate_metrics, active_metrics)

        register_candidate(
            model_file=model_path,
            class_names=class_names,
            metrics=candidate_metrics,
            model_name=model_name,
            input_shape=settings.image_shape,
        )
        shutil.rmtree(staging_dir, ignore_errors=True)

        if decision["should_promote"]:
            self.update_state(state="PROGRESS", meta={"stage": "promoting"})
            metadata = promote_candidate()
            record_retraining_event("promoted", model_version=metadata["model_version"], detail="; ".join(decision["reasons"]))
            result = {
                "status": "promoted",
                "model_version": metadata["model_version"],
                "reasons": decision["reasons"],
                "metrics": candidate_metrics,
            }
        else:
            self.update_state(state="PROGRESS", meta={"stage": "rejecting"})
            archive_path = reject_candidate(decision["reasons"])
            record_retraining_event("rejected", detail="; ".join(decision["reasons"]))
            result = {
                "status": "rejected",
                "reasons": decision["reasons"],
                "metrics": candidate_metrics,
                "archived_at": str(archive_path),
            }

        return result

    except Exception as exc:  # noqa: BLE001
        logger.exception("Retraining task failed")
        record_retraining_event("failed", detail=str(exc))
        raise
