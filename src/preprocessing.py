"""Data validation, preprocessing, splitting, and dataset statistics for RurangaSort AI."""
from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from src.config import ALLOWED_IMAGE_EXTENSIONS, DEFAULT_CLASS_NAMES, settings

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    total_files: int = 0
    accepted: int = 0
    corrupted: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    unsupported_format: list[str] = field(default_factory=list)
    unknown_class: list[str] = field(default_factory=list)
    unsafe_path: list[str] = field(default_factory=list)
    oversized: list[str] = field(default_factory=list)

    def is_clean(self) -> bool:
        return not any(
            [self.corrupted, self.unsupported_format, self.unknown_class, self.unsafe_path, self.oversized]
        )

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "accepted": self.accepted,
            "corrupted": self.corrupted,
            "duplicates": self.duplicates,
            "unsupported_format": self.unsupported_format,
            "unknown_class": self.unknown_class,
            "unsafe_path": self.unsafe_path,
            "oversized": self.oversized,
        }


def is_image_corrupted(path: Path) -> bool:
    """Return True when the file cannot be opened/verified as an image."""
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() closes the file and leaves the image unusable for further ops, reopen to be safe.
        with Image.open(path) as img:
            img.load()
        return False
    except (UnidentifiedImageError, OSError, ValueError):
        return True


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_safe_member_path(member_name: str) -> bool:
    """Reject zip-slip / path-traversal / absolute-path zip members."""
    if member_name.startswith("/") or member_name.startswith("\\"):
        return False
    if ".." in Path(member_name).parts:
        return False
    if Path(member_name).drive:  # e.g. "C:\\..."
        return False
    return True


def validate_dataset_zip(
    zip_path: Path,
    known_classes: list[str] | None = None,
    max_files: int | None = None,
    max_file_size_mb: float | None = None,
) -> ValidationReport:
    """Validate a bulk-upload ZIP of labelled images without extracting unsafe members."""
    known_classes = known_classes or DEFAULT_CLASS_NAMES
    max_files = max_files or settings.max_zip_files
    max_file_size_mb = max_file_size_mb or settings.max_upload_size_mb

    report = ValidationReport()

    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Uploaded file is not a valid ZIP archive.")

    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        report.total_files = len(members)

        if report.total_files == 0:
            raise ValueError("Uploaded ZIP file is empty.")
        if report.total_files > max_files:
            raise ValueError(f"ZIP contains {report.total_files} files, exceeding the limit of {max_files}.")

        seen_hashes: dict[str, str] = {}

        for member in members:
            name = member.filename

            if not is_safe_member_path(name):
                report.unsafe_path.append(name)
                continue

            size_mb = member.file_size / (1024 * 1024)
            if size_mb > max_file_size_mb:
                report.oversized.append(name)
                continue

            suffix = Path(name).suffix.lower()
            if suffix not in ALLOWED_IMAGE_EXTENSIONS:
                report.unsupported_format.append(name)
                continue

            parts = Path(name).parts
            class_name = parts[0] if len(parts) > 1 else None
            if class_name not in known_classes:
                report.unknown_class.append(name)
                continue

            with archive.open(member) as fh:
                data = fh.read()

            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes:
                report.duplicates.append(name)
                continue
            seen_hashes[digest] = name

            try:
                Image.open(__import__("io").BytesIO(data)).verify()
            except (UnidentifiedImageError, OSError, ValueError):
                report.corrupted.append(name)
                continue

            report.accepted += 1

    return report


def safe_extract_zip(zip_path: Path, destination: Path, known_classes: list[str] | None = None) -> ValidationReport:
    """Validate then safely extract only accepted, safe members into `destination/<class>/...`."""
    known_classes = known_classes or DEFAULT_CLASS_NAMES
    report = validate_dataset_zip(zip_path, known_classes=known_classes)
    destination.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = member.filename
            if not is_safe_member_path(name):
                continue
            suffix = Path(name).suffix.lower()
            if suffix not in ALLOWED_IMAGE_EXTENSIONS:
                continue
            parts = Path(name).parts
            class_name = parts[0] if len(parts) > 1 else None
            if class_name not in known_classes:
                continue

            with archive.open(member) as fh:
                data = fh.read()
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            target_dir = destination / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / Path(name).name
            with open(target_path, "wb") as out:
                out.write(data)

    return report


def load_and_preprocess_image(path: Path, image_size: int | None = None) -> np.ndarray:
    """Load an image, convert to RGB, resize, and scale pixel values to [0, 1]."""
    image_size = image_size or settings.image_size
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size))
        array = np.asarray(img, dtype=np.float32) / 255.0
    return array


def build_dataset_index(raw_dir: Path, class_names: list[str] | None = None) -> list[dict]:
    """Walk `raw_dir/<class>/*` and return a list of {path, label} records, skipping corrupted files."""
    class_names = class_names or DEFAULT_CLASS_NAMES
    records = []
    for class_name in class_names:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
                continue
            if is_image_corrupted(path):
                logger.warning("Skipping corrupted image: %s", path)
                continue
            records.append({"path": str(path), "label": class_name})
    return records


def deduplicate_records(records: list[dict]) -> tuple[list[dict], list[str]]:
    seen: dict[str, str] = {}
    unique_records = []
    duplicates = []
    for record in records:
        digest = file_hash(Path(record["path"]))
        if digest in seen:
            duplicates.append(record["path"])
            continue
        seen[digest] = record["path"]
        unique_records.append(record)
    return unique_records, duplicates


def split_dataset(
    records: list[dict],
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict[str, list[dict]]:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    labels = [r["label"] for r in records]
    train_records, temp_records = train_test_split(
        records, train_size=train_size, random_state=random_state, stratify=labels
    )
    temp_labels = [r["label"] for r in temp_records]
    relative_val = val_size / (val_size + test_size)
    val_records, test_records = train_test_split(
        temp_records, train_size=relative_val, random_state=random_state, stratify=temp_labels
    )
    return {"train": train_records, "validation": val_records, "test": test_records}


def materialize_split(split: dict[str, list[dict]], output_root: Path) -> None:
    """Copy split records into output_root/<split_name>/<class>/file for Keras' image_dataset_from_directory."""
    import shutil

    for split_name, records in split.items():
        for record in records:
            src_path = Path(record["path"])
            dst_dir = output_root / split_name / record["label"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / src_path.name
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)


def save_class_names(class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        return list(DEFAULT_CLASS_NAMES)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_manifest(split: dict[str, list[dict]], path: Path, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "counts": {name: len(records) for name, records in split.items()},
        "class_distribution": {
            split_name: _class_counts(records) for split_name, records in split.items()
        },
    }
    if extra:
        manifest.update(extra)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _class_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["label"]] = counts.get(record["label"], 0) + 1
    return counts


def compute_dataset_statistics(raw_dir: Path, class_names: list[str] | None = None, sample_per_class: int = 200) -> dict:
    """Compute per-class image count, brightness, RGB means, and dimension stats for visualization."""
    class_names = class_names or DEFAULT_CLASS_NAMES
    stats: dict[str, dict] = {}

    for class_name in class_names:
        class_dir = raw_dir / class_name
        if not class_dir.exists():
            stats[class_name] = {
                "count": 0, "brightness_mean": None, "rgb_mean": None,
                "avg_width": None, "avg_height": None, "avg_aspect_ratio": None,
            }
            continue

        paths = [p for p in sorted(class_dir.iterdir()) if p.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS]
        count = len(paths)
        sample_paths = paths[:sample_per_class]

        brightness_values = []
        rgb_values = []
        widths, heights, aspect_ratios = [], [], []

        for path in sample_paths:
            try:
                with Image.open(path) as img:
                    img_rgb = img.convert("RGB")
                    width, height = img_rgb.size
                    widths.append(width)
                    heights.append(height)
                    aspect_ratios.append(width / height if height else 0)

                    array = np.asarray(img_rgb, dtype=np.float32)
                    rgb_values.append(array.reshape(-1, 3).mean(axis=0))

                    grayscale = np.asarray(img_rgb.convert("L"), dtype=np.float32)
                    brightness_values.append(grayscale.mean())
            except (UnidentifiedImageError, OSError):
                continue

        rgb_mean = np.mean(rgb_values, axis=0).tolist() if rgb_values else None

        stats[class_name] = {
            "count": count,
            "brightness_mean": float(np.mean(brightness_values)) if brightness_values else None,
            "rgb_mean": {"r": rgb_mean[0], "g": rgb_mean[1], "b": rgb_mean[2]} if rgb_mean else None,
            "avg_width": float(np.mean(widths)) if widths else None,
            "avg_height": float(np.mean(heights)) if heights else None,
            "avg_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else None,
        }

    return stats


def get_augmentation_layer():
    """Return a tf.keras augmentation pipeline applied only to the training set."""
    import tensorflow as tf
    from tensorflow.keras import layers

    return tf.keras.Sequential(
        [
            layers.RandomRotation(0.08),
            layers.RandomFlip("horizontal"),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomBrightness(0.1),
        ],
        name="augmentation",
    )
