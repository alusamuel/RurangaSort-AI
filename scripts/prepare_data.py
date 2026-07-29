"""Validate data/raw/, deduplicate, split 70/15/15, and materialize data/train|validation|test.

Usage:
    python scripts/prepare_data.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_CLASS_NAMES, settings  # noqa: E402
from src.preprocessing import (  # noqa: E402
    build_dataset_index,
    deduplicate_records,
    materialize_split,
    save_class_names,
    save_manifest,
    split_dataset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(ROOT_DIR / settings.data_raw_dir))
    parser.add_argument("--output-root", default=str(ROOT_DIR))
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_root = Path(args.output_root)

    print(f"Indexing raw dataset at {raw_dir} ...")
    records = build_dataset_index(raw_dir, DEFAULT_CLASS_NAMES)
    if not records:
        print(
            "No images found. Populate data/raw/<class>/ first "
            "(scripts/download_trashnet.py or scripts/generate_synthetic_dataset.py)."
        )
        sys.exit(1)
    print(f"Found {len(records)} valid images across {len(DEFAULT_CLASS_NAMES)} classes.")

    records, duplicates = deduplicate_records(records)
    if duplicates:
        print(f"Removed {len(duplicates)} duplicate images.")

    split = split_dataset(records, args.train_size, args.val_size, args.test_size)
    for name, split_records in split.items():
        print(f"  {name}: {len(split_records)} images")

    print("Materializing split directories (data/train, data/validation, data/test) ...")
    materialize_split(split, output_root / "data")

    save_class_names(DEFAULT_CLASS_NAMES, output_root / settings.class_names_path)
    save_manifest(
        split,
        output_root / settings.data_processed_dir / "manifest.json",
        extra={"duplicates_removed": len(duplicates), "total_images": len(records)},
    )

    print("Done. Class names + manifest saved under models/ and data/processed/.")


if __name__ == "__main__":
    main()
