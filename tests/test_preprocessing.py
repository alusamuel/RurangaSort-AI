from __future__ import annotations

from pathlib import Path

from src.preprocessing import (
    build_dataset_index,
    compute_dataset_statistics,
    deduplicate_records,
    file_hash,
    is_image_corrupted,
    is_safe_member_path,
    load_and_preprocess_image,
    safe_extract_zip,
    split_dataset,
    validate_dataset_zip,
)

from tests.conftest import make_image_bytes


def test_is_image_corrupted_detects_garbage(tmp_path):
    corrupted_path = tmp_path / "not_an_image.jpg"
    corrupted_path.write_bytes(b"this is definitely not a jpeg")
    assert is_image_corrupted(corrupted_path) is True


def test_is_image_corrupted_accepts_valid_image(tmp_path):
    valid_path = tmp_path / "valid.jpg"
    valid_path.write_bytes(make_image_bytes())
    assert is_image_corrupted(valid_path) is False


def test_file_hash_is_stable_and_content_sensitive(tmp_path):
    path_a = tmp_path / "a.jpg"
    path_b = tmp_path / "b.jpg"
    path_a.write_bytes(make_image_bytes(color=(1, 2, 3)))
    path_b.write_bytes(make_image_bytes(color=(4, 5, 6)))

    assert file_hash(path_a) == file_hash(path_a)
    assert file_hash(path_a) != file_hash(path_b)


def test_is_safe_member_path_rejects_traversal_and_absolute():
    assert is_safe_member_path("cardboard/img.jpg") is True
    assert is_safe_member_path("../../etc/passwd") is False
    assert is_safe_member_path("/etc/passwd") is False
    assert is_safe_member_path("C:\\Windows\\system32\\evil.jpg") is False


def test_validate_dataset_zip_accepts_clean_archive(make_dataset_zip):
    zip_path = make_dataset_zip(
        {
            "cardboard/img1.jpg": make_image_bytes(color=(10, 20, 30)),
            "cardboard/img2.jpg": make_image_bytes(color=(40, 50, 60)),
            "glass/img1.jpg": make_image_bytes(color=(70, 80, 90)),
        }
    )
    report = validate_dataset_zip(zip_path, known_classes=["cardboard", "glass"])
    assert report.total_files == 3
    assert report.accepted == 3
    assert report.is_clean()


def test_validate_dataset_zip_rejects_path_traversal(make_dataset_zip):
    zip_path = make_dataset_zip({"../../evil.jpg": make_image_bytes()})
    report = validate_dataset_zip(zip_path, known_classes=["cardboard"])
    assert report.unsafe_path == ["../../evil.jpg"]
    assert report.accepted == 0
    assert not report.is_clean()


def test_validate_dataset_zip_rejects_unknown_class(make_dataset_zip):
    zip_path = make_dataset_zip({"not_a_class/img.jpg": make_image_bytes()})
    report = validate_dataset_zip(zip_path, known_classes=["cardboard", "glass"])
    assert "not_a_class/img.jpg" in report.unknown_class


def test_validate_dataset_zip_rejects_corrupted_image(make_dataset_zip):
    zip_path = make_dataset_zip({"cardboard/broken.jpg": b"not an image"})
    report = validate_dataset_zip(zip_path, known_classes=["cardboard"])
    assert "cardboard/broken.jpg" in report.corrupted


def test_validate_dataset_zip_detects_duplicates(make_dataset_zip):
    data = make_image_bytes(color=(5, 5, 5))
    zip_path = make_dataset_zip({"cardboard/a.jpg": data, "cardboard/b.jpg": data})
    report = validate_dataset_zip(zip_path, known_classes=["cardboard"])
    assert report.accepted == 1
    assert report.duplicates == ["cardboard/b.jpg"]


def test_validate_dataset_zip_rejects_unsupported_format(make_dataset_zip):
    zip_path = make_dataset_zip({"cardboard/notes.txt": b"hello"})
    report = validate_dataset_zip(zip_path, known_classes=["cardboard"])
    assert "cardboard/notes.txt" in report.unsupported_format


def test_validate_dataset_zip_rejects_empty_zip(make_dataset_zip):
    import pytest

    zip_path = make_dataset_zip({})
    with pytest.raises(ValueError, match="empty"):
        validate_dataset_zip(zip_path, known_classes=["cardboard"])


def test_safe_extract_zip_only_writes_accepted_safe_members(tmp_path, make_dataset_zip):
    zip_path = make_dataset_zip(
        {
            "cardboard/good.jpg": make_image_bytes(color=(1, 1, 1)),
            "../evil.jpg": make_image_bytes(color=(2, 2, 2)),
            "unknown_class/img.jpg": make_image_bytes(color=(3, 3, 3)),
        }
    )
    destination = tmp_path / "extracted"
    safe_extract_zip(zip_path, destination, known_classes=["cardboard"])

    extracted_files = list(destination.rglob("*.jpg"))
    assert len(extracted_files) == 1
    assert extracted_files[0].parent.name == "cardboard"


def test_load_and_preprocess_image_shape_and_range(tmp_path):
    path = tmp_path / "img.jpg"
    path.write_bytes(make_image_bytes(size=(64, 64)))
    array = load_and_preprocess_image(path, image_size=32)
    assert array.shape == (32, 32, 3)
    assert array.min() >= 0.0
    assert array.max() <= 1.0


def test_build_dataset_index_skips_corrupted(raw_dataset):
    raw_dir, classes = raw_dataset
    (raw_dir / "cardboard" / "corrupted.jpg").write_bytes(b"garbage")

    records = build_dataset_index(raw_dir, classes)
    paths = [Path(r["path"]).name for r in records]
    assert "corrupted.jpg" not in paths
    assert len(records) == 8  # 4 per class, corrupted one excluded


def test_deduplicate_records_removes_exact_duplicates(tmp_path):
    class_dir = tmp_path / "cardboard"
    class_dir.mkdir()
    data = make_image_bytes(color=(9, 9, 9))
    (class_dir / "a.jpg").write_bytes(data)
    (class_dir / "b.jpg").write_bytes(data)

    records = [
        {"path": str(class_dir / "a.jpg"), "label": "cardboard"},
        {"path": str(class_dir / "b.jpg"), "label": "cardboard"},
    ]
    unique, duplicates = deduplicate_records(records)
    assert len(unique) == 1
    assert len(duplicates) == 1


def test_split_dataset_ratios_and_stratification(raw_dataset):
    raw_dir, classes = raw_dataset
    records = build_dataset_index(raw_dir, classes)
    # Duplicate to get a large-enough sample for a meaningful stratified split.
    records = records * 5

    split = split_dataset(records, train_size=0.7, val_size=0.15, test_size=0.15)
    total = sum(len(v) for v in split.values())
    assert total == len(records)
    assert set(split.keys()) == {"train", "validation", "test"}
    assert len(split["train"]) > len(split["validation"])
    assert len(split["train"]) > len(split["test"])


def test_compute_dataset_statistics_reports_expected_fields(raw_dataset):
    raw_dir, classes = raw_dataset
    stats = compute_dataset_statistics(raw_dir, classes)

    for class_name in classes:
        assert stats[class_name]["count"] == 4
        assert stats[class_name]["brightness_mean"] is not None
        assert stats[class_name]["rgb_mean"] is not None
        assert stats[class_name]["avg_width"] == 64
        assert stats[class_name]["avg_height"] == 64
