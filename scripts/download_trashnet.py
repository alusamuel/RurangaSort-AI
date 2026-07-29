"""Download and lay out the real TrashNet dataset into data/raw/<class>/*.

TrashNet (Gary Thung & Mindy Yang, Stanford, CC BY 2.0) has 6 classes: cardboard,
glass, metal, paper, plastic, trash. It isn't hosted behind a single stable
anonymous-download URL, so this script supports three sources -- use whichever
you have access to:

1. --url  a direct HTTPS link to a zip/tar of the dataset (e.g. a mirror you trust,
          or a pre-signed S3/GDrive link) laid out as <root>/<class>/*.jpg
2. --kaggle "owner/dataset-slug"
          uses the `kaggle` CLI (requires ~/.kaggle/kaggle.json credentials) to
          download a Kaggle mirror, e.g. "asdasdasasdas/garbage-classification"
3. --manual-dir path
          you already downloaded/extracted the dataset yourself; this just
          validates and copies it into data/raw/ with the expected structure

Usage:
    python scripts/download_trashnet.py --url https://example.com/trashnet.zip
    python scripts/download_trashnet.py --kaggle asdasdasasdas/garbage-classification
    python scripts/download_trashnet.py --manual-dir /path/to/extracted/dataset-resized
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ROOT_DIR = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def _extract_archive(archive_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_to)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_to)
    else:
        raise ValueError(f"Unrecognized archive format: {archive_path}")


def _find_class_dirs(search_root: Path) -> dict[str, Path]:
    """Recursively look for folders named after our known classes."""
    found = {}
    for path in search_root.rglob("*"):
        if path.is_dir() and path.name.lower() in CLASS_NAMES and path.name.lower() not in found:
            found[path.name.lower()] = path
    return found


def _copy_into_raw(class_dirs: dict[str, Path], raw_dir: Path) -> None:
    for class_name in CLASS_NAMES:
        source = class_dirs.get(class_name)
        if source is None:
            print(f"  [!] class '{class_name}' not found in downloaded data, skipping")
            continue
        destination = raw_dir / class_name
        destination.mkdir(parents=True, exist_ok=True)
        count = 0
        for image_path in source.iterdir():
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                shutil.copy2(image_path, destination / image_path.name)
                count += 1
        print(f"  [ok] {class_name}: copied {count} images")


def download_via_url(url: str, raw_dir: Path, workdir: Path) -> None:
    print(f"Downloading {url} ...")
    archive_path = workdir / "trashnet_download.tmp"
    urlretrieve(url, archive_path)
    extract_dir = workdir / "extracted"
    print("Extracting archive ...")
    _extract_archive(archive_path, extract_dir)
    class_dirs = _find_class_dirs(extract_dir)
    _copy_into_raw(class_dirs, raw_dir)


def download_via_kaggle(dataset_slug: str, raw_dir: Path, workdir: Path) -> None:
    print(f"Downloading Kaggle dataset '{dataset_slug}' (requires kaggle CLI + credentials) ...")
    extract_dir = workdir / "kaggle_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(workdir), "--unzip"],
        check=True,
    )
    class_dirs = _find_class_dirs(workdir)
    if not class_dirs:
        class_dirs = _find_class_dirs(extract_dir)
    _copy_into_raw(class_dirs, raw_dir)


def use_manual_dir(manual_dir: Path, raw_dir: Path) -> None:
    class_dirs = _find_class_dirs(manual_dir)
    if not class_dirs:
        raise FileNotFoundError(
            f"Could not find any of {CLASS_NAMES} as subfolders under {manual_dir}. "
            "Make sure you pointed at the extracted dataset root."
        )
    _copy_into_raw(class_dirs, raw_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="Direct URL to a zip/tar archive of the dataset")
    parser.add_argument("--kaggle", help="Kaggle dataset slug, e.g. asdasdasasdas/garbage-classification")
    parser.add_argument("--manual-dir", help="Path to an already-downloaded/extracted dataset")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "data" / "raw"))
    args = parser.parse_args()

    if not any([args.url, args.kaggle, args.manual_dir]):
        parser.print_help()
        print(
            "\nNo source given. Pick one:\n"
            "  --url <direct archive URL>\n"
            "  --kaggle <owner/dataset-slug>   (needs `pip install kaggle` + ~/.kaggle/kaggle.json)\n"
            "  --manual-dir <path>             (dataset you already downloaded yourself)\n"
        )
        sys.exit(1)

    raw_dir = Path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    workdir = ROOT_DIR / ".download_tmp"
    workdir.mkdir(exist_ok=True)

    try:
        if args.manual_dir:
            use_manual_dir(Path(args.manual_dir), raw_dir)
        elif args.kaggle:
            download_via_kaggle(args.kaggle, raw_dir, workdir)
        elif args.url:
            download_via_url(args.url, raw_dir, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    print(f"\nDone. Verify class folders under {raw_dir} before training.")
    print("Licence reminder: TrashNet is CC BY 2.0 -- attribute Gary Thung & Mindy Yang.")


if __name__ == "__main__":
    main()
