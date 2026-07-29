"""Generate a small, deterministic, class-discriminative placeholder image dataset.

This exists so the full pipeline (preprocessing -> training -> API -> UI -> retraining)
can be exercised end-to-end without network access or the real TrashNet dataset.
Each class gets a distinct colour palette, base shade, and a class-specific mix of
shapes/lines/noise so a small CNN can learn real (if trivial) discriminative signal --
this is NOT a substitute for the real dataset; replace it before reporting real metrics.

Usage:
    python scripts/generate_synthetic_dataset.py --count-per-class 60
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parent.parent

CLASS_STYLES = {
    "cardboard": {"base": (150, 111, 51), "accent": (90, 60, 20), "shape": "lines"},
    "glass": {"base": (200, 230, 225), "accent": (60, 140, 150), "shape": "circles"},
    "metal": {"base": (170, 170, 175), "accent": (90, 90, 100), "shape": "rects"},
    "paper": {"base": (245, 245, 235), "accent": (200, 200, 180), "shape": "lines"},
    "plastic": {"base": (60, 140, 220), "accent": (240, 200, 40), "shape": "circles"},
    "trash": {"base": (70, 60, 55), "accent": (140, 40, 40), "shape": "noise"},
}


def _jitter(color: tuple[int, int, int], amount: int, rng: random.Random) -> tuple[int, int, int]:
    return tuple(max(0, min(255, c + rng.randint(-amount, amount))) for c in color)


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, size: int, accent, rng: random.Random) -> None:
    if shape == "lines":
        for _ in range(rng.randint(4, 8)):
            y = rng.randint(0, size)
            draw.line([(0, y), (size, y + rng.randint(-10, 10))], fill=accent, width=rng.randint(1, 3))
    elif shape == "circles":
        for _ in range(rng.randint(3, 6)):
            r = rng.randint(10, size // 4)
            x, y = rng.randint(0, size), rng.randint(0, size)
            draw.ellipse([x - r, y - r, x + r, y + r], outline=accent, width=3)
    elif shape == "rects":
        for _ in range(rng.randint(3, 6)):
            w, h = rng.randint(20, size // 2), rng.randint(20, size // 2)
            x, y = rng.randint(0, size - w), rng.randint(0, size - h)
            draw.rectangle([x, y, x + w, y + h], outline=accent, width=3)
    else:  # noise
        for _ in range(rng.randint(200, 400)):
            x, y = rng.randint(0, size), rng.randint(0, size)
            draw.point((x, y), fill=accent)


def generate_image(class_name: str, index: int, size: int, rng: random.Random) -> Image.Image:
    style = CLASS_STYLES[class_name]
    base = _jitter(style["base"], 15, rng)
    accent = _jitter(style["accent"], 15, rng)

    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image)
    _draw_shape(draw, style["shape"], size, accent, rng)
    return image


def generate_dataset(output_dir: Path, count_per_class: int, image_size: int = 256, seed: int = 42) -> None:
    for class_name in CLASS_STYLES:
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(f"{seed}-{class_name}")
        for i in range(count_per_class):
            image = generate_image(class_name, i, image_size, rng)
            image.save(class_dir / f"{class_name}_{i:04d}.jpg", quality=90)
        print(f"Generated {count_per_class} images for class '{class_name}' -> {class_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "data" / "raw"))
    parser.add_argument("--count-per-class", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(Path(args.output_dir), args.count_per_class, args.image_size, args.seed)
    print("\nSynthetic placeholder dataset generated. This is for pipeline development only --")
    print("replace with the real TrashNet dataset (scripts/download_trashnet.py) before reporting real metrics.")


if __name__ == "__main__":
    main()
