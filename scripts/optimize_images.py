"""
Optimize and create thumbnails for the app static images.
- Backs up originals as <name>_orig.png (only if backup doesn't exist).
- Creates a thumbnail with max size (800x600) preserving aspect ratio.
- Saves optimized PNG overwriting the served filenames.

Run: ./.venv/Scripts/python.exe scripts/optimize_images.py
"""
from PIL import Image
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "app" / "static" / "images"

IMAGES = ["setosa.png", "versicolor.png", "virginica.png"]
MAX_SIZE = (800, 600)  # max width, max height


def optimize_image(path: Path, max_size=MAX_SIZE):
    backup = path.with_name(path.stem + "_orig" + path.suffix)
    if not backup.exists():
        path.replace(backup)
        # after replace, backup exists at backup path and original moved; open backup to create optimized file
        src_path = backup
        print(f"Created backup: {backup.name}")
    else:
        src_path = backup
        print(f"Backup already exists: {backup.name}")

    try:
        with Image.open(src_path) as img:
            # Convert paletted images to RGBA first to preserve transparency when present
            img_format = img.format
            if img.mode in ("P", "LA"):
                img = img.convert("RGBA")
            elif img.mode == "CMYK":
                img = img.convert("RGB")

            # Resize keeping aspect ratio
            img.thumbnail(max_size, Image.LANCZOS)

            # Save as PNG optimized
            # For PNG, Pillow's 'optimize' does lossless compression. For extra reduction, could quantize but that loses colors.
            img.save(path, format="PNG", optimize=True)
            print(f"Optimized and saved: {path.name} (from {src_path.name})")
    except Exception as e:
        print(f"Failed to optimize {path.name}: {e}")


if __name__ == "__main__":
    if not IMAGES_DIR.exists():
        print(f"Images directory not found: {IMAGES_DIR}")
        raise SystemExit(1)

    for name in IMAGES:
        p = IMAGES_DIR / name
        if not p.exists():
            print(f"Image not found, skipping: {name}")
            continue
        optimize_image(p)

    print("Done optimizing images.")
