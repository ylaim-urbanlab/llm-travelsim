"""
Export and load cache directory to/from a folder or zip file.
Uses JSON files only (no pickle); portable and human-readable.
"""
from __future__ import annotations
import zipfile
from pathlib import Path
from typing import List


def _json_files(cache_dir: Path) -> List[Path]:
    """All .json files in cache_dir (non-recursive)."""
    if not cache_dir.is_dir():
        return []
    return sorted(cache_dir.glob("*.json"))


def export_caches(cache_dir: str, output_path: str) -> str:
    """
    Export all JSON cache files from cache_dir to output_path.

    - If output_path ends with .zip: create a zip archive with the same filenames.
    - Otherwise: create a directory and copy each JSON file into it.

    Returns the path written (output_path).
    """
    cache = Path(cache_dir)
    out = Path(output_path)
    files = _json_files(cache)
    if not files:
        raise FileNotFoundError(f"No .json files in cache dir: {cache_dir}")

    if out.suffix.lower() == ".zip":
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, f.name)
        print(f"[cache_io] Exported {len(files)} files to {out}")
        return str(out)

    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        (out / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[cache_io] Exported {len(files)} files to {out}")
    return str(out)


def load_caches(input_path: str, cache_dir: str) -> str:
    """
    Load JSON cache files from input_path into cache_dir.

    - If input_path is a .zip file: extract all members into cache_dir.
    - Otherwise: treat as a directory and copy all .json files into cache_dir.

    Overwrites existing files in cache_dir. Creates cache_dir if needed.
    Returns cache_dir.
    """
    src = Path(input_path)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    if src.suffix.lower() == ".zip":
        if not src.is_file():
            raise FileNotFoundError(f"Zip not found: {input_path}")
        count = 0
        with zipfile.ZipFile(src, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".json"):
                    target = cache / Path(name).name
                    target.write_text(zf.read(name).decode("utf-8"), encoding="utf-8")
                    count += 1
        print(f"[cache_io] Loaded {count} files from {src} into {cache}")
        return str(cache)

    if not src.is_dir():
        raise FileNotFoundError(f"Not a directory or zip: {input_path}")
    files = _json_files(src)
    if not files:
        raise FileNotFoundError(f"No .json files in {input_path}")
    for f in files:
        (cache / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[cache_io] Loaded {len(files)} files from {src} into {cache}")
    return str(cache)
