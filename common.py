"""Shared helpers for the Face-Match Photo Sorter.

Centralizes what enroll.py and faces.py both need:
  1. Decoding images (including Apple HEIC) into the BGR numpy arrays
     insightface expects, with EXIF orientation applied.
  2. Constructing the insightface model once, configured for CPU.
  3. Small filesystem/embedding utilities (collision-safe output paths,
     loading enrolled reference embeddings for cluster calibration).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Register the HEIC opener with Pillow. After this, Image.open() handles .heic
# transparently alongside .jpg/.png. iPhone class photos are all HEIC.
import pillow_heif

pillow_heif.register_heif_opener()

# Extensions we'll treat as images when scanning a folder.
IMAGE_EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Archive extensions worth flagging: a common first-run mistake is dropping the
# downloaded album archive into the folder without extracting it.
ARCHIVE_EXTS = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}


class ArchiveFoundError(Exception):
    """An un-extracted archive was found under a folder being scanned for images.

    Raised instead of silently ignoring it: a stray zip almost always means
    photos that *should* be processed are still packed up. The caller catches
    this and prints the (already-friendly) message.
    """


def rel_key(path: str | Path, root: str | Path) -> str:
    """Identity key for an album file: its path relative to the album root,
    POSIX-style (forward slashes, stable across OSes and inside CSVs).

    For a file sitting directly in the album this is just its basename, so
    artifacts written before subfolders existed (which stored basenames) keep
    matching without a re-embed. A file in a subfolder gets a `subdir/...`
    prefix, which is what lets two photos share a basename across subfolders
    (e.g. a reused `IMG_4492.HEIC`) without colliding.
    """
    return Path(path).relative_to(root).as_posix()


def empty_hint(folder: str | Path) -> str:
    """Extra hint for a 'no images found' message, or '' if none applies.

    If the (otherwise image-less) folder holds an archive, the user probably
    forgot to extract it — say so rather than just 'no images found'.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return ""
    archives = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in ARCHIVE_EXTS)
    if archives:
        shown = ", ".join(archives[:3]) + ("..." if len(archives) > 3 else "")
        return f" Found an archive ({shown}) — extract it into the folder first."
    return ""


def list_images(folder: str | Path) -> list[Path]:
    """Return image files anywhere under *folder* (recursively), sorted,
    case-insensitive on extension. Dotfiles and dot-directories (`.DS_Store`,
    `.serve_cache/`, …) are skipped.

    Recursion is what makes subfolders work: drop a batch into
    `album/<name>/` and its images are found alongside the top-level ones.
    Paths are returned in full; pair with `rel_key(p, folder)` for the
    album-relative identity key the pipeline stores.

    Raises a friendly FileNotFoundError / NotADirectoryError if *folder* is
    missing or is actually a file, and ArchiveFoundError if an un-extracted
    archive is sitting in the tree — the common mistakes (typo'd path, forgot
    to populate album/, or dropped a downloaded zip straight in) — which would
    otherwise surface as a bare traceback or silently skip packed-up photos.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder} — create it and add photos, or pass the right path."
        )
    if not folder.is_dir():
        raise NotADirectoryError(
            f"{folder} is a file, not a folder. Point this at a directory of photos "
            "(unzip an archive first if you downloaded one)."
        )
    images: list[Path] = []
    archives: list[Path] = []
    for p in folder.rglob("*"):
        rel = p.relative_to(folder)
        if any(part.startswith(".") for part in rel.parts):
            continue  # skip dotfiles / dot-dirs (.DS_Store, .serve_cache, ...)
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            images.append(p)
        elif ext in ARCHIVE_EXTS:
            archives.append(p)
    if archives:
        shown = ", ".join(sorted(a.relative_to(folder).as_posix() for a in archives)[:3])
        first = sorted(a.stem for a in archives)[0]
        raise ArchiveFoundError(
            f"Found an un-extracted archive under {folder}/ ({shown}). Unzip it into "
            f"its own subfolder, then remove the archive and re-run -- e.g.\n"
            f"    mkdir {folder}/{first} && unzip '{folder}/{first}.zip' -d {folder}/{first}\n"
            f"    rm '{folder}/{first}.zip'\n"
            f"Leaving the archive in {folder}/ would skip every photo packed inside it. "
            f"(Non-image files in the subfolder, e.g. .mp4 videos, are ignored.)"
        )
    return sorted(images)


def load_image_bgr(source) -> np.ndarray:
    """Load an image as an HxWx3 uint8 BGR array (insightface's expected format).

    *source* may be a path (str/Path) or raw bytes. EXIF orientation is applied
    so portrait iPhone shots aren't fed in sideways (which wrecks detection).
    """
    if isinstance(source, (bytes, bytearray)):
        import io

        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)

    # Honor the EXIF orientation tag, then drop it (pixels are now upright).
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    rgb = np.asarray(img)
    # PIL gives RGB; insightface / OpenCV want BGR.
    return rgb[:, :, ::-1].copy()


def load_image_rgb_pil(source) -> Image.Image:
    """Load as an upright RGB PIL image (used when writing JPEG copies of hits)."""
    if isinstance(source, (bytes, bytearray)):
        import io

        img = Image.open(io.BytesIO(source))
    else:
        img = Image.open(source)
    return ImageOps.exif_transpose(img).convert("RGB")


def build_face_app(det_size: int = 1024, det_thresh: float = 0.4, modules=("detection", "recognition")):
    """Create and prepare an insightface FaceAnalysis app on CPU.

    det_size: detector input is resized so its long side is this many pixels.
        These are 24MP photos with distant kids, so faces are small relative to
        the frame — a larger det_size finds more of them at the cost of speed.
        640 is the library default; 1024 is a reasonable balance here.
    det_thresh: minimum detector confidence to report a face. Lowering it
        surfaces more (and weaker) detections.
    modules: which buffalo_l sub-models to load. We only need detection (to find
        faces + their 5-point landmarks for alignment) and recognition (the
        embedding). Skipping the landmark and gender/age models is faster and
        leaves embeddings/scores identical. Pass None to load everything.
    """
    # Imported lazily so `--help` and arg errors don't pay the import cost.
    from insightface.app import FaceAnalysis

    kwargs = {"name": "buffalo_l", "providers": ["CPUExecutionProvider"]}
    if modules is not None:
        kwargs["allowed_modules"] = list(modules)
    app = FaceAnalysis(**kwargs)
    app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_thresh)
    return app


def largest_face(faces):
    """Return the face with the biggest bounding box, or None if empty.

    Used during enrollment: a clear reference photo's subject is assumed to be
    the most prominent face.
    """
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def load_refs(path: str) -> np.ndarray:
    """Load enrolled reference embeddings as an (N, 512) float32 stack.

    Produced by enroll.py; consumed by faces.py's cluster-calibration readout
    (does the reference person land in one clean cluster?).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Reference embeddings not found: {path}. Run enroll.py first.")
    refs = np.load(path)
    if refs.ndim == 1:  # tolerate a single saved vector
        refs = refs[None, :]
    refs = refs.astype(np.float32)
    if refs.ndim != 2 or refs.shape[1] != 512:
        raise ValueError(f"Expected (N, 512) reference embeddings, got shape {refs.shape}")
    return refs


def _unique_path(path: Path) -> Path:
    """Return *path*, or path with a _1/_2/... suffix if it already exists.

    Guards against silent overwrites when two sources map to the same output
    name — e.g. IMG_5.heic and IMG_5.jpg both becoming IMG_5.jpg under --jpeg,
    or any two query hits sharing a stem.
    """
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while (cand := parent / f"{stem}_{i}{suffix}").exists():
        i += 1
    return cand
