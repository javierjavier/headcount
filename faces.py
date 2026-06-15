#!/usr/bin/env python3
"""Whole-class face database: detect and embed *every* face in the album.

This is the foundation of the clustering pipeline described in DESIGN.md. It
keeps every face and its embedding, so faces can later be grouped by identity
(cluster), labeled (review), and sorted (assign) for the whole class.

Phases:

  embed   Detect + embed every face in every album image, once. Writes a face
          table (faces.csv) and the parallel embedding matrix (faces.npy).
          Slow and resumable. Decode is overlapped with inference via --prefetch
          (the single-threaded HEIC decode is ~half of per-image time; threading
          it gives ~1.5x on an M1).

  cluster Group faces by identity (HDBSCAN by default) -> clusters.csv.
          Re-runnable: a calibration readout shows whether the enrolled reference
          person's faces fall into one clean cluster.

  review  Crop a sample of each cluster's faces into a contact-sheet montage
          (clusters/) and write a skeleton labels.csv. You open the montages and
          type a name per cluster; that's the actual tagging step.

  assign  From the filled-in labels.csv, compute who is in each photo ->
          image_people.csv, and optionally sort copies/symlinks into by_child/.

  query   Set queries over image_people.csv, e.g. `--with Ada,Ben` (both
          present), `--any`, `--without`, `--only`, into query/<expr>/.
          `--where indoor|outdoor` adds a location filter (needs `scene`).

  scene   Tag each photo indoor/outdoor from foliage+sky colour -> scene.csv,
          and print the hour cross-tab so it can be checked against the schedule.

Artifacts:
  faces.csv   one row per face: face_id, filename, x1, y1, x2, y2, det_score
  faces.emb   append journal: raw little-endian float32, 512 per face, parallel
              to faces.csv (crash-safe; flushed per image)
  faces.npy   finalized M x 512 embedding matrix (np.save of faces.emb), the
              artifact `cluster` consumes. Row i <-> faces.csv face_id i.

The embeddings are insightface's normed_embedding (L2-normalized), so cosine
similarity between any two rows is a plain dot product.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import build_face_app, empty_hint, list_images, load_image_bgr, load_image_rgb_pil

FACES_HEADER = ["face_id", "filename", "x1", "y1", "x2", "y2", "det_score"]
EMB_DIM = 512


def read_face_filenames(csv_path: Path) -> set[str]:
    """Return the set of filenames that produced >=1 face in *csv_path*."""
    done: set[str] = set()
    if not csv_path.exists():
        return done
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("filename")
            if name:
                done.add(name)
    return done


def read_done_manifest(done_path: Path) -> set[str]:
    """Return the set of filenames recorded as processed in the .done manifest.

    faces.csv only lists images that yielded a face, so an image where the
    detector found *nothing* leaves no trace there and would be re-decoded and
    re-detected on every resume. The .done manifest records every processed
    filename (face-bearing or not), one per line, so zero-face images are skipped
    on subsequent runs too. Resume uses the union of this and faces.csv's
    filenames (see cmd_embed), which keeps it correct even if a crash lands
    between the per-image csv flush and the manifest append.
    """
    done: set[str] = set()
    if not done_path.exists():
        return done
    with done_path.open() as f:
        for line in f:
            name = line.strip()
            if name:
                done.add(name)
    return done


def _next_face_id(csv_path: Path) -> int:
    """Largest face_id in *csv_path* + 1, so a resumed run keeps numbering."""
    last = -1
    if not csv_path.exists():
        return 0
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                last = max(last, int(row["face_id"]))
            except (KeyError, ValueError):
                continue
    return last + 1


def _emb_count(emb_path: Path) -> int:
    """Number of 512-float vectors already in the embedding journal."""
    if not emb_path.exists():
        return 0
    nbytes = emb_path.stat().st_size
    if nbytes % (EMB_DIM * 4) != 0:
        raise ValueError(
            f"{emb_path} size {nbytes} is not a multiple of {EMB_DIM*4}; "
            "the embedding journal is corrupt. Re-embed with --rescan."
        )
    return nbytes // (EMB_DIM * 4)


def _recover_desync(faces_csv: Path, emb_path: Path, done_path: Path) -> bool:
    """Trim a csv/emb pair back to a consistent prefix after a mid-write crash.

    embed writes each face's csv row then its embedding, flushing per image, so a
    kill can leave faces.csv and faces.emb differing by at most the in-flight
    image's faces. Rather than force a full --rescan (throwing away a ~40-minute
    embed), keep the longest consistent prefix: truncate both to
    min(rows, vectors). The image straddling that cut (the first dropped csv row's
    filename) may have earlier faces still inside the prefix; drop those too so it
    is re-embedded in full rather than left missing some faces. face_ids stay
    0..k-1 contiguous because we only ever keep a prefix. Returns True if it
    changed anything.
    """
    rows = read_face_rows(faces_csv)
    n_csv, n_emb = len(rows), _emb_count(emb_path)
    if n_csv == n_emb:
        return False

    n = min(n_csv, n_emb)
    keep = n
    reembed_fn = None
    # If the cut lands inside faces.csv, the image at row[n] was only partially
    # written. Pull any of its earlier faces (contiguous) out of the prefix.
    if n < len(rows):
        reembed_fn = rows[n]["filename"]
        while keep > 0 and rows[keep - 1]["filename"] == reembed_fn:
            keep -= 1

    # Rewrite faces.csv to header + first `keep` rows.
    with faces_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FACES_HEADER)
        for r in rows[:keep]:
            w.writerow([r[c] for c in FACES_HEADER])
    # Truncate the embedding journal to the same `keep` vectors.
    with emb_path.open("r+b") as f:
        f.truncate(keep * EMB_DIM * 4)
    # The straddling image must be re-embedded, so make sure it isn't marked done.
    if reembed_fn is not None:
        manifest = read_done_manifest(done_path)
        if reembed_fn in manifest:
            manifest.discard(reembed_fn)
            with done_path.open("w") as f:
                for name in sorted(manifest):
                    f.write(name + "\n")

    tail = f" (re-embedding {reembed_fn!r})" if reembed_fn else ""
    print(f"Recovered a desynced journal: faces.csv had {n_csv} rows, "
          f"{emb_path.name} had {n_emb} vectors; trimmed both to {keep}{tail}.")
    return True


def _decode(path: Path):
    """Decode one image, returning the BGR array or the Exception (logged later)."""
    try:
        return load_image_bgr(path)
    except Exception as e:  # noqa: BLE001 - surface to the main thread, keep going
        return e


def _decode_pil(path: Path):
    """Decode to an upright RGB PIL image (for cropping), or the Exception."""
    try:
        return load_image_rgb_pil(path)
    except Exception as e:  # noqa: BLE001
        return e


def _prefetch(paths, workers: int, loader=None):
    """Yield (path, decoded_or_Exception), decoding up to `workers` ahead.

    HEIC decode (libheif) is ~half of per-image time and releases the GIL, so
    decoding the next images on background threads while the main thread does its
    work (inference for embed, cropping for review) overlaps the two — ~1.5x on
    an M1, in a single process. The in-flight window is bounded so decoded 24MP
    images (~70 MB each) don't pile up in memory. `loader` defaults to BGR decode.
    """
    loader = loader or _decode
    if workers <= 0:
        for p in paths:
            yield p, loader(p)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = iter(paths)
        q: deque = deque()
        for _ in range(workers + 1):  # prime the pipeline
            try:
                p = next(it)
            except StopIteration:
                break
            q.append((p, ex.submit(loader, p)))
        while q:
            p, fut = q.popleft()
            yield p, fut.result()
            try:
                nxt = next(it)
            except StopIteration:
                continue
            q.append((nxt, ex.submit(loader, nxt)))


def cmd_embed(args) -> int:
    album = Path(args.album)
    try:
        images = list_images(album)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(e, file=sys.stderr)
        return 1
    if not images:
        print(f"No images found in '{album}'.{empty_hint(album)}", file=sys.stderr)
        return 1

    faces_csv = Path(args.faces)
    emb_path = faces_csv.with_suffix(".emb")
    done_path = faces_csv.with_suffix(".done")

    # Repair a desynced csv/emb pair (e.g. a kill between the two writes) by
    # trimming to the last consistent image, instead of forcing a full --rescan.
    if faces_csv.exists():
        _recover_desync(faces_csv, emb_path, done_path)

    # Resume skip set: images that produced a face (faces.csv) OR were recorded
    # as processed (the .done manifest, which also covers zero-face images).
    done = read_face_filenames(faces_csv) | read_done_manifest(done_path)
    todo = [p for p in images if p.name not in done]

    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(images)} images, {len(done)} already embedded, {len(todo)} to embed.")
    if not todo:
        print("Nothing to do. (Use --rescan to start over.)")
        _finalize_npy(emb_path, faces_csv)
        return 0

    print(f"Loading detector (buffalo_l, det_size={args.det_size}) ...")
    app = build_face_app(det_size=args.det_size, det_thresh=args.det_thresh)

    face_id = _next_face_id(faces_csv)
    new_csv = not faces_csv.exists()

    # Decode the next images on background threads while we run inference on the
    # current one (see _prefetch). --prefetch 0 falls back to plain serial.
    stream = _prefetch(todo, args.prefetch)
    try:
        from tqdm import tqdm

        stream = tqdm(stream, total=len(todo), unit="img")
    except ImportError:
        pass

    n_faces = 0
    # Append to all three files and flush per image. Keep csv and emb in lockstep
    # so a crash leaves an equal-length, consistent pair; record the filename in
    # the .done manifest *after* that flush so a face-bearing image is never
    # double-counted (it's already in faces.csv, which the resume union also
    # consults), while zero-face images still get marked processed.
    with faces_csv.open("a", newline="") as cf, emb_path.open("ab") as ef, \
            done_path.open("a") as df:
        writer = csv.writer(cf)
        if new_csv:
            writer.writerow(FACES_HEADER)
        for path, decoded in stream:
            if isinstance(decoded, Exception):
                print(f"\n  ! {path.name}: {decoded}")
                faces = []
            else:
                try:
                    faces = app.get(decoded)
                except Exception as e:  # noqa: BLE001 - keep going on a bad image
                    print(f"\n  ! {path.name}: {e}")
                    faces = []
            for face in faces:
                x1, y1, x2, y2 = (int(round(v)) for v in face.bbox)
                emb = face.normed_embedding.astype(np.float32)
                writer.writerow(
                    [face_id, path.name, x1, y1, x2, y2, f"{float(face.det_score):.4f}"]
                )
                ef.write(emb.tobytes())
                face_id += 1
                n_faces += 1
            cf.flush()
            ef.flush()
            df.write(path.name + "\n")
            df.flush()

    print(f"\nEmbedded {n_faces} face(s) from {len(todo)} image(s) -> {faces_csv}")
    _finalize_npy(emb_path, faces_csv)
    return 0


def read_face_rows(csv_path: Path) -> list[dict]:
    """All rows of a faces.csv as dicts (small: a few tens of thousands)."""
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_faces(faces_csv: Path):
    """Return (rows, embeddings) for a face table, asserting they line up.

    Row i of faces.csv corresponds to row i of faces.npy (face_id i).
    """
    rows = read_face_rows(faces_csv)
    npy = faces_csv.with_suffix(".npy")
    if not npy.exists():
        raise FileNotFoundError(f"{npy} not found — run `faces.py embed` first.")
    mat = np.load(npy)
    if len(rows) != len(mat):
        raise ValueError(
            f"{faces_csv} has {len(rows)} rows but {npy} has {len(mat)} vectors — "
            "are they from the same embed run?"
        )
    return rows, mat


def load_cluster_map(face_rows: list[dict], clusters_csv: Path) -> dict[str, int]:
    """Read clusters.csv as {face_id: cluster_id}, checking it matches the faces.

    cluster_ids are positional over a specific embed run, so a clusters.csv from
    a different (e.g. re-embedded) faces.csv silently produces wrong assignments.
    Catch that here: the set of face_ids must match faces.csv exactly. Cheap, and
    turns a silent-corruption footgun into a clear error pointing at re-cluster.
    """
    cmap = {r["face_id"]: int(r["cluster_id"]) for r in read_face_rows(clusters_csv)}
    face_ids = {r["face_id"] for r in face_rows}
    if set(cmap) != face_ids:
        only_clu = len(set(cmap) - face_ids)
        only_faces = len(face_ids - set(cmap))
        raise ValueError(
            f"{clusters_csv} doesn't match the current face table "
            f"({len(cmap)} clustered face_ids vs {len(face_ids)} in faces.csv; "
            f"{only_clu} only in clusters, {only_faces} only in faces). "
            "They're from different embed runs — re-run `faces.py cluster`."
        )
    return cmap


def _face_size(row: dict) -> int:
    """Longer bbox side in pixels (proxy for how usable/close a face is)."""
    return max(int(row["x2"]) - int(row["x1"]), int(row["y2"]) - int(row["y1"]))


def cmd_cluster(args) -> int:
    faces_csv = Path(args.faces)
    rows, mat = load_faces(faces_csv)

    # Pre-filter (cheap, re-runnable): drop tiny / low-confidence faces before
    # clustering. Their embeddings are noisy and are the main cause of distinct
    # kids smearing into one cluster. This is the knob that most affects purity.
    keep = np.array([
        _face_size(r) >= args.min_size and float(r["det_score"]) >= args.min_det
        for r in rows
    ])
    idx = np.where(keep)[0]
    X = mat[idx]
    print(f"{len(rows)} faces total; {len(idx)} pass pre-filter "
          f"(size>={args.min_size}px, det>={args.min_det}); {len(rows) - len(idx)} dropped.")
    if len(idx) == 0:
        print("Nothing left after pre-filter — lower --min-size/--min-det.", file=sys.stderr)
        return 1

    # HDBSCAN (default) handles the variable density of class photos: a single
    # global DBSCAN eps either dumps most faces to noise or chains many kids into
    # one mega-cluster (verified on this album). HDBSCAN extracts clusters at
    # locally-appropriate density instead. Embeddings are unit-norm, so euclidean
    # distance is monotonic in cosine — same ordering, no need for a cosine metric.
    # DBSCAN stays available (--algo dbscan) for small/dense subsets.
    if args.algo == "hdbscan":
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=(args.min_samples or None),
            metric="euclidean",
        ).fit_predict(X.astype(np.float64))
        params = f"hdbscan min_cluster_size={args.min_cluster_size}"
    else:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=args.eps, min_samples=(args.min_samples or 4),
                        metric="cosine", n_jobs=-1).fit_predict(X)
        params = f"dbscan eps={args.eps} min_samples={args.min_samples or 4}"

    # Full per-face assignment: -2 = pre-filtered out, -1 = DBSCAN noise, else id.
    full = np.full(len(rows), -2, dtype=int)
    full[idx] = labels

    out = Path(args.out)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["face_id", "cluster_id"])
        for r, c in zip(rows, full):
            w.writerow([r["face_id"], int(c)])

    from collections import Counter

    sizes = Counter(labels[labels >= 0].tolist())
    n_noise = int((labels == -1).sum())
    print(f"\n{params}: {len(sizes)} clusters, {n_noise} noise, "
          f"{len(rows) - len(idx)} pre-filtered -> {out}")
    if sizes:
        print("Largest clusters (id: faces):")
        for cid, sz in sizes.most_common(15):
            print(f"  {cid:>4}: {sz}")

    # Calibration against the enrolled reference child: we know what
    # these faces look like, so we can see whether they land in ONE clean cluster.
    refs_path = Path(args.refs)
    if refs_path.exists():
        from common import load_refs

        refs = load_refs(str(refs_path))
        best = (X @ refs.T).max(axis=1)
        ref_mask = best >= args.ref_thresh
        n_ref = int(ref_mask.sum())
        print(f"\nReference-person calibration "
              f"({n_ref} faces match references at sim>={args.ref_thresh}):")
        if n_ref:
            for cid, cnt in Counter(labels[ref_mask].tolist()).most_common():
                where = "noise" if cid == -1 else f"cluster {cid}"
                total = sizes.get(cid, cnt) if cid >= 0 else cnt
                purity = f", {100*cnt/total:.0f}% of that cluster" if cid >= 0 else ""
                print(f"  {cnt} ref faces in {where} (size {total}{purity})")
            print("Goal: ref faces concentrate in one cluster that's almost all ref.\n"
                  "  split across many clusters -> raise --eps; mixed with other kids -> lower --eps.")
    return 0


def _finalize_npy(emb_path: Path, faces_csv: Path) -> None:
    """Materialize the canonical faces.npy from the append journal."""
    if not emb_path.exists():
        return
    n = _emb_count(emb_path)
    if n == 0:
        return
    mat = np.fromfile(emb_path, dtype=np.float32).reshape(n, EMB_DIM)
    npy_path = faces_csv.with_suffix(".npy")
    np.save(npy_path, mat)
    print(f"Finalized {n} embedding(s) ({mat.nbytes/1e6:.0f} MB) -> {npy_path}")


def _even_sample(items: list, n: int) -> list:
    """Up to *n* items spread evenly across *items* (keeps variety, not just the top)."""
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    out, seen = [], set()
    for i in idx:
        i = int(i)
        if i not in seen:
            seen.add(i)
            out.append(items[i])
    return out


def _square_thumb(im, size: int):
    """Pad a face crop to a square (no distortion) and resize to size x size."""
    from PIL import Image

    w, h = im.size
    s = max(w, h, 1)
    bg = Image.new("RGB", (s, s), (32, 32, 32))
    bg.paste(im, ((s - w) // 2, (s - h) // 2))
    return bg.resize((size, size))


def _montage(thumbs: list, size: int, cols: int):
    """Tile square thumbnails into a single contact-sheet image."""
    from PIL import Image

    rows = (len(thumbs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * size, rows * size), (16, 16, 16))
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        canvas.paste(t, (c * size, r * size))
    return canvas


def cmd_review(args) -> int:
    faces_csv = Path(args.faces)
    rows = read_face_rows(faces_csv)
    if not rows:
        print(f"No faces in {faces_csv} — run `embed` first.", file=sys.stderr)
        return 1
    clu_path = Path(args.clusters)
    if not clu_path.exists():
        print(f"{clu_path} not found — run `cluster` first.", file=sys.stderr)
        return 1
    try:
        assign = load_cluster_map(rows, clu_path)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    # Group faces into their (real, non-noise) clusters.
    by_cluster: dict[int, list] = {}
    for r in rows:
        cid = assign.get(r["face_id"], -2)
        if cid >= 0:
            by_cluster.setdefault(cid, []).append(r)
    if not by_cluster:
        print("No labelable clusters (all noise/pre-filtered).", file=sys.stderr)
        return 1
    ranked = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))

    # Pick a representative spread per cluster (size-sorted, then even sample so a
    # contaminant face isn't hidden by showing only the biggest crops).
    sampled = {
        cid: _even_sample(sorted(fr, key=lambda r: -_face_size(r)), args.per_cluster)
        for cid, fr in by_cluster.items()
    }

    # Crop by loading each source image once (a photo may feed several clusters).
    need: dict[str, list] = {}
    for cid, fr in sampled.items():
        for r in fr:
            need.setdefault(r["filename"], []).append((cid, r))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    album = Path(args.album)
    crops: dict[int, list] = {}

    paths = [album / fn for fn in need]
    stream = _prefetch(paths, args.prefetch, loader=_decode_pil)
    try:
        from tqdm import tqdm

        stream = tqdm(stream, total=len(paths), unit="img", desc="cropping")
    except ImportError:
        pass
    for path, img in stream:
        if isinstance(img, Exception):
            print(f"\n  ! {path.name}: {img}")
            continue
        for cid, r in need[path.name]:
            box = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
            crops.setdefault(cid, []).append((_face_size(r), _square_thumb(img.crop(box), args.thumb)))

    # Labeling is the one piece of hand work in the pipeline, and re-running
    # review (e.g. after a re-cluster) would otherwise overwrite labels.csv with
    # an empty skeleton. Carry any names you've already typed forward by
    # cluster_id, and back up the old file before rewriting so nothing is lost
    # even if some cluster_ids changed. (Cluster ids are stable only within one
    # clusters.csv; names for ids that no longer exist survive only in the backup.)
    labels_path = Path(args.labels)
    prior_names = _read_labels(labels_path)
    if prior_names:
        from common import _unique_path

        backup = _unique_path(labels_path.with_suffix(labels_path.suffix + ".bak"))
        shutil.copy2(labels_path, backup)
        print(f"Existing {labels_path} has {len(prior_names)} name(s); backed it up -> {backup}")

    # Build one montage per cluster, biggest clusters first (most-photographed
    # kids on top), and a skeleton labels.csv to type names into.
    written = 0
    written_cids: set[int] = set()
    with labels_path.open("w", newline="") as lf:
        w = csv.writer(lf)
        w.writerow(["cluster_id", "size", "montage", "name"])
        for rank, (cid, fr) in enumerate(ranked):
            thumbs = crops.get(cid)
            if not thumbs:
                continue
            thumbs = [t for _, t in sorted(thumbs, key=lambda x: -x[0])]
            name = f"c{rank:02d}__cluster{cid}__n{len(fr)}.jpg"
            _montage(thumbs, args.thumb, args.cols).save(out_dir / name, "JPEG", quality=85)
            w.writerow([cid, len(fr), name, prior_names.get(cid, "")])
            written_cids.add(cid)
            written += 1

    carried = sum(1 for cid in prior_names if cid in written_cids)
    lost = sorted(cid for cid in prior_names if cid not in written_cids)
    print(f"\nWrote {written} cluster montage(s) -> {out_dir}/ and skeleton -> {labels_path}")
    if carried:
        print(f"Carried forward {carried} existing name(s) by cluster_id.")
    if lost:
        print(f"  ! {len(lost)} prior name(s) were for cluster id(s) not in this run "
              f"({', '.join(map(str, lost))}); they remain only in the backup.")
    print("Open the montages (biggest clusters are c00, c01, ...), then type a name in\n"
          f"the 'name' column of {labels_path} for each. Same kid in two clusters? Same name.")
    return 0


def _scene_stats(im):
    """(green, sky, bright) fractions from a downscaled RGB image.

    Outdoor play means grass/trees (green) and sky (blue) — strong, cheap signals
    that an indoor classroom lacks. Brightness alone misfires (a bright window-lit
    circle time reads 'outdoor'), so foliage+sky is the reliable cue.
    """
    a = np.asarray(im.resize((64, 64))).astype(np.float32)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    green = float(((G > R + 8) & (G > B + 8) & (G > 50)).mean())
    sky = float(((B > R + 10) & (B > G + 5) & (B > 110)).mean())
    return green, sky, float(a.mean() / 255)


def _exif_hour(path: Path) -> int:
    """Hour-of-day from EXIF DateTime, or -1 if missing."""
    from PIL import Image

    try:
        dt = Image.open(path).getexif().get(306, "")
        return int(dt.split()[1].split(":")[0]) if dt else -1
    except Exception:  # noqa: BLE001
        return -1


def _exif_dt(path: Path) -> str:
    """Raw EXIF capture datetime 'YYYY:MM:DD HH:MM:SS', or '' if missing.

    Reads only the EXIF header (no full pixel decode), so it's cheap enough to
    run over the whole album — `serve` caches the result either way.
    """
    from PIL import Image

    try:
        return Image.open(path).getexif().get(306, "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _parse_hours(spec: str) -> set:
    """'10-11' -> {10, 11}; also accepts a comma list like '10,11'."""
    out: set = set()
    for part in spec.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def cmd_scene(args) -> int:
    album = Path(args.album)
    try:
        images = list_images(album)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(e, file=sys.stderr)
        return 1
    if args.faces and Path(args.faces).exists():
        wanted = {r["filename"] for r in read_face_rows(Path(args.faces))}
        images = [p for p in images if p.name in wanted]
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"No images to classify in '{album}'.{empty_hint(album)}", file=sys.stderr)
        return 1

    hours = _parse_hours(args.outdoor_hours)
    out = Path(args.out)
    rows = []

    # 'time' needs no pixels — just the EXIF hour — so it skips decode entirely
    # (instant). 'green' and 'both' decode for the foliage/sky colour signal.
    # 'both' = outdoor only if the colour says so AND it's in the outdoor window
    # (rejects green classroom decor at non-outdoor hours).
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "hour", "green", "sky", "bright", "scene"])
        if args.method == "time":
            seq = images
            try:
                from tqdm import tqdm
                seq = tqdm(images, unit="img", desc="scene(time)")
            except ImportError:
                pass
            for path in seq:
                hr = _exif_hour(path)
                scene = "outdoor" if hr in hours else "indoor"
                w.writerow([path.name, hr, "", "", "", scene])
                rows.append((hr, scene))
        else:
            stream = _prefetch(images, args.prefetch, loader=_decode_pil)
            try:
                from tqdm import tqdm
                stream = tqdm(stream, total=len(images), unit="img", desc="scene")
            except ImportError:
                pass
            for path, im in stream:
                if isinstance(im, Exception):
                    continue
                g, s, b = _scene_stats(im)
                hr = _exif_hour(path)
                green_out = (g + s) >= args.thresh
                out_flag = (green_out and hr in hours) if args.method == "both" else green_out
                scene = "outdoor" if out_flag else "indoor"
                w.writerow([path.name, hr, f"{g:.3f}", f"{s:.3f}", f"{b:.3f}", scene])
                rows.append((hr, scene))

    from collections import Counter

    sc = Counter(s for _, s in rows)
    print(f"\n{len(rows)} images -> {out}: {sc.get('outdoor',0)} outdoor, {sc.get('indoor',0)} indoor")
    print("\nby hour (validates against the daily schedule):")
    print(f"  {'hour':>4} {'outdoor':>8} {'indoor':>7}")
    for hr in sorted({h for h, _ in rows if h >= 0}):
        o = sum(1 for h, s in rows if h == hr and s == "outdoor")
        i = sum(1 for h, s in rows if h == hr and s == "indoor")
        print(f"  {hr:>4} {o:>8} {i:>7}")
    return 0


def _read_labels(path: Path) -> dict[int, str]:
    """cluster_id -> name from labels.csv, skipping rows with no name typed in."""
    out: dict[int, str] = {}
    for r in read_face_rows(path):
        name = (r.get("name") or "").strip()
        if not name:
            continue
        try:
            out[int(r["cluster_id"])] = name
        except (KeyError, ValueError):
            continue
    return out


def _materialize(filenames, name: str, album: Path, base: Path, copy: bool) -> int:
    """Put each file under base/name/ as a symlink (default) or real copy."""
    from common import _unique_path

    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for fn in sorted(filenames):
        src = (album / fn).resolve()
        if not src.exists():
            continue
        dst = _unique_path(d / fn)
        try:
            if copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src)
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ! {fn} -> {name}: {e}")
    return n


def cmd_assign(args) -> int:
    rows = read_face_rows(Path(args.faces))
    if not rows:
        print(f"No faces in {args.faces} — run `embed` first.", file=sys.stderr)
        return 1
    clusters_path = Path(args.clusters)
    if not clusters_path.exists():
        print(f"{clusters_path} not found — run `cluster` first.", file=sys.stderr)
        return 1
    try:
        clusters = load_cluster_map(rows, clusters_path)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    labels = _read_labels(Path(args.labels))
    if not labels:
        print(f"No names filled into {args.labels} yet — run `review`, then type names.",
              file=sys.stderr)
        return 1

    # Warn (don't fail) if labels.csv names cluster ids that aren't in clusters.csv
    # — a sign the labels predate the current clustering and may be misaligned.
    stale = sorted(set(labels) - set(clusters.values()))
    if stale:
        print(f"  ! {len(stale)} labeled cluster id(s) are absent from {clusters_path} "
              f"({', '.join(map(str, stale[:10]))}{'...' if len(stale) > 10 else ''}); "
              "labels may be stale — re-run `review` if assignments look wrong.",
              file=sys.stderr)

    # filename -> set of named people present (via each face's cluster's label).
    img_names: dict[str, set] = {}
    images: set = set()
    for r in rows:
        fn = r["filename"]
        images.add(fn)
        name = labels.get(clusters.get(r["face_id"], -2))
        if name:
            img_names.setdefault(fn, set()).add(name)

    # Optional noise recovery: faces that fell to noise / junk / unlabeled
    # clusters but sit close to one named cluster's centroid get pulled back in.
    # Strict by design — a face is only recovered if it clears --recover-thresh
    # AND beats the runner-up cluster by --recover-margin, so ambiguous faces
    # (including unlabeled kids) stay out rather than get misassigned.
    if args.recover:
        from collections import defaultdict

        mat = np.load(Path(args.faces).with_suffix(".npy"))
        if len(mat) != len(rows):
            print("faces.npy/csv length mismatch — can't recover.", file=sys.stderr)
            return 1
        members: dict[int, list] = defaultdict(list)
        for i, r in enumerate(rows):
            c = clusters.get(r["face_id"], -2)
            if c in labels:
                members[c].append(i)
        cids = sorted(members)
        if len(cids) < 2:
            print("Need >=2 named clusters to recover (margin test).", file=sys.stderr)
        else:
            cnames = [labels[c] for c in cids]
            cent = np.vstack([mat[members[c]].mean(0) for c in cids])
            cent /= np.linalg.norm(cent, axis=1, keepdims=True)
            pool = [i for i, r in enumerate(rows) if clusters.get(r["face_id"], -2) not in labels]
            sims = mat[pool].astype(np.float32) @ cent.T
            order = np.argsort(-sims, axis=1)
            ar = np.arange(len(pool))
            best = sims[ar, order[:, 0]]
            second = sims[ar, order[:, 1]]
            ok = (best >= args.recover_thresh) & ((best - second) >= args.recover_margin)
            n = 0
            for k, i in enumerate(pool):
                if ok[k]:
                    img_names.setdefault(rows[i]["filename"], set()).add(cnames[order[k, 0]])
                    n += 1
            print(f"recovered {n} faces (thresh {args.recover_thresh}, margin {args.recover_margin})")

    out_csv = Path(args.out)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "names"])
        for fn in sorted(images):
            w.writerow([fn, ";".join(sorted(img_names.get(fn, ())))])

    from collections import Counter

    counts = Counter(n for names in img_names.values() for n in names)
    print(f"{len(images)} images with faces; {len(img_names)} have >=1 named person "
          f"({len(images) - len(img_names)} unnamed) -> {out_csv}")
    for name, c in counts.most_common():
        print(f"  {name}: {c} photos")

    if args.folders:
        names_filter = {s.strip() for s in args.names.split(",")} if args.names else None
        base = Path(args.folders)
        per_name: dict[str, list] = {}
        for fn, names in img_names.items():
            for name in names:
                if names_filter and name not in names_filter:
                    continue
                per_name.setdefault(name, []).append(fn)
        kind = "copies" if args.copy else "symlinks"
        print(f"\nWriting {kind} into {base}/<name>/ ...")
        for name, fns in sorted(per_name.items()):
            n = _materialize(fns, name, Path(args.album), base, args.copy)
            print(f"  {name}/: {n}")
    return 0


def _name_set(spec: str) -> set:
    return {s.strip() for s in spec.split(",") if s.strip()} if spec else set()


def _query_match(names: set, want: set, any_of: set, without: set, only: set) -> bool:
    """Whether a photo's *names* satisfy the query sets. See DESIGN.md's table.

    --only is exact-set and wins outright; otherwise --with is a subset test
    (those present, others allowed), --any needs an intersection, and --without
    excludes. Pure set logic, factored out of cmd_query so it can be unit-tested.
    """
    if only:
        return names == only
    if want and not want <= names:
        return False
    if any_of and not (names & any_of):
        return False
    if without and (names & without):
        return False
    return True


def cmd_query(args) -> int:
    ip = Path(args.image_people)
    if not ip.exists():
        print(f"{ip} not found — run `assign` first.", file=sys.stderr)
        return 1
    data = {
        r["filename"]: set(filter(None, (r.get("names") or "").split(";")))
        for r in read_face_rows(ip)
    }

    want = _name_set(args.with_)
    any_of = _name_set(args.any)
    without = _name_set(args.without)
    only = _name_set(args.only)
    if not (want or any_of or only):
        print("Give at least one of --with / --any / --only.", file=sys.stderr)
        return 1

    # Optional indoor/outdoor filter from a `scene` pass.
    where_ok = None
    if args.where:
        scene_path = Path(args.scene)
        if not scene_path.exists():
            print(f"{scene_path} not found — run `scene` first (or drop --where).", file=sys.stderr)
            return 1
        where_ok = {r["filename"] for r in read_face_rows(scene_path) if r["scene"] == args.where}

    selected = sorted(
        fn for fn, names in data.items()
        if _query_match(names, want, any_of, without, only)
        and (where_ok is None or fn in where_ok)
    )

    # Human-readable folder name for the query.
    if only:
        label = "only_" + "_".join(sorted(only))
    else:
        parts = []
        if want:
            parts.append("with_" + "_".join(sorted(want)))
        if any_of:
            parts.append("any_" + "_".join(sorted(any_of)))
        label = "__".join(parts) or "query"
    if without:
        label += "__not_" + "_".join(sorted(without))
    if args.where:
        label += "__" + args.where

    print(f"{len(selected)} image(s) match {label}")
    if not selected:
        return 0
    if args.dry_run:
        for fn in selected[:50]:
            print(f"  {fn}")
        if len(selected) > 50:
            print(f"  ... (+{len(selected) - 50} more)")
        return 0

    # query results go directly in out/<label>/ (not nested under a name like
    # by_child/), so we copy/symlink here rather than reuse _materialize.
    #
    # Wipe-and-rewrite: out/<label>/ is named for this exact query, so it should
    # *be* its result set, not an append log. query is the fast re-runnable
    # tune-loop step, so re-running must be idempotent -- otherwise a stale prior
    # run lingers (and, since filenames within one run are already unique, the
    # only collisions are with that stale run, which used to spawn _1 dup links).
    # This is deliberately unlike collect, which is intentionally append-only.
    out = Path(args.out) / label
    if out.exists():
        shutil.rmtree(out)
        print(f"Cleared existing {out}/")
    out.mkdir(parents=True, exist_ok=True)

    album = Path(args.album)
    n = 0
    for fn in selected:
        src = (album / fn).resolve()
        if not src.exists():
            continue
        try:
            if args.jpeg:
                # Re-encode to JPEG. macOS Finder thumbnails HEIC unreliably
                # (its icon service bails on many of these files even though
                # Quick Look can decode them); JPEG always thumbnails. Open
                # WITHOUT exif_transpose so the original pixels + orientation tag
                # stay together, and copy EXIF verbatim unless stripped.
                from PIL import Image

                dst = out / f"{Path(fn).stem}.jpg"
                img = Image.open(src)
                exif = None if args.strip_exif else img.info.get("exif")
                img = img.convert("RGB")
                if args.max_size:
                    img.thumbnail((args.max_size, args.max_size))
                save_kw = {"quality": args.jpeg_quality}
                if exif:
                    save_kw["exif"] = exif
                img.save(dst, "JPEG", **save_kw)
            elif args.copy:
                shutil.copy2(src, out / fn)
            else:
                (out / fn).symlink_to(src)
            n += 1
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't abort the query
            print(f"  ! {fn}: {e}")
    kind = "jpegs" if args.jpeg else ("copies" if args.copy else "symlinks")
    print(f"Wrote {n} {kind} -> {out}/")
    return 0


SERVE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>headcount — browse</title>
<style>
  :root { --bg:#16181d; --panel:#1f232b; --ink:#e7e9ee; --muted:#9aa3b2; --accent:#5b9dff; --line:#2c313b; --cell:150px; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }
  .wrap { display:flex; height:100vh; }
  aside { width:280px; flex:none; background:var(--panel); border-right:1px solid var(--line); padding:16px; overflow-y:auto; }
  main { flex:1; overflow-y:auto; padding:16px 20px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:20px 0 8px; }
  h2:first-child { margin-top:0; }
  input[type=search], select { width:100%; padding:7px 9px; background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:7px; }
  .names { max-height:38vh; overflow-y:auto; border:1px solid var(--line); border-radius:7px; padding:6px; margin-top:8px; }
  .names label { display:flex; align-items:center; gap:8px; padding:4px 6px; border-radius:5px; cursor:pointer; }
  .names label:hover { background:var(--bg); }
  .names .count { margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; }
  .modes { display:flex; gap:14px; margin-top:8px; color:var(--muted); }
  .modes label { display:flex; gap:5px; align-items:center; cursor:pointer; }
  /* dual-thumb range: two transparent sliders overlaid on one shared track */
  .rangewrap { position:relative; height:20px; margin:10px 8px 0; }
  .rangewrap .track { position:absolute; top:50%; left:0; right:0; height:4px; transform:translateY(-50%); background:var(--line); border-radius:2px; }
  .rangewrap .fill { position:absolute; top:50%; height:4px; transform:translateY(-50%); background:var(--accent); border-radius:2px; }
  .rangewrap input[type=range] { position:absolute; top:50%; transform:translateY(-50%); left:0; width:100%; height:16px; margin:0; padding:0; background:none; pointer-events:none; -webkit-appearance:none; appearance:none; }
  .rangewrap input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; appearance:none; pointer-events:auto; width:16px; height:16px; border-radius:50%; background:var(--accent); border:2px solid var(--bg); cursor:pointer; }
  .rangewrap input[type=range]::-moz-range-thumb { pointer-events:auto; width:14px; height:14px; border-radius:50%; background:var(--accent); border:2px solid var(--bg); cursor:pointer; }
  .hourlab { margin-top:8px; font-variant-numeric:tabular-nums; color:var(--muted); }
  /* single-thumb range for thumbnail size */
  .sizerange { width:100%; margin:10px 0 0; height:16px; -webkit-appearance:none; appearance:none; background:none; cursor:pointer; }
  .sizerange::-webkit-slider-runnable-track { height:4px; border-radius:2px; background:var(--line); }
  .sizerange::-moz-range-track { height:4px; border-radius:2px; background:var(--line); }
  .sizerange::-webkit-slider-thumb { -webkit-appearance:none; appearance:none; margin-top:-6px; width:16px; height:16px; border-radius:50%; background:var(--accent); border:2px solid var(--bg); }
  .sizerange::-moz-range-thumb { width:14px; height:14px; border-radius:50%; background:var(--accent); border:2px solid var(--bg); }
  /* sticky header wraps the active-filter chips + the controls bar */
  .topbar { position:sticky; top:-16px; z-index:5; background:var(--bg); padding:12px 0; margin-bottom:14px; border-bottom:1px solid var(--line); }
  .active { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin-bottom:10px; }
  .active:empty { display:none; }
  .active .lead { color:var(--muted); }
  .chip { display:inline-flex; align-items:center; gap:5px; background:var(--bg); border:1px solid var(--line); border-radius:999px; padding:3px 7px 3px 11px; font-size:13px; }
  .chip button { background:none; border:0; color:var(--muted); cursor:pointer; padding:0; font-size:15px; line-height:1; }
  .chip button:hover { color:var(--ink); }
  .active .clear { background:none; border:0; color:var(--accent); cursor:pointer; font-weight:600; padding:2px 4px; margin-left:4px; }
  .active .clear:hover { text-decoration:underline; }
  .bar { display:flex; align-items:center; gap:14px; }
  .bar .n { font-weight:600; }
  .bar .spacer { flex:1; }
  button { background:var(--accent); color:#fff; border:0; padding:8px 16px; border-radius:7px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.4; cursor:default; }
  .ctl { display:flex; align-items:center; gap:8px; color:var(--muted); }
  .ctl select { width:auto; }
  .grid { display:flex; flex-wrap:wrap; gap:8px; align-items:flex-start; }
  .grid .cell { width:var(--cell); height:var(--cell); flex:none; background:var(--panel); border-radius:8px; overflow:hidden; cursor:pointer; border:0; padding:0;
    transition:width .12s ease, height .12s ease; content-visibility:auto; contain-intrinsic-size:var(--cell); }
  .grid img { width:100%; height:100%; object-fit:cover; display:block; }
  /* slim vertical date marker sitting inline between each day's photos */
  .dhead { flex:none; width:30px; height:var(--cell); transition:height .12s ease; display:flex; align-items:center; justify-content:center;
           writing-mode:vertical-rl; text-orientation:mixed; white-space:nowrap;
           font-size:11px; font-weight:600; color:var(--muted); letter-spacing:.02em;
           border-left:2px solid var(--line); }
  .empty { color:var(--muted); padding:40px 0; text-align:center; }
  /* lightbox */
  #lb { display:none; position:fixed; inset:0; background:rgba(8,9,12,.92); z-index:50;
        flex-direction:column; align-items:center; justify-content:center; }
  #lbstage { position:relative; flex:1; width:100%; display:flex; align-items:center; justify-content:center; min-height:0; padding:48px 64px 8px; }
  #lbimg { max-width:100%; max-height:100%; object-fit:contain; border-radius:6px;
           transition:filter .15s; cursor:zoom-in; }
  #lbimg.loading { filter:blur(8px); }
  #lbcap { padding:6px 16px 14px; color:var(--muted); text-align:center; }
  #lbcap b { color:var(--ink); }
  #lbcap a { color:var(--accent); text-decoration:none; margin-left:10px; }
  #lbcap a:hover { text-decoration:underline; }
  .lbbtn { position:absolute; top:50%; transform:translateY(-50%); background:rgba(255,255,255,.08);
           color:#fff; border:0; width:44px; height:64px; border-radius:8px; font-size:24px; cursor:pointer; }
  .lbbtn:hover { background:rgba(255,255,255,.16); }
  #lbprev { left:10px; } #lbnext { right:10px; }
  #lbclose { position:absolute; top:14px; right:18px; background:none; border:0; color:#fff; font-size:30px; cursor:pointer; width:auto; height:auto; }
</style>
</head>
<body>
<div class="wrap">
  <aside>
    <h2>Names</h2>
    <input type="search" id="nsearch" placeholder="filter names…" autocomplete="off">
    <div class="modes">
      <label><input type="radio" name="mode" value="all" checked> all of</label>
      <label><input type="radio" name="mode" value="any"> any of</label>
    </div>
    <div class="names" id="names"></div>
    <h2>Time of day</h2>
    <div class="rangewrap">
      <div class="track"></div>
      <div class="fill" id="hfill"></div>
      <input type="range" id="hmin" step="1">
      <input type="range" id="hmax" step="1">
    </div>
    <div class="hourlab" id="hourlab"></div>
    <h2>Scene</h2>
    <select id="scene">
      <option value="">Any</option>
      <option value="indoor">Indoor</option>
      <option value="outdoor">Outdoor</option>
    </select>
    <h2>Preview size</h2>
    <input type="range" class="sizerange" id="csize" min="90" max="300" step="10">
  </aside>
  <main>
    <div class="topbar">
    <div class="active" id="active"></div>
    <div class="bar">
      <span class="n" id="count"></span>
      <span class="ctl">sort
        <select id="sort">
          <option value="new">Newest first</option>
          <option value="old">Oldest first</option>
        </select>
      </span>
      <span class="spacer" style="flex:1"></span>
      <span class="ctl">export as
        <select id="fmt">
          <option value="orig">originals (full quality)</option>
          <option value="jpeg">JPEG 2048px (smaller, universal)</option>
        </select>
      </span>
      <button id="export">Export zip</button>
    </div>
    </div>
    <div class="grid" id="grid"></div>
  </main>
</div>

<div id="lb">
  <button id="lbclose" title="Close (Esc)">&times;</button>
  <div id="lbstage">
    <button class="lbbtn" id="lbprev" title="Previous (←)">&#8249;</button>
    <img id="lbimg" alt="">
    <button class="lbbtn" id="lbnext" title="Next (→)">&#8250;</button>
  </div>
  <div id="lbcap"></div>
</div>

<script>
const DATA = __MANIFEST__;
const S = { names:new Set(), mode:"all", hmin:0, hmax:23, scene:"", search:"", sort:"new", cell:150 };
const MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"];

// hour bounds present in the data (photos with a known hour)
const hrs = DATA.items.map(i => i.h).filter(h => h !== null);
const HMIN = hrs.length ? Math.min(...hrs) : 0, HMAX = hrs.length ? Math.max(...hrs) : 23;
S.hmin = HMIN; S.hmax = HMAX;

// --- persist filters across refreshes (per-origin localStorage) ---
const SKEY = "headcount.filters.v1";
function saveState() {
  try { localStorage.setItem(SKEY, JSON.stringify({
    names:[...S.names], mode:S.mode, hmin:S.hmin, hmax:S.hmax, scene:S.scene, sort:S.sort, cell:S.cell
  })); } catch (e) {}
}
function loadState() {
  let v; try { v = JSON.parse(localStorage.getItem(SKEY) || "null"); } catch (e) { return; }
  if (!v) return;
  const known = new Set(DATA.names);                              // drop names absent from this album
  if (Array.isArray(v.names)) S.names = new Set(v.names.filter(n => known.has(n)));
  if (v.mode === "all" || v.mode === "any") S.mode = v.mode;
  if (typeof v.hmin === "number") S.hmin = Math.min(Math.max(v.hmin, HMIN), HMAX);   // clamp to data bounds
  if (typeof v.hmax === "number") S.hmax = Math.min(Math.max(v.hmax, HMIN), HMAX);
  if (S.hmin > S.hmax) { S.hmin = HMIN; S.hmax = HMAX; }
  if (v.scene === "indoor" || v.scene === "outdoor") S.scene = v.scene;
  if (v.sort === "old" || v.sort === "new") S.sort = v.sort;
  if (typeof v.cell === "number") S.cell = Math.min(Math.max(v.cell, 90), 300);
}
loadState();

const $ = id => document.getElementById(id);
const dayOf = it => it.dt ? it.dt.slice(0,10).replace(/:/g,"-") : "";        // "2026-06-11" or ""
function prettyDay(d) {
  if (!d) return "Undated";
  const [y,m,day] = d.split("-");
  return MONTHS[+m-1] + " " + (+day) + ", " + y;
}
function prettyDayShort(d) {                       // compact mm/dd/yyyy label for the vertical marker
  if (!d) return "Undated";
  const [y,m,day] = d.split("-");
  return m + "/" + day + "/" + y;
}
function prettyTime(dt) {                           // "2026:06:03 09:23:03" -> "9:23 AM"
  const t = (dt || "").split(" ")[1];
  if (!t) return "";
  let [h, m] = t.split(":").map(Number);
  const ap = h < 12 ? "AM" : "PM";
  h = h % 12 || 12;
  return h + ":" + String(m).padStart(2,"0") + " " + ap;
}

function matches(it) {
  if (S.names.size) {
    const has = it.n.filter(n => S.names.has(n)).length;
    if (S.mode === "all" && has < S.names.size) return false;
    if (S.mode === "any" && has === 0) return false;
  }
  if (it.h !== null && (it.h < S.hmin || it.h > S.hmax)) return false;   // unknown hour always passes
  if (S.scene && it.s !== S.scene) return false;
  return true;
}

// active name filters shown above the count, each removable; plus "Clear all"
function renderActive() {
  const box = $("active"); box.innerHTML = "";
  const names = [...S.names].sort();
  if (!names.length) return;                                   // :empty hides the row
  const lead = document.createElement("span"); lead.className = "lead"; lead.textContent = "Filtering:";
  box.appendChild(lead);
  for (const n of names) {
    const chip = document.createElement("span"); chip.className = "chip";
    chip.append(document.createTextNode(n));
    const x = document.createElement("button"); x.type = "button"; x.textContent = "\\u00d7"; x.title = "Remove " + n;
    x.onclick = () => { S.names.delete(n); buildNames(); render(); };
    chip.appendChild(x); box.appendChild(chip);
  }
  const clr = document.createElement("button"); clr.type = "button"; clr.className = "clear"; clr.textContent = "Clear all";
  clr.onclick = () => { S.names.clear(); buildNames(); render(); };
  box.appendChild(clr);
}

let current = [];
function render() {
  saveState();
  renderActive();
  current = DATA.items.filter(matches);
  current.sort((a,b) => {
    const x = a.dt || "", y = b.dt || "";
    if (!x && !y) return 0;
    if (!x) return 1;            // undated sinks to the end either way
    if (!y) return -1;
    return S.sort === "old" ? (x < y ? -1 : x > y ? 1 : 0) : (x < y ? 1 : x > y ? -1 : 0);
  });
  $("count").textContent = current.length + " photo" + (current.length === 1 ? "" : "s");
  $("export").disabled = current.length === 0;
  const grid = $("grid");
  if (!current.length) { grid.innerHTML = '<div class="empty">No photos match these filters.</div>'; return; }
  grid.innerHTML = "";
  const frag = document.createDocumentFragment();
  let lastDay = null;
  current.forEach((it, i) => {
    const day = dayOf(it);
    if (day !== lastDay) {
      lastDay = day;
      const sameDay = current.filter(o => dayOf(o) === day).length;
      const h = document.createElement("div"); h.className = "dhead";
      h.textContent = prettyDayShort(day);
      h.title = prettyDay(day) + " · " + sameDay + " photo" + (sameDay === 1 ? "" : "s");
      frag.appendChild(h);
    }
    const cell = document.createElement("button");
    cell.className = "cell"; cell.title = it.n.join(", "); cell.onclick = () => openLB(i);
    const img = document.createElement("img");
    img.loading = "lazy"; img.src = "/thumb/" + encodeURIComponent(it.k);
    cell.appendChild(img); frag.appendChild(cell);
  });
  grid.appendChild(frag);
}

function hourLabel() {
  $("hourlab").textContent = (S.hmin === HMIN && S.hmax === HMAX)
    ? "Any time" : (String(S.hmin).padStart(2,"0") + ":00 – " + String(S.hmax).padStart(2,"0") + ":59");
}

function buildNames() {
  const box = $("names"); box.innerHTML = "";
  const counts = {};
  for (const n of DATA.names) counts[n] = 0;
  for (const it of DATA.items) for (const n of it.n) counts[n] = (counts[n]||0) + 1;
  for (const n of DATA.names) {
    if (S.search && !n.toLowerCase().includes(S.search)) continue;
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = S.names.has(n);
    cb.onchange = () => { cb.checked ? S.names.add(n) : S.names.delete(n); render(); };
    const span = document.createElement("span"); span.textContent = n;
    const c = document.createElement("span"); c.className = "count"; c.textContent = counts[n];
    lab.append(cb, span, c); box.appendChild(lab);
  }
}

// --- lightbox: show the (already-loaded) thumbnail instantly, swap in the
// 2048px render when it arrives; clicking the image opens the original. ---
let lbi = -1;
function openLB(i) {
  lbi = i; const it = current[i];
  const img = $("lbimg");
  img.classList.add("loading");
  img.src = "/thumb/" + encodeURIComponent(it.k);     // instant placeholder
  const orig = "/full/" + encodeURIComponent(it.k) + "?full=1";
  img.title = "Open full resolution";
  img.onclick = () => window.open(orig, "_blank", "noopener");
  const full = new Image();
  full.onload = () => { if (lbi === i) { img.src = full.src; img.classList.remove("loading"); } };
  full.src = "/full/" + encodeURIComponent(it.k);
  const day = dayOf(it), time = prettyTime(it.dt);
  $("lbcap").innerHTML = "<b>" + (it.n.join(", ") || "(no names)") + "</b> · " + prettyDay(day) +
    (time ? " · " + time : "") +
    ' <a href="' + orig + '" target="_blank" rel="noopener">full resolution &#8599;</a>';
  $("lbprev").style.visibility = i > 0 ? "visible" : "hidden";
  $("lbnext").style.visibility = i < current.length - 1 ? "visible" : "hidden";
  $("lb").style.display = "flex";
}
function closeLB() { $("lb").style.display = "none"; lbi = -1; }
function navLB(d) { const n = lbi + d; if (n >= 0 && n < current.length) openLB(n); }

$("lbclose").onclick = closeLB;
$("lbprev").onclick = () => navLB(-1);
$("lbnext").onclick = () => navLB(1);
$("lb").onclick = e => { if (e.target === $("lb") || e.target.id === "lbstage") closeLB(); };
document.addEventListener("keydown", e => {
  if ($("lb").style.display === "none") return;
  if (e.key === "Escape") closeLB();
  else if (e.key === "ArrowLeft") navLB(-1);
  else if (e.key === "ArrowRight") navLB(1);
});

$("nsearch").oninput = e => { S.search = e.target.value.trim().toLowerCase(); buildNames(); };
for (const r of document.querySelectorAll('input[name=mode]')) {
  r.checked = (r.value === S.mode);
  r.onchange = e => { S.mode = e.target.value; render(); };
}
$("sort").value = S.sort;
$("sort").onchange = e => { S.sort = e.target.value; render(); };
// dual range spans the actual hour bounds, so full extent == "Any time"
const hmin = $("hmin"), hmax = $("hmax");
hmin.min = hmax.min = HMIN; hmin.max = hmax.max = HMAX;
hmin.value = S.hmin; hmax.value = S.hmax;
function updFill() {
  const span = Math.max(1, HMAX - HMIN);
  $("hfill").style.left = (S.hmin - HMIN) / span * 100 + "%";
  $("hfill").style.right = (HMAX - S.hmax) / span * 100 + "%";
}
hmin.oninput = e => { S.hmin = Math.min(+e.target.value, S.hmax); e.target.value = S.hmin; hourLabel(); updFill(); render(); };
hmax.oninput = e => { S.hmax = Math.max(+e.target.value, S.hmin); e.target.value = S.hmax; hourLabel(); updFill(); render(); };
$("scene").value = S.scene;
$("scene").onchange = e => { S.scene = e.target.value; render(); };
// thumbnail size is pure CSS (a custom property) — no re-render needed
const csize = $("csize");
csize.value = S.cell;
document.documentElement.style.setProperty("--cell", S.cell + "px");   // reflect restored size at load
let sizeRAF = 0;   // coalesce rapid input events into one --cell write per frame
csize.oninput = e => {
  S.cell = +e.target.value;
  saveState();
  if (sizeRAF) return;
  sizeRAF = requestAnimationFrame(() => { sizeRAF = 0; document.documentElement.style.setProperty("--cell", S.cell + "px"); });
};

$("export").onclick = async () => {
  const btn = $("export"); btn.disabled = true; const was = btn.textContent; btn.textContent = "Zipping…";
  try {
    const res = await fetch("/export", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ keys: current.map(i => i.k), fmt: $("fmt").value })
    });
    if (!res.ok) throw new Error(await res.text());
    const blob = await res.blob();
    const url = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = url; a.download = "headcount_export.zip"; a.click(); URL.revokeObjectURL(url);
  } catch (err) { alert("Export failed: " + err.message); }
  finally { btn.textContent = was; btn.disabled = current.length === 0; }
};

buildNames(); hourLabel(); updFill(); render();
</script>
</body>
</html>"""


def _build_thumbs(items, album, cache, size, workers):
    """Pre-render a square-ish JPEG thumbnail per item into *cache* (idempotent).

    HEIC decode is the slow part, so we cache by key and skip ones already done
    on re-runs. EXIF (incl. GPS) is dropped from thumbnails — they only need to
    look right in a grid. Mirrors cmd_query's prefetch-threaded decode style.

    The cache records the size it was built at in a `.thumb_size` marker; when
    the requested size differs (or the marker is missing) the existing thumbs
    are stale, so we clear them and rebuild. Without this, the `exists()` skip
    below would silently keep serving thumbs at whatever size they were first
    built — a smaller `--thumb` later looks fine, a larger one stays blurry.
    """
    marker = cache / ".thumb_size"
    prev = marker.read_text().strip() if marker.exists() else None
    if prev != str(size):
        stale = list(cache.glob("*.jpg"))
        if stale:
            print(f"Thumbnails: size changed ({prev or 'unknown'} -> {size}px), "
                  f"clearing {len(stale)} stale thumbs.")
            for f in stale:
                f.unlink()
        marker.write_text(str(size))

    todo = [it for it in items if not (cache / f"{it['k']}.jpg").exists()]
    if not todo:
        print(f"Thumbnails: {len(items)} cached, 0 to build.")
        return
    print(f"Thumbnails: building {len(todo)} (of {len(items)}) -> {cache}/ ...")
    done = 0

    def one(it):
        src = (album / it["f"]).resolve()
        if not src.exists():
            return
        img = load_image_rgb_pil(src)   # upright RGB; EXIF dropped on save below
        img.thumbnail((size, size))
        img.save(cache / f"{it['k']}.jpg", "JPEG", quality=80)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for _ in ex.map(one, todo):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(todo)}")
    print(f"  done ({len(todo)} built).")


def cmd_serve(args) -> int:
    import http.server
    import io
    import json
    import socketserver
    import tempfile
    import webbrowser
    import zipfile

    ip = Path(args.image_people)
    if not ip.exists():
        print(f"{ip} not found — run `assign` first.", file=sys.stderr)
        return 1

    people = {
        r["filename"]: sorted(filter(None, (r.get("names") or "").split(";")))
        for r in read_face_rows(ip)
    }

    # Optional scene/hour overlay — present iff a `scene` pass has been run.
    hours, scenes = {}, {}
    sp = Path(args.scene)
    if sp.exists():
        for r in read_face_rows(sp):
            h = r.get("hour")
            hours[r["filename"]] = int(h) if h not in (None, "") and str(h).isdigit() else None
            scenes[r["filename"]] = r.get("scene") or ""

    album = Path(args.album)
    # `k` is a filesystem-safe key (the stem) used for thumb/full URLs and as the
    # cache filename; `f` is the real album filename used to read original bytes.
    # Stems can collide (e.g. IMG_1.HEIC vs IMG_1.JPG — the same case _unique_path
    # guards) — suffix dupes so neither photo gets silently shadowed.
    items, by_key = [], {}
    for fn, names in sorted(people.items()):
        if not (album / fn).exists():
            continue
        key = Path(fn).stem
        if key in by_key:
            i = 1
            while f"{key}-{i}" in by_key:
                i += 1
            key = f"{key}-{i}"
        it = {"k": key, "f": fn, "n": names,
              "h": hours.get(fn), "s": scenes.get(fn, "")}
        items.append(it)
        by_key[key] = it
    if not items:
        print(f"No album files found for the {len(people)} indexed photos under {album}/.", file=sys.stderr)
        return 1

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    _build_thumbs(items, album, cache, args.thumb, args.prefetch)

    # Capture-date sidecar for sort + date grouping. Re-reading EXIF for every
    # photo each launch is slow, so cache filename -> datetime in the thumb cache
    # and only extract ones we haven't recorded yet.
    dates_path = cache / "dates.json"
    dates = {}
    if dates_path.exists():
        try:
            dates = json.loads(dates_path.read_text())
        except Exception:  # noqa: BLE001
            dates = {}
    missing = [it for it in items if it["f"] not in dates]
    if missing:
        print(f"Reading capture dates for {len(missing)} photo(s) ...")
        with ThreadPoolExecutor(max_workers=max(1, args.prefetch)) as ex:
            for fn, dt in ex.map(lambda it: (it["f"], _exif_dt(album / it["f"])), missing):
                dates[fn] = dt
        dates_path.write_text(json.dumps(dates))
    for it in items:
        it["dt"] = dates.get(it["f"], "")

    all_names = sorted({n for it in items for n in it["n"]})
    manifest = {"names": all_names,
                "items": [{"k": it["k"], "n": it["n"], "h": it["h"], "s": it["s"], "dt": it["dt"]}
                          for it in items]}
    page = SERVE_PAGE.replace("__MANIFEST__", json.dumps(manifest))

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet: no per-request console spam
            pass

        def _send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            from urllib.parse import unquote, urlparse
            parts = urlparse(self.path)
            path, query = unquote(parts.path), parts.query
            if path == "/":
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif path.startswith("/thumb/"):
                it = by_key.get(path[len("/thumb/"):])
                f = cache / f"{it['k']}.jpg" if it else None
                if f and f.exists():
                    self._send(200, f.read_bytes(), "image/jpeg",
                               {"Cache-Control": "max-age=3600"})
                else:
                    self._send(404, b"not found", "text/plain")
            elif path.startswith("/full/"):
                it = by_key.get(path[len("/full/"):])
                src = (album / it["f"]) if it else None
                if src and src.exists():
                    img = load_image_rgb_pil(src)
                    if "full=1" not in query:          # default: 2048px preview; full=1 serves full res
                        img.thumbnail((2048, 2048))
                    buf = io.BytesIO()
                    img.save(buf, "JPEG", quality=90)
                    self._send(200, buf.getvalue(), "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            from urllib.parse import urlparse
            if urlparse(self.path).path != "/export":
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
                keys = [k for k in req.get("keys", []) if k in by_key]   # validate against known set
                fmt = "jpeg" if req.get("fmt") == "jpeg" else "orig"
            except Exception as e:  # noqa: BLE001
                self._send(400, str(e).encode(), "text/plain")
                return

            # Stream via a temp file so a large selection isn't held in RAM.
            # Images are already compressed, so ZIP_STORED (no recompress) is fastest.
            tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            try:
                with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
                    for k in keys:
                        it = by_key[k]
                        src = album / it["f"]
                        if not src.exists():
                            continue
                        if fmt == "jpeg":
                            img = load_image_rgb_pil(src)
                            img.thumbnail((2048, 2048))
                            buf = io.BytesIO()
                            img.save(buf, "JPEG", quality=90)   # drops EXIF/GPS
                            z.writestr(f"{it['k']}.jpg", buf.getvalue())
                        else:
                            z.write(src, it["f"])
                tmp.flush()
                size = tmp.tell()
                tmp.seek(0)
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Disposition", 'attachment; filename="headcount_export.zip"')
                self.end_headers()
                shutil.copyfileobj(tmp, self.wfile)
            finally:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)

    class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    httpd = Server((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"\n  headcount browsing {len(items)} photos, {len(all_names)} names")
    print(f"  serving at {url}  (Ctrl-C to stop)\n")
    if not args.no_open:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_emb = sub.add_parser("embed", help="detect+embed every face -> faces.csv/.npy (slow, once)")
    p_emb.add_argument("--album", default="album", help="album folder (default: album/)")
    p_emb.add_argument("--faces", default="faces.csv", help="face table output (default: faces.csv)")
    p_emb.add_argument("--det-size", type=int, default=1024, help="detector input size (default: 1024)")
    p_emb.add_argument("--det-thresh", type=float, default=0.4, help="detector confidence (default: 0.4)")
    p_emb.add_argument("--prefetch", type=int, default=2,
                       help="background HEIC-decode threads to overlap with inference (~1.5x; 0=serial)")
    p_emb.add_argument("--limit", type=int, default=0, help="only embed first N images (for testing)")
    p_emb.add_argument("--rescan", action="store_true", help="ignore existing faces.csv and start over")
    p_emb.set_defaults(func=cmd_embed)

    p_clu = sub.add_parser("cluster", help="group faces by identity -> clusters.csv (re-runnable)")
    p_clu.add_argument("--faces", default="faces.csv", help="face table (default: faces.csv)")
    p_clu.add_argument("--out", default="clusters.csv", help="cluster assignment output")
    p_clu.add_argument("--algo", choices=["hdbscan", "dbscan"], default="hdbscan",
                       help="clustering algorithm (default: hdbscan; dbscan for small/dense sets)")
    p_clu.add_argument("--min-cluster-size", type=int, default=15,
                       help="HDBSCAN: smallest cluster to keep (default: 15)")
    p_clu.add_argument("--eps", type=float, default=0.45,
                       help="DBSCAN cosine-distance radius; smaller=tighter (default: 0.45)")
    p_clu.add_argument("--min-samples", type=int, default=0,
                       help="core-point neighbours; 0=auto (default)")
    p_clu.add_argument("--min-size", type=int, default=40, help="pre-filter: min bbox side in px (default: 40)")
    p_clu.add_argument("--min-det", type=float, default=0.5, help="pre-filter: min det_score (default: 0.5)")
    p_clu.add_argument("--refs", default="reference_embeddings.npy", help="enrolled refs for calibration readout")
    p_clu.add_argument("--ref-thresh", type=float, default=0.35, help="sim>= this counts as the reference person")
    p_clu.set_defaults(func=cmd_cluster)

    p_rev = sub.add_parser("review", help="montage each cluster + skeleton labels.csv to name them")
    p_rev.add_argument("--album", default="album", help="album folder (default: album/)")
    p_rev.add_argument("--faces", default="faces.csv", help="face table (default: faces.csv)")
    p_rev.add_argument("--clusters", default="clusters.csv", help="cluster assignment (default: clusters.csv)")
    p_rev.add_argument("--out", default="clusters", help="montage output folder (default: clusters/)")
    p_rev.add_argument("--labels", default="labels.csv", help="skeleton label file to fill in")
    p_rev.add_argument("--per-cluster", type=int, default=25, help="faces shown per montage (default: 25)")
    p_rev.add_argument("--cols", type=int, default=5, help="montage grid columns (default: 5)")
    p_rev.add_argument("--thumb", type=int, default=110, help="thumbnail px per face (default: 110)")
    p_rev.add_argument("--prefetch", type=int, default=4, help="background decode threads for cropping (default: 4)")
    p_rev.set_defaults(func=cmd_review)

    p_asg = sub.add_parser("assign", help="labels.csv -> image_people.csv (+ optional by_child/ folders)")
    p_asg.add_argument("--album", default="album", help="album folder (default: album/)")
    p_asg.add_argument("--faces", default="faces.csv", help="face table (default: faces.csv)")
    p_asg.add_argument("--clusters", default="clusters.csv", help="cluster assignment (default: clusters.csv)")
    p_asg.add_argument("--labels", default="labels.csv", help="filled-in labels (default: labels.csv)")
    p_asg.add_argument("--out", default="image_people.csv", help="who-is-in-each-photo index")
    p_asg.add_argument("--folders", nargs="?", const="by_child", default="",
                       help="also write per-child folders here (default dir: by_child/)")
    p_asg.add_argument("--names", default="", help="with --folders, only these names (comma list)")
    p_asg.add_argument("--copy", action="store_true", help="real copies instead of symlinks (uses disk)")
    p_asg.add_argument("--recover", action="store_true",
                       help="pull noise/junk faces into their nearest named cluster (boosts recall)")
    p_asg.add_argument("--recover-thresh", type=float, default=0.45,
                       help="min similarity to a cluster centroid to recover (default: 0.45)")
    p_asg.add_argument("--recover-margin", type=float, default=0.05,
                       help="recovered face must beat 2nd-nearest cluster by this (default: 0.05)")
    p_asg.set_defaults(func=cmd_assign)

    p_qry = sub.add_parser("query", help="set queries over image_people.csv (e.g. --with Ada,Ben)")
    p_qry.add_argument("--album", default="album", help="album folder (default: album/)")
    p_qry.add_argument("--image-people", default="image_people.csv", help="index from `assign`")
    p_qry.add_argument("--with", dest="with_", default="", help="all of these present (comma list)")
    p_qry.add_argument("--any", default="", help="at least one of these present")
    p_qry.add_argument("--without", default="", help="none of these present")
    p_qry.add_argument("--only", default="", help="exactly this set present")
    p_qry.add_argument("--where", choices=["indoor", "outdoor"], default="",
                       help="restrict to indoor/outdoor (needs a `scene` pass)")
    p_qry.add_argument("--scene", default="scene.csv", help="scene index from `scene`")
    p_qry.add_argument("--out", default="query", help="output base folder (default: query/)")
    p_qry.add_argument("--copy", action="store_true", help="real HEIC copies instead of symlinks")
    p_qry.add_argument("--jpeg", action="store_true",
                       help="re-encode to JPEG (reliable Finder thumbnails; HEIC ones are flaky)")
    p_qry.add_argument("--max-size", type=int, default=0,
                       help="with --jpeg, downscale to this long edge (0 = full res)")
    p_qry.add_argument("--jpeg-quality", type=int, default=90, help="with --jpeg, JPEG quality (default: 90)")
    p_qry.add_argument("--strip-exif", action="store_true",
                       help="with --jpeg, drop EXIF (incl. GPS); default preserves it")
    p_qry.add_argument("--dry-run", action="store_true", help="list matches, don't write files")
    p_qry.set_defaults(func=cmd_query)

    p_scn = sub.add_parser("scene", help="tag each photo indoor/outdoor (foliage+sky) -> scene.csv")
    p_scn.add_argument("--album", default="album", help="album folder (default: album/)")
    p_scn.add_argument("--faces", default="faces.csv", help="limit to images in this face table ('' = all)")
    p_scn.add_argument("--out", default="scene.csv", help="scene index output (default: scene.csv)")
    p_scn.add_argument("--method", choices=["green", "time", "both"], default="time",
                       help="time=hour window (default; instant, no decode, best when the "
                            "daily schedule is rigid), green=foliage/sky colour, both=AND")
    p_scn.add_argument("--outdoor-hours", default="10-11",
                       help="outdoor hour window for time/both, e.g. 10-11 (default: 10-11)")
    p_scn.add_argument("--thresh", type=float, default=0.12,
                       help="green+sky fraction >= this is outdoor (default: 0.12)")
    p_scn.add_argument("--limit", type=int, default=0, help="only first N images (for testing)")
    p_scn.add_argument("--prefetch", type=int, default=4, help="background decode threads (default: 4)")
    p_scn.set_defaults(func=cmd_scene)

    p_srv = sub.add_parser("serve", help="local web browser: name/time filters + zip export (localhost only)")
    p_srv.add_argument("--album", default="album", help="album folder (default: album/)")
    p_srv.add_argument("--image-people", default="image_people.csv", help="index from `assign`")
    p_srv.add_argument("--scene", default="scene.csv", help="scene/hour index from `scene` (optional)")
    p_srv.add_argument("--cache", default=".serve_cache", help="thumbnail cache dir (default: .serve_cache/)")
    p_srv.add_argument("--thumb", type=int, default=768, help="thumbnail long edge in px (default: 768)")
    p_srv.add_argument("--prefetch", type=int, default=4, help="background decode threads for thumbs (default: 4)")
    p_srv.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1 — localhost only)")
    p_srv.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    p_srv.add_argument("--no-open", action="store_true", help="don't auto-open a browser tab")
    p_srv.set_defaults(func=cmd_serve)

    args = ap.parse_args()

    if getattr(args, "rescan", False):
        faces_csv = Path(args.faces)
        faces_csv.unlink(missing_ok=True)
        faces_csv.with_suffix(".emb").unlink(missing_ok=True)
        faces_csv.with_suffix(".npy").unlink(missing_ok=True)
        faces_csv.with_suffix(".done").unlink(missing_ok=True)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
