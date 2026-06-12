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
