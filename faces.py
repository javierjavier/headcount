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
import hashlib
import shutil
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from common import (
    ArchiveFoundError,
    build_face_app,
    empty_hint,
    list_images,
    list_videos,
    load_image_bgr,
    load_image_rgb_pil,
    rel_key,
)

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


def _file_hash(path: Path) -> str:
    """Content hash (md5 hex) of a file's raw bytes, read in chunks.

    Used to skip re-embedding byte-identical photos that arrive under a second
    path — the common case when several teachers re-share an overlapping batch
    (`embed` otherwise keys only on the album-relative path, so the same image
    in two subfolders is embedded twice). This catches *byte-identical* copies,
    not visually-equal re-encodes (those decode differently and would need a
    perceptual hash); raw bytes are cheap (no decode) and cover the re-upload
    case we actually see. md5 is for dedup, not security.
    """
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_hash_manifest(path: Path) -> dict[str, str]:
    """Read the `faces.hashes` sidecar as {album-relative filename: content hash}."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.reader(f):
            if len(row) == 2:
                out[row[0]] = row[1]
    return out


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
    except (FileNotFoundError, NotADirectoryError, ArchiveFoundError) as e:
        print(e, file=sys.stderr)
        return 1
    if not images:
        print(f"No images found in '{album}'.{empty_hint(album)}", file=sys.stderr)
        return 1

    faces_csv = Path(args.faces)
    emb_path = faces_csv.with_suffix(".emb")
    done_path = faces_csv.with_suffix(".done")
    hash_path = faces_csv.with_suffix(".hashes")

    # Repair a desynced csv/emb pair (e.g. a kill between the two writes) by
    # trimming to the last consistent image, instead of forcing a full --rescan.
    if faces_csv.exists():
        _recover_desync(faces_csv, emb_path, done_path)

    # Resume skip set: images that produced a face (faces.csv) OR were recorded
    # as processed (the .done manifest, which also covers zero-face images).
    done = read_face_filenames(faces_csv) | read_done_manifest(done_path)
    todo = [p for p in images if rel_key(p, album) not in done]

    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(images)} images, {len(done)} already embedded, {len(todo)} to embed.")

    # Content-dedup: skip a to-embed image whose raw bytes match one already
    # embedded (or an earlier image this run). `embed` keys only on the album
    # path, so the same photo re-shared under a second subfolder would otherwise
    # be embedded — and clustered/shown/exported — twice. We persist a
    # filename->hash sidecar so this stays incremental; the first guarded run
    # over an existing album back-fills hashes for what's already embedded (a
    # one-time read), after which only new images are hashed. --no-dedup skips
    # the whole thing. See _file_hash for what "duplicate" means (byte-identical).
    seen_hash: dict[str, str] = {}
    hash_of: dict[str, str] = {}
    dups: list[tuple[str, str]] = []   # (duplicate key, canonical key it matches)
    if not args.no_dedup and todo:
        key_to_path = {rel_key(p, album): p for p in images}
        recorded = read_hash_manifest(hash_path)
        backfill = [k for k in done if k not in recorded and k in key_to_path]
        if backfill:
            print(f"Indexing content hashes for {len(backfill)} already-embedded "
                  f"image(s) (one-time, enables dedup) ...")
            with ThreadPoolExecutor(max_workers=max(1, args.prefetch)) as ex:
                for k, hh in zip(backfill, ex.map(lambda k: _file_hash(key_to_path[k]), backfill)):
                    recorded[k] = hh
            with hash_path.open("a", newline="") as hf:
                w = csv.writer(hf)
                for k in backfill:
                    w.writerow([k, recorded[k]])
        for k, hh in recorded.items():
            seen_hash.setdefault(hh, k)   # first key wins as the canonical copy

        kept = []
        with ThreadPoolExecutor(max_workers=max(1, args.prefetch)) as ex:
            todo_keys = [rel_key(p, album) for p in todo]
            for p, key, hh in zip(todo, todo_keys, ex.map(_file_hash, todo)):
                canon = seen_hash.get(hh)
                if canon is not None:
                    dups.append((key, canon))
                else:
                    seen_hash[hh] = key
                    hash_of[key] = hh
                    kept.append(p)
        if dups:
            print(f"Skipping {len(dups)} byte-identical duplicate(s); "
                  f"{len(kept)} unique image(s) to embed.")
            for dk, ck in dups[:10]:
                print(f"  dup: {dk}  ==  {ck}")
            if len(dups) > 10:
                print(f"  ... and {len(dups) - 10} more")
        todo = kept

    if not todo:
        print("Nothing to do. (Use --rescan to start over.)")
        # Still record any duplicates as processed so they aren't re-hashed next run.
        if dups:
            with done_path.open("a") as df:
                for dk, _ in dups:
                    df.write(dk + "\n")
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
            done_path.open("a") as df, hash_path.open("a", newline="") as hf:
        writer = csv.writer(cf)
        hash_writer = csv.writer(hf)
        if new_csv:
            writer.writerow(FACES_HEADER)
        # Record skipped duplicates as processed up front so a crash mid-run
        # doesn't leave them to be re-hashed (their bytes are already embedded
        # under the canonical key).
        for dk, _ in dups:
            df.write(dk + "\n")
        df.flush()
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
            key = rel_key(path, album)  # album-relative path; basename for top-level files
            for face in faces:
                x1, y1, x2, y2 = (int(round(v)) for v in face.bbox)
                emb = face.normed_embedding.astype(np.float32)
                writer.writerow(
                    [face_id, key, x1, y1, x2, y2, f"{float(face.det_score):.4f}"]
                )
                ef.write(emb.tobytes())
                face_id += 1
                n_faces += 1
            cf.flush()
            ef.flush()
            df.write(key + "\n")
            df.flush()
            if key in hash_of:                 # persist this image's content hash
                hash_writer.writerow([key, hash_of[key]])
                hf.flush()

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
    # Back up the prior membership before overwriting. Re-clustering renumbers
    # HDBSCAN ids, so the next `review` needs the OLD face_id->cluster map to carry
    # labels forward by identity (see remap_labels_by_face_id). A single rolling
    # backup is enough: it holds the clustering the current labels.csv was made
    # against. (clusters.csv is regenerable, unlike hand-typed labels, so no need
    # for _unique_path history here.)
    if out.exists():
        backup = out.with_suffix(out.suffix + ".bak")
        shutil.copy2(out, backup)
        print(f"Backed up prior {out} -> {backup} (lets `review` remap labels by face_id).")
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


def remap_labels_by_face_id(old_face_cluster: dict[str, int], old_cluster_name: dict[int, str],
                            new_face_cluster: dict[str, int]) -> dict[int, tuple[str, float, int]]:
    """Infer each NEW cluster's name from the OLD labeling, joining on stable
    face_id rather than cluster_id.

    HDBSCAN renumbers cluster ids whenever the face set changes, so carrying a
    name by cluster_id can silently put it on a different person after a
    re-cluster. face_id is stable, so instead we look at each new cluster's member
    faces, find what they were named before, and take the majority.

    Args:
        old_face_cluster: face_id -> cluster_id from the PRIOR clusters.csv.
        old_cluster_name: cluster_id -> name from the PRIOR labels.csv (named only).
        new_face_cluster: face_id -> cluster_id from the CURRENT clusters.csv.

    Returns {new_cluster_id: (name, purity, n_named)} where purity is the winning
    name's share among the new cluster's faces that *had* a name before, and
    n_named is how many voted. A new cluster with no previously-named faces is
    absent (it's new or junk -> left blank for manual labeling).
    """
    from collections import Counter, defaultdict

    votes: dict[int, Counter] = defaultdict(Counter)
    for fid, nc in new_face_cluster.items():
        if nc < 0:
            continue
        oc = old_face_cluster.get(fid)
        name = old_cluster_name.get(oc) if oc is not None else None
        if name:
            votes[nc][name] += 1
    out = {}
    for nc, counter in votes.items():
        name, n = counter.most_common(1)[0]
        out[nc] = (name, n / sum(counter.values()), sum(counter.values()))
    return out


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
        for cid, r in need[rel_key(path, album)]:
            box = (int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))
            crops.setdefault(cid, []).append((_face_size(r), _square_thumb(img.crop(box), args.thumb)))

    # Labeling is the one piece of hand work in the pipeline, and re-running
    # review (e.g. after a re-cluster) would otherwise overwrite labels.csv with
    # an empty skeleton. So carry the names you've already typed forward, and back
    # up the old file before rewriting so nothing is lost.
    #
    # Cluster ids are NOT stable across a re-cluster (HDBSCAN renumbers), so a
    # by-id carry can mislabel. When the prior membership is available (cluster
    # backs up clusters.csv -> .bak), carry by face_id majority vote instead —
    # robust to renumbering — and report the vote purity so any uncertain remap is
    # visible. Falls back to by-id only on a first run with no backup.
    labels_path = Path(args.labels)
    prior_names = _read_labels(labels_path)
    remap = None
    if prior_names:
        from common import _unique_path

        backup = _unique_path(labels_path.with_suffix(labels_path.suffix + ".bak"))
        shutil.copy2(labels_path, backup)
        print(f"Existing {labels_path} has {len(prior_names)} name(s); backed it up -> {backup}")

        clusters_bak = clu_path.with_suffix(clu_path.suffix + ".bak")
        if clusters_bak.exists():
            # Keep face_id as a str key to match `assign` (load_cluster_map ->
            # dict[str, int]); int-keying here would silently never join.
            with clusters_bak.open() as bf:
                old_assign = {r["face_id"]: int(r["cluster_id"])
                              for r in csv.DictReader(bf)}
            remap = remap_labels_by_face_id(old_assign, prior_names, assign)
        else:
            print(f"  (no {clusters_bak} — carrying names by cluster_id; re-run after "
                  "`cluster` for an identity-stable remap.)")

    # Build one montage per cluster, biggest clusters first (most-photographed
    # kids on top), and a skeleton labels.csv to type names into.
    written = 0
    report = []  # (cid, name, purity|None) for the carried-forward summary
    kept = set()  # montage filenames written this run, to sweep stale ones below
    with labels_path.open("w", newline="") as lf:
        w = csv.writer(lf)
        w.writerow(["cluster_id", "size", "montage", "name"])
        for rank, (cid, fr) in enumerate(ranked):
            thumbs = crops.get(cid)
            if not thumbs:
                continue
            thumbs = [t for _, t in sorted(thumbs, key=lambda x: -x[0])]
            montage = f"c{rank:02d}__cluster{cid}__n{len(fr)}.jpg"
            _montage(thumbs, args.thumb, args.cols).save(out_dir / montage, "JPEG", quality=85)
            kept.add(montage)
            if remap is not None:
                name, purity, _ = remap.get(cid, ("", None, 0))
            else:
                name, purity = prior_names.get(cid, ""), None
            w.writerow([cid, len(fr), montage, name])
            report.append((cid, name, purity))
            written += 1

    # Re-clustering renumbers clusters, so a prior run's montages (different rank,
    # cid, or face count in the name) linger and clutter the folder you label in.
    # Sweep any montage-pattern file we didn't just write — same idea as serve's
    # thumbnail stale-sweep. Glob is specific to our `c..__cluster..__n..jpg`
    # names, so it never touches unrelated files a user dropped in the folder.
    stale = [p for p in out_dir.glob("c*__cluster*__n*.jpg") if p.name not in kept]
    for p in stale:
        p.unlink()

    print(f"\nWrote {written} cluster montage(s) -> {out_dir}/ and skeleton -> {labels_path}")
    if stale:
        print(f"Removed {len(stale)} stale montage(s) from a prior run.")
    carried = sum(1 for _, n, _ in report if n)
    if remap is not None:
        print(f"Carried {carried} name(s) forward by face_id vote (robust to re-cluster renumbering).")
        low = [(cid, n, pur) for cid, n, pur in report if n and pur is not None and pur < 0.90]
        if low:
            print(f"  ! {len(low)} cluster(s) labeled with <90% vote agreement — eyeball the montage:")
            for cid, n, pur in low:
                print(f"      cluster {cid}: {n} ({pur:.0%})")
        dropped = sorted(set(prior_names.values()) - {n for _, n, _ in report if n})
        if dropped:
            print(f"  ! prior name(s) not matched to any cluster this run: {', '.join(dropped)} "
                  f"(still in {backup}).")
    elif prior_names:
        print(f"Carried forward {carried} existing name(s) by cluster_id.")
        lost = sorted(set(prior_names) - {cid for cid, _, _ in report})
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


def _hour_from_dt(dt: str) -> int:
    """Hour-of-day from an EXIF datetime 'YYYY:MM:DD HH:MM:SS', or -1."""
    try:
        return int(dt.split()[1].split(":")[0])
    except (IndexError, ValueError):
        return -1


def _exif_hour(path: Path) -> int:
    """Hour-of-day from EXIF DateTime, or -1 if missing."""
    return _hour_from_dt(_exif_dt(path))


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


def merge_scene_rows(existing: list[list[str]], new: list[list[str]]) -> list[list[str]]:
    """Merge freshly-classified scene rows into existing ones, keyed by filename
    (column 0); the new row wins on conflict. Returns rows sorted by filename.

    This is what lets `scene --subdir 20260618` re-tag only that import's photos —
    which may have a different outdoor block than earlier batches — and merge the
    result back into scene.csv without disturbing the other batches' rows.
    """
    by_fn = {r[0]: r for r in existing}
    for r in new:
        by_fn[r[0]] = r
    return [by_fn[k] for k in sorted(by_fn)]


def cmd_scene(args) -> int:
    album = Path(args.album)
    try:
        images = list_images(album)
    except (FileNotFoundError, NotADirectoryError, ArchiveFoundError) as e:
        print(e, file=sys.stderr)
        return 1
    if args.faces and Path(args.faces).exists():
        wanted = {r["filename"] for r in read_face_rows(Path(args.faces))}
        images = [p for p in images if rel_key(p, album) in wanted]
    # --subdir scopes classification to one import folder (album/<subdir>/) and
    # merges into scene.csv, so a batch whose outdoor block differs from earlier
    # ones can be re-tagged without re-tagging (and mis-tagging) the rest.
    if args.subdir:
        pref = args.subdir.strip("/") + "/"
        images = [p for p in images if rel_key(p, album).startswith(pref)]
        if not images:
            print(f"No images under album/{pref} (in the face table?).", file=sys.stderr)
            return 1
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"No images to classify in '{album}'.{empty_hint(album)}", file=sys.stderr)
        return 1

    hours = _parse_hours(args.outdoor_hours)
    out = Path(args.out)
    new_rows = []   # full csv rows just classified
    summary = []    # (hour, scene) for the by-hour readout

    # 'time' needs no pixels — just the EXIF hour — so it skips decode entirely
    # (instant). 'green' and 'both' decode for the foliage/sky colour signal.
    # 'both' = outdoor only if the colour says so AND it's in the outdoor window
    # (rejects green classroom decor at non-outdoor hours).
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
            new_rows.append([rel_key(path, album), str(hr), "", "", "", scene])
            summary.append((hr, scene))
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
            new_rows.append([rel_key(path, album), str(hr), f"{g:.3f}", f"{s:.3f}", f"{b:.3f}", scene])
            summary.append((hr, scene))

    header = ["filename", "hour", "green", "sky", "bright", "scene"]
    note = ""
    if args.subdir and out.exists():
        with out.open() as f:
            rd = csv.reader(f)
            next(rd, None)
            existing = [r for r in rd if r]
        final_rows = merge_scene_rows(existing, new_rows)
        note = f" (merged batch into {len(final_rows)} total rows)"
    else:
        final_rows = new_rows
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(final_rows)

    from collections import Counter, defaultdict

    sc = Counter(s for _, s in summary)
    print(f"\n{len(summary)} images classified -> {out}{note}: "
          f"{sc.get('outdoor',0)} outdoor, {sc.get('indoor',0)} indoor")
    print("\nby hour (validates against the daily schedule):")
    print(f"  {'hour':>4} {'outdoor':>8} {'indoor':>7}")
    by_hour: dict[int, Counter] = defaultdict(Counter)
    for h, s in summary:  # one pass over the just-classified batch
        if h >= 0:
            by_hour[h][s] += 1
    for hr in sorted(by_hour):
        c = by_hour[hr]
        print(f"  {hr:>4} {c.get('outdoor',0):>8} {c.get('indoor',0):>7}")
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
        # fn may be a subfolder-relative path; flatten to the basename so each
        # child's folder stays flat, and let _unique_path break basename clashes
        # between subfolders (e.g. two reused IMG_4492.HEIC).
        dst = _unique_path(d / Path(fn).name)
        try:
            if copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src)
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ! {fn} -> {name}: {e}")
    return n


def match_to_centroids(embs: np.ndarray, cents: np.ndarray,
                       thresh: float, margin: float) -> np.ndarray:
    """For each row of *embs*, the index of its nearest centroid — or -1.

    A match is accepted only if it clears *thresh* AND beats the second-nearest
    centroid by *margin*, the same strict two-test rule `assign --recover` uses to
    keep ambiguous faces out. Both *embs* and *cents* are assumed L2-normalized,
    so the dot product is cosine similarity. With a single centroid the margin is
    vacuous (no runner-up), so only the threshold applies. Pure NumPy — unit-tested.
    """
    out = np.full(len(embs), -1, dtype=int)
    if len(embs) == 0 or len(cents) == 0:
        return out
    sims = embs.astype(np.float32) @ cents.T
    order = np.argsort(-sims, axis=1)
    ar = np.arange(len(embs))
    best_idx = order[:, 0]
    best = sims[ar, best_idx]
    if cents.shape[0] >= 2:
        second = sims[ar, order[:, 1]]
    else:
        second = np.full(len(embs), -np.inf, dtype=np.float32)
    ok = (best >= thresh) & ((best - second) >= margin)
    out[ok] = best_idx[ok]
    return out


def build_name_centroids(rows: list[dict], clusters: dict[str, int],
                         labels: dict[int, str], mat: np.ndarray):
    """Per-NAME mean embedding (L2-normalized) over every labeled face.

    Unlike `assign --recover`'s per-cluster centroids, clusters that share a name
    (a kid split across several clusters — the expected case, see DESIGN.md) are
    merged into one centroid, so `match_to_centroids`' margin test is genuinely
    name-vs-name rather than cluster-vs-cluster. Returns (names, cents) with
    `cents[i]` the centroid for `names[i]`; names sorted for determinism.
    """
    from collections import defaultdict

    by_name: dict[str, list] = defaultdict(list)
    for i, r in enumerate(rows):
        name = labels.get(clusters.get(r["face_id"], -2))
        if name:
            by_name[name].append(i)
    names = sorted(by_name)
    if not names:
        return [], np.zeros((0, EMB_DIM), dtype=np.float32)
    cents = np.vstack([mat[by_name[nm]].mean(0) for nm in names]).astype(np.float32)
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    return names, cents


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
            picks = match_to_centroids(mat[pool], cent, args.recover_thresh, args.recover_margin)
            n = 0
            for k, i in enumerate(pool):
                if picks[k] >= 0:
                    img_names.setdefault(rows[i]["filename"], set()).add(cnames[picks[k]])
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


VIDEO_PEOPLE_HEADER = ["filename", "names", "n_named", "peaks"]


def _write_video_people(out_csv: Path, results: dict[str, list]) -> None:
    """Rewrite video_people.csv from the *results* map (filename -> row).

    Called after each processed clip and ordered *before* the .done append, so the
    csv is never behind the manifest: a clip marked done is guaranteed to have its
    row persisted, which is what makes resume lossless.
    """
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(VIDEO_PEOPLE_HEADER)
        for fn in sorted(results):
            w.writerow(results[fn])


def cmd_video(args) -> int:
    album = Path(args.album)
    try:
        videos = list_videos(album)
    except Exception as e:  # noqa: BLE001
        print(e, file=sys.stderr)
        return 1
    if not videos:
        print(f"No videos found under '{album}'.", file=sys.stderr)
        return 1

    # Name faces against the ALREADY-labeled photo clusters — no re-clustering, so
    # the calibrated photo pipeline is untouched (see DESIGN.md, "faces in video").
    faces_csv = Path(args.faces)
    try:
        rows, mat = load_faces(faces_csv)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
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
        print(f"No names in {args.labels} — run `review`/`assign` first, then label.",
              file=sys.stderr)
        return 1
    names, cents = build_name_centroids(rows, clusters, labels, mat)
    if not names:
        print("No labeled faces to match against — fill in labels.csv first.", file=sys.stderr)
        return 1

    out_csv = Path(args.out)
    done_path = out_csv.with_suffix(".done")
    done = read_done_manifest(done_path)
    # Preserve rows already computed (resume / re-run); new scans overwrite their key.
    results: dict[str, list] = {r["filename"]: [r.get(c, "") for c in VIDEO_PEOPLE_HEADER]
                                for r in read_face_rows(out_csv)}
    todo = [v for v in videos if rel_key(v, album) not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(videos)} videos, {len(done)} already scanned, {len(todo)} to scan; "
          f"matching against {len(names)} labeled names "
          f"(thresh {args.thresh}, margin {args.margin}, {args.fps} fps).")
    if not todo:
        print("Nothing to do. (Delete the .done manifest to re-scan.)")
        return 0

    print(f"Loading detector (buffalo_l, det_size={args.det_size}) ...")
    app = build_face_app(det_size=args.det_size, det_thresh=args.det_thresh)

    seq = todo
    try:
        from tqdm import tqdm
        seq = tqdm(todo, unit="vid")
    except ImportError:
        pass

    n_named_videos = 0
    with done_path.open("a") as df:
        for v in seq:
            key = rel_key(v, album)
            # name -> [count, best_sim, best_t]; a name added many times is still one
            # set member, so no per-clip dedup is needed for the name set itself.
            hits: dict[str, list] = {}
            for t, frame in _sample_video_frames(v, args.fps, args.max_frames):
                try:
                    faces = app.get(frame)
                except Exception as e:  # noqa: BLE001 - one bad frame shouldn't abort the clip
                    print(f"\n  ! {key} @ {t:.0f}s: {e}")
                    continue
                faces = [f for f in faces
                         if max(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) >= args.min_size]
                if not faces:
                    continue
                embs = np.vstack([f.normed_embedding for f in faces]).astype(np.float32)
                sims = embs @ cents.T
                picks = match_to_centroids(embs, cents, args.thresh, args.margin)
                for j, pick in enumerate(picks):
                    if pick < 0:
                        continue
                    nm = names[pick]
                    sim = float(sims[j, pick])
                    rec = hits.setdefault(nm, [0, -1.0, 0.0])
                    rec[0] += 1
                    if sim > rec[1]:
                        rec[1], rec[2] = sim, t
            present = sorted(hits)
            n_named = sum(int(hits[nm][0]) for nm in present)
            peaks = ";".join(f"{nm}@{hits[nm][2]:.1f}" for nm in present)
            results[key] = [key, ";".join(present), str(n_named), peaks]
            if present:
                n_named_videos += 1
            # csv first, then mark done — so the manifest never gets ahead of the csv.
            _write_video_people(out_csv, results)
            df.write(key + "\n")
            df.flush()

    from collections import Counter

    counts = Counter(nm for r in results.values() for nm in filter(None, r[1].split(";")))
    print(f"\nScanned {len(todo)} video(s); {n_named_videos} have >=1 named person -> {out_csv}")
    for nm, c in counts.most_common():
        print(f"  {nm}: {c} videos")
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


def _export_one(src: Path, fn: str, dstdir: Path, args) -> None:
    """Materialize one source image into *dstdir* per the query's output mode.

    Flatten subfolder-relative names to a basename and dedup basename clashes so
    neither is shadowed. Raises on failure; the caller logs and continues.
    """
    from common import _unique_path

    dstdir.mkdir(parents=True, exist_ok=True)
    if args.jpeg:
        # Re-encode to JPEG. macOS Finder thumbnails HEIC unreliably (its icon
        # service bails on many of these files even though Quick Look can decode
        # them); JPEG always thumbnails. Open WITHOUT exif_transpose so the
        # original pixels + orientation tag stay together, and copy EXIF verbatim
        # unless stripped.
        from PIL import Image

        dst = _unique_path(dstdir / f"{Path(fn).stem}.jpg")
        img: Image.Image = Image.open(src)
        exif = None if args.strip_exif else img.info.get("exif")
        img = img.convert("RGB")
        if args.max_size:
            img.thumbnail((args.max_size, args.max_size))
        save_kw = {"quality": args.jpeg_quality}
        if exif:
            save_kw["exif"] = exif
        img.save(dst, "JPEG", **save_kw)
    elif args.copy:
        shutil.copy2(src, _unique_path(dstdir / Path(fn).name))
    else:
        (_unique_path(dstdir / Path(fn).name)).symlink_to(src)


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

    # Optional indoor/outdoor data from a `scene` pass — drives both --where
    # (filter to one scene) and --split-scene (fan into indoor/outdoor subfolders).
    if args.where and args.split_scene:
        print("Use either --where or --split-scene, not both.", file=sys.stderr)
        return 1
    scene_of = None
    if args.where or args.split_scene:
        scene_path = Path(args.scene)
        if not scene_path.exists():
            print(f"{scene_path} not found — run `scene` first (or drop --where/--split-scene).",
                  file=sys.stderr)
            return 1
        scene_of = {r["filename"]: r["scene"] for r in read_face_rows(scene_path)}
    where_ok = {fn for fn, sc in scene_of.items() if sc == args.where} if args.where else None

    name_matched = [
        fn for fn, names in data.items()
        if _query_match(names, want, any_of, without, only)
    ]
    # Guard against a stale scene.csv. If a scene split/filter is requested but
    # some matched photos aren't in it (typically a new import that `scene` hasn't
    # processed yet), refuse loudly rather than quietly dropping them (--where) or
    # dumping them in unscored/ (--split-scene). --allow-unscored opts into that
    # soft path knowingly.
    if scene_of is not None and not args.allow_unscored:
        missing = [fn for fn in name_matched if fn not in scene_of]
        if missing:
            print(f"{len(missing)} of {len(name_matched)} matched photo(s) are missing from "
                  f"{scene_path} — it is stale. Re-run `scene`, or pass --allow-unscored.",
                  file=sys.stderr)
            for fn in missing[:5]:
                print(f"  {fn}", file=sys.stderr)
            if len(missing) > 5:
                print(f"  ... (+{len(missing) - 5} more)", file=sys.stderr)
            return 1

    selected = sorted(
        fn for fn in name_matched
        if where_ok is None or fn in where_ok
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
    unscored = 0
    for fn in selected:
        src = (album / fn).resolve()
        if not src.exists():
            continue
        if args.split_scene:
            # Route into out/<scene>/. Reachable with a missing scene row only
            # under --allow-unscored (the pre-flight check aborts otherwise);
            # those land in out/unscored/ rather than vanish.
            sub = scene_of.get(fn)
            unscored += sub is None
            dstdir = out / (sub or "unscored")
        else:
            dstdir = out
        try:
            _export_one(src, fn, dstdir, args)
            n += 1
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't abort the query
            print(f"  ! {fn}: {e}")
    kind = "jpegs" if args.jpeg else ("copies" if args.copy else "symlinks")
    print(f"Wrote {n} {kind} -> {out}/")
    if unscored:
        print(f"  ({unscored} had no scene tag -> {out}/unscored/)")

    if args.zip:
        import zipfile

        # Pack the materialized folder into a sibling <label>.zip, preserving the
        # split-scene subfolders. is_file() follows symlinks, so symlinked results
        # are stored as real content -- a zip is self-contained by definition.
        zpath = out.with_name(out.name + ".zip")
        if zpath.exists():
            zpath.unlink()
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(out.parent))
        print(f"Zipped -> {zpath}")
    return 0


# Two head-and-shoulders silhouettes on the accent-blue tile — "counting heads".
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <rect width="32" height="32" rx="7" fill="#5b9dff"/>
  <circle cx="11.5" cy="12.5" r="4" fill="#fff" opacity="0.55"/>
  <path d="M4.5 26.5c0-3.9 3.1-7 7-7s7 3.1 7 7z" fill="#fff" opacity="0.55"/>
  <circle cx="20.5" cy="13.5" r="4.6" fill="#fff"/>
  <path d="M12 27.5c0-4.7 3.8-8.5 8.5-8.5s8.5 3.8 8.5 8.5z" fill="#fff"/>
</svg>"""

SERVE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>headcount</title>
<link rel="icon" href="/favicon.svg">
<link rel="mask-icon" href="/favicon.svg" color="#5b9dff">
<style>
  :root { --bg:#16181d; --panel:#1f232b; --ink:#e7e9ee; --muted:#9aa3b2; --accent:#5b9dff; --line:#2c313b; --cell:150px; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--ink); }
  .wrap { display:flex; height:100vh; }
  aside { width:280px; flex:none; background:var(--panel); border-right:1px solid var(--line); padding:16px; overflow-y:auto; }
  main { flex:1; overflow-y:auto; padding:16px 20px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:20px 0 8px; }
  h2:first-child { margin-top:0; }
  h2.toggle { cursor:pointer; user-select:none; display:flex; align-items:center; gap:6px; }
  h2.toggle::before { content:"\\25be"; font-size:11px; transition:transform .12s; }   /* ▾ */
  h2.toggle.closed::before { transform:rotate(-90deg); }                                /* ▸ when collapsed */
  h2.toggle.closed + * { display:none !important; }
  input[type=search], select { width:100%; padding:7px 9px; background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:7px; }
  .names { max-height:38vh; overflow-y:auto; border:1px solid var(--line); border-radius:7px; padding:6px; margin-top:8px; }
  .names label { display:flex; align-items:center; gap:8px; padding:4px 6px; border-radius:5px; cursor:pointer; }
  .names label:hover { background:var(--bg); }
  .names .count { margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; }
  .modes { display:flex; gap:14px; margin-top:8px; color:var(--muted); }
  .modes.col { flex-direction:column; gap:6px; align-items:flex-start; }
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
  .grid .cell { position:relative; width:var(--cell); height:var(--cell); flex:none; background:var(--panel); border-radius:8px; overflow:hidden; cursor:pointer; border:0; padding:0;
    transition:width .12s ease, height .12s ease; content-visibility:auto; contain-intrinsic-size:var(--cell); }
  .grid img { width:100%; height:100%; object-fit:cover; display:block; }
  /* play badge on video cells (CSS-only, so it scales with the size slider) */
  .grid .cell.vid::after { content:"\\25B6"; position:absolute; left:6px; bottom:6px;
    width:22px; height:22px; border-radius:50%; background:rgba(8,9,12,.62); color:#fff;
    font-size:9px; display:flex; align-items:center; justify-content:center; padding-left:2px; pointer-events:none; }
  /* duration badge, bottom-right opposite the play badge (text from data-dur) */
  .grid .cell.vid[data-dur]::before { content:attr(data-dur); position:absolute; right:6px; bottom:6px;
    background:rgba(8,9,12,.62); color:#fff; font-size:10px; line-height:1; padding:3px 5px;
    border-radius:4px; font-variant-numeric:tabular-nums; pointer-events:none; }
  /* slim vertical date marker sitting inline between each day's photos */
  .dhead { flex:none; width:30px; height:var(--cell); transition:height .12s ease; display:flex; align-items:center; justify-content:center;
           writing-mode:vertical-rl; text-orientation:mixed; white-space:nowrap;
           font-size:11px; font-weight:600; color:var(--muted); letter-spacing:.02em;
           border-left:2px solid var(--line); cursor:pointer; user-select:none; }
  .dhead:hover { color:var(--text); border-left-color:var(--accent); }
  /* collapsed day: a full-width horizontal bar standing in for the hidden run of cells */
  .dhead.collapsed { width:100%; flex-basis:100%; height:auto; writing-mode:horizontal-tb;
                     justify-content:flex-start; gap:6px; padding:7px 10px; background:var(--panel);
                     border-radius:8px; border-left:0; }
  .dhead.collapsed:hover { border-left:0; background:var(--line); }
  .dhead.collapsed .cnt { color:var(--muted); font-weight:400; }
  .empty { color:var(--muted); padding:40px 0; text-align:center; }
  /* lightbox */
  #lb { display:none; position:fixed; inset:0; background:rgba(8,9,12,.92); z-index:50;
        flex-direction:column; align-items:center; justify-content:center; }
  #lbstage { position:relative; flex:1; width:100%; display:flex; align-items:center; justify-content:center; min-height:0; padding:48px 64px 8px; }
  #lbimg { max-width:100%; max-height:100%; object-fit:contain; border-radius:6px;
           transition:filter .15s; cursor:zoom-in; }
  #lbimg.loading { filter:blur(8px); }
  #lbvid { max-width:100%; max-height:100%; border-radius:6px; background:#000; }
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
    <select id="nsort" style="margin-top:8px">
      <option value="az">Name A → Z</option>
      <option value="za">Name Z → A</option>
      <option value="hi">Most photos</option>
      <option value="lo">Fewest photos</option>
    </select>
    <div class="names" id="names"></div>
    <h2>Faces</h2>
    <div class="rangewrap" id="fcwrap">
      <div class="track"></div>
      <div class="fill" id="ffill"></div>
      <input type="range" id="fmin" step="1">
      <input type="range" id="fmax" step="1">
    </div>
    <div class="hourlab" id="faclab"></div>
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
    <h2>Media</h2>
    <div class="modes col" id="media">
      <label><input type="checkbox" data-media="photo" checked> photos</label>
      <label><input type="checkbox" data-media="live" checked> live photos</label>
      <label><input type="checkbox" data-media="video" checked> videos</label>
    </div>
    <h2 id="datehead">Date</h2>
    <div class="rangewrap" id="datewrap">
      <div class="track"></div>
      <div class="fill" id="dfill"></div>
      <input type="range" id="dmin" step="1">
      <input type="range" id="dmax" step="1">
    </div>
    <div class="hourlab" id="datelab"></div>
    <h2 id="foldhead" class="toggle closed">Folder</h2>
    <div class="names" id="folders"></div>
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
    <video id="lbvid" controls playsinline preload="metadata" style="display:none"></video>
    <button class="lbbtn" id="lbnext" title="Next (→)">&#8250;</button>
  </div>
  <div id="lbcap"></div>
</div>

<script>
const DATA = __MANIFEST__;
const S = { names:new Set(), mode:"all", fmin:0, fmax:0, hmin:0, hmax:23, dmin:0, dmax:0, scene:"", media:{photo:true, live:true, video:true}, foldersOff:new Set(), foldOpen:false, collapsedDays:new Set(), search:"", sort:"new", nsort:"az", cell:150 };
// album subfolders present in the data (""=album root). Stored as an *exclude*
// set so the default (nothing excluded) shows everything and a freshly imported
// subfolder is visible without clearing saved filters.
const FOLDERS = DATA.folders || [];
const LIVE_MAX = DATA.liveMax || 3.5;   // videos <= this many seconds are "live photos"
// media bucket for an item: photo / live (short clip) / video (everything else,
// incl. clips whose duration is unknown so they're never hidden as "live").
function mediaCat(it) { return !it.v ? "photo" : ((it.d && it.d <= LIVE_MAX) ? "live" : "video"); }
const MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"];

// hour bounds present in the data (photos with a known hour)
const hrs = DATA.items.map(i => i.h).filter(h => h !== null);
const HMIN = hrs.length ? Math.min(...hrs) : 0, HMAX = hrs.length ? Math.max(...hrs) : 23;
S.hmin = HMIN; S.hmax = HMAX;

// distinct capture days present (ISO "YYYY-MM-DD", sorted ascending). The date
// slider works on indices into this array; undated items (no dt) ignore it. We
// persist the day *strings* (not indices) so saved filters survive new imports
// shifting the index positions.
const DAYS = [...new Set(DATA.items.map(i => i.dt ? i.dt.slice(0,10).replace(/:/g,"-") : "").filter(Boolean))].sort();
const DMIN = 0, DMAX = DAYS.length ? DAYS.length - 1 : 0;
S.dmin = DMIN; S.dmax = DMAX;

// face-count bounds present in the data (items with a known count; videos/
// uncounted photos are null and ignored here — they always pass the filter)
const fcs = DATA.items.map(i => i.fc).filter(c => c !== null && c !== undefined);
const FCMIN = fcs.length ? Math.min(...fcs) : 0, FCMAX = fcs.length ? Math.max(...fcs) : 0;
S.fmin = FCMIN; S.fmax = FCMAX;

// --- persist filters across refreshes (per-origin localStorage) ---
const SKEY = "headcount.filters.v1";
function saveState() {
  try { localStorage.setItem(SKEY, JSON.stringify({
    names:[...S.names], mode:S.mode, fmin:S.fmin, fmax:S.fmax, hmin:S.hmin, hmax:S.hmax, dlo:DAYS[S.dmin] || "", dhi:DAYS[S.dmax] || "", scene:S.scene, media:S.media, foldersOff:[...S.foldersOff], foldOpen:S.foldOpen, collapsedDays:[...S.collapsedDays], sort:S.sort, nsort:S.nsort, cell:S.cell
  })); } catch (e) {}
}
function loadState() {
  let v; try { v = JSON.parse(localStorage.getItem(SKEY) || "null"); } catch (e) { return; }
  if (!v) return;
  const known = new Set(DATA.names);                              // drop names absent from this album
  if (Array.isArray(v.names)) S.names = new Set(v.names.filter(n => known.has(n)));
  if (v.mode === "all" || v.mode === "any") S.mode = v.mode;
  if (typeof v.fmin === "number") S.fmin = Math.min(Math.max(v.fmin, FCMIN), FCMAX);   // clamp to data bounds
  if (typeof v.fmax === "number") S.fmax = Math.min(Math.max(v.fmax, FCMIN), FCMAX);
  if (S.fmin > S.fmax) { S.fmin = FCMIN; S.fmax = FCMAX; }
  if (typeof v.hmin === "number") S.hmin = Math.min(Math.max(v.hmin, HMIN), HMAX);   // clamp to data bounds
  if (typeof v.hmax === "number") S.hmax = Math.min(Math.max(v.hmax, HMIN), HMAX);
  if (S.hmin > S.hmax) { S.hmin = HMIN; S.hmax = HMAX; }
  if (typeof v.dlo === "string") { const i = DAYS.indexOf(v.dlo); if (i >= 0) S.dmin = i; }   // re-anchor by day
  if (typeof v.dhi === "string") { const i = DAYS.indexOf(v.dhi); if (i >= 0) S.dmax = i; }
  if (S.dmin > S.dmax) { S.dmin = DMIN; S.dmax = DMAX; }
  if (v.scene === "indoor" || v.scene === "outdoor") S.scene = v.scene;
  if (v.media && typeof v.media === "object") {
    for (const k of ["photo", "live", "video"]) if (typeof v.media[k] === "boolean") S.media[k] = v.media[k];
  }
  if (Array.isArray(v.foldersOff)) {                              // drop folders absent from this album
    const kf = new Set(FOLDERS);
    S.foldersOff = new Set(v.foldersOff.filter(f => kf.has(f)));
  }
  if (typeof v.foldOpen === "boolean") S.foldOpen = v.foldOpen;
  if (Array.isArray(v.collapsedDays)) {                          // drop days absent from this album
    const kd = new Set(DAYS);
    S.collapsedDays = new Set(v.collapsedDays.filter(d => kd.has(d)));
  }
  if (v.sort === "old" || v.sort === "new") S.sort = v.sort;
  if (["az","za","hi","lo"].includes(v.nsort)) S.nsort = v.nsort;
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
function fmtDur(s) {                                // seconds -> "0:02", "1:24"
  s = Math.max(1, Math.round(s));
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
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
  if (it.fc != null && (it.fc < S.fmin || it.fc > S.fmax)) return false; // uncounted item always passes
  if (it.h !== null && (it.h < S.hmin || it.h > S.hmax)) return false;   // unknown hour always passes
  if (DAYS.length && it.dt) {                                            // undated item always passes
    const d = it.dt.slice(0,10).replace(/:/g,"-");
    if (d < DAYS[S.dmin] || d > DAYS[S.dmax]) return false;             // ISO days compare lexically
  }
  if (S.scene && it.s && it.s !== S.scene) return false;                 // unknown scene always passes

  if (!S.media[mediaCat(it)]) return false;
  if (S.foldersOff.has(it.sf)) return false;                            // deselected subfolder
  return true;
}

// --- preview warming: decode each on-screen thumbnail's 2048px preview in the
// background so the lightbox opens instantly. The server caches each render to
// disk (decoded once) and the browser caches the response, so warming a key is
// cheap after the first hit and openLB reuses whatever's already in flight. ---
const warmed = new Set();
const VIDKEYS = new Set(DATA.items.filter(i => i.v).map(i => i.k));   // videos have no /full preview
function warm(k) {
  if (!k || warmed.has(k) || VIDKEYS.has(k)) return;
  warmed.add(k);
  new Image().src = "/full/" + encodeURIComponent(k);   // 2048px preview (no full=1)
}
// observe thumbnails near the viewport; warm as they scroll into view
const warmIO = ("IntersectionObserver" in window) ? new IntersectionObserver((entries) => {
  for (const e of entries) if (e.isIntersecting) { warm(e.target.dataset.k); warmIO.unobserve(e.target); }
}, { rootMargin: "300px" }) : null;

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
  $("count").textContent = current.length + " item" + (current.length === 1 ? "" : "s");
  $("export").disabled = current.length === 0;
  const grid = $("grid");
  if (!current.length) { grid.innerHTML = '<div class="empty">Nothing matches these filters.</div>'; return; }
  if (warmIO) warmIO.disconnect();   // drop observations on the cells we're about to discard
  grid.innerHTML = "";
  const frag = document.createDocumentFragment();
  const dayCount = new Map();                              // matched items per day, computed once
  for (const it of current) { const d = dayOf(it); dayCount.set(d, (dayCount.get(d) || 0) + 1); }
  let lastDay = null;
  current.forEach((it, i) => {
    const day = dayOf(it);
    if (day !== lastDay) {
      lastDay = day;
      const sameDay = dayCount.get(day);
      const collapsed = S.collapsedDays.has(day);
      const h = document.createElement("div"); h.className = collapsed ? "dhead collapsed" : "dhead";
      h.textContent = prettyDayShort(day);
      if (collapsed) {                                     // horizontal bar: "06/26/2026  (22 items)"
        const c = document.createElement("span"); c.className = "cnt";
        c.textContent = "(" + sameDay + " item" + (sameDay === 1 ? "" : "s") + ")";
        h.appendChild(c);
      }
      h.title = prettyDay(day) + " · " + sameDay + " item" + (sameDay === 1 ? "" : "s")
              + (collapsed ? " — click to expand" : " — click to collapse");
      h.onclick = () => {
        if (S.collapsedDays.has(day)) S.collapsedDays.delete(day); else S.collapsedDays.add(day);
        render();
      };
      frag.appendChild(h);
    }
    if (S.collapsedDays.has(day)) return;                  // collapsed day: skip its cells (count/export unaffected)
    const cell = document.createElement("button");
    cell.className = it.v ? "cell vid" : "cell";
    cell.title = it.n.join(", ") || (it.v ? "video" : "");
    cell.onclick = () => openLB(i);
    cell.dataset.k = it.k;
    if (it.v && it.d) cell.dataset.dur = fmtDur(it.d);
    const img = document.createElement("img");
    img.loading = "lazy"; img.src = "/thumb/" + encodeURIComponent(it.k);
    cell.appendChild(img); frag.appendChild(cell);
  });
  grid.appendChild(frag);
  if (warmIO) for (const c of grid.children) if (c.dataset.k) warmIO.observe(c);
}

function hourLabel() {
  $("hourlab").textContent = (S.hmin === HMIN && S.hmax === HMAX)
    ? "Any time" : (String(S.hmin).padStart(2,"0") + ":00 – " + String(S.hmax).padStart(2,"0") + ":59");
}

function dateLabel() {
  $("datelab").textContent = (S.dmin === DMIN && S.dmax === DMAX)
    ? "Any date" : (prettyDayShort(DAYS[S.dmin]) + " – " + prettyDayShort(DAYS[S.dmax]));
}

function faceLabel() {
  const lab = $("faclab");
  if (S.fmin === FCMIN && S.fmax === FCMAX) lab.textContent = "Any number";
  else if (S.fmin === S.fmax) lab.textContent = S.fmin + (S.fmin === 1 ? " face" : " faces");
  else lab.textContent = S.fmin + " – " + S.fmax + " faces";
}

function buildNames() {
  const box = $("names"); box.innerHTML = "";
  const counts = {};
  for (const n of DATA.names) counts[n] = 0;
  for (const it of DATA.items) for (const n of it.n) counts[n] = (counts[n]||0) + 1;
  // Order the roster by the chosen key; ties (and equal counts) fall back to A→Z.
  const order = [...DATA.names].sort((a, b) =>
    S.nsort === "za" ? b.localeCompare(a) :
    S.nsort === "hi" ? (counts[b] - counts[a]) || a.localeCompare(b) :
    S.nsort === "lo" ? (counts[a] - counts[b]) || a.localeCompare(b) :
    a.localeCompare(b));
  for (const n of order) {
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

const folderLabel = f => f === "" ? "album root" : f;
function buildFolders() {
  const box = $("folders"); box.innerHTML = "";
  const counts = {};
  for (const f of FOLDERS) counts[f] = 0;
  for (const it of DATA.items) counts[it.sf] = (counts[it.sf] || 0) + 1;
  for (const f of FOLDERS) {
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = !S.foldersOff.has(f);
    cb.onchange = () => { cb.checked ? S.foldersOff.delete(f) : S.foldersOff.add(f); render(); };
    const span = document.createElement("span"); span.textContent = folderLabel(f);
    const c = document.createElement("span"); c.className = "count"; c.textContent = counts[f];
    lab.append(cb, span, c); box.appendChild(lab);
  }
}

// --- lightbox: show the (already-loaded) thumbnail instantly, swap in the
// 2048px render when it arrives; clicking the image opens the original. ---
let lbi = -1;
function stopVid() {                                  // pause + detach so audio never plays on
  const v = $("lbvid"); v.pause(); v.removeAttribute("src"); v.load(); v.style.display = "none";
}
function openLB(i) {
  lbi = i; const it = current[i];
  const img = $("lbimg");
  const day = dayOf(it), time = prettyTime(it.dt);
  $("lbprev").style.visibility = i > 0 ? "visible" : "hidden";
  $("lbnext").style.visibility = i < current.length - 1 ? "visible" : "hidden";
  if (it.v) {
    // Video: stream from /video/ (Range-served so seeking works). Some codecs
    // (e.g. HEVC .mov outside Safari) won't decode; the download link is the
    // fallback. No /full preview warming for videos.
    img.style.display = "none"; img.classList.remove("loading");
    const url = "/video/" + encodeURIComponent(it.k);
    const vid = $("lbvid"); vid.style.display = ""; vid.src = url; vid.play().catch(() => {});
    $("lbcap").innerHTML = "<b>" + (it.n.join(", ") || "Video") + "</b> · " + prettyDay(day) +
      (time ? " · " + time : "") + (it.d ? " · " + fmtDur(it.d) : "") +
      ' <a href="' + url + '" target="_blank" rel="noopener">download &#8599;</a>';
  } else {
    stopVid();
    img.style.display = "";
    img.classList.add("loading");
    img.src = "/thumb/" + encodeURIComponent(it.k);     // instant placeholder
    const orig = "/full/" + encodeURIComponent(it.k) + "?full=1";
    img.title = "Open full resolution";
    img.onclick = () => window.open(orig, "_blank", "noopener");
    const full = new Image();
    full.onload = () => { if (lbi === i) { img.src = full.src; img.classList.remove("loading"); } };
    full.src = "/full/" + encodeURIComponent(it.k);
    $("lbcap").innerHTML = "<b>" + (it.n.join(", ") || "(no names)") + "</b> · " + prettyDay(day) +
      (time ? " · " + time : "") +
      ' <a href="' + orig + '" target="_blank" rel="noopener">full resolution &#8599;</a>';
    if (i > 0) warm(current[i - 1].k);                     // warm neighbors so arrow-nav is instant
    if (i < current.length - 1) warm(current[i + 1].k);
  }
  $("lb").style.display = "flex";
}
function closeLB() { stopVid(); $("lb").style.display = "none"; lbi = -1; }
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
$("nsort").value = S.nsort;
$("nsort").onchange = e => { S.nsort = e.target.value; buildNames(); saveState(); };
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
// dual range over distinct capture days (by index). Hidden when there's nothing
// to filter on (0 or 1 distinct day), like the Faces slider.
const dmin = $("dmin"), dmax = $("dmax");
if (DAYS.length < 2) {
  $("datehead").style.display = "none"; $("datewrap").style.display = "none"; $("datelab").style.display = "none";
} else {
  dmin.min = dmax.min = DMIN; dmin.max = dmax.max = DMAX;
  dmin.value = S.dmin; dmax.value = S.dmax;
  const updDFill = () => {
    const span = Math.max(1, DMAX - DMIN);
    $("dfill").style.left = (S.dmin - DMIN) / span * 100 + "%";
    $("dfill").style.right = (DMAX - S.dmax) / span * 100 + "%";
  };
  dmin.oninput = e => { S.dmin = Math.min(+e.target.value, S.dmax); e.target.value = S.dmin; dateLabel(); updDFill(); render(); };
  dmax.oninput = e => { S.dmax = Math.max(+e.target.value, S.dmin); e.target.value = S.dmax; dateLabel(); updDFill(); render(); };
  dateLabel(); updDFill();
}
// dual range over the face-count bounds. Hidden when there's no spread to filter
// on (no faces.csv, or every counted item has the same count).
const fmin = $("fmin"), fmax = $("fmax");
if (FCMIN === FCMAX) {
  $("fcwrap").previousElementSibling.style.display = "none";   // the "Faces" <h2>
  $("fcwrap").style.display = "none"; $("faclab").style.display = "none";
} else {
  fmin.min = fmax.min = FCMIN; fmin.max = fmax.max = FCMAX;
  fmin.value = S.fmin; fmax.value = S.fmax;
  const updFFill = () => {
    const span = Math.max(1, FCMAX - FCMIN);
    $("ffill").style.left = (S.fmin - FCMIN) / span * 100 + "%";
    $("ffill").style.right = (FCMAX - S.fmax) / span * 100 + "%";
  };
  fmin.oninput = e => { S.fmin = Math.min(+e.target.value, S.fmax); e.target.value = S.fmin; faceLabel(); updFFill(); render(); };
  fmax.oninput = e => { S.fmax = Math.max(+e.target.value, S.fmin); e.target.value = S.fmax; faceLabel(); updFFill(); render(); };
  faceLabel(); updFFill();
}
$("scene").value = S.scene;
$("scene").onchange = e => { S.scene = e.target.value; render(); };
for (const cb of document.querySelectorAll('#media input[type=checkbox]')) {
  cb.checked = S.media[cb.dataset.media] !== false;
  cb.onchange = () => { S.media[cb.dataset.media] = cb.checked; render(); };
}
// One folder (or none) means nothing to filter on — hide the section, like Faces.
// Otherwise it's a collapsible section (CSS hides the list while .closed is set).
const foldhead = $("foldhead");
if (FOLDERS.length < 2) {
  foldhead.style.display = "none"; $("folders").style.display = "none";
} else {
  foldhead.classList.toggle("closed", !S.foldOpen);   // reflect restored open/closed state
  foldhead.onclick = () => {
    S.foldOpen = !S.foldOpen;
    foldhead.classList.toggle("closed", !S.foldOpen);
    saveState();
  };
}
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

buildNames(); buildFolders(); hourLabel(); updFill(); render();
</script>
</body>
</html>"""


# MIME types for the bytes `serve` streams from /video/. Anything else falls
# back to octet-stream (the browser sniffs / offers a download).
VIDEO_CTYPES = {
    ".mp4": "video/mp4", ".m4v": "video/x-m4v", ".mov": "video/quicktime",
    ".webm": "video/webm", ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
}


def parse_byte_range(header: str | None, file_size: int):
    """Parse one HTTP Range header against *file_size*. Pure, so it's unit-tested.

    Browsers won't reliably play a <video> served as a plain 200 — they expect
    `Accept-Ranges: bytes` and 206 partial responses so seeking only fetches the
    needed slice. Returns:
      None         -> no/ill-formed range; caller serves the whole file (200).
      False        -> a syntactically valid but unsatisfiable range (-> 416).
      (start, end) -> inclusive byte bounds clamped to the file (-> 206).
    Only the first range of a multi-range request is honored (enough for media
    scrubbing; multipart/byteranges isn't worth the complexity here).
    """
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes="):].split(",")[0].strip()
    if "-" not in spec:
        return None
    a, b = (s.strip() for s in spec.split("-", 1))
    try:
        if a == "":                       # suffix form "bytes=-N" -> last N bytes
            if b == "":
                return None
            length = int(b)
            if length <= 0:
                return False
            start, end = max(0, file_size - length), file_size - 1
        else:
            start = int(a)
            end = int(b) if b else file_size - 1
    except ValueError:
        return None
    end = min(end, file_size - 1)
    if start > end or start >= file_size:
        return False
    return (start, end)


def _video_dt(path: Path) -> str:
    """Capture datetime 'YYYY:MM:DD HH:MM:SS' for a video, or '' if unknown.

    Mirrors `_exif_dt`'s format so videos sort and date-group alongside photos.
    Prefers the container's `creation_time` tag (ffprobe), which iPhone records in
    UTC — converted to local time so it lines up with the photos' local EXIF
    clock. Falls back to the file's mtime if there's no tag (or no ffprobe).
    """
    import json
    import subprocess
    from datetime import datetime, timezone

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_entries", "format_tags=creation_time", str(path)],
            capture_output=True, text=True, timeout=20,
        ).stdout
        ct = (json.loads(out or "{}").get("format", {})
              .get("tags", {}).get("creation_time", ""))
        if ct:
            dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone().strftime("%Y:%m:%d %H:%M:%S")
    except Exception:  # noqa: BLE001 - ffprobe missing / odd container -> mtime
        pass
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y:%m:%d %H:%M:%S")
    except Exception:  # noqa: BLE001
        return ""


def _video_duration(path: Path) -> float:
    """Duration of a video in seconds (ffprobe), or 0.0 if unknown / no ffprobe.

    Feeds the grid's length badge — and makes Live-Photo motion clips (≈2s) easy
    to tell apart from real videos at a glance.
    """
    import json
    import subprocess

    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_entries", "format=duration", str(path)],
            capture_output=True, text=True, timeout=20,
        ).stdout
        return float(json.loads(out or "{}").get("format", {}).get("duration") or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


def _video_poster(path: Path):
    """A representative frame of *path* as an upright RGB PIL image, or None.

    Used as the grid thumbnail for a video. Seeks ~1s in (skips black intro
    frames) with a fast input seek; retries at 0s for clips shorter than that.
    Returns None if ffmpeg is missing or can't decode — the caller then draws a
    generic placeholder, so a video is never dropped from the gallery.
    """
    import io
    import subprocess

    from PIL import Image

    for ss in ("1", "0"):
        try:
            out = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-ss", ss, "-i", str(path),
                 "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
                capture_output=True, timeout=30,
            ).stdout
            if out:
                return Image.open(io.BytesIO(out)).convert("RGB")
        except Exception:  # noqa: BLE001 - no ffmpeg / undecodable -> placeholder
            return None
    return None


def _sample_video_frames(path: Path, fps: float, max_frames: int = 0):
    """Yield (t_seconds, BGR ndarray) frames sampled from *path* at ~*fps*.

    ffmpeg decodes the whole clip once and writes the sampled frames to a temp
    dir (cheaper and simpler than seeking per frame); we then hand them out one at
    a time as the BGR arrays insightface expects. Yields nothing if ffmpeg is
    missing or the clip can't be decoded — the caller just records zero faces.
    With *max_frames* > 0 a long clip is evenly down-sampled to that many frames
    so one outlier video can't dominate the run.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                 "-vf", f"fps={fps}", "-qscale:v", "3", str(tdp / "%06d.jpg")],
                capture_output=True, timeout=600, check=False,
            )
        except Exception:  # noqa: BLE001 - no ffmpeg / hang -> no frames
            return
        frames = sorted(tdp.glob("*.jpg"))
        if max_frames and len(frames) > max_frames:
            frames = _even_sample(frames, max_frames)
        for fp in frames:
            try:
                t = (int(fp.stem) - 1) / fps          # %06d is 1-indexed
                yield t, load_image_bgr(fp)
            except Exception:  # noqa: BLE001 - skip a single unreadable frame
                continue


def _placeholder_thumb(size: int):
    """A neutral film-strip tile for a video with no extractable poster frame."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (size, size), (38, 41, 48))
    d = ImageDraw.Draw(im)
    s = max(6, size // 12)                       # sprocket-hole strip down each edge
    for y in range(s, size - s, 2 * s):
        d.rectangle([s // 2, y, s, y + s], fill=(28, 30, 36))
        d.rectangle([size - s, y, size - s // 2, y + s], fill=(28, 30, 36))
    c, r = size // 2, size // 7                   # centered play triangle
    d.polygon([(c - r, c - r), (c - r, c + r), (c + r, c)], fill=(154, 163, 178))
    return im


def assign_thumb_keys(filenames: list[str]) -> dict[str, str]:
    """Map each album-relative filename -> a unique, filesystem-safe key used for
    its thumbnail cache file and its /thumb//full URLs in `serve`.

    A bare stem identifies a photo uniquely ONLY when no other indexed photo
    shares it. With subfolders two photos can share a stem (a reused
    IMG_4492.HEIC in different import folders, or IMG_1.HEIC vs IMG_1.JPG); a
    stem-named cache file would then alias the wrong photo, and a stale thumb
    from a prior run would be served for the new one. So any stem shared by >1
    photo is keyed by stem + a short hash of the full path: stable per photo,
    distinct between the sharers, and distinct from the bare-stem name so stale
    stem-named cache entries are orphaned rather than reused. Unique stems keep
    their bare-stem key, so an existing cache stays valid and only genuinely
    ambiguous thumbs rebuild.
    """
    from collections import Counter

    counts = Counter(Path(fn).stem for fn in filenames)
    keys = {}
    for fn in filenames:
        stem = Path(fn).stem
        if counts[stem] == 1:
            keys[fn] = stem
        else:
            keys[fn] = f"{stem}-{hashlib.sha1(fn.encode()).hexdigest()[:8]}"
    return keys


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

    # Prune thumbs whose photo is no longer indexed under its current key — e.g. a
    # stem that became shared after a new import now uses a path-hash key, orphaning
    # its old bare-stem thumb. The hash scheme already prevents a stale thumb being
    # *served* for the wrong photo; this just keeps orphans from piling up on disk.
    valid = {it["k"] for it in items}
    orphans = [p for p in cache.glob("*.jpg") if p.stem not in valid]
    if orphans:
        print(f"Thumbnails: pruning {len(orphans)} orphaned thumb(s).")
        for p in orphans:
            p.unlink()

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
        if it.get("v"):                 # video: poster frame, or a placeholder tile
            img = _video_poster(src) or _placeholder_thumb(size)
        else:
            img = load_image_rgb_pil(src)   # upright RGB; EXIF dropped on save below
        img.thumbnail((size, size))
        img.save(cache / f"{it['k']}.jpg", "JPEG", quality=80)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for _ in ex.map(one, todo):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(todo)}")
    print(f"  done ({len(todo)} built).")


def _load_json_cache(path: Path) -> dict:
    """Read a JSON dict sidecar from the thumb cache, or {} if missing/corrupt."""
    import json

    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def cmd_serve(args) -> int:
    import http.server
    import io
    import json
    import os
    import socketserver
    import tempfile
    import threading
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

    # Per-photo face count (how crowded the shot is): the number of faces the
    # detector found, i.e. every faces.csv row for the photo. Powers the gallery's
    # min/max-faces slider. NOTE: this deliberately does NOT reuse `cluster`'s
    # size/det pre-filter — that threshold drops faces whose *embeddings* are too
    # noisy to identify, which is unrelated to a headcount. A distant group shot
    # has many small-but-clearly-detected faces and the slider should count them
    # all; the detector already gates on confidence at embed time (det_thresh), so
    # the raw row count is the right "how many faces are in this photo" answer.
    # A photo not in faces.csv (or when faces.csv is missing) has a genuinely
    # unknown count -> null -> always passes, like unknown-hour items. (Videos get
    # their count from the `video` pass — see video items below.)
    face_counts: dict[str, int] = {}
    fcsv = Path(args.faces)
    if fcsv.exists():
        from collections import Counter

        fc = Counter()
        for r in read_face_rows(fcsv):
            fc[r["filename"]] += 1
        face_counts = dict(fc)

    # Optional scene/hour overlay — present iff a `scene` pass has been run.
    hours, scenes = {}, {}
    sp = Path(args.scene)
    if sp.exists():
        for r in read_face_rows(sp):
            h = r.get("hour")
            hours[r["filename"]] = int(h) if h and h.isdigit() else None
            scenes[r["filename"]] = r.get("scene") or ""

    # scene.csv only covers photos, so videos would have no indoor/outdoor tag and
    # vanish whenever a scene is selected. The album's scene is time-derived (see
    # DESIGN.md), so reconstruct an hour -> scene map from the photos that DO have a
    # scene and reuse it to tag videos by their capture hour — no new config, same
    # rule. (Falls back to "unknown" for hours no photo covers; those pass the
    # filter, just as unknown-hour items pass the time filter.)
    hour_scene: dict[int, str] = {}
    if scenes:
        from collections import Counter, defaultdict

        votes: dict[int, Counter] = defaultdict(Counter)
        for fn, sc in scenes.items():
            h = hours.get(fn)
            if sc and h is not None:
                votes[h][sc] += 1
        hour_scene = {h: c.most_common(1)[0][0] for h, c in votes.items()}

    # Optional video-name overlay — present iff a `video` pass has been run, so
    # clips get the same name filter/caption treatment as photos.
    vpeople = {}
    vp = Path(args.video_people)
    if vp.exists():
        vpeople = {r["filename"]: sorted(filter(None, (r.get("names") or "").split(";")))
                   for r in read_face_rows(vp)}

    album = Path(args.album)
    # `k` is a filesystem-safe key used for thumb/full/video URLs and as the cache
    # filename; `f` is the real album-relative path used to read original bytes.
    # See assign_thumb_keys for why a bare stem isn't always safe with subfolders.
    visible = [(fn, names) for fn, names in sorted(people.items())
               if (album / fn).exists()]
    # Videos are an additive view layer: the face pipeline ignores them, so they
    # carry no names and are discovered straight from album/ (not image_people).
    video_fns = [] if args.no_videos else [rel_key(p, album) for p in list_videos(album)]
    # Key photos and videos together so a video and photo sharing a stem
    # (IMG_1.MP4 vs IMG_1.HEIC) still get distinct, collision-free cache keys.
    keys = assign_thumb_keys([fn for fn, _ in visible] + video_fns)
    items, by_key = [], {}
    for fn, names in visible:
        it = {"k": keys[fn], "f": fn, "n": names,
              "fc": face_counts.get(fn),  # None when not in faces.csv -> unknown count
              "h": hours.get(fn), "s": scenes.get(fn, ""), "v": False}
        items.append(it)
        by_key[keys[fn]] = it
    for fn in video_fns:
        # Videos aren't in the face pipeline, so their "face count" is the number
        # of distinct *named* people from the `video` pass (the only per-clip head
        # count available). This undercounts unnamed kids vs a photo's all-faces
        # count, but it's far better than leaving videos uncounted — that made a
        # clip with several kids show up under a "1–2 faces" filter. A video the
        # `video` pass never scanned has no entry -> null -> always passes.
        names_v = vpeople.get(fn)
        it = {"k": keys[fn], "f": fn, "n": names_v or [],
              "fc": len(names_v) if names_v is not None else None,
              "h": None, "s": "", "v": True}
        items.append(it)
        by_key[keys[fn]] = it
    if not items:
        print(f"No album files found for the {len(people)} indexed photos under {album}/.", file=sys.stderr)
        return 1

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    # 2048px lightbox previews are decoded on demand and cached here so each
    # original is HEIC-decoded at most once ever (the slow part). Kept in a
    # subdir so _build_thumbs's `*.jpg` stale-sweep never touches them.
    pcache = cache / "preview"
    pcache.mkdir(exist_ok=True)
    _build_thumbs(items, album, cache, args.thumb, args.prefetch)

    # Capture-date sidecar for sort + date grouping. Re-reading EXIF for every
    # photo each launch is slow, so cache filename -> datetime in the thumb cache
    # and only extract ones we haven't recorded yet.
    dates_path = cache / "dates.json"
    dates = _load_json_cache(dates_path)
    missing = [it for it in items if it["f"] not in dates]
    if missing:
        print(f"Reading capture dates for {len(missing)} item(s) ...")

        def _read_dt(it):
            p = album / it["f"]
            return it["f"], (_video_dt(p) if it["v"] else _exif_dt(p))

        with ThreadPoolExecutor(max_workers=max(1, args.prefetch)) as ex:
            for fn, dt in ex.map(_read_dt, missing):
                dates[fn] = dt
        dates_path.write_text(json.dumps(dates))
    for it in items:
        it["dt"] = dates.get(it["f"], "")
        # Videos have no scene.csv hour; derive it from the capture time so the
        # time-of-day filter applies to them just like photos.
        if it["v"] and it["dt"]:
            h = _hour_from_dt(it["dt"])
            if h >= 0:
                it["h"] = h
        # ...then tag the video indoor/outdoor by that hour (see hour_scene above).
        if it["v"] and not it["s"] and it["h"] is not None:
            it["s"] = hour_scene.get(it["h"], "")

    # Video durations (seconds) for the grid length badge — ffprobe once, cached
    # in the same way as dates. Only videos need it.
    durs_path = cache / "durations.json"
    durs = _load_json_cache(durs_path)
    miss_d = [it for it in items if it["v"] and it["f"] not in durs]
    if miss_d:
        print(f"Reading durations for {len(miss_d)} video(s) ...")
        with ThreadPoolExecutor(max_workers=max(1, args.prefetch)) as ex:
            for fn, d in ex.map(lambda it: (it["f"], _video_duration(album / it["f"])), miss_d):
                durs[fn] = d
        durs_path.write_text(json.dumps(durs))
    for it in items:
        it["d"] = float(durs.get(it["f"], 0.0)) if it["v"] else 0.0

    all_names = sorted({n for it in items for n in it["n"]})
    # Album-relative parent dir of each item (POSIX, see rel_key) — drives the
    # gallery's per-subfolder filter. Files sitting directly in album/ have no
    # subfolder; "" groups them together (labelled "album root" client-side).
    def _subfolder(fn: str) -> str:
        return fn.rsplit("/", 1)[0] if "/" in fn else ""

    all_folders = sorted({_subfolder(it["f"]) for it in items})
    manifest = {"names": all_names, "folders": all_folders, "liveMax": args.live_max,
                "items": [{"k": it["k"], "n": it["n"], "fc": it["fc"],
                           "h": it["h"], "s": it["s"], "sf": _subfolder(it["f"]),
                           "dt": it["dt"], "v": 1 if it["v"] else 0,
                           "d": round(it["d"], 1) if it["v"] else 0}
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

        def _send_file_range(self, src: Path, ctype: str):
            """Stream *src* with HTTP Range support so <video> can seek/scrub.

            Browsers play media by requesting byte ranges and expect 206 partial
            responses; without `Accept-Ranges`/206 many won't play at all. Reads
            in chunks so a multi-GB clip never lands in RAM, and swallows the
            client-aborted-connection errors that seeking routinely triggers.
            """
            size = src.stat().st_size
            rng = parse_byte_range(self.headers.get("Range"), size)
            if rng is False:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            if rng is None:
                start, end, code = 0, size - 1, 200
            else:
                start, end, code = rng[0], rng[1], 206
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            remaining = end - start + 1
            try:
                with src.open("rb") as f:
                    f.seek(start)
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client seeked away / closed the tab mid-stream — normal

        def do_GET(self):
            from urllib.parse import unquote, urlparse
            parts = urlparse(self.path)
            path, query = unquote(parts.path), parts.query
            if path == "/":
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif path == "/favicon.svg":
                self._send(200, FAVICON_SVG.encode(), "image/svg+xml",
                           {"Cache-Control": "max-age=86400"})
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
                src = (album / it["f"]) if (it and not it["v"]) else None   # videos use /video, not /full
                if not (src and src.exists()):
                    self._send(404, b"not found", "text/plain")
                elif "full=1" in query:                # full res: rare path, decode each time
                    img = load_image_rgb_pil(src)
                    buf = io.BytesIO()
                    img.save(buf, "JPEG", quality=90)
                    self._send(200, buf.getvalue(), "image/jpeg",
                               {"Cache-Control": "max-age=3600"})
                else:                                  # 2048px preview: build once to disk, then serve cached
                    pf = pcache / f"{it['k']}.jpg"
                    if not pf.exists():
                        img = load_image_rgb_pil(src)
                        img.thumbnail((2048, 2048))
                        tmp = pf.with_suffix(f".{threading.get_ident()}.tmp")   # per-thread temp; atomic rename so a concurrent reader never sees a partial file
                        img.save(tmp, "JPEG", quality=90)
                        os.replace(tmp, pf)
                    self._send(200, pf.read_bytes(), "image/jpeg",
                               {"Cache-Control": "max-age=3600"})
            elif path.startswith("/video/"):
                it = by_key.get(path[len("/video/"):])
                src = (album / it["f"]) if (it and it["v"]) else None
                if not (src and src.exists()):
                    self._send(404, b"not found", "text/plain")
                else:
                    ctype = VIDEO_CTYPES.get(src.suffix.lower(), "application/octet-stream")
                    self._send_file_range(src, ctype)
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
                        if it["v"]:                  # videos export as-is (no re-encode)
                            z.write(src, it["f"])
                        elif fmt == "jpeg":
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
    n_vid = sum(1 for it in items if it["v"])
    summary = f"{len(items) - n_vid} photos" + (f", {n_vid} videos" if n_vid else "")
    print(f"\n  headcount browsing {summary}, {len(all_names)} names")
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
    p_emb.add_argument("--no-dedup", action="store_true",
                       help="embed every image even if byte-identical to one already embedded")
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
    p_qry.add_argument("--split-scene", action="store_true",
                       help="fan matches into indoor/ and outdoor/ subfolders (needs a `scene` "
                            "pass; mutually exclusive with --where)")
    p_qry.add_argument("--allow-unscored", action="store_true",
                       help="with --where/--split-scene, proceed even if scene.csv is missing some "
                            "matches (else the query aborts on a stale scene.csv); untagged photos "
                            "go to unscored/")
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
    p_qry.add_argument("--zip", action="store_true",
                       help="also pack the result folder into <out>/<label>.zip")
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
    p_scn.add_argument("--subdir", default="",
                       help="only classify images under album/<subdir>/ and MERGE into "
                            "scene.csv, keeping other batches' rows. Use when an import's "
                            "outdoor block differs from earlier ones (e.g. --subdir 20260618 "
                            "--outdoor-hours 13-14).")
    p_scn.add_argument("--thresh", type=float, default=0.12,
                       help="green+sky fraction >= this is outdoor (default: 0.12)")
    p_scn.add_argument("--limit", type=int, default=0, help="only first N images (for testing)")
    p_scn.add_argument("--prefetch", type=int, default=4, help="background decode threads (default: 4)")
    p_scn.set_defaults(func=cmd_scene)

    p_vid = sub.add_parser("video",
                           help="detect + name faces in album videos -> video_people.csv (uses existing labels)")
    p_vid.add_argument("--album", default="album", help="album folder (default: album/)")
    p_vid.add_argument("--faces", default="faces.csv", help="face table (default: faces.csv)")
    p_vid.add_argument("--clusters", default="clusters.csv", help="cluster assignment (default: clusters.csv)")
    p_vid.add_argument("--labels", default="labels.csv", help="filled-in labels (default: labels.csv)")
    p_vid.add_argument("--out", default="video_people.csv", help="who-is-in-each-video index")
    p_vid.add_argument("--fps", type=float, default=1.0,
                       help="frames sampled per second of video (default: 1.0)")
    p_vid.add_argument("--max-frames", type=int, default=0,
                       help="evenly down-sample to at most this many frames per clip (0 = no cap)")
    p_vid.add_argument("--thresh", type=float, default=0.35,
                       help="min cosine similarity to a name centroid to count (default: 0.35, calibrated)")
    p_vid.add_argument("--margin", type=float, default=0.05,
                       help="matched name must beat the runner-up name by this (default: 0.05)")
    p_vid.add_argument("--min-size", type=int, default=40,
                       help="ignore detected faces whose bbox side is under this many px (default: 40)")
    p_vid.add_argument("--det-size", type=int, default=640,
                       help="detector input size; video frames are smaller than photos (default: 640)")
    p_vid.add_argument("--det-thresh", type=float, default=0.4, help="detector confidence (default: 0.4)")
    p_vid.add_argument("--limit", type=int, default=0, help="only scan first N videos (for testing)")
    p_vid.set_defaults(func=cmd_video)

    p_srv = sub.add_parser("serve", help="local web browser: name/time filters + zip export (localhost only)")
    p_srv.add_argument("--album", default="album", help="album folder (default: album/)")
    p_srv.add_argument("--image-people", default="image_people.csv", help="index from `assign`")
    p_srv.add_argument("--faces", default="faces.csv",
                       help="face index from `embed` — powers the face-count filter (optional)")
    p_srv.add_argument("--scene", default="scene.csv", help="scene/hour index from `scene` (optional)")
    p_srv.add_argument("--video-people", default="video_people.csv",
                       help="per-video name index from `video` (optional)")
    p_srv.add_argument("--cache", default=".serve_cache", help="thumbnail cache dir (default: .serve_cache/)")
    p_srv.add_argument("--thumb", type=int, default=768, help="thumbnail long edge in px (default: 768)")
    p_srv.add_argument("--prefetch", type=int, default=4, help="background decode threads for thumbs (default: 4)")
    p_srv.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1 — localhost only)")
    p_srv.add_argument("--port", type=int, default=8765, help="port (default: 8765)")
    p_srv.add_argument("--no-open", action="store_true", help="don't auto-open a browser tab")
    p_srv.add_argument("--no-videos", action="store_true",
                       help="don't surface album videos in the gallery (photos only)")
    p_srv.add_argument("--live-max", type=float, default=3.5,
                       help="videos this many seconds or shorter count as 'live photos' "
                            "in the Media filter (default: 3.5)")
    p_srv.set_defaults(func=cmd_serve)

    args = ap.parse_args()

    if getattr(args, "rescan", False):
        faces_csv = Path(args.faces)
        faces_csv.unlink(missing_ok=True)
        faces_csv.with_suffix(".emb").unlink(missing_ok=True)
        faces_csv.with_suffix(".npy").unlink(missing_ok=True)
        faces_csv.with_suffix(".done").unlink(missing_ok=True)
        faces_csv.with_suffix(".hashes").unlink(missing_ok=True)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
