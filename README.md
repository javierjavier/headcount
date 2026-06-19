# headcount

Sort a big class photo album by **who's in each picture** — locally and
privately. `faces.py` detects every face, clusters them by identity (HDBSCAN),
you label each cluster once, then it answers combination queries like "photos
with both Ada and Ben, taken outdoors." `enroll.py` is an optional cold-start
helper that calibrates the clustering on a brand-new album (see *Calibration*
below). The *why* — algorithm choices, tradeoffs, hard cases — is in `DESIGN.md`.

All face data is biometric and stays local — nothing is committed or uploaded.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On first run, insightface downloads the `buffalo_l` model (~300 MB) into
`~/.insightface/models/`. Everything runs on **CPU** — no GPU needed.

The codebase was built around an album that happens to be Apple **HEIC**, but the pipeline accepts any format
Pillow can read (`.jpg`, `.png`, `.webp`, `.tiff`, …) — nothing downstream
assumes HEIC. HEIC decoding specifically is handled by `pillow-heif`; for every
format, EXIF orientation is applied automatically (portrait iPhone shots would
otherwise be fed in sideways and fail to detect).

### Folders

```
album/       # [PUT THE FULL SET OF PHOTOS HERE]
             #   scanned recursively, so you can drop each batch in its own
             #   subfolder (album/2026-spring/, album/photos-3/, ...). A stray
             #   archive left in album/ is a hard error — unzip into a subfolder.
reference/   # optional: a few clear photos of one known child, used only for
             #   cold-start cluster calibration (see enroll below)
clusters/    # created by `review` — one montage per cluster, for labeling
by_child/    # created by `assign --folders` (opt-in) — one folder per named child
query/       # created by `query` — the slices you pull out
```

## Workflow (`faces.py`)

First, put all the photos in `album/` (any Pillow-readable format — see Setup).
Then the pipeline is expensive-once / tune-cheap: one slow `embed`, then fast
re-runnable steps.

```bash
# 0. populate album/ — drop the full set of photos in (or symlink a folder:
#    `ln -s /path/to/photos album`). embed reads everything in album/.

# 1. embed — detect + embed EVERY face, once (slow, resumable). HEIC decode is
#    overlapped with inference on background threads (--prefetch, ~1.5x on M1).
python faces.py embed

# 2. cluster — group faces by identity (HDBSCAN). Fast and re-runnable.
#    --min-cluster-size tunes granularity (15 is a good default). If you enrolled
#    a reference child (see Calibration), it prints a readout showing whether
#    those known faces land in ONE clean cluster.
python faces.py cluster

# 3. review — montage each cluster into clusters/ + write a skeleton labels.csv
python faces.py review

# 4. label — open clusters/ (Finder Quick Look), then type a name per cluster
#    into the 'name' column of labels.csv. Skip the junk cluster (backs of heads)
#    and any you don't know. Same kid split across two clusters? Same name.

# 5. assign — who's in each photo -> image_people.csv (+ per-child counts).
#    --recover pulls noise/profile faces into their nearest named cluster,
#    lifting recall (~84%->97% for a well-photographed kid) at ~97% purity.
#    --folders [dir] also sorts copies into by_child/<name>/ (opt-in, default
#    off; a photo with 3 named kids lands in all 3 folders). --jpeg works here too.
python faces.py assign --recover
python faces.py assign --recover --folders     # also fan out into by_child/

# 6. query — copy any slice into query/<expr>/. Default is symlinks; --copy for
#    real HEIC files; --jpeg to re-encode (most reliable Finder thumbnails, since
#    macOS thumbnails HEIC unreliably even as real copies). --jpeg also takes
#    --max-size N to downscale and --strip-exif to drop metadata.
#    Each query/<expr>/ dir is wiped and rewritten per run, so it always reflects
#    the current query.
python faces.py query --with ada,ben          # both present
python faces.py query --with ada,ben --jpeg   # browsable: reliable thumbnails
python faces.py query --with ada --without ben
python faces.py query --only ada,ben          # exactly those two

# 7. serve — browse the whole album in a localhost-only web gallery instead of
#    Finder: name + time-of-day filters, a preview-size slider, and zip export
#    (originals, or re-encoded 2048px JPEGs). Nothing leaves the machine; it
#    binds 127.0.0.1 only. First run builds a thumbnail cache (.serve_cache/),
#    which is the slow part (re-decodes each HEIC); later runs reuse it.
python faces.py serve                          # -> http://127.0.0.1:8765
python faces.py serve --thumb 1024             # sharper previews (see note)
```

Previews come from `.serve_cache/` thumbnails built at `--thumb` long-edge
(default 768). The grid fills cells by the photo's *short* edge and HiDPI/Retina
screens want ~2× the CSS pixels, so a too-small thumb upscales and looks blurry —
raise `--thumb` for crisper previews at the cost of build time and disk; lower it
to save both. Changing the size rebuilds the cache automatically (it records the
build size and clears stale thumbs); no manual `rm -rf .serve_cache` needed.

### Calibration (`enroll.py`) — optional, for a new album

Clustering is unsupervised, so on a fresh album you have no ground truth to tell
whether your `--min-cluster-size` / `--eps` produced clean clusters before you
sink time into labeling. Enrolling one child you can recognize gives `cluster` an
anchor:

```bash
# Drop 5–10 clear photos of one known child into reference/, then:
python enroll.py --reference reference/        # -> reference_embeddings.npy
python faces.py cluster                         # readout: do those faces land
                                                #   in ONE clean cluster?
```

The readout's goal: the reference faces concentrate in a single cluster that's
almost all reference — split across many clusters means raise `--eps`; mixed with
other kids means lower it. Check the "reference-to-reference similarity" that
`enroll.py` prints too: if one photo barely matches the others, it's probably the
wrong kid or a bad shot — remove it and re-enroll. Once the clustering is dialed
in and labeled, you don't need this again for routine re-runs.

A tightly-cropped close-up where the face fills the frame can *fail* to detect at
a large `det-size` (the upscaled face exceeds the detector's anchor range), so
`enroll.py` defaults to a **smaller** `det-size` (640) than `embed`. If an
obviously-clear reference reports "no face detected," it's cropped too tight —
give it margin, or lower `enroll.py --det-size`.

### Indoor / outdoor (`scene`)

Optional location dimension. If the album's GPS is stripped but the daily
schedule is rigid (e.g. an outdoor block at a fixed hour), classify by EXIF
time — instant, no decode:

```bash
python faces.py scene --method time --outdoor-hours 10-11   # -> scene.csv
python faces.py query --with ada --where outdoor
```

A foliage/sky colour method (`--method green`) also exists, but green classroom
decor (a leafy rug, a green wall) makes it leak ~20%; time wins for this album.
See `DESIGN.md` / commit history.

### Speed & adding photos

`embed` is the only slow step (~40 min for ~4.7k photos on an M1 Max, runs cool);
everything downstream is seconds. Per image the cost splits roughly in half
between decoding the 24 MP HEIC (single-threaded, via libheif) and running
detection + recognition. That 50/50 split is the HEIC worst case: a non-HEIC
input decodes through its own Pillow codec (e.g. libjpeg for a JPEG), which is
usually cheaper, so decode is a smaller share. Overlapping the decode with
inference on background threads (`--prefetch`, the default) buys ~1.5×. Running multiple independent
processes over disjoint slices *was* tried and measured at ~1.0× on a single
machine — a lone onnxruntime process plus the OS scheduler already oversubscribe
the cores — so that path was removed rather than left in as a tempting
non-speedup.

To add photos later: drop the new batch into **its own subfolder** under
`album/` (e.g. `album/photos-3/`), then re-run `embed` (skips already-done
files), `cluster`, and `assign`. Resume tracks done images in a `faces.done`
manifest (one path per line) alongside `faces.csv`/`.npy`/`.emb`. `faces.csv`
only lists images that yielded a face, so the manifest is what lets a re-run also
skip images where *no* face was detected — otherwise those would be re-decoded
every time. It's generated and gitignored; delete it (or use `embed --rescan`)
only if you want a clean rebuild.

The skip logic keys on the image's path **relative to `album/`** (e.g.
`photos-3/IMG_4492.HEIC`), not its bare basename. This is why subfolders matter:
phone counters reset and reuse old numbers, so a brand-new photo can arrive named
`IMG_4492.HEIC` while an unrelated older `IMG_4492.HEIC` already sits in the
album. Same basename, *different* relative path → no collision, no silent skip,
no overwrite. Drop each import in a fresh subfolder and reused numbers never
clash. (A top-level file's relative path *is* its basename, so artifacts written
before subfolders existed keep matching — no re-embed.) `query` and
`assign --folders` flatten these back to basenames in their output dirs, deduping
any clash with a `_1`/`_2` suffix.

If you leave a downloaded `.zip` (or other archive) sitting in `album/`, the
tools stop with an error telling you to unzip it into a subfolder and remove the
archive — rather than silently skipping every photo packed inside it. Videos and
other non-image files in a subfolder are simply ignored.

## Files

| File                      | Purpose                                            |
| ------------------------- | -------------------------------------------------- |
| `faces.py`                | The tool: `embed`/`cluster`/`review`/`assign`/`query`/`scene` |
| `enroll.py`               | Cold-start calibration: build `reference_embeddings.npy` from `reference/` |
| `common.py`               | Shared HEIC/EXIF loading, model setup, small utilities |
| `requirements.txt`        | Dependencies                                       |
| `DESIGN.md`               | Pipeline rationale, tradeoffs, hard cases          |
| `reference_embeddings.npy`| Enrolled calibration anchor (generated)            |
| `faces.csv` / `faces.npy` | Every face's metadata + embedding (generated)      |
| `faces.done`              | filenames already embedded, incl. zero-face images, for resume (generated) |
| `clusters.csv`            | face → cluster id (generated)                      |
| `labels.csv`              | cluster → name (you fill in during `review`)       |
| `image_people.csv`        | filename → people present (generated)              |
| `scene.csv`               | filename → indoor/outdoor (generated)              |
