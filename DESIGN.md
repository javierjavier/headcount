# Design: whole-class clustering & multi-person queries

Status: **built** in `faces.py`. This document originally sketched the whole-class
step — detect every face, group faces by identity, label the groups once, then
sort the whole class and answer multi-person queries ("photos with both Ada and
Ben") — as the successor to an earlier single-child matcher (`match.py`). That
step is now the tool, and `match.py` has been removed (see *Unification* below);
`enroll.py` remains only as the cold-start cluster-calibration seeder.

Build status: **shipped** in `faces.py`
(`embed` → `cluster` → `review` → `assign` → `query`, plus an optional `scene`
indoor/outdoor dimension). Calibration is settled: the default clusterer is
HDBSCAN (`min_cluster_size=15`), chosen after a single global DBSCAN `eps` proved
unworkable on the full 18k-face album (early sample tuning had landed on
`eps≈0.50`, clean for the reference child but over-merging younger kids — see
*cluster* below for why HDBSCAN replaced it).

It follows the same two-phase philosophy as the existing pipeline: do the
expensive compute once, then iterate cheaply.

## Goal

Turn the one-child matcher into a whole-class sorter:

1. Detect every face in the album.
2. Cluster faces by identity (one cluster ≈ one child).
3. Label clusters with names (human-in-the-loop, once).
4. Emit `by_child/<name>/` folders, and answer set queries over who-is-in-what.

The existing single-child path stays — it's the simpler tool when you "just want
Ada." Both paths share the face-embedding backbone (`faces.npy`).

## Pipeline

```
album/ ──embed──▶ faces.npy + faces.csv ──cluster──▶ clusters.csv
                                                          │
                                              review (montages) ──▶ labels.csv  ◀── you edit
                                                          │
                                   assign ──▶ image_people.csv ──▶ by_child/<name>/
                                                          │
                                                       query ──▶ query/<expr>/
```

### 1. `embed` — expensive, once-only  ✅ built

Like today's `scan`, but stores *every* face, not just the best score per image.
Implemented as `faces.py embed`. Speed comes from `--prefetch` (overlapping the
single-threaded HEIC decode with inference on background threads, ~1.4–1.5× on an
M1). Multi-process sharding was tried and measured at ~1.0× on a single machine,
so it was dropped — see the README's "Speed & adding photos" note.

- `faces.npy` — `M × 512` array of L2-normalized embeddings.
  M ≈ 25–40k faces for this album; ~80 MB. Trivial.
- `faces.csv` — one row per face, parallel to `faces.npy`:
  `face_id, filename, x1, y1, x2, y2, det_score`
- `faces.emb` — the append journal it's built from: raw float32, 512 per face,
  flushed per image so a crash leaves a consistent csv/emb pair. `faces.npy` is
  finalized from it (`np.save`) at the end of `embed`/`merge`. Row *i* of
  `faces.npy` ↔ `face_id` *i* in `faces.csv`.

This is the artifact the current `scan` throws away (it keeps only the best score
per image). Building it also retroactively makes the single-child matcher instant
to re-reference — no re-detection when references change.

**Pre-filter — deferred to `cluster`, not `embed`.** The plan was to drop tiny /
low-`det_score` faces here, before clustering. In the build it moved to `cluster`
instead: `embed` is the expensive once-only phase, and the purity filter is a
knob you'll want to re-tune (like `eps`) without paying a full re-embed. So
`embed` stores *every* detected face above the detector's `det_thresh` floor, and
`cluster` applies the cheap, re-runnable min-size / min-`det_score` pre-filter.
Same expensive-once / tune-cheap split as `scan` / `collect`. (Tiny/blurry faces
still matter most for cluster purity — that reasoning is unchanged; only *where*
the filter runs moved.)

### 2. `cluster` — fast, re-runnable (the `collect`-equivalent)

Load `faces.npy` (already normalized → cosine = dot product), cluster, write
`clusters.csv` (`face_id → cluster_id`).

- **Algorithm: HDBSCAN** (scikit-learn ≥1.3), with DBSCAN kept as `--algo
  dbscan`. Both need no guess at the number of kids and route junk/uncertain
  faces to a noise bucket (`-1`). Embeddings are unit-norm, so euclidean distance
  is monotonic in cosine — same ordering.
- **Why HDBSCAN won, empirically.** DBSCAN's single global `eps` failed on the
  full 18k-face album: *no* value worked. Tight `eps` (≤0.31) kept the reference
  child 100% pure but dumped ~70% of faces to noise; loose `eps` (≥0.42) chained
  many kids into one 5k–15k-face mega-blob. Class photos have very uneven density
  (kids photographed wildly different amounts), which is exactly DBSCAN's blind
  spot. HDBSCAN (`min_cluster_size=15`) extracts clusters at locally-appropriate
  density: 22 clean clusters, largest 892 (a plausible single kid, no blob), the
  reference child 100% pure at 84% recall. Cost: conservative (~40% of faces to
  noise, recoverable by nearest-cluster assignment) and slower (~4 min vs DBSCAN's
  seconds) — a good trade.
- **Calibrated against ground truth** like the 0.35 threshold: the reference
  child's known faces should land in one clean cluster. The one knob is
  `min_cluster_size` (HDBSCAN) or `eps` (DBSCAN); re-running is cheap (no
  re-embed) — same embed-once / tune-fast split as `scan`/`collect`.

### 3. `review` — human-in-the-loop (reuses montage tooling)

For each cluster, emit a contact sheet of representative crops (medoid + a few
samples) into `clusters/<cluster_id>/`, sorted by cluster size (most-photographed
kids first). Write a skeleton `labels.csv` (`cluster_id, name`) for you to fill
in. Minimal — folders of crops + a CSV you edit; no GUI required. Could later
become a small local HTML page.

Re-running after a new import is the common case, and HDBSCAN does **not** keep
cluster ids stable across runs (the same integer can denote a different child once
the face set changes), so carrying labels forward by cluster_id silently
mislabels. Instead `cluster` snapshots the prior membership (`clusters.csv.bak`)
and `review` re-attaches each name by **face_id majority vote**: face_id is stable
across re-embeds, so for each new cluster we look at what its member faces were
named before and take the majority, reporting the vote purity so any uncertain
remap is visible. This is the identity-stable version of the carry-forward — see
`remap_labels_by_face_id`.

### 4. `assign` — fast export (reuses `collect` copy logic)

From `labels.csv` + `clusters.csv` + `faces.csv`, compute the set of named kids
per image and persist it as `image_people.csv`:

```
filename         names
IMG_0195.HEIC    Ada;Cleo
IMG_0241.HEIC    Ada;Ben;Cleo
IMG_0940.HEIC    Ada
```

Optionally (`assign --folders`, off by default) also copy each image into
`by_child/<name>/` for every labeled face it contains (a photo with 3 named kids
lands in 3 folders). Originals by default, `--jpeg` optional.

## Multi-person queries

`image_people.csv` makes "who is in this photo" a set per image, so any
combination query is a one-line set operation. A `query` command copies matches
into `query/<expr>/`:

| Query | Meaning | Predicate |
| --- | --- | --- |
| `--with Ada,Ben` | both present | `{Ada,Ben} ⊆ names` |
| `--any Ada,Ben` | either present | `names ∩ {Ada,Ben} ≠ ∅` |
| `--with Ada --without Ben` | Ada but not Ben | `Ada ∈ names and Ben ∉ names` |
| `--only Ada,Ben` | exactly these two | `names == {Ada,Ben}` |
| `--with Ada` (alone) | Ada present (others allowed) | `{Ada} ⊆ names` |

Derived analytics are cheap too — e.g. "who is Ada most often photographed with?"
is just ranking co-occurrence counts against Ada's set.

### Caveat: AND queries have lower recall

A photo qualifies for `--with Ada,Ben` only if *both* faces were detected,
clustered, and labeled correctly. Detection misses compound:

```
P(photo with both qualifies) ≈ P(catch Ada) × P(catch Ben)
```

If each child is reliably caught ~85% of the time (turned away, tiny, blurry, or
dumped to noise otherwise), both-present recall is ~0.85 × 0.85 ≈ **72%**.
Single-person queries don't compound like this. So an AND query finds *most* true
co-appearances, not all — fine for "find photos of the two friends," but not
exhaustive. False positives are correspondingly rarer (both must be mislabeled).

## Browsing: the `serve` gallery

`faces.py serve` is a presentation layer over `image_people.csv` (+ optional
`scene.csv`): a localhost-only web app for name/time filtering and zip export,
an alternative to fanning copies into Finder folders with `query`. It binds
`127.0.0.1` only — the same local-only privacy stance as everything else.

**Thumbnail size (`--thumb`, default 768) is deliberate — don't drop it back to
320.** Previews are pre-rendered into `.serve_cache/` once (re-decoding each HEIC
is the slow part; everything else is instant). The blur trap: the grid lays out
square cells with `object-fit: cover`, which scales the image to fill by its
*short* edge, and HiDPI/Retina screens render ~2× the CSS pixels. So a cell shown
at 150 CSS px wants ~300 device px on its short edge, and the size slider goes to
300 CSS px (~600 device px). A 320px-long-edge thumb (~240px short edge on a
landscape shot) is already below that at the *default* zoom, so it upscales and
looks soft. 768px long-edge (~576px short edge) covers the largest cell at 2×.
The tradeoff is build time and disk (~5–6× vs 320); raise/lower to taste.

The cache is **size-aware**: it records the build size in a `.serve_cache/.thumb_size`
marker and clears + rebuilds when `--thumb` changes. Without this the per-key
`exists()` skip would silently keep serving whatever size was built first — a
smaller value later looks fine, a larger one stays blurry, with no signal why.

### Videos: a view-only layer over `serve`

`serve` also lists videos found in `album/` (`common.VIDEO_EXTS`). This is
deliberately a **presentation-only** addition — the face pipeline is untouched:
`embed`/`cluster`/`assign`/`query` still read images only, and videos carry no
names (until the optional `video` command names them — see below). They're
discovered straight from disk (not from `image_people.csv`), keyed
through the same `assign_thumb_keys` collision scheme as photos (a video and photo
can share a stem), and merged into the same grid/lightbox/export/filter machinery.

Three things needed handling that photos didn't:

- **Thumbnails.** No Pillow path decodes a video, so a poster frame is pulled with
  `ffmpeg` (seek ~1s in to skip black intros). ffmpeg is treated as an *optional
  system tool*, not a new pip dependency — missing it degrades to a film-strip
  placeholder tile rather than dropping the video, keeping the pure-pip install
  story intact. Posters cache in `.serve_cache/` next to photo thumbs, so the
  existing size-aware sweep and orphan-prune cover them for free.
- **Capture time.** Photos read EXIF; videos have none. `_video_dt` reads the
  container `creation_time` via `ffprobe` and converts it from UTC (how iPhone
  stores it) to local so videos date-group consistently with the photos' local
  EXIF clock, falling back to file mtime. The hour is derived from that for the
  time-of-day filter.
- **Streaming.** A `<video>` element won't play off a plain 200 — browsers expect
  `Accept-Ranges`/206 partial responses so a seek fetches only the needed slice.
  The handler grows a Range-aware file streamer; the header parsing is factored
  into the pure, unit-tested `parse_byte_range` (full / suffix / open-ended /
  unsatisfiable→416). Codec support is the browser's: H.264 `.mp4` is universal,
  HEVC `.mov` is Safari-only — so the lightbox always offers a download link.

## Scene tagging: the time window and per-folder overrides

`scene` defaults to `--method time`: a photo is `outdoor` iff its EXIF capture
hour falls in `--outdoor-hours` (default `10-11`, the daily outdoor block). This
beats the pixel (`green`) method on the regular album because green classroom
decor reads as foliage and leaks false "outdoor"; the rigid daily schedule is a
cleaner signal than the pixels. The window is **one value per run**, applied to
every photo — what varies per photo is its capture hour, compared against that
window.

That breaks for a one-off event on a *different* schedule — e.g. an all-outdoor
4pm graduation. `--subdir` lets you re-tag just that folder with a different
`--outdoor-hours`, but that is a **one-shot** rewrite, not a stored setting:
nothing records *why* those rows are outdoor, so the next full `scene` run
re-judges every row against the global window and silently reverts them. (This
bit us once: the graduation folder reverted to indoor and the cause was
invisible — the EXIF hour column still read `16`, because that hour is re-read
from the photo every run; only the *verdict* had flipped.)

`scene_overrides.csv` (CSV: `subdir,scene`) makes per-folder scene **durable**.
Every `scene` run applies it *after* time/green classification, so a listed
folder is forced to its tag (`indoor`/`outdoor`) regardless of the window and
survives full re-runs. Longest matching prefix wins (a nested override beats a
broader one); the match is prefix-on-`subdir/` so a sibling sharing a date
prefix isn't caught. It's gitignored under the `scene*` rule like all other
scene data — the folder names are event-identifying, and it's local config, not
shared state.

## Data artifacts

| File | Phase | Contents |
| --- | --- | --- |
| `faces.npy` | embed | `M × 512` face embeddings (finalized from `faces.emb`) |
| `faces.emb` | embed | raw float32 append journal, 512/face (crash-safe working file) |
| `faces.csv` | embed | `face_id, filename, bbox, det_score` (parallel to npy) |
| `faces.done` | embed | one processed filename per line, incl. zero-face images, so resume skips them (faces.csv only lists images that yielded a face) |
| `clusters.csv` | cluster | `face_id → cluster_id` |
| `labels.csv` | review | `cluster_id → name` (human-edited) |
| `image_people.csv` | assign | `filename → set of names` |
| `video_people.csv` | video | `video → set of names` (+ `n_named`, peak frame per name); `video_people.done` is its resume manifest |
| `by_child/<name>/`, `query/<expr>/` | assign/query | output copies |

All of these contain biometric data and stay gitignored (see below).

**`filename` is the path relative to `album/`**, POSIX-style — a bare basename
for a top-level file, a `subdir/name` for one in a subfolder. `album/` is scanned
recursively so each import can live in its own subfolder, which is what keeps
reused phone numbers (counter resets hand a new photo an old `IMG_4492.HEIC`)
from colliding: same basename, different relative path. Because a top-level
file's relative path equals its basename, artifacts written before subfolders
existed stay valid with no re-embed. A stray archive in the scanned tree is a hard
error (`ArchiveFoundError`), not a silent skip.

## Hard parts (and how they're handled)

- **A kid split across several clusters** (lighting/age/hair) — expected and
  normal. Handled by **many-to-one labeling**: put the same name on multiple
  clusters. The data model assumes this from the start.
- **Two similar kids merged into one cluster** — rarer. Re-cluster that cluster
  at lower `eps`, or split manually. Flag impure clusters during review.
- **Junk faces** — handled by the embed-phase pre-filter plus DBSCAN noise.
- **Singletons** (a kid photographed once) — small clusters or noise; still
  surfaced in review so they can be labeled.

## Dependencies, scale, privacy

- **Dependency:** add `scikit-learn` (DBSCAN). One line in `requirements.txt`.
- **Scale:** ~40k faces cluster in-memory in seconds. Millions would need an ANN
  index (`faiss`) — not needed here.
- **Privacy:** this builds a face database of *every child in the class*, not
  just one. It stays local and gitignored, but it is other families' children's
  biometric data. Sorting your own class photos to share back is a reasonable
  use; it deserves a conscious note, and none of it should ever go to a remote.

## Effort

A moderate build, not a tweak:

- `embed` — ✅ done (`faces.py`): stores all faces + embeddings, resumable, with
  a crash-safe csv/emb journal and a `--prefetch` decode pipeline (~1.4–1.5×).
- `cluster` — ~20 lines of scikit-learn, plus the deferred min-size/`det_score`
  pre-filter (cheap, re-runnable).
- `review` / `assign` / `query` — reuse existing montage + `collect` code.

The fiddly time is `eps` calibration and the labeling loop — the same kind of
tuning the 0.35 threshold needed. Roughly an afternoon to a day for a working v1,
plus tuning.

## Unification (done)

As anticipated, once `faces.npy` existed the single-child and whole-class paths
converged: `match.py` scoring turned out to be just a special case of querying
for one name, so it was removed — `faces.py query --with <name>` subsumes it.
Empirically the two agreed ~96% on a well-photographed child, disagreeing only in
the threshold noise band. `enroll.py` survives only as the cold-start
calibration seeder: a brand-new album has no labeled cluster to anchor on, so
external reference photos seed the `cluster` readout instead.

## Deferred: persistent name anchors

Re-importing relies on the **face_id remap** (see `review` above): face_ids are
stable across an append-style re-embed, so a re-cluster's renumbered ids don't
lose labels — each new cluster inherits the majority name of its faces' prior
labels. That covers the normal "add more photos" path completely.

A more durable scheme — **persistent name anchors** — was designed and
deliberately *not* built. The idea generalizes `reference_embeddings.npy` (today a
single enrolled child) to a `name -> medoid embedding` table over *every* labeled
kid:

- A builder (`faces.py anchor`, or folded into `assign`) groups faces by name via
  `labels.csv` + `clusters.csv`, takes each name's medoid embedding (median-like,
  robust to a few contaminant faces), and writes `name_anchors.npy` +
  `name_anchors.csv` (gitignored — biometric, like everything else).
- On a later run, each cluster's medoid is matched to its nearest name anchor by
  cosine similarity; above the calibrated **0.35** match threshold it's auto-named,
  below it's flagged as genuinely new for manual labeling.

Where it would pay off *beyond* the face_id remap:

1. **A full re-embed** (`embed --rescan`) renumbers face_ids, so the face_id join
   breaks entirely — embedding anchors still match by identity.
2. **Cross-check / blank-filling** — suggest a name (by identity proximity) for a
   cluster the face_id vote left blank because its faces were mostly old-noise
   (e.g. a kid who graduates from the noise bucket into a real cluster).
3. **New-album cold start** that shares children with this one — drop the anchors
   in and label nothing.

Why deferred: for this album's remaining append-style import(s), the face_id
remap suffices and the vote already reports purity / flags low-confidence
clusters, so anchors are insurance against a re-embed and a reusable asset for a
future album — not load-bearing now. The narrow, cheap slice is the cross-check in
(2): computable inside `review` from data already in hand (`faces.npy` +
`clusters.csv.bak` + prior labels), needing no new artifact or workflow step.

## Detecting faces *inside* videos — the `video` command ✅ built

The `serve` video support above is view-only. The follow-on — "who is in each
video?" — is **built** as `faces.py video`, deliberately as a *standalone naming
pass* rather than full pipeline integration. It samples frames, runs the existing
`buffalo_l` detector+recognizer on each, matches every face to the **already-labeled**
photo clusters, and writes `video_people.csv` (`filename, names, n_named, peaks`).
`serve` overlays those names onto clips. The face pipeline (`embed`/`cluster`/
`assign`/`query`) is byte-for-byte untouched — `video` only *reads* `faces.npy` +
`clusters.csv` + `labels.csv`.

### Why standalone, not full integration

The tempting design folds video faces into `faces.csv` so a clip flows through
`cluster`/`assign`/`query` like a photo. That was **rejected for v1** because the
album is already labeled, which collapses the cost:

- **It protects the calibration.** Dumping thousands of motion-blurred video faces
  into `cluster` perturbs HDBSCAN's density — the exact failure mode this doc warns
  about — risking the *photo* clusters. Matching against existing centroids instead
  leaves them alone.
- **It sidesteps the load-bearing problem (per-clip dedup).** Folding faces in
  means a kid on screen for 10s at 1 fps adds ~10 near-duplicate embeddings that
  swamp clustering and inflate counts, so they'd need collapsing into one medoid per
  (video, identity) track. But for a *name set per clip*, a name matched 50× is still
  one set member — so **no dedup is needed at all**. The hardest piece simply
  disappears.
- **It reuses calibrated machinery.** Naming = nearest labeled centroid at the
  calibrated **0.35** threshold with a runner-up **margin** — literally
  `assign --recover`'s logic. That math is now the shared, unit-tested
  `match_to_centroids`; `build_name_centroids` builds one centroid per *name*
  (merging a kid's split clusters, so the margin test is name-vs-name).

### Mechanics

- **Frames** via `ffmpeg` (`_sample_video_frames`, default 1 fps, optional
  `--max-frames` cap so one long clip can't dominate). ffmpeg-gated like the poster
  thumbnails: no ffmpeg → no frames → the clip just records zero names.
- **Resumable** like `embed`: a `video_people.done` manifest records each scanned
  clip. The csv is rewritten *before* the manifest append, so the manifest never
  gets ahead of the csv and resume is lossless. It's opt-in (a separate command),
  since at ~detector-inference-per-frame it's a mini-embed (~18 min for this
  album's 704 clips at 1 fps).
- **Noise control** is the `cluster`-style lever applied up front: `--min-size`
  drops tiny faces, and the threshold+margin keep ambiguous matches out. Video
  faces are weaker (blur, angle) so the defaults stay conservative.

### Still deferred: full integration

If "`query` videos by who's in them" (or discovering a kid who appears *only* in
videos) ever becomes a real need, the full-integration path remains open: give each
video face a frame-qualified key (`trip/clip.mov#t=12.0`), teach every `album /
filename` consumer to split the `#t=` and extract that frame, roll a clip's frames
up in `assign`, and add the per-(video, identity) dedup described above. The
standalone pass covers the day-to-day "who's in this clip" need at a fraction of the
cost and zero risk, so that larger change waits until it's actually warranted.
