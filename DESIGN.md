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
| `by_child/<name>/`, `query/<expr>/` | assign/query | output copies |

All of these contain biometric data and stay gitignored (see below).

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
