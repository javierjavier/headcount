# headcount

Sort a big class photo album by **who's in each picture** — locally and
privately. `faces.py` detects every face, clusters them by identity (HDBSCAN),
you label each cluster once, then it answers combination queries like "photos
with both Ada and Ben, taken outdoors." `enroll.py` is an optional cold-start
helper that calibrates the clustering on a brand-new album (see *Calibration*
below). The *why* — algorithm choices, tradeoffs, hard cases — is in `DESIGN.md`.

All face data is biometric and stays local — nothing is committed or uploaded.

## Setup

You need a C/C++ compiler: `insightface` ships only as source, so pip builds it
during install. On macOS, install the Xcode Command Line Tools once:

```bash
xcode-select --install
```

Then:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On first run, insightface downloads the `buffalo_l` model (~300 MB) into
`~/.insightface/models/`. Everything runs on **CPU** — no GPU needed. The first
`embed` also prints "Matplotlib is building the font cache" and a download
progress bar before it starts on your photos; on a slow connection that step
can take a few minutes.

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
work/        # created on first run — everything the tool generates from the
             #   album: face data, montages (work/clusters/), work/labels.csv,
             #   the gallery's thumbnail cache. Delete it with album/ to remove
             #   every trace of the photos.
by_child/    # created by `assign --folders` (opt-in) — one folder per named child
query/       # created by `query` — the slices you pull out
```

Checkouts from before `work/` existed kept these files at the top level; the
first run moves them into `work/` and prints what it moved.

## Workflow

Put all the photos in `album/` (any Pillow-readable format, see Setup), or
symlink a folder there: `ln -s /path/to/photos album`. Then run three commands
and open the gallery:

```bash
# 1. embed — find and embed every face. Slow (about 40 min for 4.7k photos) and
#    done once; if it stops, running it again picks up where it left off.
python faces.py embed

# 2. cluster — group faces by who they look like. Fast; safe to re-run.
python faces.py cluster

# 3. review — make one montage per group and a blank work/labels.csv.
python faces.py review

# 4. serve — opens the browser. The first time, it asks you to name each
#    montage (see "Labeling" below). Press Done and it opens the gallery.
python faces.py serve                          # -> http://127.0.0.1:8765
```

From then on, `python faces.py serve` opens the gallery directly. Everything
else is in the browser: filter by name, time of day, date and number of faces,
export a zip, and change names with **Edit names** in the sidebar. Nothing
leaves the machine; the server listens on 127.0.0.1 only.

The options that still need the command line are under *Advanced* below.

### Labeling (step 4)

`cluster` sorts every face into groups it thinks are the same person. `review`
then makes one montage per group: a 5×5 grid of up to 25 face crops from that
group. The first time you run `python faces.py serve`, the browser shows these
montages one at a time, biggest group first, and asks who each one is:

- **One child, or mostly one child** → type their name and press Enter.
- **Same child as an earlier montage** → the same name again; pick it from the
  suggestions. One child is often split across two or three groups.
- **Mixed children, adults you don't need, blurry or turned-away faces** → Skip.

One group, often the first and biggest, collects blurry, turned-away and partly
hidden faces. They look more like each other than like any one child. That's
normal: skip it.

Names are matched exactly, including capitalization, so use one short spelling
per child (`ada`, not `Ada` in one place and `ada` in another). The page warns
you when a name differs from an earlier one only in capitals. A name with a
space has to be quoted on the command line (`--with "ada b"`), so single words
are easier.

You don't need to name every montage, only the children you care about. Names
save as you go, so you can close the page and come back. Press **Done** to build
the gallery. To change a name later, use **Edit names** in the gallery sidebar.

The names are stored in `work/labels.csv`, one `montage,name` row per montage:

```
montage,name
c00__cluster22__n209.jpg,
c01__cluster20__n188.jpg,ada
```

The montage filename says which group it is: `c00` is the rank (biggest first),
`cluster22` the cluster id, `n209` the number of faces. You can edit the file by
hand instead of using the page (on a Mac, `open -e work/labels.csv` edits it in
TextEdit; Numbers would convert it). Don't change the filenames: the cluster id
in them is how the tool knows which faces a name belongs to. `serve` picks up
the edits on its next start.

### Gallery (`serve`)

The first run builds a thumbnail cache in `work/serve_cache/`, which is the slow
part (it decodes every photo again). Most of it builds while you label, and the
browser shows progress until the gallery is ready. Later runs reuse the cache.
Zip export gives the originals or 2048px JPEGs (EXIF removed).

Previews come from `work/serve_cache/` thumbnails built at `--thumb` long-edge
(default 768). The grid fills cells by the photo's *short* edge and HiDPI/Retina
screens want ~2× the CSS pixels, so a too-small thumb upscales and looks blurry —
raise `--thumb` for crisper previews at the cost of build time and disk; lower it
to save both. Changing the size rebuilds the cache automatically (it records the
build size and clears stale thumbs); no manual `rm -rf work/serve_cache` needed.

#### Videos in the gallery

`serve` also surfaces any videos in `album/` (`.mp4`, `.mov`, `.m4v`, `.webm`,
`.avi`, `.mkv`), discovered straight from disk — they carry no face data (the
face pipeline still ignores them entirely; `embed`/`cluster`/`assign`/`query`
only ever touch images). In the grid each video gets a poster-frame thumbnail
with a ▶ play badge plus a length badge (e.g. `0:02`) and plays inline in the
lightbox. The **Media** checkboxes toggle photos / live photos / videos
independently (all on by default) — "live photos" are clips at or under
`--live-max` seconds (default 3.5), which on a phone album is almost all of them.
They're included in date sorting/grouping (by
container `creation_time`, converted from UTC to local, falling back to file
mtime) and the hour filter, and in zip export (always as the original file, never
re-encoded). Use `--no-videos` to leave them out.

If you've run `faces.py video` (see *Advanced*), `serve` also overlays each clip's
detected names from `video_people.csv` — so videos become name-filterable and
show names in the grid tooltip and lightbox caption, just like photos. Without
that pass, videos simply show with no names. If names have changed since `video`
ran, the gallery shows a note above the grid until you run it again.

Two notes:

- **ffmpeg** (`ffmpeg`/`ffprobe` on `PATH`) is used to extract poster frames and
  read capture times. It's an optional *system* tool, not a pip dependency — if
  it's missing, videos still appear with a generic film-strip placeholder tile
  and mtime-based dates. Install via e.g. `brew install ffmpeg`.
- **Codec/playback** depends on the browser. `.mp4`/H.264 plays everywhere; HEVC
  (common in iPhone `.mov`) plays in Safari but often not Chrome. When a browser
  can't decode a clip, the lightbox shows native controls plus a **download**
  link so the original is always reachable. Streaming is HTTP Range-served, so
  seeking only fetches the needed bytes.

### Sending photos to a parent

1. In the gallery, tick the child's name.
2. Tick **confident matches only**.
3. Press **Export zip**.

The box leaves out photos where the child was matched only by the step that
pulls in faces that didn't fit any group (noise recovery, see `assign` below).
Those matches are right about 97% of the time, so a large export without the
box usually includes a few photos of other children. Only those photos are left
out; every photo where the child's face was grouped with their other photos
stays in. Video names are all matched this way, so videos never count as
confident.

To look through the left-out photos before deciding, use
`query --recovered split` (see *Advanced*).

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
files), `cluster`, `review` and `serve`. Resume tracks done images in a `faces.done`
manifest (one path per line) alongside `faces.csv`/`.npy`/`.emb`. `faces.csv`
only lists images that yielded a face, so the manifest is what lets a re-run also
skip images where *no* face was detected — otherwise those would be re-decoded
every time. It's generated and gitignored; delete it (or use `embed --rescan`)
only if you want a clean rebuild.

Re-running `cluster` renumbers HDBSCAN's cluster ids (the same integer can mean a
different child after the face set changes), which would break a label
carry-forward done by id. So `cluster` backs up the prior `clusters.csv` to
`clusters.csv.bak`, and `review` carries your existing names forward by **face_id
majority vote** against that backup — robust to renumbering. It prints how many
names it carried and flags any cluster labeled with <90% vote agreement so you
can eyeball just those montages. (First run, with no backup, it falls back to the
old by-id carry.) So after adding photos the labeling step is usually a quick
confirm, not a redo: open **Edit names** in the gallery and name only the genuinely new children.

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
archive — rather than silently skipping every photo packed inside it. The face
pipeline ignores videos and other non-image files; videos do, however, show up in
the `serve` gallery (see *Videos in the gallery* above).

## Advanced: command-line options

None of these are needed for the workflow above.

### `assign` — who's in each photo

`serve` runs `assign` for you whenever the names change. Run it yourself only for
these options:

```bash
python faces.py assign --folders               # also copy photos into by_child/<name>/
python faces.py assign --no-recover            # only faces in a named group count
```

Noise recovery is on by default: faces that didn't fit any group are matched to
the nearest named child if they're close enough. It raises how many of a child's
photos are found (about 84% to 97% for a well-photographed child), and about 97%
of those extra matches are right. `--folders [dir]` copies photos into one
folder per child; a photo with 3 named children lands in all 3 folders.

### `query` — repeatable exports

Copies a set of photos into `query/<expr>/` as the original files. Each run
wipes and rewrites that folder. It covers what the gallery can't: "none of" and
"exactly these" filters, subfolders inside the export, a separate folder for the
lower-confidence matches, and other JPEG settings.

```bash
python faces.py query --with ada,ben          # both present
python faces.py query --any ada,ben           # at least one
python faces.py query --with ada --without ben
python faces.py query --only ada,ben          # exactly those two
python faces.py query --with ada --dry-run    # list matches, write nothing
```

Output options:

- `--jpeg` re-encodes to JPEG. macOS Finder shows HEIC thumbnails unreliably and
  JPEG ones always. With `--jpeg`: `--max-size N` downscales, `--jpeg-quality`
  sets quality, `--strip-exif` drops metadata (kept by default).
- `--zip` also writes `query/<expr>.zip`.
- `--split-scene` sorts into `indoor/` and `outdoor/` (needs `scene`).
  `--where indoor|outdoor` keeps only one.
- `--split-size` sorts into `candid/` and `large-group/` by the number of faces
  detected (`--large-min`, default 5).
- `--recovered drop` leaves out matches from noise recovery, like the gallery's
  **confident matches only** box; the folder name gets `__clustered`.
  `--recovered split` puts them in a `recovered/` folder to look through first.

The same export for a parent, split into folders:

```bash
python faces.py query --with ada --split-size --recovered split --zip
# -> query/with_ada/{candid,large-group,recovered}/ + a .zip alongside
```

See DESIGN.md "Output binning" for how the bins are chosen.

### `video` — names inside videos

Finds the named children in album videos and writes `work/video_people.csv`.
The gallery then shows those names on clips and lets you filter videos by name.
It samples frames (default 1 per second) and matches each face against the
groups you've named, so it only works after labeling. Needs ffmpeg. It takes a
while (about 18 minutes for 704 clips), so it isn't part of the main workflow.
If it stops, running it again continues.

```bash
python faces.py video
python faces.py video --fps 2 --limit 20       # more frames; first 20 clips
```

If you change names afterwards, the gallery says the video names are out of
date. Running `video` again re-scans every clip with the new names.

### Indoor / outdoor (`scene`)

Optional location dimension. If the album's GPS is stripped but the daily
schedule is rigid (e.g. an outdoor block at a fixed hour), classify by EXIF
time — instant, no decode:

```bash
python faces.py scene --method time --outdoor-hours 10-11   # -> scene.csv
python faces.py query --with ada --where outdoor
```

`--outdoor-hours` counts whole clock hours and includes both ends: `10-11` means
10:00–11:59. So an outdoor block from 10am to noon is `--outdoor-hours 10-11`,
not `10-12` (which would also tag 12:00–12:59).

`scene` rewrites the whole `scene.csv` with one outdoor-hours rule. If a later
import's outdoor block differs from earlier batches (different day, different
schedule), re-tagging everything would mis-tag the old batches. Use `--subdir` to
classify just that import's folder and **merge** the result, leaving other
batches' rows untouched:

```bash
python faces.py scene --subdir 20260618 --outdoor-hours 13-14   # only that batch
```

A foliage/sky colour method (`--method green`) also exists, but green classroom
decor (a leafy rug, a green wall) makes it leak ~20%; time wins for this album.
See `DESIGN.md` / commit history.

### Calibration (`enroll.py`) — optional, for a new album

Clustering is unsupervised, so on a fresh album you have no ground truth to tell
whether your `--min-cluster-size` / `--eps` produced clean clusters before you
sink time into labeling. Enrolling one child you can recognize gives `cluster` an
anchor:

```bash
# Drop 5–10 clear photos of one known child into reference/, then:
python enroll.py --reference reference/        # -> work/reference_embeddings.npy
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

## When you're done

Get photos out with **Export zip** in the gallery (see *Sending photos to a
parent*); the zip downloads through your browser. For exports split into
folders, or ones you want to repeat the same way, use `query` (see *Advanced*).

Deleting `album/` does not delete the face data: face embeddings, montages, names
and thumbnails of every child are in `work/`. To delete all of it but keep the
install for next time:

```bash
rm -rf work album reference
```

Your exports in `query/` and `by_child/` are photos of the children too. Delete
them once you've copied out what you need.

## Files

| File                             | Purpose                                            |
| -------------------------------- | -------------------------------------------------- |
| `faces.py`                       | The tool: `embed`/`cluster`/`review`/`assign`/`query`/`scene`/`video`/`serve` |
| `enroll.py`                      | Cold-start calibration: build `work/reference_embeddings.npy` from `reference/` |
| `common.py`                      | Shared HEIC/EXIF loading, model setup, small utilities |
| `requirements.txt`               | Dependencies                                       |
| `DESIGN.md`                      | Pipeline rationale, tradeoffs, hard cases          |
| `work/`                          | Everything the tool generates from the album (the rows below), plus montages (`work/clusters/`) and the thumbnail cache (`work/serve_cache/`) |
| `work/reference_embeddings.npy`  | Enrolled calibration anchor (generated)            |
| `work/faces.csv` / `faces.npy`   | Every face's metadata + embedding (generated)      |
| `work/faces.done`                | filenames already embedded, incl. zero-face images, for resume (generated) |
| `work/clusters.csv`              | face → cluster id (generated)                      |
| `work/labels.csv`                | montage → name (filled in on the `serve` labeling page) |
| `work/image_people.csv`          | filename → people present (generated)              |
| `work/video_people.csv`          | video → people present, from `video` (generated)   |
| `work/scene.csv`                 | filename → indoor/outdoor (generated)              |
