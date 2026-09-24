# headcount

Sort a big class photo album by **who's in each picture**, locally and
privately. `faces.py` finds every face, groups the faces by person, and after
you name each group once, it answers questions like "photos with both Ada and
Ben, taken outdoors." How it works and why is in `DESIGN.md`.

All face data is biometric and stays on your machine. Nothing is committed or
uploaded.

New here? Pick one:

- **Guided setup:** open this folder in [Claude Code](https://claude.com/claude-code)
  and say "set this up for me". It installs everything, asks where your photos
  are, runs each step, and tells you when it needs you (naming the children,
  choosing whose photos to export).
- **By hand:** follow **Quick start** below.

The sections after Quick start are reference.

## Quick start

### 1. Install (once)

You need macOS or Linux, Python 3 (tested with 3.11 and 3.14), and a C/C++
compiler (pip builds `insightface` from source). No GPU is needed.

```bash
git clone git@github.com:javierjavier/headcount.git
cd headcount
xcode-select --install          # macOS only: the compiler; skip if already installed
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional: `brew install ffmpeg` gives videos thumbnails and capture dates in the
gallery.

In every new terminal, run `cd headcount && source .venv/bin/activate` before
the commands below.

### 2. Add photos

```bash
mkdir -p album/2026-fall        # any subfolder name
# copy the photos (and videos) into album/2026-fall/
```

Or link a folder you already have instead: `ln -s /path/to/photos album`.
Link it as `album` itself, not as a subfolder inside `album/`: the scan skips
links inside `album/`, so none of those photos get processed.

- Any common photo format works (HEIC, JPEG, PNG, WebP, TIFF, …).
- Put each batch in its own subfolder of `album/`. Phones reuse file names
  like `IMG_4492.HEIC`, and separate subfolders keep a new photo from being
  mistaken for an old one.
- Unzip downloads first. A `.zip` left in `album/` stops the tools with an error.
- After step 3, don't move or rename folders inside `album/`. Photos are
  recorded by their path, and moved ones drop out of the gallery (it shows a
  warning when this happens).

### 3. Run the pipeline

```bash
python faces.py embed           # finds every face; the slow step
python faces.py cluster         # groups faces by person
python faces.py review          # makes one face montage per group
```

`embed` took 0.5–1 second per photo on an Apple M1 Max: 40 minutes for 4,700
photos in one album, 16 minutes for 1,060 in another. Speed varies by machine;
the progress bar shows an estimate once it starts. The first `embed` also
downloads a ~300 MB face model into `~/.insightface/models/` (and may print
"Matplotlib is building the font cache"). If `embed` stops, run it again and it
continues where it left off. The other steps take seconds.

### 4. Name the children and open the gallery

```bash
python faces.py serve           # opens http://127.0.0.1:8765
```

The first time, the browser shows one montage of faces at a time: type the
child's name, reuse the same name if a child shows up again, and press **Skip**
for mixed or blurry montages (details in [Naming the children](#naming-the-children)).
Press **Done** and the gallery opens. After that, `serve` opens the gallery
directly. Press Ctrl+C in the terminal to stop it.

### 5. Export a child's photos

In the gallery:

1. Tick the child's name.
2. Tick **confident matches only** (leaves out the less certain matches; see
   [Exporting](#exporting)).
3. Press **Export zip**. The zip downloads through the browser.

### Later

- **More photos:** put them in a new subfolder of `album/`, then run steps 3 and
  4 again. Only the new photos go through the slow step, and your names carry
  over; name any new children with **Edit names** in the gallery.
- **Done with the album:** see [When you're done](#when-youre-done) to delete
  the face data.

## Naming the children

`cluster` sorts every face into groups it thinks are the same person, and
`review` makes one montage per group: up to 25 face crops. The naming page
shows them one at a time, biggest group first:

- **One child, or mostly one child** → type their name and press Enter.
- **Same child as an earlier montage** → the same name again; pick it from the
  suggestions. One child is often split across two or three groups.
- **Mixed children, adults you don't need, blurry or turned-away faces** → Skip.

One group, often the first and biggest, collects blurry, turned-away and partly
hidden faces. They look more like each other than like any one child. That's
normal: skip it.

Names are matched exactly, including capitalization, so use one short spelling
per child (`ada`, not `Ada` in one place and `ada` in another). The page warns
you when a name differs from an earlier one only in capitals. Single words are
easiest: a name with a space has to be quoted on the command line
(`--with "ada b"`).

You only need to name the children you care about. Names save as you go, so you
can close the page and come back. To change a name later, or name a group you
skipped, use **Edit names** in the gallery sidebar. The names are stored in
`work/labels.csv`.

## The gallery

`python faces.py serve` opens the gallery at http://127.0.0.1:8765. It listens
on this machine only. In the sidebar you can filter by name, date, time of day,
number of faces and, after `scene` has run, indoor/outdoor. Change names with
**Edit names**.

The first run decodes every photo again to build thumbnails in
`work/serve_cache/`. Most of that happens while you name the children, and later
runs reuse it. If previews look blurry on a Retina screen, restart with a larger
`--thumb` (default 768); the cache rebuilds on its own.

**Videos** in `album/` (`.mp4`, `.mov`, `.m4v`, `.webm`, `.avi`, `.mkv`) show in
the gallery and play in the lightbox. The **Media** checkboxes switch photos,
live photos and videos on and off; clips of `--live-max` seconds or less (default
3.5) count as live photos. Use `--no-videos` to hide them.

- Without ffmpeg, videos get a placeholder tile and the file's modified date.
- iPhone `.mov` files (HEVC) usually play in Safari but not Chrome. The lightbox
  always has a download link.
- Videos have no names until you run `faces.py video` (see *Advanced*).

## Exporting

In the gallery, tick one or more names and press **Export zip**. Choose the
original files or 2048px JPEGs (EXIF removed). Videos are always exported as the
original file.

**Confident matches only.** Some faces don't fit any group well, and a later
step matches them to the nearest named child. Those matches are right about 97%
of the time, so a large export without the box usually includes a few photos of
other children. The box leaves out photos where the child was found only that
way. Every photo where the child's face was grouped with their other photos
stays in. Video names are all matched the looser way, so videos never count as
confident.

For folders inside the export, a separate folder of the less certain matches,
or exports you want to repeat the same way, use `query` (see *Advanced*).

## Advanced: command-line options

None of these are needed for the Quick start.

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

Optional location dimension. The gallery shows its **Scene** filter only after
`scene` has run; the time-of-day filter works without it. `scene` tags a photo
outdoor when its EXIF capture hour falls in the class's scheduled outdoor time.
It reads only the EXIF header, so it's instant:

```bash
python faces.py scene --outdoor-hours 10-11   # -> work/scene.csv
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

For a folder that should always be one scene regardless of the hour, such as a
field trip, list it in `work/scene_overrides.csv`. Every `scene` run applies it,
so it survives full re-runs:

```
subdir,scene
2026-10-03 zoo trip,outdoor
```

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

Deleting `album/` does not delete the face data: face embeddings, montages, names
and thumbnails of every child are in `work/`. To delete all of it but keep the
install for next time:

```bash
rm -rf work album reference
```

If `album` is a link to your photo folder, `rm -rf album` removes only the link,
not your photos.

Your exports in `query/` and `by_child/` are photos of the children too. Delete
them once you've copied out what you need.
