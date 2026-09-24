# CLAUDE.md

`headcount` — sorts a class photo album by who's in each picture, locally. See
`README.md` for the workflow and `DESIGN.md` for the *why* (algorithm choices,
tradeoffs, hard cases). **Read `DESIGN.md` before changing pipeline behavior** —
most non-obvious choices (threshold, recovery, hour-based scene tagging) are
deliberate and explained there.

To run the tool for someone (install, add photos, run the pipeline, open the
gallery, export), follow README.md *Quick start*.

## Guided setup for new users

When someone asks how to use the tool rather than how to change it (e.g. "how do
I export photos of my kid"), offer to set it up for them step by step. If they
accept, or ask for it ("set this up for me"), run every step yourself and stop
only where they have to act. Before each wait, say what is running and roughly
how long it takes.

1. **Install.** Check `python3 --version`, `xcode-select -p` (macOS) and
   `which ffmpeg` (optional). Create `.venv`, `pip install -r requirements.txt`,
   then verify: `python -c "import insightface, onnxruntime, pillow_heif; from
   sklearn.cluster import HDBSCAN"`.
2. **Photos (user acts).** Ask for the folder path. Count files by extension and
   look for `.zip` files; ask them to unzip any first. Link the whole folder as
   `album` (`ln -s "<path>" album`). A link inside `album/` is not scanned. If
   `album/` already exists, ask before changing it.
3. **Embed.** Run `python faces.py embed` in the background, logging to a file.
   After about a minute, read the progress bar and give the user the ETA. When
   it finishes, run `cluster` and `review`.
4. **Name the children (user acts).** Run `serve` in the background and check
   that 127.0.0.1:8765 answers. Summarize the README *Labeling* rules (name
   your child, reuse the exact spelling, Skip everything else, press Done) and
   wait for the user to say they are done.
5. **Export (user chooses).** Read the named rows of `work/labels.csv`. If
   there is more than one name, show photo counts per name
   (`query --with <name> --dry-run`) and ask which child. Offer the gallery's
   **Export zip** with **confident matches only**, or run
   `query --with <name> --recovered drop --jpeg --zip` for them
   (`--recovered split` to include the less certain matches in their own folder).

## Privacy invariant (non-negotiable)

All face data is biometric, and the album is a child's photos. **Never commit**
`album/`, `reference/`, `matches/`, `work/` (every generated artifact: `faces.*`,
`clusters*`, `labels*`, `image_people*`, `scene*`, the serve cache), or exports
(`query*/`, `by_child*/`). All are gitignored — keep it that way; never
`git add -f` them. New generated files go in `work/` (`common.WORK`), not the
top level.

## Layout

- `faces.py` — the tool: `embed`/`cluster`/`review`/`assign`/`query`/`scene`/
  `video`/`serve` (`serve` also hosts the labeling page).
- `enroll.py` — optional cold-start calibration anchor from `reference/`.
- `common.py` — shared image loading (any Pillow-readable format, EXIF-aware),
  model setup, small utils.
- `tests/test_logic.py` — pure-logic tests (no model load / no insightface).

## Conventions

- Style: double-quoted, 4-space. Lint: `ruff check .` (rule set `F`/`E`/`W` plus
  `N999`, `E501` ignored). Ruff isn't installed as a binary here — run it via
  `uvx ruff check .`. A tracked pre-commit hook (`.githooks/pre-commit`) runs
  the same check and blocks the commit on failure; a fresh clone opts in once
  with `git config core.hooksPath .githooks` (bypass a single commit with
  `--no-verify`).
- Tests: `pytest`, or `python tests/test_logic.py` (runs without pytest).
- `embed` is the only slow step (~40 min for ~4.7k photos) and is the
  expensive-once stage — everything else is fast and re-runnable. Don't trigger
  a re-embed casually; it resumes via the `faces.done` manifest.
- Input format is not assumed to be HEIC — `common.py` handles any Pillow
  format. HEIC-specific notes (libheif decode cost, `--jpeg` for Finder
  thumbnails) are about this album, not hard requirements.
