# CLAUDE.md

`headcount` — sorts a class photo album by who's in each picture, locally. See
`README.md` for the workflow and `DESIGN.md` for the *why* (algorithm choices,
tradeoffs, hard cases). **Read `DESIGN.md` before changing pipeline behavior** —
most non-obvious choices (threshold, recovery, time-vs-green scene tagging) are
deliberate and explained there.

## Privacy invariant (non-negotiable)

All face data is biometric, and the album is a child's photos. **Never commit**
`album/`, `reference/`, `matches/`, or any generated artifact (`faces.*`,
`clusters*`, `labels*`, `image_people*`, `scene*`, `query*/`, `*.jpg`
validations). All are gitignored — keep it that way; never `git add -f` them.

## Layout

- `faces.py` — the tool: `embed`/`cluster`/`review`/`assign`/`query`/`scene`.
- `enroll.py` — optional cold-start calibration anchor from `reference/`.
- `common.py` — shared image loading (any Pillow-readable format, EXIF-aware),
  model setup, small utils.
- `tests/test_logic.py` — pure-logic tests (no model load / no insightface).

## Conventions

- Style: double-quoted, 4-space. Lint: `ruff check .` (rule set `F`/`E`/`W`,
  `E501` ignored).
- Tests: `pytest`, or `python tests/test_logic.py` (runs without pytest).
- `embed` is the only slow step (~40 min for ~4.7k photos) and is the
  expensive-once stage — everything else is fast and re-runnable. Don't trigger
  a re-embed casually; it resumes via the `faces.done` manifest.
- Input format is not assumed to be HEIC — `common.py` handles any Pillow
  format. HEIC-specific notes (libheif decode cost, `--jpeg` for Finder
  thumbnails) are about this album, not hard requirements.
