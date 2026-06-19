"""Unit tests for the pure-logic helpers (no insightface / model load needed).

These cover the set-query predicate, the resume/skip bookkeeping, the desync
recovery, and the small parsing/sampling utilities — the parts where a silent
bug would quietly corrupt output or waste a long re-embed.

Run either way:
    pytest
    python tests/test_logic.py        # no pytest required
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402
import faces  # noqa: E402


# --- helpers ---------------------------------------------------------------

def _write_faces_csv(path: Path, items: list[tuple[int, str]]) -> None:
    """items: (face_id, filename); bbox/det_score filled with dummies."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(faces.FACES_HEADER)
        for fid, fn in items:
            w.writerow([fid, fn, 0, 0, 10, 10, "0.9000"])


def _write_emb(path: Path, n: int) -> None:
    """Write n dummy 512-float32 vectors to an .emb journal."""
    arr = np.arange(n * faces.EMB_DIM, dtype=np.float32).reshape(n, faces.EMB_DIM)
    path.write_bytes(arr.tobytes())


# --- _query_match (DESIGN.md's table) --------------------------------------

def test_query_with_is_subset_not_exact():
    # The bug item #3 fixed: `--with Ada` alone is a subset test, not --only.
    names = {"Ada", "Ben"}
    assert faces._query_match(names, {"Ada"}, set(), set(), set()) is True
    assert faces._query_match(names, {"Ada", "Ben"}, set(), set(), set()) is True
    assert faces._query_match(names, {"Ada", "Cleo"}, set(), set(), set()) is False


def test_query_only_is_exact():
    assert faces._query_match({"Ada"}, set(), set(), set(), {"Ada"}) is True
    assert faces._query_match({"Ada", "Ben"}, set(), set(), set(), {"Ada"}) is False


def test_query_any_and_without():
    assert faces._query_match({"Ada"}, set(), {"Ada", "Ben"}, set(), set()) is True
    assert faces._query_match({"Cleo"}, set(), {"Ada", "Ben"}, set(), set()) is False
    # --with Ada --without Ben
    assert faces._query_match({"Ada"}, {"Ada"}, set(), {"Ben"}, set()) is True
    assert faces._query_match({"Ada", "Ben"}, {"Ada"}, set(), {"Ben"}, set()) is False


# --- _name_set / _parse_hours ----------------------------------------------

def test_name_set_strips_and_drops_empties():
    assert faces._name_set(" Ada , Ben ,") == {"Ada", "Ben"}
    assert faces._name_set("") == set()


def test_parse_hours_range_and_list():
    assert faces._parse_hours("10-11") == {10, 11}
    assert faces._parse_hours("10,11") == {10, 11}
    assert faces._parse_hours("9-9") == {9}
    assert faces._parse_hours("") == set()


# --- _even_sample ----------------------------------------------------------

def test_even_sample_keeps_endpoints_and_count():
    out = faces._even_sample(list(range(10)), 3)
    assert out[0] == 0 and out[-1] == 9
    assert len(out) <= 3
    assert faces._even_sample([1, 2], 5) == [1, 2]


# --- _unique_path ----------------------------------------------------------

def test_unique_path_avoids_clobber(tmp_path):
    p = tmp_path / "a.jpg"
    p.write_text("x")
    got = common._unique_path(p)
    assert got.name == "a_1.jpg"
    got.write_text("y")
    assert common._unique_path(p).name == "a_2.jpg"


# --- list_images first-run guards ------------------------------------------

def test_list_images_finds_and_sorts(tmp_path):
    for n in ["b.JPG", "a.heic", "note.txt", "c.png"]:
        (tmp_path / n).write_bytes(b"")
    got = [p.name for p in common.list_images(tmp_path)]
    assert got == ["a.heic", "b.JPG", "c.png"]  # sorted, .txt dropped


def test_list_images_missing_folder_raises_friendly(tmp_path):
    try:
        common.list_images(tmp_path / "nope")
    except FileNotFoundError as e:
        assert "Folder not found" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError for a missing folder")


def test_list_images_file_not_dir_raises_friendly(tmp_path):
    f = tmp_path / "album.zip"
    f.write_bytes(b"PK\x03\x04")
    try:
        common.list_images(f)
    except NotADirectoryError as e:
        assert "not a folder" in str(e)
    else:
        raise AssertionError("expected NotADirectoryError when pointed at a file")


def test_list_images_recurses_into_subfolders(tmp_path):
    # Subfolder images are found alongside top-level ones; dot-dirs are skipped.
    (tmp_path / "top.heic").write_bytes(b"")
    sub = tmp_path / "photos-3"
    sub.mkdir()
    (sub / "IMG_4492.HEIC").write_bytes(b"")
    cache = tmp_path / ".serve_cache"
    cache.mkdir()
    (cache / "x.jpg").write_bytes(b"")  # dot-dir contents must not leak in
    got = sorted(common.rel_key(p, tmp_path) for p in common.list_images(tmp_path))
    assert got == ["photos-3/IMG_4492.HEIC", "top.heic"]


def test_list_images_raises_on_archive_in_tree(tmp_path):
    # A stray zip alongside real images must hard-stop, not be silently skipped.
    (tmp_path / "a.heic").write_bytes(b"")
    (tmp_path / "Photos-3-001.zip").write_bytes(b"PK\x03\x04")
    try:
        common.list_images(tmp_path)
    except common.ArchiveFoundError as e:
        assert "Photos-3-001.zip" in str(e) and "subfolder" in str(e)
    else:
        raise AssertionError("expected ArchiveFoundError for a stray archive")


def test_rel_key_toplevel_equals_basename(tmp_path):
    # Migration hinge: a top-level file's key is its bare basename, so artifacts
    # written before subfolders existed keep matching (no re-embed). Subfolder
    # files get a POSIX-style prefix, which is what defeats basename collisions.
    assert common.rel_key(tmp_path / "IMG_1.HEIC", tmp_path) == "IMG_1.HEIC"
    assert common.rel_key(tmp_path / "sub" / "IMG_1.HEIC", tmp_path) == "sub/IMG_1.HEIC"


# --- empty_hint ------------------------------------------------------------

def test_empty_hint_flags_archive(tmp_path):
    (tmp_path / "photos.zip").write_bytes(b"PK")
    hint = common.empty_hint(tmp_path)
    assert "archive" in hint and "photos.zip" in hint


def test_empty_hint_silent_without_archive(tmp_path):
    (tmp_path / "stray.txt").write_text("x")
    assert common.empty_hint(tmp_path) == ""
    assert common.empty_hint(tmp_path / "missing") == ""


# --- resume manifest -------------------------------------------------------

def test_read_done_manifest(tmp_path):
    m = tmp_path / "faces.done"
    assert faces.read_done_manifest(m) == set()
    m.write_text("A.HEIC\nB.HEIC\n\n")
    assert faces.read_done_manifest(m) == {"A.HEIC", "B.HEIC"}


def test_resume_union_skips_zero_face_images(tmp_path):
    # faces.csv lists only face-bearing images; the manifest also covers the
    # zero-face one. The union is what makes a re-run skip everything.
    fcsv = tmp_path / "faces.csv"
    _write_faces_csv(fcsv, [(0, "A.HEIC"), (1, "B.HEIC")])
    done = tmp_path / "faces.done"
    done.write_text("A.HEIC\nB.HEIC\nC.HEIC\n")  # C had no faces
    skip = faces.read_face_filenames(fcsv) | faces.read_done_manifest(done)
    assert skip == {"A.HEIC", "B.HEIC", "C.HEIC"}


# --- load_cluster_map integrity check --------------------------------------

def test_load_cluster_map_ok(tmp_path):
    face_rows = [{"face_id": "0"}, {"face_id": "1"}, {"face_id": "2"}]
    clu = tmp_path / "clusters.csv"
    with clu.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["face_id", "cluster_id"])
        w.writerows([["0", "5"], ["1", "-1"], ["2", "5"]])
    cmap = faces.load_cluster_map(face_rows, clu)
    assert cmap == {"0": 5, "1": -1, "2": 5}


def test_load_cluster_map_rejects_mismatch(tmp_path):
    face_rows = [{"face_id": "0"}, {"face_id": "1"}, {"face_id": "2"}]
    clu = tmp_path / "clusters.csv"
    with clu.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["face_id", "cluster_id"])
        w.writerows([["0", "5"], ["1", "5"]])  # missing face 2 -> stale
    try:
        faces.load_cluster_map(face_rows, clu)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on a stale clusters.csv")


# --- _recover_desync -------------------------------------------------------

def _emb_n(p: Path) -> int:
    return p.stat().st_size // (faces.EMB_DIM * 4)


def test_recover_noop_when_in_sync(tmp_path):
    fcsv = tmp_path / "faces.csv"
    emb = tmp_path / "faces.emb"
    done = tmp_path / "faces.done"
    _write_faces_csv(fcsv, [(0, "A"), (1, "A"), (2, "B")])
    _write_emb(emb, 3)
    assert faces._recover_desync(fcsv, emb, done) is False
    assert len(faces.read_face_rows(fcsv)) == 3 and _emb_n(emb) == 3


def test_recover_drops_partial_multiface_image(tmp_path):
    # A(0,1) complete; B(2,3,4) only its first face embedded (emb=3). B must be
    # fully dropped so it re-embeds; A is preserved.
    fcsv = tmp_path / "faces.csv"
    emb = tmp_path / "faces.emb"
    done = tmp_path / "faces.done"
    _write_faces_csv(fcsv, [(0, "A"), (1, "A"), (2, "B"), (3, "B"), (4, "B")])
    _write_emb(emb, 3)
    assert faces._recover_desync(fcsv, emb, done) is True
    rows = faces.read_face_rows(fcsv)
    assert [r["filename"] for r in rows] == ["A", "A"]
    assert _emb_n(emb) == 2
    # face_ids stay a contiguous prefix
    assert [r["face_id"] for r in rows] == ["0", "1"]


def test_recover_preserves_complete_image_at_exact_boundary(tmp_path):
    # The over-drop bug: cut lands exactly between complete A and partial B.
    # A must survive; only B (one stray csv row, no embedding) is dropped.
    fcsv = tmp_path / "faces.csv"
    emb = tmp_path / "faces.emb"
    done = tmp_path / "faces.done"
    _write_faces_csv(fcsv, [(0, "A"), (1, "A"), (2, "B")])
    _write_emb(emb, 2)
    assert faces._recover_desync(fcsv, emb, done) is True
    rows = faces.read_face_rows(fcsv)
    assert [r["filename"] for r in rows] == ["A", "A"]
    assert _emb_n(emb) == 2


def test_recover_truncates_extra_embeddings(tmp_path):
    # Defensive: more embeddings than csv rows -> trim emb to match.
    fcsv = tmp_path / "faces.csv"
    emb = tmp_path / "faces.emb"
    done = tmp_path / "faces.done"
    _write_faces_csv(fcsv, [(0, "A"), (1, "A")])
    _write_emb(emb, 3)
    assert faces._recover_desync(fcsv, emb, done) is True
    assert len(faces.read_face_rows(fcsv)) == 2 and _emb_n(emb) == 2


def test_recover_unmarks_reembedded_file_in_manifest(tmp_path):
    fcsv = tmp_path / "faces.csv"
    emb = tmp_path / "faces.emb"
    done = tmp_path / "faces.done"
    _write_faces_csv(fcsv, [(0, "A"), (1, "B")])
    _write_emb(emb, 1)  # B's row has no embedding
    done.write_text("A\nB\n")
    faces._recover_desync(fcsv, emb, done)
    # B is being re-embedded, so it must no longer count as done.
    assert faces.read_done_manifest(done) == {"A"}


# --- _read_labels ----------------------------------------------------------

def test_read_labels_skips_blank_names(tmp_path):
    lp = tmp_path / "labels.csv"
    with lp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster_id", "size", "montage", "name"])
        w.writerows([["5", "100", "c00.jpg", "Ada"], ["6", "80", "c01.jpg", ""]])
    assert faces._read_labels(lp) == {5: "Ada"}


# --- serve thumbnail keys --------------------------------------------------

def test_thumb_keys_unique_stems_stay_bare():
    # No collisions -> bare stems, so an existing stem-named cache stays valid.
    keys = faces.assign_thumb_keys(["IMG_1.HEIC", "sub/IMG_2.HEIC", "a/b/IMG_3.jpg"])
    assert keys == {"IMG_1.HEIC": "IMG_1", "sub/IMG_2.HEIC": "IMG_2", "a/b/IMG_3.jpg": "IMG_3"}


def test_thumb_keys_shared_stem_disambiguated_and_unique():
    # A reused IMG_4492.HEIC across import folders: both must get distinct keys,
    # and neither may keep the bare stem (which a stale cache file could alias).
    fns = ["IMG_4492.HEIC", "20260618/IMG_4492.HEIC"]
    keys = faces.assign_thumb_keys(fns)
    assert keys["IMG_4492.HEIC"] != keys["20260618/IMG_4492.HEIC"]
    assert all(k != "IMG_4492" and k.startswith("IMG_4492-") for k in keys.values())
    assert len(set(keys.values())) == 2  # collision-free


def test_thumb_keys_same_stem_different_ext_also_split():
    # IMG_1.HEIC vs IMG_1.JPG share a stem in one folder -> still disambiguated.
    keys = faces.assign_thumb_keys(["IMG_1.HEIC", "IMG_1.JPG"])
    assert len(set(keys.values())) == 2
    assert all(k.startswith("IMG_1-") for k in keys.values())


def test_thumb_keys_stable_across_runs():
    # Keys are a pure function of the filename set -> deterministic (a cache built
    # one run is still addressable the next).
    fns = ["IMG_4492.HEIC", "20260618/IMG_4492.HEIC", "lone.png"]
    assert faces.assign_thumb_keys(fns) == faces.assign_thumb_keys(list(reversed(fns)))


# --- standalone runner (no pytest) -----------------------------------------

def _run_standalone() -> int:
    import inspect
    import tempfile

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        needs_tmp = "tmp_path" in inspect.signature(fn).parameters
        try:
            if needs_tmp:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  ok   {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
