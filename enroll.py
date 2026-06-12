#!/usr/bin/env python3
"""Enroll a known person from a handful of reference photos.

Runs each reference image through face detection + embedding, stacks the
per-photo embeddings, and saves them to reference_embeddings.npy.

This is the cold-start calibration seeder for `faces.py`. On a brand-new album
there are no labels yet, so there's no ground truth to check the clustering
against. Enrolling one child you can recognize gives `faces.py cluster` an
anchor: its calibration readout reports whether those known faces land in ONE
clean cluster, telling you if --min-cluster-size / --eps are dialed in before
you sink time into labeling.

We save the *stack* of per-reference embeddings (N x 512), not a single mean,
so the check scores against the best-matching reference — more robust to varied
angles/lighting than averaging everything into one blurry vector.

Usage:
    python enroll.py --reference reference/ --out reference_embeddings.npy
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from common import build_face_app, empty_hint, largest_face, list_images, load_image_bgr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default="reference", help="folder of reference photos (default: reference/)")
    ap.add_argument("--out", default="reference_embeddings.npy", help="output .npy path")
    # Reference photos are close-up crops where the face fills much of the frame.
    # A large det_size upscales such faces past the detector's anchors and misses
    # them, so enrollment defaults small (640). The album scan uses a larger size
    # for distant faces. See README.
    ap.add_argument("--det-size", type=int, default=640, help="detector input size (default: 640)")
    ap.add_argument("--det-thresh", type=float, default=0.4, help="detector confidence threshold (default: 0.4)")
    args = ap.parse_args()

    try:
        images = list_images(args.reference)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(e, file=sys.stderr)
        return 1
    if not images:
        print(f"No images found in {args.reference!r}.{empty_hint(args.reference)}", file=sys.stderr)
        return 1

    print(f"Loading detector (buffalo_l, det_size={args.det_size}) ...")
    app = build_face_app(det_size=args.det_size, det_thresh=args.det_thresh)

    embeddings: list[np.ndarray] = []
    used: list[str] = []
    for path in images:
        try:
            img = load_image_bgr(path)
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't abort enrollment
            print(f"  ! {path.name}: failed to load ({e})")
            continue

        faces = app.get(img)
        if not faces:
            print(f"  ! {path.name}: no face detected — skipping")
            continue
        if len(faces) > 1:
            print(f"  ~ {path.name}: {len(faces)} faces, using the largest")

        face = largest_face(faces)
        # normed_embedding is already L2-normalized, so cosine == dot product.
        embeddings.append(face.normed_embedding.astype(np.float32))
        used.append(path.name)
        print(f"  + {path.name}")

    if not embeddings:
        print("No usable faces found in any reference photo.", file=sys.stderr)
        return 1

    stack = np.vstack(embeddings)  # (N, 512), each row unit-norm
    np.save(args.out, stack)
    print(f"\nSaved {stack.shape[0]} reference embedding(s) -> {args.out}")

    # Sanity check: references should be mutually similar. A low value flags a
    # photo of the wrong child or a bad detection that will pollute matching.
    if stack.shape[0] > 1:
        sims = stack @ stack.T
        off_diag = sims[~np.eye(stack.shape[0], dtype=bool)]
        print(
            f"Reference-to-reference cosine similarity: "
            f"min={off_diag.min():.3f} mean={off_diag.mean():.3f} max={off_diag.max():.3f}"
        )
        # Surface the least-consistent reference so the user can eyeball it.
        per_ref = (sims.sum(axis=1) - 1) / (stack.shape[0] - 1)
        worst = int(np.argmin(per_ref))
        print(f"Least-consistent reference: {used[worst]} (avg sim {per_ref[worst]:.3f})")
        if off_diag.min() < 0.2:
            print("  ! Some references barely match each other — check for a wrong/poor photo.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
