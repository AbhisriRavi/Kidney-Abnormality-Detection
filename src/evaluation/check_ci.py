#!/usr/bin/env python3
"""
Find which results files contain Set A vs Set B for the four ROI per-class
confidence intervals, so the current run can be identified.

Usage:
    python check_ci.py                       # searches $SCRATCH/kidney-results
    python check_ci.py /some/other/dir ...   # or give paths explicitly

Searches .csv, .json and .txt under the given roots for the interval bounds
and reports every match with the file's modification time.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Candidate interval sets. Values are (low, high) as they appear in the draft.
SETS = {
    "A (currently in table AND text)": {
        "Normal": (0.080, 0.280),
        "Cyst": (-0.242, -0.042),
        "Tumor": (-0.075, -0.021),
        "Stone": (0.013, 0.117),
    },
    "B (was in the prose only)": {
        "Normal": (0.079, 0.285),
        "Cyst": (-0.235, -0.044),
        "Tumor": (-0.077, -0.021),
        "Stone": (0.013, 0.119),
    },
}

TOL = 5e-4  # values are quoted to 3 dp, so match within half of the last digit
EXTS = {".csv", ".json", ".txt", ".tsv"}


def roots():
    if len(sys.argv) > 1:
        return [Path(p) for p in sys.argv[1:]]
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        sys.exit("SCRATCH is not set; pass one or more directories instead.")
    return [Path(scratch) / "kidney-results"]


def numbers_in(text):
    """Yield every float-looking token in the text."""
    out = []
    tok = ""
    for ch in text:
        if ch.isdigit() or ch in ".-+eE":
            tok += ch
        else:
            if tok:
                try:
                    out.append(float(tok))
                except ValueError:
                    pass
                tok = ""
    if tok:
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def scan(path):
    """Return {set_name: {class: [matched bounds]}} for one file."""
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return {}
    vals = numbers_in(text)
    if not vals:
        return {}
    hits = {}
    for set_name, classes in SETS.items():
        for cls, (lo, hi) in classes.items():
            got_lo = any(abs(v - lo) < TOL for v in vals)
            got_hi = any(abs(v - hi) < TOL for v in vals)
            if got_lo and got_hi:
                hits.setdefault(set_name, []).append(cls)
    return hits


def main():
    found = []
    for root in roots():
        if not root.exists():
            print(f"skip (missing): {root}")
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in EXTS:
                continue
            hits = scan(path)
            if hits:
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                found.append((mtime, path, hits))

    if not found:
        print("No file contained either interval set.")
        print("Widen the search: pass the directory holding your robust_stats "
              "output, or add its file extension to EXTS.")
        return

    found.sort(reverse=True)
    print(f"{len(found)} file(s) matched, newest first.\n")
    for mtime, path, hits in found:
        print(f"{mtime:%Y-%m-%d %H:%M}  {path}")
        for set_name, classes in hits.items():
            print(f"    {set_name}: matches {len(classes)}/4 -> "
                  f"{', '.join(sorted(classes))}")
        print()

    # Summary: which set has the newest full 4/4 match
    print("-" * 60)
    for set_name in SETS:
        full = [(m, p) for m, p, h in found
                if set_name in h and len(h[set_name]) == 4]
        if full:
            m, p = max(full)
            print(f"{set_name}: newest complete match {m:%Y-%m-%d %H:%M}  {p}")
        else:
            print(f"{set_name}: no file matched all four classes")
    print("\nThe set with the newer complete match is the current run.")
    print("If both appear, the older one is the superseded bootstrap seed.")


if __name__ == "__main__":
    main()