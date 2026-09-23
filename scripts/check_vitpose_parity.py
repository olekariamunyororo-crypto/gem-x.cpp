#!/usr/bin/env python3
"""
Stronger ViTPose heatmap / keypoint parity checker for the automatic tile work.

Compares every heatmap produced by two independently loaded backends on the
same set of normalised real-video crops (including flips). Designed to be
called from CI or from experiment_vitpose_tiles_extended.py.

Exit codes:
  0 – all compared tensors byte-identical
  1 – usage / missing files
  2 – numerical mismatch
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(dir: Path, pattern: str) -> Dict[str, Path]:
    return {p.name: p for p in sorted(dir.rglob(pattern))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--pattern", default="*heatmap*")
    args = ap.parse_args()

    if not args.baseline.is_dir() or not args.candidate.is_dir():
        print("Both --baseline and --candidate must be directories", file=sys.stderr)
        return 1

    base = collect(args.baseline, args.pattern)
    cand = collect(args.candidate, args.pattern)

    if not base:
        print("No files matched the pattern in baseline", file=sys.stderr)
        return 1

    missing = set(base) - set(cand)
    extra = set(cand) - set(base)
    if missing or extra:
        print(f"File set mismatch: missing={missing} extra={extra}", file=sys.stderr)
        return 2

    mismatches: List[Tuple[str, str, str]] = []
    for name, bp in base.items():
        bhash = sha256(bp)
        chash = sha256(cand[name])
        if bhash != chash:
            mismatches.append((name, bhash, chash))

    if mismatches:
        print(f"{len(mismatches)} mismatched files:")
        for name, bh, ch in mismatches:
            print(f"  {name}\n    baseline  {bh}\n    candidate {ch}")
        return 2

    print(f"All {len(base)} files byte-identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
