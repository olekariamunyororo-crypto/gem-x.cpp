#!/usr/bin/env python3
"""Fetch the pinned official GEM-X visual exemplar with byte verification."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("generated/reference/e2e"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads((root / "reference/sources.json").read_text())
    revision = inventory["source"]["revision"]
    base = f"https://raw.githubusercontent.com/NVlabs/GEM-X/{revision}/assets"
    args.output.mkdir(parents=True, exist_ok=True)
    for name, expected in inventory["official_demo"].items():
        if name == "note":
            continue
        destination = args.output / name
        if not destination.exists() or digest(destination) != expected["sha256"]:
            temporary = destination.with_suffix(destination.suffix + ".part")
            urllib.request.urlretrieve(f"{base}/{name}", temporary)
            temporary.replace(destination)
        if destination.stat().st_size != expected["bytes"] or digest(destination) != expected["sha256"]:
            raise ValueError(f"official exemplar verification failed: {name}")
        print(destination)


if __name__ == "__main__":
    main()
