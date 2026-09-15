#!/usr/bin/env python3
"""Download and extract the exact YOLOX-X HumanArt ONNX used by GEM-X."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("generated/reference/yolox-humanart.onnx"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "reference/sources.json").read_text())["detector"]
    output = args.output.resolve()
    if output.exists():
        if output.stat().st_size != source["onnx_bytes"] or digest(output) != source["onnx_sha256"]:
            raise ValueError(f"existing asset identity mismatch: {output}")
        print(output)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_path = extracted_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".yolox-archive-", delete=False) as archive:
            archive_path = Path(archive.name)
            with urllib.request.urlopen(source["url"], timeout=60) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != source["zip_bytes"]:
                    raise ValueError("unexpected YOLOX archive length")
                total = 0
                while block := response.read(8 * 1024 * 1024):
                    total += len(block)
                    if total > source["zip_bytes"]:
                        raise ValueError("YOLOX archive exceeds pinned size")
                    archive.write(block)
            archive.flush()
            os.fsync(archive.fileno())
        if archive_path.stat().st_size != source["zip_bytes"] or digest(archive_path) != source["zip_sha256"]:
            raise ValueError("YOLOX archive identity mismatch")
        with zipfile.ZipFile(archive_path) as bundle:
            info = bundle.getinfo(source["onnx_member"])
            if info.is_dir() or info.flag_bits & 1 or info.file_size != source["onnx_bytes"]:
                raise ValueError("unexpected YOLOX archive member")
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".yolox-onnx-", delete=False) as extracted:
                extracted_path = Path(extracted.name)
                total = 0
                with bundle.open(info) as model:
                    while block := model.read(8 * 1024 * 1024):
                        total += len(block)
                        if total > source["onnx_bytes"]:
                            raise ValueError("YOLOX ONNX exceeds pinned size")
                        extracted.write(block)
                extracted.flush()
                os.fsync(extracted.fileno())
        if extracted_path.stat().st_size != source["onnx_bytes"] or digest(extracted_path) != source["onnx_sha256"]:
            raise ValueError("YOLOX ONNX identity mismatch")
        os.link(extracted_path, output)
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
        if extracted_path is not None:
            extracted_path.unlink(missing_ok=True)
    print(output)


if __name__ == "__main__":
    main()
