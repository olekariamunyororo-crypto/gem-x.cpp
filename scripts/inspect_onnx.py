#!/usr/bin/env python3
"""Inventory the pinned NVIDIA GEM-X denoiser without executing it."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import onnx

EXPECTED_ONNX_SHA256 = "20aab83c01bbd909a258ad0fa465458eda382c63dc80d56952b2ec952d6e192c"
EXPECTED_DATA_SHA256 = "720c2400281b730574bf2a446c33ecd77bf494ac8161b081b09bb122b9fad2f8"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def shape(value) -> dict:
    tensor = value.type.tensor_type
    dims = []
    for dim in tensor.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
    return {"dtype": onnx.TensorProto.DataType.Name(tensor.elem_type), "shape": dims}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_path = args.model.resolve()
    data_path = model_path.with_name(model_path.name + ".data")
    if digest(model_path) != EXPECTED_ONNX_SHA256:
        raise ValueError("ONNX hash does not match the pinned NVIDIA artifact")
    if digest(data_path) != EXPECTED_DATA_SHA256:
        raise ValueError("ONNX external-data hash does not match the pinned NVIDIA artifact")

    onnx.checker.check_model(str(model_path), full_check=False)
    model = onnx.load(str(model_path), load_external_data=False)
    inventory = {
        "schema": "gem-x.cpp.onnx-inventory.v1",
        "source": {
            "onnx_sha256": EXPECTED_ONNX_SHA256,
            "external_data_sha256": EXPECTED_DATA_SHA256,
            "ir_version": model.ir_version,
            "opsets": [{"domain": item.domain, "version": item.version}
                       for item in model.opset_import],
        },
        "inputs": {item.name: shape(item) for item in model.graph.input},
        "outputs": {item.name: shape(item) for item in model.graph.output},
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "operators": dict(sorted(collections.Counter(
            item.op_type for item in model.graph.node).items())),
        "initializers": [
            {
                "name": item.name,
                "dtype": onnx.TensorProto.DataType.Name(item.data_type),
                "shape": list(item.dims),
                "external": {field.key: field.value for field in item.external_data},
            }
            for item in model.graph.initializer
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
