#!/usr/bin/env python3
"""Convert the exact YOLOX-X HumanArt model used by NVIDIA GEM-X to GGUF.

The ONNX protobuf is treated as data.  The converter accepts only the pinned
OpenMMLab artifact and validates the complete operator vocabulary and the raw
head boundary before serialising weights and a small, bounded graph recipe.
"""

import argparse
import collections
import hashlib
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import numpy_helper

ONNX_SHA256 = "8e9ea96a176bd48501eaaa77216e49ee30794d2f8ba80c7b9862beca4ea972da"
ZIP_SHA256 = "1c6fccc5cd450eb563b1fe4eba13c9bdf9ea8263b6ae1cb3ada765aa4ebea2c7"
SOURCE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "yolox_x_8xb8-300e_humanart-a39d44ed.zip"
)
RAW_OUTPUTS = ("1548", "1549", "1550", "1567", "1568", "1569", "1586", "1587", "1588")
EXPECTED_OPERATORS = {
    "Add": 32, "Concat": 20, "Conv": 155, "Div": 2, "Exp": 1,
    "Flatten": 1, "Gather": 14, "Less": 1, "MaxPool": 3, "Mul": 150,
    "NonMaxSuppression": 1, "ReduceMax": 1, "Reshape": 12, "Resize": 2,
    "Shape": 1, "Sigmoid": 148, "Slice": 2, "Squeeze": 2, "Sub": 2,
    "TopK": 2, "Transpose": 13, "Unsqueeze": 7, "Where": 1,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def shape(value):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in value.type.tensor_type.shape.dim]


def attributes(node):
    return {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}


def graph_recipe(nodes) -> tuple[str, list[str]]:
    # The export begins with YOLOX Focus expressed as reshape/transpose/reshape.
    # Native preprocessing creates that 12x320x320 tensor directly.
    if [node.op_type for node in nodes[:3]] != ["Reshape", "Transpose", "Reshape"] or \
       list(nodes[2].output) != ["941"]:
        raise ValueError("unexpected YOLOX focus prefix")
    refs = {"941": -1}
    lines = []
    conv_tensors = []
    for node in nodes[3:496]:
        inputs = []
        for name in node.input:
            if name in refs:
                inputs.append(refs[name])
        attr = attributes(node)
        if node.op_type == "Conv":
            if len(inputs) != 1 or len(node.input) != 3 or attr.get("group") != 1 or \
               attr.get("dilations") != [1, 1]:
                raise ValueError("unsupported YOLOX convolution")
            kernel = attr.get("kernel_shape")
            stride = attr.get("strides")
            pads = attr.get("pads")
            if kernel not in ([1, 1], [3, 3]) or stride not in ([1, 1], [2, 2]) or \
               pads != [kernel[0] // 2] * 4:
                raise ValueError("unsupported YOLOX convolution geometry")
            conv_tensors.extend(node.input[1:])
            line = f"C,{inputs[0]},{stride[0]}"
        elif node.op_type in ("Sigmoid", "Mul", "Add"):
            expected = 1 if node.op_type == "Sigmoid" else 2
            if len(inputs) != expected or len(node.input) != expected:
                raise ValueError(f"unsupported YOLOX {node.op_type}")
            line = {"Sigmoid": "S", "Mul": "M", "Add": "A"}[node.op_type]
            line += "," + ",".join(map(str, inputs))
        elif node.op_type == "Concat":
            if len(inputs) not in (2, 4) or len(inputs) != len(node.input) or attr != {"axis": 1}:
                raise ValueError("unsupported YOLOX concat")
            line = "T," + ",".join(map(str, inputs))
        elif node.op_type == "MaxPool":
            kernel = attr.get("kernel_shape")
            if len(inputs) != 1 or len(node.input) != 1 or kernel not in ([5, 5], [9, 9], [13, 13]) or \
               attr.get("strides") != [1, 1] or attr.get("pads") != [kernel[0] // 2] * 4:
                raise ValueError("unsupported YOLOX max pool")
            line = f"P,{inputs[0]},{kernel[0]}"
        elif node.op_type == "Resize":
            if len(inputs) != 1 or inputs[0] != refs[node.input[0]] or \
               attr.get("mode") != b"nearest" or attr.get("coordinate_transformation_mode") != b"asymmetric" or \
               attr.get("nearest_mode") != b"floor":
                raise ValueError("unsupported YOLOX resize")
            line = f"U,{inputs[0]}"
        else:
            raise ValueError(f"unexpected learned YOLOX operator {node.op_type}")
        refs[node.output[0]] = len(lines)
        lines.append(line)
    if len(lines) != 493 or len(conv_tensors) != 310 or any(name not in refs for name in RAW_OUTPUTS):
        raise ValueError("unexpected YOLOX learned graph boundary")
    recipe = ";".join(lines)
    outputs = [str(refs[name]) for name in RAW_OUTPUTS]
    return recipe, conv_tensors, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gguf-py", type=Path)
    args = parser.parse_args()
    if args.gguf_py:
        sys.path.insert(0, str(args.gguf_py.resolve()))
    import gguf

    source = args.model.resolve()
    if digest(source) != ONNX_SHA256:
        raise ValueError("refusing a YOLOX model that is not the pinned HumanArt artifact")
    graph = onnx.load(str(source), load_external_data=False)
    if [(value.name, shape(value)) for value in graph.graph.input] != [("input", [1, 3, 640, 640])] or \
       [(value.name, shape(value)) for value in graph.graph.output] != [
           ("dets", [1, "Gatherdets_dim_1", 5]), ("labels", [1, "Gatherlabels_dim_1"])
       ]:
        raise ValueError("unexpected YOLOX graph signature")
    operators = collections.Counter(node.op_type for node in graph.graph.node)
    if len(graph.graph.node) != 573 or len(graph.graph.initializer) != 335 or operators != EXPECTED_OPERATORS:
        raise ValueError("unexpected YOLOX graph structure")
    recipe, tensor_names, output_refs = graph_recipe(graph.graph.node)
    initializers = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    if len(initializers) != 335:
        raise ValueError("duplicate YOLOX initializer")

    converted = {}
    for index in range(0, len(tensor_names), 2):
        weight = np.asarray(initializers[tensor_names[index]], dtype=np.float32)
        bias = np.asarray(initializers[tensor_names[index + 1]], dtype=np.float32)
        if weight.ndim != 4 or weight.shape[2:] not in ((1, 1), (3, 3)) or \
           bias.shape != (weight.shape[0],) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError(f"invalid YOLOX convolution {index // 2}")
        converted[f"conv.{index // 2:03d}.weight"] = np.ascontiguousarray(weight)
        converted[f"conv.{index // 2:03d}.bias"] = np.ascontiguousarray(bias)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.output), arch="gemx_yolox")
    writer.add_string("general.name", "OpenMMLab YOLOX-X HumanArt for NVIDIA GEM-X")
    writer.add_string("gemx.yolox.source_url", SOURCE_URL)
    writer.add_string("gemx.yolox.onnx_sha256", ONNX_SHA256)
    writer.add_string("gemx.yolox.zip_sha256", ZIP_SHA256)
    writer.add_uint32("gemx.yolox.image_width", 640)
    writer.add_uint32("gemx.yolox.image_height", 640)
    writer.add_uint32("gemx.yolox.convolution_count", 155)
    writer.add_uint32("gemx.yolox.graph_node_count", 493)
    writer.add_string("gemx.yolox.graph", recipe)
    writer.add_string("gemx.yolox.raw_outputs", ",".join(output_refs))
    writer.add_file_type(0)
    for name in sorted(converted):
        writer.add_tensor(name, converted[name])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output} ({len(converted)} tensors, {len(recipe)} graph bytes)")


if __name__ == "__main__":
    main()
