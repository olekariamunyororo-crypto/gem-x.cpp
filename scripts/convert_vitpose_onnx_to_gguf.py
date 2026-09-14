#!/usr/bin/env python3
"""Convert NVIDIA's pinned GEM-X ViTPose ONNX export to a native GGUF.

The protobuf and external-data file are non-executable inputs.  This converter
accepts only the published hashes and the exact released graph structure.
"""

import argparse
import collections
import hashlib
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import numpy_helper

ONNX_SHA256 = "0982dbf4f1e8a48446a6fe35329711522b60210cce2a7499ca6ab93458c87f34"
DATA_SHA256 = "b20ba3077ba2341d76c16dde3e58d3b37c66c3b0ec8346901a8bac14319e20fe"
MODEL_REVISION = "5ccf5ca3746c3620aa4016114f069a5f6ae399cd"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def shape(value):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in value.type.tensor_type.shape.dim]


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
    data = source.with_name(source.name + ".data")
    if digest(source) != ONNX_SHA256 or digest(data) != DATA_SHA256:
        raise ValueError("refusing a ViTPose model that is not the pinned NVIDIA artifact")
    graph = onnx.load(str(source), load_external_data=True)
    if [(x.name, shape(x)) for x in graph.graph.input] != [("imgs", ["B", 3, 256, 192])] or \
       [(x.name, shape(x)) for x in graph.graph.output] != [("heatmaps", [1, 77, 64, 48])]:
        raise ValueError("unexpected ViTPose graph signature")
    operators = collections.Counter(node.op_type for node in graph.graph.node)
    expected_operators = {"Add": 288, "Concat": 200, "Conv": 2, "ConvTranspose": 2,
        "Cos": 32, "Expand": 2, "LayerNormalization": 65, "MatMul": 224,
        "Mul": 320, "Neg": 64, "Relu": 2, "Reshape": 132, "Shape": 33,
        "Sigmoid": 32, "Sin": 32, "Slice": 321, "Softmax": 32, "Split": 64,
        "Squeeze": 96, "Transpose": 162}
    if len(graph.graph.node) != 2105 or len(graph.graph.initializer) != 573 or \
       operators != expected_operators:
        raise ValueError("unexpected ViTPose graph structure")
    tensors = {item.name: numpy_helper.to_array(item) for item in graph.graph.initializer}
    if len(tensors) != 573:
        raise ValueError("duplicate ViTPose initializer")

    converted = {}
    def take(source_name, target_name, expected_shape, transpose=False):
        value = np.asarray(tensors[source_name], dtype=np.float32)
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise ValueError(f"invalid ViTPose initializer {source_name}: {value.shape}")
        if transpose:
            value = value.T
        converted[target_name] = np.ascontiguousarray(value)

    take("add_22", "backbone.cls_token", (1, 1, 1280))
    take("model.backbone.storage_tokens", "backbone.storage_tokens", (1, 4, 1280))
    take("model.backbone.patch_embed.proj.weight", "backbone.patch.weight", (1280, 3, 16, 16))
    take("model.backbone.patch_embed.proj.bias", "backbone.patch.bias", (1280,))
    angles = np.asarray(tensors["tile"], dtype=np.float32)
    if angles.shape != (192, 64) or not np.isfinite(angles).all():
        raise ValueError("invalid ViTPose RoPE angles")
    converted["backbone.rope_angles"] = np.ascontiguousarray(
        np.concatenate([np.zeros((5, 64), dtype=np.float32), angles], axis=0))
    for block in range(32):
        prefix = f"model.backbone.blocks.{block}"
        target = f"backbone.block.{block}"
        for suffix in ("norm1.weight", "norm1.bias", "attn.proj.bias", "ls1.gamma",
                       "norm2.weight", "norm2.bias", "mlp.w1.bias", "mlp.w2.bias",
                       "mlp.w3.bias", "ls2.gamma"):
            expected = (5120,) if suffix in ("mlp.w1.bias", "mlp.w2.bias") else (1280,)
            take(f"{prefix}.{suffix}", f"{target}.{suffix}", expected)
        number = 79 if block == 0 else 85 + 183 * block
        bias_number = 40 + 111 * block
        take(f"val_{number}", f"{target}.attn.qkv.weight", (1280, 3840), True)
        take(f"mul_{bias_number}", f"{target}.attn.qkv.bias", (3840,))
        take(f"val_{219 + 183 * block}", f"{target}.attn.proj.weight", (1280, 1280), True)
        take(f"val_{223 + 183 * block}", f"{target}.mlp.w1.weight", (1280, 5120), True)
        take(f"val_{225 + 183 * block}", f"{target}.mlp.w2.weight", (1280, 5120), True)
        take(f"val_{228 + 183 * block}", f"{target}.mlp.w3.weight", (5120, 1280), True)
        angle_name = "tile" if block == 0 else f"tile_{block}"
        angle = np.asarray(tensors[angle_name], dtype=np.float32)
        if angle.shape != (192, 64) or not np.array_equal(angle, tensors["tile"]):
            raise ValueError(f"block {block} has unexpected RoPE angles")
    take("model.backbone.norm.weight", "backbone.norm.weight", (1280,))
    take("model.backbone.norm.bias", "backbone.norm.bias", (1280,))
    take("model.keypoint_head.deconv_layers.0.weight", "head.deconv.0.weight", (1280, 256, 4, 4))
    take("model.keypoint_head.deconv_layers.0.weight_bias", "head.deconv.0.bias", (256,))
    take("model.keypoint_head.deconv_layers.3.weight", "head.deconv.1.weight", (256, 256, 4, 4))
    take("model.keypoint_head.deconv_layers.3.weight_bias", "head.deconv.1.bias", (256,))
    take("model.keypoint_head.final_layer.weight", "head.final.weight", (77, 256, 1, 1))
    take("model.keypoint_head.final_layer.bias", "head.final.bias", (77,))
    if len(converted) != 525:
        raise ValueError(f"expected 525 native tensors, found {len(converted)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.output), arch="gemx_vitpose")
    writer.add_string("general.name", "NVIDIA GEM-X ViTPose SOMA-77")
    writer.add_string("gemx.model_revision", MODEL_REVISION)
    writer.add_string("gemx.vitpose.onnx_sha256", ONNX_SHA256)
    writer.add_string("gemx.vitpose.data_sha256", DATA_SHA256)
    writer.add_uint32("gemx.vitpose.image_width", 192)
    writer.add_uint32("gemx.vitpose.image_height", 256)
    writer.add_uint32("gemx.vitpose.embedding_length", 1280)
    writer.add_uint32("gemx.vitpose.feed_forward_length", 5120)
    writer.add_uint32("gemx.vitpose.block_count", 32)
    writer.add_uint32("gemx.vitpose.attention.head_count", 20)
    writer.add_uint32("gemx.vitpose.joint_count", 77)
    writer.add_float32("gemx.vitpose.layer_norm_epsilon", 1e-6)
    writer.add_file_type(0)
    for name in sorted(converted):
        writer.add_tensor(name, converted[name])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output} ({len(converted)} tensors)")


if __name__ == "__main__":
    main()
