#!/usr/bin/env python3
"""Convert the pinned official GEM-X ONNX denoiser to a native GGUF model.

The ONNX protobuf is a safe, non-pickle source. The converter accepts only the
published hashes and exact graph signature, then rewrites ONNX [in,out] linear
matrices to the [out,in] arrays expected by GGUF/GGML.
"""

import argparse
import ast
import hashlib
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import numpy_helper

EXPECTED_ONNX_SHA256 = "20aab83c01bbd909a258ad0fa465458eda382c63dc80d56952b2ec952d6e192c"
EXPECTED_DATA_SHA256 = "720c2400281b730574bf2a446c33ecd77bf494ac8161b081b09bb122b9fad2f8"
SOURCE_REVISION = "32992550dba114c62243fb55e361311972dce8f9"
MODEL_REVISION = "5ccf5ca3746c3620aa4016114f069a5f6ae399cd"
STATS_SHA256 = "dafe4ef6a62e824b0b325f54e42d508015785509bc478ef76e4208ae1f95ba7c"
SOMA_RIG_SHA256 = "515f7d5bb74be4e370e9adf5e779760ec3581556374c0b33212a32d13ab3b53f"
MHR_SHA256 = "8072ec25449c95ea18b3f747b944d8d43377d9114bbd72ef34028610cc3d58e1"
SOMA_REVISION = "e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def linear_names() -> dict[str, str]:
    result = {
        "val_81": "obs.xy.weight",
        "val_102": "obs.mlp.0.weight",
        "val_111": "obs.mlp.2.weight",
        "val_113": "cliff.mlp.0.weight",
        "val_116": "cliff.mlp.2.weight",
        "val_118": "cliff.exists.0.weight",
        "val_121": "cliff.exists.2.weight",
        "val_125": "image.proj.weight",
        "val_127": "image.exists.0.weight",
        "val_130": "image.exists.2.weight",
        "val_132": "angular.mlp.0.weight",
        "val_135": "angular.mlp.2.weight",
        "val_137": "angular.exists.0.weight",
        "val_140": "angular.exists.2.weight",
        "val_166": "time.mlp.0.weight",
        "val_169": "time.mlp.2.weight",
        "val_2274": "output.mlp.0.weight",
        "val_2283": "output.mlp.2.weight",
        "val_2341": "camera.mlp.0.weight",
        "val_2350": "camera.mlp.2.weight",
    }
    # Export IDs advance by 174 per block after block zero's projection. Keep
    # the explicit table derivation readable and independently shape-checked.
    qkv = [184, 362, 536, 710, 884, 1058, 1232, 1406, 1580, 1754, 1928, 2102]
    proj = [340, 514, 688, 862, 1036, 1210, 1384, 1558, 1732, 1906, 2080, 2254]
    fc1 = [344, 518, 692, 866, 1040, 1214, 1388, 1562, 1736, 1910, 2084, 2258]
    fc2 = [358, 532, 706, 880, 1054, 1228, 1402, 1576, 1750, 1924, 2098, 2272]
    for block in range(12):
        for offset, label in enumerate(("query", "key", "value")):
            result[f"val_{qkv[block] + 2 * offset}"] = f"block.{block}.attn.{label}.weight"
        result[f"val_{proj[block]}"] = f"block.{block}.attn.output.weight"
        result[f"val_{fc1[block]}"] = f"block.{block}.mlp.0.weight"
        result[f"val_{fc2[block]}"] = f"block.{block}.mlp.2.weight"
    return result


def named_initializers() -> dict[str, str]:
    result = {
        "learned_pos_params": "obs.missing",
        "gem.learned_pos_linear.bias": "obs.xy.bias",
        "gem.embed_noisyobs.fc1.bias": "obs.mlp.0.bias",
        "gem.embed_noisyobs.fc2.bias": "obs.mlp.2.bias",
        "gem.cliffcam_embedder.0.bias": "cliff.mlp.0.bias",
        "gem.cliffcam_embedder.3.bias": "cliff.mlp.2.bias",
        "gem.cond_exists_embedder.f_cliffcam.0.bias": "cliff.exists.0.bias",
        "gem.cond_exists_embedder.f_cliffcam.2.bias": "cliff.exists.2.bias",
        "gem.imgseq_embedder.0.weight": "image.norm.weight",
        "gem.imgseq_embedder.0.bias": "image.norm.bias",
        "gem.imgseq_embedder.1.bias": "image.proj.bias",
        "gem.cond_exists_embedder.f_imgseq.0.bias": "image.exists.0.bias",
        "gem.cond_exists_embedder.f_imgseq.2.bias": "image.exists.2.bias",
        "gem.cam_angvel_embedder.0.bias": "angular.mlp.0.bias",
        "gem.cam_angvel_embedder.3.bias": "angular.mlp.2.bias",
        "gem.cond_exists_embedder.f_cam_angvel.0.bias": "angular.exists.0.bias",
        "gem.cond_exists_embedder.f_cam_angvel.2.bias": "angular.exists.2.bias",
        "cam_angvel_mean": "angular.mean",
        "cam_angvel_std": "angular.std",
        "gem.pipeline.denoiser3d.denoiser.embed_timestep.time_embed.0.bias": "time.mlp.0.bias",
        "gem.pipeline.denoiser3d.denoiser.embed_timestep.time_embed.2.bias": "time.mlp.2.bias",
        "gem.pipeline.denoiser3d.denoiser.add_cond_linear.bias": "input.bias",
        "denoiser.scale_comps": "output.scale_components",
        "denoiser.scale_mean": "output.scale_mean",
        "gem.pipeline.denoiser3d.denoiser.final_layer.fc1.bias": "output.mlp.0.bias",
        "gem.pipeline.denoiser3d.denoiser.final_layer.fc2.bias": "output.mlp.2.bias",
        "gem.pipeline.denoiser3d.denoiser.pred_cam_head.fc1.bias": "camera.mlp.0.bias",
        "gem.pipeline.denoiser3d.denoiser.pred_cam_head.fc2.bias": "camera.mlp.2.bias",
        "denoiser.pred_cam_std": "camera.std",
        "denoiser.pred_cam_mean": "camera.mean",
    }
    prefix = "gem.pipeline.denoiser3d.denoiser.blocks"
    for block in range(12):
        source = f"{prefix}.{block}"
        target = f"block.{block}"
        for suffix in ("gate_msa", "gate_mlp", "norm1.weight", "norm1.bias",
                       "attn.query.bias", "attn.key.bias", "attn.value.bias",
                       "attn.proj.bias", "norm2.weight", "norm2.bias",
                       "mlp.fc1.bias", "mlp.fc2.bias"):
            renamed = suffix.replace("gate_msa", "attn.gate").replace("gate_mlp", "mlp.gate")
            renamed = renamed.replace("attn.proj", "attn.output")
            renamed = renamed.replace("mlp.fc1", "mlp.0").replace("mlp.fc2", "mlp.2")
            result[f"{source}.{suffix}"] = f"{target}.{renamed}"
    return result


EXPECTED_SHAPES = {
    "obs.xy.weight": (32, 2), "obs.mlp.0.weight": (1024, 1056),
    "obs.mlp.2.weight": (512, 1024), "cliff.mlp.0.weight": (512, 3),
    "image.proj.weight": (512, 1024), "angular.mlp.0.weight": (512, 6),
    "input.weight": (512, 512), "output.mlp.2.weight": (585, 512),
    "camera.mlp.2.weight": (3, 512), "output.scale_components": (69, 28),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--upstream-root", type=Path, required=True,
                        help="GEM-X checkout at the pinned source revision")
    parser.add_argument("--soma-rig", type=Path, required=True,
                        help="safe NVIDIA SOMA_neutral.npz asset")
    parser.add_argument("--identity-data", type=Path, required=True,
                        help="safe NPZ emitted by extract_soma_identity.py")
    parser.add_argument("--gguf-py", type=Path,
                        help="llama.cpp gguf-py directory (or install the gguf package)")
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional hash-verified official checkpoint for the omitted contact head")
    args = parser.parse_args()
    if args.gguf_py:
        sys.path.insert(0, str(args.gguf_py.resolve()))
    import gguf

    model_path = args.model.resolve()
    data_path = model_path.with_name(model_path.name + ".data")
    if digest(model_path) != EXPECTED_ONNX_SHA256 or digest(data_path) != EXPECTED_DATA_SHA256:
        raise ValueError("refusing an ONNX model that is not the pinned NVIDIA artifact")
    model = onnx.load(str(model_path), load_external_data=True)
    stats_path=(args.upstream_root/"gem/network/stats_compose.py").resolve()
    if digest(stats_path)!=STATS_SHA256:
        raise ValueError("GEM-X motion statistics do not match the pinned source revision")
    tree=ast.parse(stats_path.read_text(),filename=str(stats_path))
    stats=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and
                target.id=="MM_V2_SOMA_METROSIM" for target in node.targets):
            stats=ast.literal_eval(node.value)
            break
    if not isinstance(stats,dict):
        raise ValueError("pinned GEM-X motion statistics are missing")
    soma_path=args.soma_rig.resolve()
    if digest(soma_path)!=SOMA_RIG_SHA256:
        raise ValueError("SOMA neutral rig does not match the pinned safe asset")
    with np.load(soma_path,allow_pickle=False) as soma:
        soma_local=np.asarray(soma["t_pose_local"],dtype=np.float32)
        soma_world=np.asarray(soma["t_pose_world"],dtype=np.float32)
        soma_parents=np.asarray(soma["joint_parent_ids"],dtype=np.int32)
        soma_names=[str(value) for value in soma["joint_names"].tolist()]
    if soma_local.shape!=(78,4,4) or soma_world.shape!=(78,4,4) or \
       soma_parents.shape!=(78,) or len(soma_names)!=78:
        raise ValueError("unexpected SOMA neutral rig layout")
    if [(x.name, [d.dim_param or d.dim_value for d in x.type.tensor_type.shape.dim])
            for x in model.graph.input] != [
        ("obs", ["B", "L", 77, 3]), ("bbx_xys", ["B", "L", 3]),
        ("K_fullimg", ["B", "L", 3, 3]), ("f_imgseq", ["B", "L", 1024]),
        ("f_cam_angvel", ["B", "L", 6])]:
        raise ValueError("unexpected GEM-X input signature")

    matrices = linear_names()
    names = named_initializers()
    tensors = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    converted: dict[str, np.ndarray] = {}
    for source, target in matrices.items():
        value = np.asarray(tensors[source], dtype=np.float32)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ValueError(f"invalid linear initializer {source}")
        converted[target] = np.ascontiguousarray(value.T)
    for source, target in names.items():
        value = np.asarray(tensors[source], dtype=np.float32)
        if target.endswith(".gate"):
            value = value.reshape(-1)
        if target == "output.scale_components":
            value = value.T
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite initializer {source}")
        converted[target] = np.ascontiguousarray(value)
    # One GEMM per attention block preserves each output row's reduction while
    # removing two launches and duplicate input traversal. GGML views split the
    # result back into Q/K/V without copies.
    for block in range(12):
        prefix=f"block.{block}.attn"
        converted[f"{prefix}.qkv.weight"]=np.ascontiguousarray(np.concatenate([
            converted.pop(f"{prefix}.query.weight"),converted.pop(f"{prefix}.key.weight"),
            converted.pop(f"{prefix}.value.weight")],axis=0))
        converted[f"{prefix}.qkv.bias"]=np.ascontiguousarray(np.concatenate([
            converted.pop(f"{prefix}.query.bias"),converted.pop(f"{prefix}.key.bias"),
            converted.pop(f"{prefix}.value.bias")],axis=0))
    # The absent-image branch masks the projected feature before its existence
    # MLP. Its entire contribution is constant, independent of the input token.
    bias = converted["image.exists.0.bias"]
    hidden = bias / (1 + np.exp(-bias))
    converted["image.absent"] = np.ascontiguousarray(
        converted["image.exists.2.weight"] @ hidden + converted["image.exists.2.bias"])
    # The released regression path always marks these conditions present. Fold
    # the constant final input column into the first-layer bias exactly once.
    for condition in ("cliff","image","angular"):
        weight=converted[f"{condition}.exists.0.weight"]
        if weight.shape!=(512,513):
            raise ValueError(f"unexpected {condition} presence projection")
        if condition == "cliff":
            # The released checkpoint suppresses the box-camera condition when
            # a frame has fewer than four confident 2D joints. The published
            # ONNX export accidentally hard-codes this presence input to one.
            # Preserve the column so native inference can follow checkpoint
            # semantics on lost detections.
            converted["cliff.exists.presence"] = np.ascontiguousarray(weight[:,512,None])
        else:
            converted[f"{condition}.exists.0.bias"]=np.ascontiguousarray(
                converted[f"{condition}.exists.0.bias"]+weight[:,512])
        converted[f"{condition}.exists.0.weight"]=np.ascontiguousarray(weight[:,:512])
    # Regression export fixes xt to zero. Its 585 columns therefore have no
    # effect; retaining only the condition columns removes 1.2 MiB and one
    # zero-input concatenation without changing the function.
    input_weight = np.asarray(tensors["val_171"], dtype=np.float32)
    converted["input.weight"] = np.ascontiguousarray(input_weight[:512].T)
    # Only timestep 999 is reachable in the published regression graph.
    converted["time.position"] = np.ascontiguousarray(
        np.asarray(tensors["val_159"], dtype=np.float32)[999, 0])
    converted["motion.mean"] = np.ascontiguousarray(np.asarray(stats["mean"],dtype=np.float32))
    # configs/endecoder/v2_soma_local_cam.yaml sets clip_std: true.
    converted["motion.std"] = np.ascontiguousarray(
        np.maximum(np.asarray(stats["std"], dtype=np.float32), 1.0))
    converted["soma.rest_local"] = np.ascontiguousarray(soma_local)
    converted["soma.rest_world"] = np.ascontiguousarray(soma_world)
    converted["soma.parents"] = np.ascontiguousarray(soma_parents)
    identity_names = {
        "mhr_offsets": "mhr.offsets",
        "mhr_prerotations": "mhr.prerotations",
        "mhr_parents": "mhr.parents",
        "mhr_parameter_matrix": "mhr.parameter_matrix",
        "mhr_inverse_bind": "mhr.inverse_bind",
        "mhr_skin_joints": "mhr.skin_joints",
        "mhr_skin_weights": "mhr.skin_weights",
        "mhr_skin_vertices": "mhr.skin_vertices",
        "mhr_shape_vectors": "mhr.shape_vectors",
        "mhr_base_shape": "mhr.base_shape",
        "mhr_faces": "mhr.faces",
        "transfer_face_ids": "soma.transfer_face_ids",
        "transfer_barycentric": "soma.transfer_barycentric",
        "soma_rbf_crow": "soma.rbf_crow",
        "soma_rbf_columns": "soma.rbf_columns",
        "soma_rbf_values": "soma.rbf_values",
        "soma_bind_world": "soma.bind_world",
        "soma_rotation_crow": "soma.rotation_crow",
        "soma_rotation_vertices": "soma.rotation_vertices",
        "soma_rotation_reference": "soma.rotation_reference",
    }
    expected_identity = {
        "mhr_offsets": (127, 3), "mhr_prerotations": (127, 4),
        "mhr_parents": (127,), "mhr_parameter_matrix": (889, 204),
        "mhr_inverse_bind": (127, 8), "mhr_skin_joints": (1747,),
        "mhr_skin_weights": (1747,), "mhr_skin_vertices": (1747,),
        "mhr_shape_vectors": (45, 595, 3), "mhr_base_shape": (595, 3),
        "mhr_faces": (1186, 3), "transfer_face_ids": (4505,),
        "transfer_barycentric": (4505, 4), "soma_rbf_crow": (79,),
        "soma_rbf_columns": (14725,), "soma_rbf_values": (14725,),
        "soma_bind_world": (78, 4, 4), "soma_rotation_crow": (79,),
        "soma_rotation_vertices": (7709,), "soma_rotation_reference": (7709, 3),
    }
    with np.load(args.identity_data, allow_pickle=False) as identity_data:
        revision = bytes(identity_data["source_soma_revision"]).decode("ascii")
        mhr_hash = bytes(identity_data["source_mhr_sha256"]).decode("ascii")
        if revision != SOMA_REVISION or mhr_hash != MHR_SHA256:
            raise ValueError("identity constants do not identify pinned NVIDIA sources")
        for source, target in identity_names.items():
            value = np.asarray(identity_data[source])
            if value.shape != expected_identity[source]:
                raise ValueError(f"unexpected identity tensor {source}: {value.shape}")
            if np.issubdtype(value.dtype, np.floating):
                if value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f"invalid floating identity tensor {source}")
            elif not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"invalid identity tensor type {source}")
            converted[target] = np.ascontiguousarray(value.astype(
                np.int32 if np.issubdtype(value.dtype, np.integer) else np.float32,
                copy=False))
    bounded = {
        "mhr.parents": (-1, 126), "mhr.skin_joints": (0, 126),
        "mhr.skin_vertices": (0, 594), "mhr.faces": (0, 594),
        "soma.transfer_face_ids": (0, 1185), "soma.rbf_columns": (0, 4504),
        "soma.rotation_vertices": (0, 4504),
    }
    for name, (low, high) in bounded.items():
        if np.any(converted[name] < low) or np.any(converted[name] > high):
            raise ValueError(f"out-of-range identity indices in {name}")
    for name, final in (("soma.rbf_crow", 14725), ("soma.rotation_crow", 7709)):
        value = converted[name]
        if value[0] != 0 or value[-1] != final or np.any(value[1:] < value[:-1]):
            raise ValueError(f"invalid CSR offsets in {name}")
    if converted["mhr.parents"][0] != -1 or np.any(
            converted["mhr.parents"][1:] >= np.arange(1, 127)):
        raise ValueError("MHR parents are not topological")
    if converted["motion.mean"].shape!=(585,) or converted["motion.std"].shape!=(585,) or \
       not np.isfinite(converted["motion.mean"]).all() or \
       not np.isfinite(converted["motion.std"]).all() or \
       np.any(converted["motion.std"]<=0):
        raise ValueError("invalid pinned GEM-X motion statistics")
    for name, expected in EXPECTED_SHAPES.items():
        if converted[name].shape != expected:
            raise ValueError(f"{name} shape {converted[name].shape} != {expected}")
    if args.checkpoint:
        if digest(args.checkpoint) != "4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e":
            raise ValueError("Contact head requires the pinned official checkpoint")
        import torch
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        for layer, shape in (("0", (512, 512)), ("2", (6, 512))):
            source = "pipeline.denoiser3d.denoiser.static_conf_head.fc" + ("1" if layer == "0" else "2")
            for suffix, expected in (("weight", shape), ("bias", (shape[0],))):
                value = checkpoint[source + "." + suffix].float().numpy()
                if value.shape != expected or not np.isfinite(value).all():
                    raise ValueError("Invalid contact head tensor")
                converted["contact.mlp." + layer + "." + suffix] = np.ascontiguousarray(value)
    expected_count = 251 if args.checkpoint else 247
    if len(converted) != expected_count:
        raise ValueError(f"expected {expected_count} model tensors, found {len(converted)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.output), arch="gemx")
    writer.add_string("general.name", "NVIDIA GEM-X SOMA regression denoiser")
    writer.add_string("gemx.source_revision", SOURCE_REVISION)
    writer.add_string("gemx.model_revision", MODEL_REVISION)
    writer.add_string("gemx.onnx_sha256", EXPECTED_ONNX_SHA256)
    writer.add_string("gemx.stats_sha256", STATS_SHA256)
    writer.add_string("gemx.soma_rig_sha256", SOMA_RIG_SHA256)
    writer.add_string("gemx.soma_revision", SOMA_REVISION)
    writer.add_string("gemx.mhr_sha256", MHR_SHA256)
    writer.add_array("gemx.soma_joint_names",soma_names)
    writer.add_uint32("gemx.context_length", 120)
    writer.add_uint32("gemx.embedding_length", 512)
    writer.add_uint32("gemx.feed_forward_length", 2048)
    writer.add_uint32("gemx.block_count", 12)
    writer.add_uint32("gemx.attention.head_count", 8)
    writer.add_uint32("gemx.motion_length", 585)
    writer.add_uint32("gemx.observation.joint_count", 33)
    writer.add_uint32("gemx.timestep", 999)
    writer.add_float32("gemx.layer_norm_epsilon", 1e-6)
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
