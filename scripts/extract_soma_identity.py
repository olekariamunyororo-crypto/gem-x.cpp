#!/usr/bin/env python3
"""Extract bounded native MHR/SOMA constants from pinned NVIDIA assets.

This reference-only step loads the official TorchScript and SOMA Python code.
It first verifies the exact upstream checkout and every input asset.  The output
is a non-executable NPZ which the GGUF converter validates again by shape.
"""

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

SOMA_REVISION = "e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5"
SOMA_SHA256 = "515f7d5bb74be4e370e9adf5e779760ec3581556374c0b33212a32d13ab3b53f"
MHR_SHA256 = "8072ec25449c95ea18b3f747b944d8d43377d9114bbd72ef34028610cc3d58e1"
BASE_SHA256 = "cd46620ec46e968658527f25e2b54b8f2292bfd486eac59c596f8b5d5229b9e0"
WRAP_SHA256 = "9ab7a2328d5b026c17ef1d09ed692ffeb68f96d3fc74970cba72e667dbe0f728"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", source, *args], check=True,
                          text=True, stdout=subprocess.PIPE).stdout.strip()


def array(value, dtype):
    result = value.detach().cpu().numpy().astype(dtype, copy=False)
    if not np.isfinite(result).all() if np.issubdtype(result.dtype, np.floating) else False:
        raise ValueError("non-finite identity transfer constant")
    return np.ascontiguousarray(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path,
                        help="SOMA asset root containing SOMA_neutral.npz and MHR/")
    parser.add_argument("output", type=Path)
    parser.add_argument("--soma-source", type=Path, required=True)
    args = parser.parse_args()
    source = args.soma_source.resolve()
    if git(source, "rev-parse", "HEAD") != SOMA_REVISION or git(source, "status", "--porcelain"):
        raise ValueError("SOMA source must be the clean pinned NVIDIA revision")
    required = {
        args.assets / "SOMA_neutral.npz": SOMA_SHA256,
        args.assets / "MHR/mhr_model_lod6.pt": MHR_SHA256,
        args.assets / "MHR/base_body_lod6.obj": BASE_SHA256,
        args.assets / "MHR/SOMA_wrap_lod1.obj": WRAP_SHA256,
    }
    for path, expected in required.items():
        if digest(path) != expected:
            raise ValueError(f"asset does not match pinned NVIDIA file: {path}")

    sys.path.insert(0, str(source))
    import soma.soma as soma_module  # pylint: disable=import-outside-toplevel
    soma_module.CorrectivesMLP.load_checkpoint = staticmethod(lambda *unused, **kwargs: None)
    layer = soma_module.SOMALayer(
        args.assets, low_lod=True, device="cpu", identity_model_type="mhr", mode="torch")
    mhr = layer.identity_model.identity_model.module
    buffers = dict(mhr.named_buffers())
    prefix = "character_torch."
    get = lambda name: buffers[prefix + name]

    interp = layer.identity_model._to_soma_interp
    rbf = layer.skeleton_transfer.sparse_rbf_matrix
    parents = layer.joint_parent_ids.detach().cpu().numpy().astype(np.int32)
    children = [[] for _ in range(78)]
    for joint in range(1, 78):
        children[int(parents[joint])].append(joint)
    rotation_crow = [0]
    rotation_vertices = []
    rotation_reference = []
    for joint in range(78):
        ids = torch.where(layer.skinning_weights[:, joint] > 0.01)[0]
        rotation_vertices.append(ids.detach().cpu().numpy().astype(np.int32))
        ref = layer.bind_shape[ids] - layer.bind_pose_world[joint, :3, 3]
        rotation_reference.append(array(ref, np.float32))
        rotation_crow.append(rotation_crow[-1] + len(ids))

    values = {
        "mhr_offsets": array(get("skeleton.joint_translation_offsets"), np.float32),
        "mhr_prerotations": array(get("skeleton.joint_prerotations"), np.float32),
        "mhr_parents": array(get("skeleton.joint_parents"), np.int32),
        "mhr_parameter_matrix": array(
            get("parameter_transform.parameter_transform")[:, :204], np.float32),
        "mhr_inverse_bind": array(
            get("linear_blend_skinning.inverse_bind_pose"), np.float32),
        "mhr_skin_joints": array(
            get("linear_blend_skinning.skin_indices_flattened"), np.int32),
        "mhr_skin_weights": array(
            get("linear_blend_skinning.skin_weights_flattened"), np.float32),
        "mhr_skin_vertices": array(
            get("linear_blend_skinning.vert_indices_flattened"), np.int32),
        "mhr_shape_vectors": array(get("blend_shape.shape_vectors")[:45], np.float32),
        "mhr_base_shape": array(get("blend_shape.base_shape"), np.float32),
        "mhr_faces": array(get("mesh.faces"), np.int32),
        "transfer_face_ids": array(interp.face_ids, np.int32),
        "transfer_barycentric": array(interp.bary_coords, np.float32),
        "soma_rbf_crow": array(rbf.crow_indices(), np.int32),
        "soma_rbf_columns": array(rbf.col_indices(), np.int32),
        "soma_rbf_values": array(rbf.values(), np.float32),
        "soma_bind_world": array(layer.bind_pose_world, np.float32),
        "soma_rotation_crow": np.asarray(rotation_crow, dtype=np.int32),
        "soma_rotation_vertices": np.ascontiguousarray(np.concatenate(rotation_vertices)),
        "soma_rotation_reference": np.ascontiguousarray(np.concatenate(rotation_reference)),
        "source_soma_revision": np.frombuffer(SOMA_REVISION.encode("ascii"), dtype=np.uint8),
        "source_mhr_sha256": np.frombuffer(MHR_SHA256.encode("ascii"), dtype=np.uint8),
    }
    expected = {
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
    for name, shape in expected.items():
        if values[name].shape != shape:
            raise ValueError(f"unexpected {name} shape {values[name].shape}, expected {shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **values)
    print(f"wrote {args.output} ({digest(args.output)})")


if __name__ == "__main__":
    main()
