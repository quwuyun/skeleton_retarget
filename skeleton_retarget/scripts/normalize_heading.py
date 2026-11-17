"""Save a separate motion whose first-frame horizontal facing is world +X."""

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from skeleton_retarget.retargeting.heading import heading_transform
from skeleton_retarget.retargeting.validation import sha256

FIELDS = ("fps", "root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list")


def normalize_motion(
    motion_path, output_prefix, *, metadata_path=None, target_heading=0.0, zero_xy=True
):
    """Create NPZ/PKL/JSON without changing the input or any source references."""
    source = Path(motion_path).resolve()
    metadata_path = (
        Path(metadata_path).resolve() if metadata_path else source.with_suffix(".json")
    )
    prefix = Path(output_prefix).resolve()
    if prefix.suffix in {".npz", ".pkl", ".json"}:
        prefix = prefix.with_suffix("")
    outputs = {
        extension: Path(f"{prefix}.{extension}") for extension in ("npz", "pkl", "json")
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("Output NPZ/PKL/JSON already exists; choose a new prefix")
    metadata = json.loads(metadata_path.read_text())
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != set(FIELDS):
            raise ValueError("Expected exactly the six standard motion fields")
        motion = {field: archive[field].copy() for field in FIELDS}
    frames = len(motion["root_pos"])
    if motion["fps"].shape != () or float(motion["fps"]) <= 0:
        raise ValueError("fps must be a positive finite scalar")
    names = motion["link_body_list"]
    if names.ndim != 1 or names.dtype.kind not in "US":
        raise ValueError("link_body_list must contain strings")
    if (
        motion["dof_pos"].ndim != 2
        or len(motion["dof_pos"]) != frames
        or motion["local_body_pos"].shape != (frames, len(names), 3)
    ):
        raise ValueError("Joint/body shapes must agree with the root frame count")
    for field in FIELDS:
        if motion[field].dtype.hasobject:
            raise ValueError("Object arrays are prohibited")
        if field != "link_body_list" and not np.isfinite(motion[field]).all():
            raise ValueError(f"Non-finite values in {field}")
    root_pos, root_rot, info = heading_transform(
        motion["root_pos"],
        motion["root_rot"],
        target_heading=target_heading,
        zero_xy=zero_xy,
    )
    previous = metadata.get("heading_normalization", {})
    if previous.get("enabled"):
        # Metadata always maps original calibrated/exported targets directly to
        # this result, even when an already normalized motion is rotated again.
        rotation = Rotation.from_quat(info["world_rotation_xyzw"])
        previous_rotation = Rotation.from_quat(previous["world_rotation_xyzw"])
        info["world_rotation_xyzw"] = (rotation * previous_rotation).as_quat().tolist()
        info["world_translation_m"] = (
            rotation.apply(previous["world_translation_m"])
            + info["world_translation_m"]
        ).tolist()
        info["previous_normalization"] = copy.deepcopy(previous)
        info["transform_scope"] = (
            "Composed transform from original IK export to this motion"
        )
    motion["root_pos"], motion["root_rot"] = root_pos, root_rot
    info.update(
        {
            "parent_motion": str(source),
            "parent_motion_sha256": sha256(source),
            "parent_metadata": str(metadata_path),
            "parent_metadata_sha256": sha256(metadata_path),
        }
    )
    metadata["heading_normalization"] = info
    metadata["outputs"] = {key: str(value) for key, value in outputs.items()}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outputs["npz"], **motion)
    pickled = {**motion, "fps": float(motion["fps"]), "link_body_list": names.tolist()}
    with outputs["pkl"].open("wb") as stream:
        pickle.dump(pickled, stream, protocol=pickle.HIGHEST_PROTOCOL)
    outputs["json"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    return outputs, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion", required=True, type=Path, help="Input six-field NPZ"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New output prefix; input is preserved",
    )
    parser.add_argument(
        "--metadata", type=Path, help="Default: input NPZ's same-stem JSON"
    )
    parser.add_argument(
        "--target_heading_deg",
        type=float,
        default=0.0,
        help="First-frame root +X facing in world XY: 0=+X, 90=+Y",
    )
    parser.add_argument(
        "--keep_xy",
        action="store_true",
        help="Keep first root XY; default starts each clip at XY=(0,0)",
    )
    args = parser.parse_args()
    outputs, info = normalize_motion(
        args.motion,
        args.output,
        metadata_path=args.metadata,
        target_heading=np.deg2rad(args.target_heading_deg),
        zero_xy=not args.keep_xy,
    )
    print(f"World yaw rotation: {np.rad2deg(info['yaw_rotation_radians']):.6f} deg")
    print(outputs["npz"])


if __name__ == "__main__":
    main()
