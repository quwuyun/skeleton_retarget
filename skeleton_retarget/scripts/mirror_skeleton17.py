"""Create left-right mirrored skeleton17 motions and validate their FK.

Run from the project root::

    uv run python -m skeleton_retarget skeleton17 mirror

The default batch mirrors the seven long walking/start-stop clips. Originals
are read-only; each derived motion gets its own NPZ, PKL, JSON and reference.
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from skeleton_retarget.retargeting.motion import inspect_model, local_body_positions
from skeleton_retarget.retargeting.paths import (
    MODELS_ROOT,
    REPO_ROOT,
    resolve_recorded_path,
)
from skeleton_retarget.retargeting.validation import sha256
from skeleton_retarget.scripts.build_behavior_set import write_json
from skeleton_retarget.scripts.validate_skeleton17 import FIELD_ORDER, validate

DEFAULT_INPUT_ROOT = REPO_ROOT / "data/outputs/skeleton17_collision/long_segments_x_v02"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/outputs/skeleton17_collision/long_mirror"
DEFAULT_MODEL = MODELS_ROOT / "skeleton17_collision/human_skeleton.xml"
DEFAULT_CLIPS = (
    "long_forward_walk2_subject4_a",
    "long_forward_walk2_subject4_b",
    "long_forward_walk3_subject4",
    "stand_start_pause_restart_walk1_subject2",
    "walk_pause_restart_walk2_subject3",
    "walk_to_stop_walk3_subject2",
    "stand_to_turn_start_walk3_subject2",
)
REFLECTION = np.diag([1.0, -1.0, 1.0])


def mirror_name(name):
    """Swap common left/right labels in BVH and MJCF body names."""
    value = name.replace("Left", "\0").replace("Right", "Left").replace("\0", "Right")
    value = value.replace("left", "\0").replace("right", "left").replace("\0", "right")
    value = value.replace("lvs", "\0").replace("rvs", "lvs").replace("\0", "rvs")

    return re.sub(r"(?<![A-Za-z0-9])([lr])(?=_|$)|(?<=_)([lr])(?=_|$)", _token_swap, value)


def _token_swap(match):
    token = match.group(1) or match.group(2)
    return "r" if token == "l" else "l"


def mirror_quaternions_xyzw(quaternions):
    """Reflect proper rotations across world Y=0, retaining XYZW order."""
    result = np.asarray(quaternions, dtype=float).copy()
    if result.shape[-1] != 4:
        raise ValueError("Expected quaternion arrays ending in four XYZW components")
    result[..., 0] *= -1
    result[..., 2] *= -1
    return result


def mirror_joint_positions(dof_pos, joint_names):
    """Exchange bilateral hinge coordinates and apply reflection signs."""
    values = np.asarray(dof_pos, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(joint_names):
        raise ValueError("dof_pos shape does not match the ordered joint names")
    indices = {name: index for index, name in enumerate(joint_names)}
    if len(indices) != len(joint_names):
        raise ValueError("Joint names must be unique")
    result = np.empty_like(values)
    for destination, name in enumerate(joint_names):
        match = re.fullmatch(r"(hip|knee|ankle)_([xyz])_([rl])", name)
        back_match = re.fullmatch(r"back_([xyz])", name)
        if match:
            group, axis, side = match.groups()
            source_name = f"{group}_{axis}_{'l' if side == 'r' else 'r'}"
        elif back_match:
            axis = back_match.group(1)
            source_name = name
        else:
            raise ValueError(f"Unsupported skeleton17 hinge name: {name}")
        if source_name not in indices:
            raise ValueError(f"Missing mirror joint pair for {name}: {source_name}")
        sign = -1.0 if axis in {"x", "z"} else 1.0
        result[:, destination] = sign * values[:, indices[source_name]]
    return result


def mirror_reference(reference_path, output_path):
    """Reflect source FK and exchange anatomical left/right body columns."""
    with np.load(reference_path, allow_pickle=False) as archive:
        fields = {key: archive[key].copy() for key in archive.files}
    names = fields.get("body_names")
    if names is None or "positions" not in fields or "rotations" not in fields:
        raise ValueError(f"Unsupported source-reference fields: {reference_path}")
    names = names.tolist()
    indices = {name: index for index, name in enumerate(names)}
    if len(indices) != len(names):
        raise ValueError("Source reference body names must be unique")
    if {mirror_name(name) for name in names} != set(names):
        raise ValueError("Source reference does not contain all left/right body pairs")
    positions = fields["positions"]
    rotations = fields["rotations"]
    if positions.shape != (len(positions), len(names), 3):
        raise ValueError("Invalid source reference position shape")
    if rotations.shape != (len(positions), len(names), 4):
        raise ValueError("Invalid source reference quaternion shape")
    mirrored_positions = np.empty_like(positions)
    mirrored_rotations = np.empty_like(rotations)
    for destination, name in enumerate(names):
        source = indices[mirror_name(name)]
        mirrored_positions[:, destination] = positions[:, source] @ REFLECTION.T
        source_xyzw = rotations[:, source][:, [1, 2, 3, 0]]
        reflected_xyzw = mirror_quaternions_xyzw(source_xyzw)
        mirrored_rotations[:, destination] = reflected_xyzw[:, [3, 0, 1, 2]]
    fields["positions"] = mirrored_positions
    fields["rotations"] = mirrored_rotations
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **fields)
    return output_path


def _resolve_reference(source_motion, metadata):
    recorded = Path(metadata["source_reference"]).expanduser()
    if recorded.is_file():
        reference = recorded.resolve()
    else:
        reference = source_motion.parent / "references" / recorded.name
        if not reference.is_file():
            raise FileNotFoundError(
                f"Recorded source reference is missing and no local copy exists: {recorded}"
            )
        reference = reference.resolve()
    expected_hash = metadata.get("source_reference_sha256")
    if expected_hash and sha256(reference) != expected_hash:
        raise ValueError(f"Source-reference hash mismatch: {reference}")
    return reference


def _mirror_metadata(metadata, source_motion, source_reference, output_prefix, output_reference):
    result = json.loads(json.dumps(metadata))
    root_translation = np.asarray(result.get("world_translation_m", [0, 0, 0]), dtype=float)
    if root_translation.shape == (3,):
        root_translation[1] *= -1
        result["world_translation_m"] = root_translation.tolist()
    heading = result.get("heading_normalization", {})
    if heading.get("enabled"):
        quat = np.asarray(heading["world_rotation_xyzw"], dtype=float)
        trans = np.asarray(heading["world_translation_m"], dtype=float)
        reflected = mirror_quaternions_xyzw(quat)
        trans[1] *= -1
        heading["world_rotation_xyzw"] = reflected.tolist()
        heading["world_translation_m"] = trans.tolist()
        for field in ("original_heading_radians", "yaw_rotation_radians"):
            if field in heading:
                heading[field] = -float(heading[field])
    output_paths = {
        "npz": output_prefix.with_suffix(".npz"),
        "pkl": output_prefix.with_suffix(".pkl"),
        "json": output_prefix.with_suffix(".json"),
    }
    result["outputs"] = {key: str(path) for key, path in output_paths.items()}
    result["source_reference"] = str(output_reference.resolve())
    result["source_reference_sha256"] = sha256(output_reference)
    result["mirror_augmentation"] = {
        "method": "World-Y reflection with left/right anatomical exchange",
        "plane": "world Y=0",
        "reflection_matrix": REFLECTION.tolist(),
        "frame_order_reversed": False,
        "source_motion": str(source_motion.resolve()),
        "source_motion_sha256": sha256(source_motion),
        "source_reference_original": str(source_reference.resolve()),
        "source_reference_original_sha256": sha256(source_reference),
        "source_reference_mirrored": str(output_reference.resolve()),
        "source_reference_mirrored_sha256": sha256(output_reference),
        "source_tracking_note": "Source tracking is not evaluated for the mirrored derivative because the IK configuration retains original side-specific offsets.",
    }
    return result


def _decorate_validation_report(report, original, mirrored, fk_error):
    original_rotations = Rotation.from_quat(original["root_rot"])
    original_facing = original_rotations.apply([1.0, 0.0, 0.0])
    original_yaw = np.unwrap(np.arctan2(original_facing[:, 1], original_facing[:, 0]))
    mirrored_rotations = Rotation.from_quat(mirrored["root_rot"])
    mirrored_facing = mirrored_rotations.apply([1.0, 0.0, 0.0])
    mirrored_yaw = np.unwrap(np.arctan2(mirrored_facing[:, 1], mirrored_facing[:, 0]))
    report["source_tracking"] = {
        "status": "not_evaluated",
        "reason": "The mirrored derivative preserves the original IK configuration, whose side-specific offsets are calibrated for the unmirrored motion. Mirror correctness is checked from root pose, joint mapping, mirrored source-reference hashes, and fresh target-model FK.",
    }
    report["mirror_checks"] = {
        "root_xy_reflection_matches": bool(
            np.allclose(mirrored["root_pos"], original["root_pos"] @ REFLECTION.T)
        ),
        "root_rotation_reflection_matches": bool(
            np.allclose(mirrored["root_rot"], mirror_quaternions_xyzw(original["root_rot"]))
        ),
        "heading_change_sign_reversed": bool(
            np.isclose(
                mirrored_yaw[-1] - mirrored_yaw[0],
                -(original_yaw[-1] - original_yaw[0]),
                atol=1e-8,
                rtol=0,
            )
        ),
        "local_fk_matches_mirrored_source_fk": fk_error < 1e-8,
        "mirrored_source_reference_written": True,
    }
    return report


def _summary_row(
    clip_name, source_motion, output_paths, validation_path, mirrored, fk_error, report
):
    rotations = Rotation.from_quat(mirrored["root_rot"])
    facing = rotations.apply([1.0, 0.0, 0.0])
    yaw = np.unwrap(np.arctan2(facing[:, 1], facing[:, 0]))
    displacement = mirrored["root_pos"][-1] - mirrored["root_pos"][0]
    return {
        "name": clip_name,
        "mirror_of": str(source_motion.resolve()),
        "npz": str(output_paths["npz"].resolve()),
        "validation": str(validation_path.resolve()),
        "frames": len(mirrored["root_pos"]),
        "fps": float(mirrored["fps"]),
        "sample_span_seconds": (len(mirrored["root_pos"]) - 1) / float(mirrored["fps"]),
        "heading_change_deg": float(np.rad2deg(yaw[-1] - yaw[0])),
        "net_displacement_xyz_m": displacement.tolist(),
        "max_local_fk_mirror_error_m": fk_error,
        "validation_pass": report["format_and_kinematic_constraints_pass"],
        "warnings": report["warnings"],
    }


def mirror_motion(source_motion, output_dir, model_path):
    source_motion = source_motion.resolve(strict=True)
    source_metadata_path = source_motion.with_suffix(".json")
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"Missing motion metadata: {source_metadata_path}")
    source_metadata = json.loads(source_metadata_path.read_text())
    source_reference = _resolve_reference(source_motion, source_metadata)
    model_path = model_path.resolve(strict=True)
    if source_metadata.get("robot_xml_sha256") != sha256(model_path):
        raise ValueError(f"Source motion uses a different model: {source_motion}")

    model = mj.MjModel.from_xml_path(str(model_path))
    body_ids, body_names, joint_columns = inspect_model(model)
    joint_names = [column["name"] for column in joint_columns]
    with np.load(source_motion, allow_pickle=False) as archive:
        if list(archive.files) != FIELD_ORDER:
            raise ValueError(f"Expected official six-field ordering: {source_motion}")
        original = {key: archive[key].copy() for key in archive.files}
    if not np.array_equal(original["link_body_list"], np.asarray(body_names)):
        raise ValueError(f"Source body order differs from the target MJCF: {source_motion}")

    qpos = np.column_stack(
        (
            original["root_pos"],
            original["root_rot"][:, [3, 0, 1, 2]],
            original["dof_pos"],
        )
    )
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Motion qpos width does not match the target MJCF: {source_motion}")

    mirrored = {
        "fps": original["fps"],
        "root_pos": original["root_pos"] @ REFLECTION.T,
        "root_rot": mirror_quaternions_xyzw(original["root_rot"]),
        "dof_pos": mirror_joint_positions(original["dof_pos"], joint_names),
        "local_body_pos": None,
        "link_body_list": original["link_body_list"],
    }
    mirrored_qpos = np.column_stack(
        (
            mirrored["root_pos"],
            mirrored["root_rot"][:, [3, 0, 1, 2]],
            mirrored["dof_pos"],
        )
    )
    mirrored["local_body_pos"] = local_body_positions(model, mirrored_qpos, body_ids)

    # Verify the target MJCF is symmetric under the same anatomical mapping.
    source_body_indices = {name: index for index, name in enumerate(body_names)}
    expected_local = np.empty_like(mirrored["local_body_pos"])
    for destination, name in enumerate(body_names):
        paired = mirror_name(name)
        if paired not in source_body_indices:
            raise ValueError(f"Target MJCF is missing mirrored body pair for {name}")
        expected_local[:, destination] = (
            original["local_body_pos"][:, source_body_indices[paired]] @ REFLECTION.T
        )
    fk_mirror_error = float(
        np.linalg.norm(mirrored["local_body_pos"] - expected_local, axis=-1).max()
    )
    if fk_mirror_error > 1e-8:
        raise ValueError(
            f"Target model FK is not mirror symmetric (max error {fk_mirror_error:.3g} m)"
        )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_name = source_motion.stem.removesuffix("_120hz")
    output_name = f"{clip_name}_mirror_120hz"
    output_prefix = output_dir / clip_name / output_name
    output_paths = {
        "npz": output_prefix.with_suffix(".npz"),
        "pkl": output_prefix.with_suffix(".pkl"),
        "json": output_prefix.with_suffix(".json"),
    }
    validation_path = output_prefix.with_name(output_prefix.name + "_validation.json")
    mirror_reference_path = (
        output_dir / clip_name / "references" / (source_reference.stem + "_mirror.npz")
    )
    all_outputs = (*output_paths.values(), validation_path, mirror_reference_path)
    if any(path.exists() for path in all_outputs):
        if not all(path.is_file() for path in all_outputs):
            raise FileExistsError(
                f"Incomplete mirrored clip; refusing to overwrite: {output_prefix}"
            )
        existing_metadata = json.loads(output_paths["json"].read_text())
        provenance = existing_metadata.get("mirror_augmentation", {})
        if (
            provenance.get("source_motion_sha256") != sha256(source_motion)
            or provenance.get("source_reference_original_sha256") != sha256(source_reference)
            or existing_metadata.get("robot_xml_sha256") != sha256(model_path)
            or resolve_recorded_path(existing_metadata.get("source_reference", ""))
            != mirror_reference_path.resolve()
        ):
            raise FileExistsError(
                f"Existing mirrored clip has different provenance: {output_prefix}"
            )
        with np.load(output_paths["npz"], allow_pickle=False) as archive:
            existing_motion = {key: archive[key].copy() for key in archive.files}
        if list(existing_motion) != FIELD_ORDER or any(
            not np.array_equal(existing_motion[field], mirrored[field]) for field in FIELD_ORDER
        ):
            raise ValueError(f"Existing mirrored NPZ differs from its source: {output_prefix}")
        config_path = Path(existing_metadata["ik_config"]).expanduser().resolve(strict=True)
        report = validate(
            output_paths["npz"],
            model_path,
            output_paths["json"],
            config_path,
            mirror_reference_path,
            expected_frames=len(mirrored["root_pos"]),
            expected_fps=float(mirrored["fps"]),
        )
        if not report["format_and_kinematic_constraints_pass"]:
            failures = [name for name, passed in report["checks"].items() if not passed]
            raise ValueError(f"Existing mirrored motion failed validation: {failures}")
        report = _decorate_validation_report(report, original, existing_motion, fk_mirror_error)
        write_json(validation_path, report)
        return _summary_row(
            clip_name,
            source_motion,
            output_paths,
            validation_path,
            mirrored,
            fk_mirror_error,
            report,
        )

    mirror_reference(source_reference, mirror_reference_path)
    metadata = _mirror_metadata(
        source_metadata,
        source_motion,
        source_reference,
        output_prefix,
        mirror_reference_path,
    )
    metadata["mirror_augmentation"]["max_local_fk_mirror_error_m"] = fk_mirror_error
    np.savez_compressed(output_paths["npz"], **mirrored)
    with output_paths["pkl"].open("wb") as stream:
        pickle.dump(mirrored, stream, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(output_paths["json"], metadata)

    config_path = Path(metadata["ik_config"]).expanduser().resolve(strict=True)
    report = validate(
        output_paths["npz"],
        model_path,
        output_paths["json"],
        config_path,
        mirror_reference_path,
        expected_frames=len(mirrored["root_pos"]),
        expected_fps=float(mirrored["fps"]),
    )
    report = _decorate_validation_report(report, original, mirrored, fk_mirror_error)
    write_json(validation_path, report)
    if not report["format_and_kinematic_constraints_pass"]:
        failures = [name for name, passed in report["checks"].items() if not passed]
        raise ValueError(f"Mirrored motion validation failed: {failures}")
    return _summary_row(
        clip_name,
        source_motion,
        output_paths,
        validation_path,
        mirrored,
        fk_mirror_error,
        report,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--robot_xml", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIPS))
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    robot_xml = args.robot_xml.expanduser().resolve(strict=True)
    if output_dir == input_root:
        parser.error("output_dir must differ from input_root")
    rows = []
    for clip in args.clips:
        candidates = sorted(input_root.rglob(f"{clip}_120hz.npz"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one source NPZ for {clip}, found {len(candidates)}"
            )
        rows.append(mirror_motion(candidates[0], output_dir, robot_xml))
        print(f"Mirrored {clip}: {rows[-1]['frames']} frames at {rows[-1]['fps']:g} Hz")

    summary = {
        "model": str(robot_xml),
        "mirror_plane": "world Y=0",
        "source_root": str(input_root),
        "clips": rows,
        "all_pass": all(row["validation_pass"] for row in rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary)
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
