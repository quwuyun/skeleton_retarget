"""Build the five-behavior set at 120 Hz, with initial facing world +X.

Reuse forward/backward trajectories, generate a model-neutral static pose, and
retarget complete turns. Existing results are validated and reused only when
their selection and parent provenance match.
"""

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from skeleton_retarget.retargeting.motion import inspect_model, local_body_positions
from skeleton_retarget.retargeting.paths import (
    CONFIGS_ROOT,
    DATA_ROOT,
    MODELS_ROOT,
    REPO_ROOT,
    resolve_recorded_path,
)
from skeleton_retarget.retargeting.sources import read_bvh_fps
from skeleton_retarget.retargeting.validation import sha256
from skeleton_retarget.scripts.normalize_heading import normalize_motion
from skeleton_retarget.scripts.validate_skeleton17 import FIELD_ORDER, validate

DEFAULT_SELECTION = DATA_ROOT / "selections/lafan1_five_behaviors.json"
DEFAULT_OUTPUT = DATA_ROOT / "outputs/skeleton17_collision/five_behaviors_x"
FPS = 120.0


def build_neutral_stand(clip, selection_record, output_dir, model_path):
    """Export a constant upright pose from the model's zero hinge coordinates."""
    name = clip["name"]
    frames = int(clip["frames"])
    if frames < 2 or not math.isclose(clip["fps"], FPS):
        raise ValueError("Neutral standing requires at least two frames at 120 Hz")
    prefix = output_dir / name / f"{name}_120hz"
    outputs = {ext: prefix.with_suffix(f".{ext}") for ext in ("npz", "pkl", "json")}
    validation_path = prefix.with_name(prefix.name + "_validation.json")
    model_path = Path(model_path).resolve(strict=True)
    existing = any(path.exists() for path in (*outputs.values(), validation_path))
    if existing and not all(
        path.is_file() for path in (*outputs.values(), validation_path)
    ):
        raise FileExistsError(f"Incomplete neutral output: {prefix}")

    model = mj.MjModel.from_xml_path(str(model_path))
    body_ids, body_names, columns = inspect_model(model)
    qpos = model.qpos0.copy()
    qpos[:3] = 0
    qpos[3:7] = [1, 0, 0, 0]
    qpos[7:] = 0
    data = mj.MjData(model)
    data.qpos[:] = qpos
    mj.mj_forward(model, data)
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
        dtype=float,
    )
    foot_minimums_at_root_zero = {}
    for side in "lr":
        geom_id = model.geom(f"foot_{side}_collision").id
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        corners = (signs * model.geom_size[geom_id]) @ rotation.T + data.geom_xpos[
            geom_id
        ]
        foot_minimums_at_root_zero[side] = float(corners[:, 2].min())
    target_clearance = float(clip.get("foot_clearance_m", 0.001))
    root_height = target_clearance - min(foot_minimums_at_root_zero.values())
    qpos[2] = root_height
    data.qpos[:] = qpos
    mj.mj_forward(model, data)
    foot_geometry = {}
    for side in "lr":
        geom_id = model.geom(f"foot_{side}_collision").id
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        corners = (signs * model.geom_size[geom_id]) @ rotation.T + data.geom_xpos[
            geom_id
        ]
        foot_geometry[side] = {
            "center_xyz_m": data.geom_xpos[geom_id].tolist(),
            "minimum_corner_height_m": float(corners[:, 2].min()),
            "forward_axis_world": rotation[:, 0].tolist(),
            "up_axis_world": rotation[:, 2].tolist(),
        }
    motion = {
        "fps": np.asarray(FPS),
        "root_pos": np.repeat(qpos[None, :3], frames, axis=0),
        "root_rot": np.repeat(np.array([[0.0, 0.0, 0.0, 1.0]]), frames, axis=0),
        "dof_pos": np.zeros((frames, len(columns)), dtype=float),
        "local_body_pos": local_body_positions(
            model, np.repeat(qpos[None, :], frames, axis=0), body_ids
        ),
        "link_body_list": np.asarray(body_names, dtype=str),
    }
    metadata = {
        "target_model": "skeleton17_collision",
        "robot_xml": str(model_path),
        "robot_xml_sha256": sha256(model_path),
        "model_manifest": str(model_path.parent / "manifest.json"),
        "model_manifest_sha256": sha256(model_path.parent / "manifest.json"),
        "source_format": "synthetic_model_neutral_stand",
        "motion_origin": "MJCF qpos0 with all scalar hinge coordinates set to zero; constant kinematic pose",
        "fps": FPS,
        "frames": frames,
        "saved_interval_duration_seconds": (frames - 1) / FPS,
        "position_unit": "meter",
        "root_rotation_order": "xyzw",
        "root_orientation": "upright identity, root local +X facing world +X",
        "root_position_m": qpos[:3].tolist(),
        "root_linear_velocity_m_s": [0.0, 0.0, 0.0],
        "root_angular_velocity_rad_s": [0.0, 0.0, 0.0],
        "joint_columns": [
            {
                **column,
                "locked_zero": column["name"]
                in {"knee_x_r", "knee_z_r", "knee_x_l", "knee_z_l"},
            }
            for column in columns
        ],
        "internal_scalar_coordinates": len(columns),
        "independent_internal_dof_count": len(columns) - 4,
        "active_joint_names_xml_order": [
            column["name"]
            for column in columns
            if column["name"] not in {"knee_x_r", "knee_z_r", "knee_x_l", "knee_z_l"}
        ],
        "locked_zero_joint_names": ["knee_x_r", "knee_z_r", "knee_x_l", "knee_z_l"],
        "link_body_list": body_names,
        "foot_clearance_target_m": target_clearance,
        "foot_collision_geometry": foot_geometry,
        "behavior_selection": selection_record,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    pickled = {**motion, "fps": float(FPS), "link_body_list": body_names}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if existing:
        existing_metadata = json.loads(outputs["json"].read_text())
        if existing_metadata.get(
            "behavior_selection"
        ) != selection_record or existing_metadata.get("robot_xml_sha256") != sha256(
            model_path
        ):
            raise ValueError(
                f"Existing neutral output has different provenance: {prefix}"
            )
        with np.load(outputs["npz"], allow_pickle=False) as archive:
            loaded = {field: archive[field].copy() for field in archive.files}
        with outputs["pkl"].open("rb") as stream:
            existing_pkl = pickle.load(stream)
        if (
            list(loaded) != FIELD_ORDER
            or any(
                not np.array_equal(loaded[field], motion[field])
                for field in FIELD_ORDER
            )
            or not all(
                np.array_equal(np.asarray(existing_pkl[field]), motion[field])
                for field in FIELD_ORDER
            )
        ):
            raise ValueError(
                f"Existing neutral output differs from the MJCF pose: {prefix}"
            )
        metadata = existing_metadata
    else:
        np.savez_compressed(outputs["npz"], **motion)
        with outputs["pkl"].open("wb") as stream:
            pickle.dump(pickled, stream, protocol=pickle.HIGHEST_PROTOCOL)
        write_json(outputs["json"], metadata)

    with np.load(outputs["npz"], allow_pickle=False) as archive:
        loaded = {field: archive[field].copy() for field in archive.files}
    with outputs["pkl"].open("rb") as stream:
        pickled = pickle.load(stream)
    expected_shapes = {
        "fps": (),
        "root_pos": (frames, 3),
        "root_rot": (frames, 4),
        "dof_pos": (frames, len(columns)),
        "local_body_pos": (frames, len(body_names), 3),
        "link_body_list": (len(body_names),),
    }
    qpos_frames = np.column_stack(
        (
            loaded["root_pos"],
            loaded["root_rot"][:, [3, 0, 1, 2]],
            loaded["dof_pos"],
        )
    )
    fresh_local_positions = local_body_positions(model, qpos_frames, body_ids)
    checks = {
        "official_six_fields_and_shapes": (
            list(loaded) == FIELD_ORDER
            and all(
                loaded[field].shape == shape for field, shape in expected_shapes.items()
            )
            and all(not loaded[field].dtype.hasobject for field in FIELD_ORDER)
        ),
        "pkl_matches_npz": all(
            np.array_equal(np.asarray(pickled[field]), loaded[field])
            for field in FIELD_ORDER
        ),
        "local_body_positions_match_fresh_fk": bool(
            np.allclose(loaded["local_body_pos"], fresh_local_positions, atol=1e-10)
        ),
        "all_joint_coordinates_zero": bool(np.all(motion["dof_pos"] == 0)),
        "joint_limits_and_knee_locks_satisfied": all(
            (
                not model.jnt_limited[column["joint_id"]]
                or (
                    model.jnt_range[column["joint_id"], 0]
                    <= 0
                    <= model.jnt_range[column["joint_id"], 1]
                )
            )
            and (
                column["name"] not in {"knee_x_r", "knee_z_r", "knee_x_l", "knee_z_l"}
                or np.all(loaded["dof_pos"][:, column["column"]] == 0)
            )
            for column in columns
        ),
        "root_pose_constant": bool(
            np.all(motion["root_pos"] == motion["root_pos"][0])
            and np.all(motion["root_rot"] == motion["root_rot"][0])
        ),
        "root_is_upright_and_faces_positive_x": bool(
            np.allclose(motion["root_rot"][0], [0, 0, 0, 1])
        ),
        "both_feet_at_target_clearance": all(
            abs(value["minimum_corner_height_m"] - target_clearance) <= 1e-9
            for value in foot_geometry.values()
        ),
        "feet_share_fore_aft_center": abs(
            foot_geometry["l"]["center_xyz_m"][0]
            - foot_geometry["r"]["center_xyz_m"][0]
        )
        < 1e-9,
        "feet_are_parallel_and_upright": all(
            np.allclose(value["forward_axis_world"], [1, 0, 0], atol=1e-12)
            and np.allclose(value["up_axis_world"], [0, 0, 1], atol=1e-12)
            for value in foot_geometry.values()
        ),
        "cross_leg_clearance_positive": bool(
            min(
                mj.mj_geomDistance(
                    model,
                    data,
                    model.geom(left).id,
                    model.geom(right).id,
                    1.0,
                    None,
                )
                for left in (
                    "thigh_l_collision",
                    "shin_l_collision",
                    "foot_l_collision",
                )
                for right in (
                    "thigh_r_collision",
                    "shin_r_collision",
                    "foot_r_collision",
                )
            )
            >= 0
        ),
    }
    report = {
        "motion": str(outputs["npz"]),
        "metadata": str(outputs["json"]),
        "frames": frames,
        "fps": FPS,
        "sample_span_seconds": (frames - 1) / FPS,
        "root_height_m": root_height,
        "foot_geometry": foot_geometry,
        "joint_coordinate_max_abs_rad": float(np.max(np.abs(motion["dof_pos"]))),
        "checks": checks,
        "format_and_kinematic_constraints_pass": bool(all(checks.values())),
        "physical_validation": "Static FK and collision geometry only; no gravity/contact-force or balance simulation.",
    }
    write_json(validation_path, report)
    if not report["format_and_kinematic_constraints_pass"]:
        raise ValueError(f"Neutral stand validation failed: {validation_path}")
    print(f"Passed {name}: {frames} static frames at {FPS:g} Hz", flush=True)
    return {
        "name": name,
        "behavior": "idle",
        "fps": FPS,
        "frames": frames,
        "sample_span_seconds": (frames - 1) / FPS,
        "source": metadata["motion_origin"],
        "npz": str(outputs["npz"]),
        "validation": str(validation_path),
        "validation_pass": True,
        "warnings": [],
        "initial_heading_deg": 0.0,
        "heading_change_deg": 0.0,
        "net_displacement_xyz_m": [0.0, 0.0, 0.0],
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def build_clip(clip, selection_path, output_dir, model):
    name = clip["name"]
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Clip names must be single directory names")
    selection_record = {
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "clip": clip,
    }
    if clip.get("mode") == "neutral_model_pose":
        return build_neutral_stand(clip, selection_record, output_dir, model)
    source = resolve_recorded_path(clip["source_bvh"], strict=True)
    if sha256(source) != clip["source_sha256"]:
        raise ValueError(f"Source hash differs from selection: {source}")
    source_fps = read_bvh_fps(source)
    start, stop = clip["start_frame"], clip["stop_frame_exclusive"]
    if start < 0 or stop <= start + 1:
        raise ValueError(f"Invalid source interval: {name}")
    frames = math.floor((stop - start - 1) / source_fps * 120 + 1e-10) + 1
    config = CONFIGS_ROOT / f"skeleton17_collision/{source.stem}.json"
    prefix = output_dir / name / f"{name}_120hz"
    outputs = {ext: prefix.with_suffix(f".{ext}") for ext in ("npz", "pkl", "json")}
    parent = None
    if "parent_motion" in clip:
        parent = resolve_recorded_path(clip["parent_motion"], strict=True)
        if sha256(parent) != clip["parent_motion_sha256"]:
            raise ValueError(f"Parent motion hash differs from selection: {parent}")
    existing = any(path.exists() for path in outputs.values())
    if existing:
        if not all(path.is_file() for path in outputs.values()):
            raise FileExistsError(f"Incomplete existing export: {prefix}")
        metadata = json.loads(outputs["json"].read_text())
        recorded_selection = metadata.get("behavior_selection", {})
        if (
            recorded_selection.get("selection") != selection_record["selection"]
            or recorded_selection.get("clip") != clip
        ):
            raise ValueError(f"Existing output belongs to another selection: {prefix}")
    else:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        if parent:
            normalize_motion(parent, prefix)
        else:
            command = [
                sys.executable,
                "-m",
                "skeleton_retarget",
                "skeleton17",
                "retarget",
                "--bvh_file",
                str(source),
                "--robot_xml",
                str(model),
                "--ik_config",
                str(config),
                "--start_frame",
                str(start),
                "--max_frames",
                str(stop - start),
                "--output_fps",
                "120",
                "--speed",
                "1",
                "--align_heading",
                "--quiet",
                "--output",
                str(prefix),
            ]
            environment = os.environ.copy()
            environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            result = subprocess.run(
                command, cwd=REPO_ROOT, env=environment, capture_output=True, text=True
            )
            (prefix.parent / "retarget.log").write_text(
                result.stdout + "\n" + result.stderr
            )
            if result.returncode:
                raise RuntimeError(
                    f"Retarget failed; see {prefix.parent / 'retarget.log'}"
                )
        metadata = json.loads(outputs["json"].read_text())
        metadata["behavior_selection"] = selection_record
        write_json(outputs["json"], metadata)

    expected = {
        "source_bvh_sha256": sha256(source),
        "robot_xml_sha256": sha256(model),
        "ik_config_sha256": sha256(config),
        "start_frame": start,
        "stop_frame_exclusive": stop,
        "frames": frames,
        "fps": 120,
        "speed_factor": 1,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Export differs from requested {key}: {name}")
    heading = metadata.get("heading_normalization", {})
    if not heading.get("enabled"):
        raise ValueError(f"Missing heading normalization: {name}")
    reference = resolve_recorded_path(metadata["source_reference"], strict=True)
    report = validate(
        outputs["npz"],
        model,
        outputs["json"],
        config,
        reference,
        expected_frames=frames,
        expected_fps=120,
    )
    with np.load(outputs["npz"], allow_pickle=False) as motion:
        facing = Rotation.from_quat(motion["root_rot"]).apply([1.0, 0.0, 0.0])
        yaw = np.unwrap(np.arctan2(facing[:, 1], facing[:, 0]))
        positions = motion["root_pos"]
        report["checks"]["initial_facing_is_world_positive_x"] = bool(
            abs(yaw[0]) < 1e-8
        )
        report["checks"]["initial_root_xy_is_zero"] = bool(
            np.max(abs(positions[0, :2])) < 1e-8
        )
        if parent:
            with np.load(parent, allow_pickle=False) as original:
                for field in ("fps", "dof_pos", "local_body_pos", "link_body_list"):
                    report["checks"][f"normalization_preserves_{field}"] = bool(
                        np.array_equal(motion[field], original[field])
                    )
                old_facing = Rotation.from_quat(original["root_rot"]).apply(
                    [1.0, 0.0, 0.0]
                )
                old_yaw = np.unwrap(np.arctan2(old_facing[:, 1], old_facing[:, 0]))
                report["checks"]["normalization_preserves_relative_yaw"] = bool(
                    np.allclose(yaw - yaw[0], old_yaw - old_yaw[0], atol=1e-10, rtol=0)
                )
        if clip["behavior"] == "turn_left":
            report["checks"]["left_turn_positive_yaw"] = bool(
                yaw[-1] - yaw[0] > np.pi / 4
            )
        elif clip["behavior"] == "turn_right":
            report["checks"]["right_turn_negative_yaw"] = bool(
                yaw[-1] - yaw[0] < -np.pi / 4
            )
        elif clip["behavior"] in {"forward", "backward"}:
            direction = 1 if clip["behavior"] == "forward" else -1
            report["checks"]["travel_has_expected_x_sign"] = bool(
                direction * (positions[-1, 0] - positions[0, 0]) > 0.5
            )
        elif clip["behavior"] == "idle":
            report["checks"]["idle_both_foot_min_corner_heights_within_2cm"] = all(
                report["ground_geometry"]["per_geom"][f"foot_{side}_collision"][
                    "clearance_m"
                ]["max"]
                <= 0.02
                for side in "lr"
            )
            report["checks"]["idle_mean_horizontal_root_speed_below_0p1_m_s"] = (
                report["root_motion"]["horizontal_speed_m_s"]["mean"] < 0.1
            )
        statistics = {
            "initial_heading_deg": float(np.rad2deg(yaw[0])),
            "heading_change_deg": float(np.rad2deg(yaw[-1] - yaw[0])),
            "net_displacement_xyz_m": (positions[-1] - positions[0]).tolist(),
        }
    report["format_and_kinematic_constraints_pass"] = bool(
        all(report["checks"].values())
    )
    validation_path = prefix.with_name(prefix.name + "_validation.json")
    write_json(validation_path, report)
    if not report["format_and_kinematic_constraints_pass"]:
        failed = [key for key, passed in report["checks"].items() if not passed]
        raise ValueError(
            f"Validation failed for {name}: {failed}; see {validation_path}"
        )
    print(f"Passed {name}: {frames} frames, {(frames - 1) / 120:.3f} s", flush=True)
    return {
        "name": name,
        "behavior": clip["behavior"],
        "fps": 120,
        "frames": frames,
        "sample_span_seconds": (frames - 1) / 120,
        "source_interval": [start, stop],
        "source_bvh": str(source),
        "npz": str(outputs["npz"]),
        "validation": str(validation_path),
        "validation_pass": True,
        "warnings": report["warnings"],
        **statistics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    selection_path = args.selection.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    selection = json.loads(selection_path.read_text())
    clips = selection["clips"]
    names = [clip["name"] for clip in clips]
    if len(names) != len(set(names)):
        parser.error("Duplicate clip names")
    model = MODELS_ROOT / "skeleton17_collision/human_skeleton.xml"
    rows = [build_clip(clip, selection_path, output_dir, model) for clip in clips]
    write_json(
        output_dir / "summary.json",
        {
            "model": str(model),
            "selection": str(selection_path),
            "selection_sha256": sha256(selection_path),
            "clips": rows,
            "all_pass": True,
        },
    )
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
