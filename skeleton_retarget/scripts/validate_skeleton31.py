"""Independently check a six-field motion against its MJCF without changing it.

Example:
    python -m skeleton_retarget.scripts.validate_skeleton31 \
        --motion data/outputs/skeleton31/walk1_subject1_120hz.npz \
        --robot_xml assets/skeleton31/walks/walk1_subject1/human_skeleton.xml \
        --source_reference data/outputs/skeleton31/references/walk1_subject1_source.npz

Geometry and inferred stance metrics describe kinematic playback, not dynamics.
The optional source reference contains world positions and wxyz rotations from
an independently verified source FK, with body_names, fps and optional absolute
start_frame arrays. Native-rate IK output is checked at the saved sample times.
"""

import argparse
import itertools
import json
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from skeleton_retarget.retargeting.heading import apply_heading_transform
from skeleton_retarget.retargeting.paths import resolve_recorded_path
from skeleton_retarget.retargeting.validation import (
    ArrayUnpickler,
    sha256,
    source_reference_at_motion_times,
    summary,
)

FIELDS = {"fps", "root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list"}


def foot_metrics(model, data, qpos, fps, threshold):
    report = {}
    clearances = []
    sign = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
    floor = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    mj.mj_forward(model, data)
    floor_z = float(data.geom_xpos[floor, 2]) if floor >= 0 else 0.0
    for side in ("r", "l"):
        geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, f"{side}_foot_box")
        if geom_id < 0:
            continue
        if model.geom_type[geom_id] != int(mj.mjtGeom.mjGEOM_BOX):
            raise ValueError("Foot geometry must be a box for corner-based checks")
        site_ids = [
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, f"{side}_foot_box-P{i}")
            for i in range(1, 5)
        ]
        site_ids = [site_id for site_id in site_ids if site_id >= 0]
        local_corners = sign * model.geom_size[geom_id]
        corners, centers, markers = [], [], []
        for frame in qpos:
            data.qpos[:] = frame
            mj.mj_forward(model, data)
            corners.append(
                local_corners @ data.geom_xmat[geom_id].reshape(3, 3).T
                + data.geom_xpos[geom_id]
            )
            centers.append(data.geom_xpos[geom_id].copy())
            markers.append(data.site_xpos[site_ids].copy())
        corners, centers, markers = map(np.asarray, (corners, centers, markers))
        clearance = corners[:, :, 2].min(axis=1) - floor_z
        clearances.append(clearance)
        stance = clearance <= threshold
        intervals = stance[:-1] & stance[1:]
        center_speed = np.linalg.norm(np.diff(centers[:, :2], axis=0), axis=1) * fps
        corner_speed = np.linalg.norm(np.diff(corners[:, :, :2], axis=0), axis=2) * fps
        low_corners = (corners[:-1, :, 2] <= floor_z + threshold) & (
            corners[1:, :, 2] <= floor_z + threshold
        )
        best_corner = np.where(low_corners, corner_speed, np.inf).min(axis=1)
        report[side] = {
            "box_name": f"{side}_foot_box",
            "box_min_z_m": float(corners[:, :, 2].min()),
            "box_sole_clearance_m": summary(clearance),
            "maximum_penetration_m": float(max(0.0, -clearance.min())),
            "penetrating_frames_gt_1mm": int((clearance < -0.001).sum()),
            "marker_names": [model.site(i).name for i in site_ids],
            "marker_min_z_m": float(markers[:, :, 2].min()) if site_ids else None,
            "candidate_stance_frames": int(stance.sum()),
            "candidate_stance_intervals": int(intervals.sum()),
            "stance_box_center_horizontal_speed_m_s": summary(center_speed[intervals]),
            "low_material_corner_horizontal_speed_m_s": summary(
                corner_speed[low_corners]
            ),
            "best_low_corner_horizontal_speed_m_s": summary(
                best_corner[np.isfinite(best_corner)]
            ),
        }
    return {
        "floor_z_m": floor_z,
        "candidate_stance_maximum_clearance_m": threshold,
        "support_foot_minimum_clearance_m": summary(np.min(clearances, axis=0))
        if clearances
        else summary([]),
        "stance_definition": "Box sole <= threshold above floor, including penetrating poses; consecutive frames for velocities.",
        "limitations": "No contact-force truth. Center motion includes rolling; low material corner velocities better indicate kinematic slip.",
        "feet": report,
    }


def source_tracking(reference_path, metadata, model, world_positions, world_rotations):
    """Rebuild calibrated foot targets directly from source FK and IK JSON."""
    names, positions, rotations = source_reference_at_motion_times(
        reference_path, metadata, len(world_positions)
    )
    config = json.loads(resolve_recorded_path(metadata["ik_config"]).read_text())
    # BVH supplies the actual height to GMR; the native MimicKit path does not.
    actual_height = metadata.get("human_height_m")
    height_ratio = (
        float(actual_height) / config["human_height_assumption"]
        if actual_height is not None
        else 1.0
    )
    if not np.isfinite(height_ratio) or height_ratio <= 0:
        raise ValueError("Invalid human height ratio in metadata")
    scale = {
        name: value * height_ratio
        for name, value in config["human_scale_table"].items()
    }
    aliases = {
        "Hips": "pelvis",
        "LeftFootMod": "left_foot",
        "RightFootMod": "right_foot",
    }

    def source_name(human):
        if human in names:
            return human
        alias = aliases.get(human)
        if alias in names:
            return alias
        raise ValueError(f"Source reference has no {human!r} node or native alias")

    root_name = source_name(config["human_root_name"])
    root = positions[:, names.index(root_name)]
    root_scaled = root * scale[config["human_root_name"]]
    # GMR subtracts config ground_height inside the rotated local offset, then
    # subtracts its runtime ground_offset along world Z after that rotation.
    runtime_ground_offset = float(metadata.get("ground_offset_m", 0.0))
    if not np.isfinite(runtime_ground_offset):
        raise ValueError("Invalid runtime ground offset in metadata")
    world_translation = np.asarray(
        metadata.get("world_translation_m", [0.0, 0.0, 0.0]), dtype=float
    )
    if world_translation.shape != (3,) or not np.isfinite(world_translation).all():
        raise ValueError("Invalid export world translation in metadata")
    results = {}
    resolved_names = {config["human_root_name"]: root_name}
    for side, human in (
        ("r", "RightFootMod"),
        ("l", "LeftFootMod"),
    ):
        body = f"calcn_{side}"
        entry = config["ik_match_table1"][body]
        if entry[0] != human:
            raise ValueError("Source foot aliases do not match IK configuration")
        native = source_name(human)
        resolved_names[human] = native
        index = names.index(native)
        target_rotation = Rotation.from_quat(
            rotations[:, index], scalar_first=True
        ) * Rotation.from_quat(entry[4], scalar_first=True)
        scaled = (positions[:, index] - root) * scale[human] + root_scaled
        offset = np.asarray(entry[3]) - np.array([0, 0, config["ground_height"]])
        target_position = scaled + target_rotation.apply(offset)
        target_position[:, 2] -= runtime_ground_offset
        target_position += world_translation
        target_position, target_rotation = apply_heading_transform(
            metadata, target_position, target_rotation
        )
        body_id = model.body(body).id
        position_error = np.linalg.norm(
            world_positions[:, body_id] - target_position, axis=1
        )
        angular_error = (
            target_rotation.inv()
            * Rotation.from_quat(world_rotations[:, body_id], scalar_first=True)
        ).magnitude()
        results[body] = {
            "position_error_m": summary(position_error),
            "orientation_error_radians": summary(angular_error),
        }
        alignment = metadata.get("ground_alignment", {})
        if alignment.get("enabled"):
            shifted_target = target_position.copy()
            shifted_target[:, 2] += np.asarray(alignment["root_z_translation_m"])
            results[body][
                "position_error_after_applying_export_ground_shift_to_target_m"
            ] = summary(
                np.linalg.norm(world_positions[:, body_id] - shifted_target, axis=1)
            )
    return {
        "source_reference": str(reference_path),
        "source_reference_sha256": sha256(reference_path),
        "source_nodes": resolved_names,
        "human_height_ratio": height_ratio,
        "runtime_ground_offset_m": runtime_ground_offset,
        "export_world_translation_m": world_translation.tolist(),
        "heading_normalization": metadata.get("heading_normalization"),
        "method": "Source FK positions are evaluated by linear interpolation and rotations by SLERP at recorded native source sample times; JSON height/segment scales and local offsets define diagnostic targets. Output poses were solved by IK at native rate and then interpolated, so this does not imply IK was rerun at the output rate. position_error_m includes export ground translation; the second position metric shifts the target by that recorded translation.",
        "feet": results,
    }


def validate(
    motion_path, xml_path, pkl_path, metadata_path, source_reference, threshold
):
    model = mj.MjModel.from_xml_path(str(xml_path))
    free = np.flatnonzero(model.jnt_type == int(mj.mjtJoint.mjJNT_FREE))
    if len(free) != 1 or model.jnt_qposadr[free[0]] != 0:
        raise ValueError("Expected exactly one floating joint at qpos[0:7]")
    with np.load(motion_path, allow_pickle=False) as archive:
        if set(archive.files) != FIELDS:
            raise ValueError("NPZ must have exactly the six GMR fields")
        motion = {field: archive[field].copy() for field in archive.files}
    frames = len(motion["root_pos"])
    names = motion["link_body_list"].tolist()
    body_ids = np.array([model.body(name).id for name in names])
    shapes = {
        "fps": (),
        "root_pos": (frames, 3),
        "root_rot": (frames, 4),
        "dof_pos": (frames, model.nq - 7),
        "local_body_pos": (frames, len(names), 3),
        "link_body_list": (len(names),),
    }
    for field, shape in shapes.items():
        if motion[field].shape != shape:
            raise ValueError(f"Unexpected {field} shape: {motion[field].shape}")
        if motion[field].dtype.hasobject:
            raise ValueError("Object arrays are prohibited")
        if field != "link_body_list" and not np.isfinite(motion[field]).all():
            raise ValueError(f"Nonfinite {field}")
    fps = float(motion["fps"])
    if fps <= 0 or frames < 2:
        raise ValueError("Expected positive FPS and at least two frames")
    qpos = np.column_stack(
        (motion["root_pos"], motion["root_rot"][:, [3, 0, 1, 2]], motion["dof_pos"])
    )
    checks = {"safe_npz_six_fields_finite_and_shapes": True}
    report = {
        "motion": str(motion_path),
        "motion_sha256": sha256(motion_path),
        "robot_xml": str(xml_path),
        "robot_xml_sha256": sha256(xml_path),
        "frames": frames,
        "fps": fps,
        "sample_span_seconds": (frames - 1) / fps,
        "field_shapes": {field: list(array.shape) for field, array in motion.items()},
        "checks": checks,
    }
    if pkl_path.exists():
        with pkl_path.open("rb") as stream:
            pickled = ArrayUnpickler(stream).load()
        matches = set(pickled) == FIELDS and all(
            np.array_equal(np.asarray(pickled[field]), motion[field])
            for field in FIELDS
        )
        checks["pkl_matches_npz_exactly"] = bool(matches)
        report["pkl"] = {
            "path": str(pkl_path),
            "sha256": sha256(pkl_path),
            "numpy_globals_only": True,
        }
    else:
        checks["pkl_matches_npz_exactly"] = False
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if metadata:
        checks["metadata_model_hash_matches"] = metadata.get(
            "robot_xml_sha256"
        ) == sha256(xml_path)
        checks["metadata_timing_matches"] = bool(
            metadata.get("frames") == frames and np.isclose(metadata.get("fps", 0), fps)
        )
        for field in ("source_bvh", "source_motion", "source_xml", "ik_config"):
            expected_hash = metadata.get(field + "_sha256")
            if expected_hash is not None:
                checks[f"metadata_{field}_hash_matches"] = (
                    sha256(resolve_recorded_path(metadata[field])) == expected_hash
                )
    resampling = metadata.get("resampling", {})
    if resampling.get("enabled"):
        output_times = np.asarray(
            resampling["output_sample_times_seconds"], dtype=float
        )
        source_times = np.asarray(
            resampling["source_sample_times_seconds"], dtype=float
        )
        if output_times.shape != (frames,) or source_times.shape != (frames,):
            raise ValueError("Resampling time arrays must match output frames")
        native_fps = float(resampling["input_fps"])
        speed = float(resampling.get("speed_factor", 1.0))
        input_frames = int(resampling["input_frames"])
        duration = (input_frames - 1) / native_fps
        checks["resampling_uniform_output_time_grid"] = bool(
            np.isfinite(output_times).all()
            and np.allclose(output_times, np.arange(frames) / fps, atol=1e-12, rtol=0)
        )
        checks["resampling_native_source_time_mapping"] = bool(
            np.isfinite(source_times).all()
            and np.allclose(source_times, output_times * speed, atol=1e-10, rtol=0)
            and source_times.min() >= -1e-10
            and source_times.max() <= duration + 1e-10
            and np.all(np.diff(source_times) > 0)
        )
        expected_frames = int(np.floor(duration / speed * fps + 1e-10)) + 1
        tail = duration - source_times[-1]
        checks["resampling_duration_and_frame_count"] = bool(
            frames == expected_frames == resampling["output_frames"]
            and input_frames == metadata.get("source_frames", input_frames)
            and np.isclose(fps, resampling["output_fps"])
            and np.isclose(native_fps, metadata.get("source_fps", native_fps))
            and np.isclose(duration, resampling["source_duration_seconds"])
            and np.isclose(output_times[-1], resampling["output_duration_seconds"])
            and np.isclose(tail, resampling["unused_source_tail_seconds"])
            and -1e-10 <= tail < speed / fps + 1e-10
            and resampling["extrapolates"] is False
        )
        report["resampling"] = {
            "input_frames": input_frames,
            "input_fps": native_fps,
            "output_frames": frames,
            "output_fps": fps,
            "speed_factor": speed,
            "source_duration_seconds": duration,
            "output_duration_seconds": float(output_times[-1]),
            "unused_source_tail_seconds": float(tail),
            "output_sample_interval_seconds": summary(np.diff(output_times)),
            "note": "Native-rate IK qpos interpolated to the output timeline; local body positions checked against fresh FK. This adds samples, not new captured observations or output-rate IK solutions.",
        }
    norm_error = abs(np.linalg.norm(motion["root_rot"], axis=1) - 1)
    checks["root_quaternion_unit_norm"] = bool(norm_error.max() < 1e-6)
    report["root_quaternion_norm_error"] = summary(norm_error)
    limited = []
    for jid in np.flatnonzero(model.jnt_limited):
        if model.jnt_type[jid] not in (
            int(mj.mjtJoint.mjJNT_HINGE),
            int(mj.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError("Validator expects scalar non-root joints")
        values = qpos[:, model.jnt_qposadr[jid]]
        low, high = model.jnt_range[jid]
        violation = np.maximum(np.maximum(low - values, values - high), 0)
        limited.append(
            {
                "joint": model.joint(jid).name,
                "max_violation": float(violation.max()),
                "range": [float(low), float(high)],
                "observed_min": float(values.min()),
                "observed_max": float(values.max()),
                "frames_within_1e_minus5_of_lower_limit": int(
                    (abs(values - low) < 1e-5).sum()
                ),
                "frames_within_1e_minus5_of_upper_limit": int(
                    (abs(values - high) < 1e-5).sum()
                ),
            }
        )
    checks["joint_limits"] = all(item["max_violation"] < 1e-6 for item in limited)
    report["joint_limits"] = {
        "max_violation": max(item["max_violation"] for item in limited),
        "per_joint": limited,
    }
    equalities = []
    for eid in np.flatnonzero(model.eq_active0):
        if model.eq_type[eid] != int(mj.mjtEq.mjEQ_JOINT):
            raise ValueError("Validator currently expects active joint equalities only")
        j1, j2 = model.eq_obj1id[eid], model.eq_obj2id[eid]
        i1 = model.jnt_qposadr[j1]
        x = (
            0
            if j2 < 0
            else qpos[:, model.jnt_qposadr[j2]] - model.qpos0[model.jnt_qposadr[j2]]
        )
        expected = model.qpos0[i1] + np.polynomial.polynomial.polyval(
            x, model.eq_data[eid, :5]
        )
        equalities.append(
            {
                "equality": model.equality(eid).name,
                "joint": model.joint(j1).name,
                "max_abs_residual": float(abs(qpos[:, i1] - expected).max()),
            }
        )
    checks["active_joint_equalities"] = all(
        item["max_abs_residual"] < 1e-8 for item in equalities
    )
    report["active_equalities"] = {"count": len(equalities), "per_equality": equalities}
    data = mj.MjData(model)
    world_positions, world_rotations, local_positions = [], [], []
    for frame in qpos:
        data.qpos[:] = frame
        mj.mj_forward(model, data)
        world_positions.append(data.xpos.copy())
        world_rotations.append(data.xquat.copy())
        data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        mj.mj_forward(model, data)
        local_positions.append(data.xpos[body_ids].copy())
    world_positions, world_rotations, local_positions = map(
        np.asarray, (world_positions, world_rotations, local_positions)
    )
    reconstructed = (
        np.einsum(
            "tij,tbj->tbi",
            Rotation.from_quat(motion["root_rot"]).as_matrix(),
            motion["local_body_pos"],
        )
        + motion["root_pos"][:, None]
    )
    local_error = np.linalg.norm(local_positions - motion["local_body_pos"], axis=-1)
    world_error = np.linalg.norm(reconstructed - world_positions[:, body_ids], axis=-1)
    checks["local_body_positions_match_fk"] = bool(local_error.max() < 1e-8)
    checks["world_reconstruction_matches_fk"] = bool(world_error.max() < 1e-8)
    report["local_body_fk_error_m"] = summary(local_error)
    report["world_body_reconstruction_error_m"] = summary(world_error)
    displacement = np.diff(motion["root_pos"], axis=0)
    report["root_motion"] = {
        "net_displacement_xyz_m": (
            motion["root_pos"][-1] - motion["root_pos"][0]
        ).tolist(),
        "horizontal_path_length_m": float(
            np.linalg.norm(displacement[:, :2], axis=1).sum()
        ),
        "horizontal_speed_m_s": summary(
            np.linalg.norm(displacement[:, :2], axis=1) * fps
        ),
        "vertical_speed_m_s": summary(abs(displacement[:, 2]) * fps),
    }
    report["ground_and_sliding"] = foot_metrics(model, data, qpos, fps, threshold)
    report["ground_and_sliding_strict_support"] = foot_metrics(
        model, data, qpos, fps, 0.003
    )
    alignment = metadata.get("ground_alignment", {})
    if alignment.get("enabled"):
        translation = np.asarray(alignment["root_z_translation_m"], dtype=float)
        if translation.shape != (frames,) or not np.isfinite(translation).all():
            raise ValueError("Invalid ground alignment translation metadata")
        report["ground_alignment_root_z_translation_m"] = summary(translation)
        clearance = report["ground_and_sliding"]["support_foot_minimum_clearance_m"]
        target_clearance = alignment["support_foot_box_clearance_m"]
        checks["recorded_ground_alignment_clearance"] = bool(
            abs(clearance["min"] - target_clearance) < 1e-8
            and abs(clearance["max"] - target_clearance) < 1e-8
        )
        before = qpos.copy()
        before[:, 2] -= translation
        report["before_recorded_ground_alignment"] = foot_metrics(
            model, data, before, fps, threshold
        )
    report["warnings"] = [
        f"{side} foot box penetrates floor by {foot['maximum_penetration_m']:.6f} m"
        for side, foot in report["ground_and_sliding"]["feet"].items()
        if foot["maximum_penetration_m"] > 0.001
    ]
    if alignment.get("enabled") and np.max(abs(translation)) > 0.05:
        report["warnings"].append(
            "Export ground alignment moves the root by more than 5 cm vertically; final positions differ from the original IK targets by that recorded translation."
        )
    for side, foot in report["ground_and_sliding_strict_support"]["feet"].items():
        speed = foot["low_material_corner_horizontal_speed_m_s"]["mean"]
        if speed is not None and speed > 0.05:
            report["warnings"].append(
                f"{side} foot material corners within 3 mm of floor have mean horizontal speed {speed:.6f} m/s; kinematic sliding remains."
            )
    if source_reference is not None:
        report["foot_task_tracking"] = source_tracking(
            source_reference, metadata, model, world_positions, world_rotations
        )
    report["format_and_kinematic_constraints_pass"] = all(checks.values())
    report["physical_validation"] = (
        "Not tested: inverse dynamics, actuation, support forces, balance and simulated tracking."
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--robot_xml", type=Path, required=True)
    parser.add_argument("--pkl", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--source_reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stance_height", type=float, default=0.02)
    args = parser.parse_args()
    if not np.isfinite(args.stance_height) or args.stance_height <= 0:
        parser.error("stance_height must be finite and positive")
    motion = args.motion.resolve(strict=True)
    report = validate(
        motion,
        args.robot_xml.resolve(strict=True),
        args.pkl or motion.with_suffix(".pkl"),
        args.metadata or motion.with_suffix(".json"),
        args.source_reference,
        args.stance_height,
    )
    output = args.output or motion.with_name(motion.stem + "_validation.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Saved {output}; format/constraints pass: {report['format_and_kinematic_constraints_pass']}"
    )
    for warning in report["warnings"]:
        print(warning)
    if not report["format_and_kinematic_constraints_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
