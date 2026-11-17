"""Check skeleton17 motion using fresh MuJoCo FK and recorded source poses.

This reads the six-field NPZ/PKL and never changes the model or motion. The only
written artifact is the requested JSON report. Ground clearance and self
penetration are checked; finite differences and source tracking are also
reported. No PhysX simulation or dynamic tracking runs.
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

FIELD_ORDER = [
    "fps",
    "root_pos",
    "root_rot",
    "dof_pos",
    "local_body_pos",
    "link_body_list",
]
LOCKED_JOINTS = {"knee_x_r", "knee_z_r", "knee_x_l", "knee_z_l"}
COLLISION_NAMES = {
    f"{part}_{side}_collision"
    for part in ("foot", "shin", "thigh")
    for side in ("l", "r")
}
BOX_SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))


def _finite_vector(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid finite {label}, expected shape {shape}")
    return result


def _model_layout(model):
    free = np.flatnonzero(model.jnt_type == int(mj.mjtJoint.mjJNT_FREE))
    if len(free) != 1 or model.jnt_qposadr[free[0]] != 0:
        raise ValueError("Expected one floating root occupying qpos[0:7]")
    root = int(model.jnt_bodyid[free[0]])
    body_ids = np.flatnonzero(model.body_rootid == root)
    columns = []
    for jid in np.argsort(model.jnt_qposadr):
        if jid == free[0]:
            continue
        kind = int(model.jnt_type[jid])
        if kind != int(mj.mjtJoint.mjJNT_HINGE):
            raise ValueError("skeleton17 internal joints must all be scalar hinges")
        columns.append(
            {
                "column": len(columns),
                "name": model.joint(int(jid)).name,
                "joint_id": int(jid),
                "qpos_index": int(model.jnt_qposadr[jid]),
                "type": "hinge",
                "unit": "radian",
                "locked_zero": model.joint(int(jid)).name in LOCKED_JOINTS,
            }
        )
    return root, body_ids, columns


def _time_checks(metadata, frames, fps, expected_frames, expected_fps):
    resampling = metadata.get("resampling", {})
    if not resampling.get("enabled"):
        raise ValueError("Expected native-rate IK with recorded output resampling")
    out_times = _finite_vector(
        resampling["output_sample_times_seconds"], (frames,), "output times"
    )
    source_times = _finite_vector(
        resampling["source_sample_times_seconds"], (frames,), "source times"
    )
    native_fps = float(resampling["input_fps"])
    input_frames = int(resampling["input_frames"])
    speed = float(resampling.get("speed_factor", metadata.get("speed_factor", 1)))
    if not all(np.isfinite(x) and x > 0 for x in (native_fps, speed)):
        raise ValueError("Invalid native FPS or speed")
    if input_frames < 2:
        raise ValueError("Expected at least two native IK frames")
    duration = (input_frames - 1) / native_fps
    tail = duration - source_times[-1]
    expected_count = int(np.floor(duration / speed * fps + 1e-10)) + 1
    checks = {
        "expected_120hz": bool(np.isclose(fps, expected_fps, atol=1e-12, rtol=0)),
        "expected_frame_count": frames == expected_frames,
        "metadata_frame_count_and_fps": bool(
            frames == metadata.get("frames")
            and np.isclose(fps, metadata.get("fps", -1), atol=1e-12, rtol=0)
        ),
        "uniform_output_time_grid": bool(
            np.allclose(out_times, np.arange(frames) / fps, atol=1e-12, rtol=0)
        ),
        "native_source_time_mapping_no_extrapolation": bool(
            np.allclose(source_times, out_times * speed, atol=1e-10, rtol=0)
            and np.all(np.diff(source_times) > 0)
            and source_times[0] >= -1e-10
            and source_times[-1] <= duration + 1e-10
            and -1e-10 <= tail < speed / fps + 1e-10
            and resampling.get("extrapolates") is False
        ),
        "resampling_frame_count_and_duration": bool(
            frames == expected_count == resampling.get("output_frames")
            and input_frames == metadata.get("source_frames", input_frames)
            and np.isclose(native_fps, metadata.get("source_fps", native_fps))
            and np.isclose(fps, resampling.get("output_fps", -1))
            and np.isclose(duration, resampling.get("source_duration_seconds", -1))
            and np.isclose(out_times[-1], resampling.get("output_duration_seconds", -1))
            and np.isclose(tail, resampling.get("unused_source_tail_seconds", -1))
            and np.isclose(
                out_times[-1], metadata.get("saved_interval_duration_seconds", -1)
            )
        ),
    }
    return checks, {
        "input_fps": native_fps,
        "input_frames": input_frames,
        "output_fps": fps,
        "output_frames": frames,
        "output_sample_span_seconds": float(out_times[-1]),
        "native_sample_span_seconds": float(duration),
        "unused_source_tail_seconds": float(tail),
        "speed_factor": speed,
        "sample_interval_seconds": summary(np.diff(out_times)),
        "note": "Native IK was interpolated; these checks do not imply IK at 120 Hz.",
    }


def _hash_checks(metadata, xml_path, config_path, reference_path):
    checks = {}
    records = {}
    for name, actual in (("robot_xml", xml_path), ("ik_config", config_path)):
        recorded_path = metadata.get(name)
        actual_hash = sha256(actual)
        checks[f"metadata_{name}_hash_matches"] = (
            metadata.get(name + "_sha256") == actual_hash
        )
        checks[f"metadata_{name}_path_matches"] = bool(
            isinstance(recorded_path, str)
            and resolve_recorded_path(recorded_path) == actual
        )
        records[name] = {"path": str(actual), "sha256": actual_hash}
    for name in ("source_bvh", "source_xml", "source_motion"):
        if metadata.get(name + "_sha256") is None:
            continue
        path = resolve_recorded_path(metadata[name])
        exists = path.is_file()
        actual_hash = sha256(path) if exists else None
        checks[f"metadata_{name}_hash_matches"] = bool(
            exists and actual_hash == metadata[name + "_sha256"]
        )
        records[name] = {"path": str(path), "sha256": actual_hash}
    records["source_reference"] = {
        "path": str(reference_path),
        "sha256": sha256(reference_path),
    }
    if "source_reference_sha256" in metadata:
        checks["metadata_source_reference_hash_matches"] = (
            metadata["source_reference_sha256"] == records["source_reference"]["sha256"]
        )
    calibration_present = "ik_calibration" in metadata
    if calibration_present:
        calibration_path = resolve_recorded_path(metadata["ik_calibration"], strict=True)
        calibration = json.loads(calibration_path.read_text())
        checks["metadata_ik_calibration_hash_matches"] = metadata.get(
            "ik_calibration_sha256"
        ) == sha256(calibration_path)
        for name, path in (
            ("robot_xml", xml_path),
            ("ik_config", config_path),
            ("source_bvh", resolve_recorded_path(metadata["source_bvh"], strict=True)),
        ):
            checks[f"calibration_{name}_hash_matches"] = calibration.get(
                name + "_sha256"
            ) == sha256(path)
        static = calibration.get("calibration_checks", {})
        checks["calibration_static_reference_aligned"] = bool(
            calibration.get("calibrated")
            and 0 <= static.get("max_position_error_m", float("inf")) < 1e-10
            and 0 <= static.get("max_orientation_error_rad", float("inf")) < 1e-10
        )
        records["ik_calibration"] = {
            "path": str(calibration_path),
            "sha256": sha256(calibration_path),
        }
    if "model_manifest" in metadata:
        manifest_path = resolve_recorded_path(metadata["model_manifest"], strict=True)
        manifest = json.loads(manifest_path.read_text())
        checks["metadata_model_manifest_hash_matches"] = metadata.get(
            "model_manifest_sha256"
        ) == sha256(manifest_path)
        checks["manifest_copied_xml_hash_matches"] = manifest.get(
            "copied_xml_sha256"
        ) == sha256(xml_path)
        # The shared model manifest describes its initial walk3 calibration.
        # Later BVHs have their own config + static calibration provenance.
        if calibration_present:
            initial_config = resolve_recorded_path(manifest["ik_config"])
            checks["manifest_initial_ik_config_hash_matches"] = bool(
                initial_config.is_file()
                and manifest.get("ik_config_sha256") == sha256(initial_config)
            )
        else:
            checks["manifest_ik_config_hash_matches"] = manifest.get(
                "ik_config_sha256"
            ) == sha256(config_path)
        source_xml = resolve_recorded_path(manifest["source_xml"])
        checks["manifest_original_xml_hash_matches"] = bool(
            source_xml.is_file()
            and sha256(source_xml) == manifest.get("source_xml_sha256")
        )
        dependencies = manifest.get("copied_dependencies", [])
        bad_dependencies = []
        for dependency in dependencies:
            copied = manifest_path.parent / dependency["relative_path"]
            source = resolve_recorded_path(dependency["source"], strict=False)
            if not (
                copied.is_file()
                and source.is_file()
                and sha256(copied) == dependency["copied_sha256"]
                and sha256(source) == dependency["source_sha256"]
                and dependency["source_sha256"] == dependency["copied_sha256"]
            ):
                bad_dependencies.append(dependency["relative_path"])
        checks["manifest_copied_asset_hashes_match_originals"] = bool(
            dependencies and not bad_dependencies
        )
        records["model_manifest"] = {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "copied_dependencies_checked": len(dependencies),
            "mismatching_dependencies": bad_dependencies,
        }
    return checks, records


def _joint_diagnostics(model, qpos, columns, fps, metadata=None):
    checks = {}
    limits = []
    rates = np.diff(qpos[:, 7:], axis=0) * fps
    acceleration = np.diff(rates, axis=0) * fps
    for column in columns:
        jid, index = column["joint_id"], column["column"]
        values = qpos[:, column["qpos_index"]]
        entry = {
            "joint": column["name"],
            "q_radians": summary(values),
            "abs_dq_rad_s": summary(abs(rates[:, index])),
            "abs_ddq_rad_s2": summary(abs(acceleration[:, index])),
            "limited": bool(model.jnt_limited[jid]),
        }
        if model.jnt_limited[jid]:
            low, high = model.jnt_range[jid]
            violation = np.maximum(np.maximum(low - values, values - high), 0)
            entry.update(
                range_radians=[float(low), float(high)],
                max_limit_violation_radians=float(violation.max()),
                fraction_within_1_degree_of_lower_limit=float(
                    np.mean(abs(values - low) <= np.deg2rad(1))
                ),
                fraction_within_1_degree_of_upper_limit=float(
                    np.mean(abs(values - high) <= np.deg2rad(1))
                ),
            )
        limits.append(entry)
    checks["scalar_joint_limits"] = all(
        entry.get("max_limit_violation_radians", 0) < 1e-6 for entry in limits
    )
    equalities = []
    for eid in np.flatnonzero(model.eq_active0):
        if model.eq_type[eid] != int(mj.mjtEq.mjEQ_JOINT):
            raise ValueError("Expected only active scalar joint equalities")
        jid = int(model.eq_obj1id[eid])
        if model.eq_obj2id[eid] >= 0 or np.any(model.eq_data[eid, :5]):
            raise ValueError("Expected independent zero-position equality locks")
        values = qpos[:, model.jnt_qposadr[jid]]
        equalities.append(
            {
                "equality": model.equality(int(eid)).name,
                "joint": model.joint(jid).name,
                "max_abs_q_radians": float(abs(values).max()),
            }
        )
    checks["exactly_four_expected_zero_knee_locks"] = bool(
        len(equalities) == 4
        and {entry["joint"] for entry in equalities} == LOCKED_JOINTS
        and all(entry["max_abs_q_radians"] < 1e-8 for entry in equalities)
    )
    velocities = np.empty((len(qpos) - 1, model.nv))
    for index in range(len(velocities)):
        mj.mj_differentiatePos(
            model, velocities[index], 1 / fps, qpos[index], qpos[index + 1]
        )
    report = {
        "per_joint": limits,
        "active_equalities": equalities,
        "generalized_root_linear_speed_m_s": summary(
            np.linalg.norm(velocities[:, :3], axis=1)
        ),
        "generalized_root_angular_speed_rad_s": summary(
            np.linalg.norm(velocities[:, 3:6], axis=1)
        ),
        "all_scalar_joint_abs_speed_rad_s": summary(abs(rates)),
        "all_scalar_joint_abs_acceleration_rad_s2": summary(abs(acceleration)),
        "note": "Finite differences of saved poses. A recorded scalar joint speed limit is checked when present; torque and balance feasibility are not tested.",
    }
    correction = (metadata or {}).get("resampling_collision_correction", {})
    speed_limit = correction.get("max_joint_speed_rad_s")
    if speed_limit is not None:
        speed_limit = float(speed_limit)
        if not np.isfinite(speed_limit) or speed_limit <= 0:
            raise ValueError("Invalid maximum scalar joint speed")
        tolerance = 1e-7
        violation = np.maximum(abs(rates) - speed_limit, 0)
        checks["scalar_joint_speed_within_recorded_limit"] = bool(
            np.all(violation <= tolerance)
        )
        max_interval, max_column = np.unravel_index(abs(rates).argmax(), rates.shape)
        report["recorded_speed_limit"] = {
            "max_joint_speed_rad_s": speed_limit,
            "tolerance_rad_s": tolerance,
            "speed_limit_violation_rad_s": summary(violation),
            "intervals_with_any_speed_violation": int(
                np.sum(np.any(violation > tolerance, axis=1))
            ),
            "largest_abs_speed_interval": {
                "start_frame": int(max_interval),
                "end_frame": int(max_interval + 1),
                "joint": columns[max_column]["name"],
                "signed_speed_rad_s": float(rates[max_interval, max_column]),
            },
            "per_joint": [
                {
                    "joint": column["name"],
                    "fraction_of_intervals_within_0_001_rad_s_of_speed_limit": float(
                        np.mean(abs(abs(rates[:, index]) - speed_limit) <= 0.001)
                    ),
                }
                for index, column in enumerate(columns)
            ],
        }
    return checks, report


def _sampled_cross_leg_clearance(model, qpos):
    """Query geometry directly at saved poses and three interior interval samples."""
    pairs = [
        (f"{left}_l_collision", f"{right}_r_collision")
        for left, right in itertools.product(("foot", "shin", "thigh"), repeat=2)
    ]
    pair_ids = [(model.geom(left).id, model.geom(right).id) for left, right in pairs]
    data = mj.MjData(model)
    fractions = (0.25, 0.5, 0.75)
    distance_cap = 1.0

    def query(frame):
        data.qpos[:] = frame
        mj.mj_forward(model, data)
        return [
            float(mj.mj_geomDistance(model, data, left, right, distance_cap, None))
            for left, right in pair_ids
        ]

    saved = np.asarray([query(frame) for frame in qpos])
    interior = np.empty((len(qpos) - 1, len(fractions), len(pairs)))
    tangent = np.empty(model.nv)
    for index in range(len(qpos) - 1):
        mj.mj_differentiatePos(model, tangent, 1.0, qpos[index], qpos[index + 1])
        for fraction_index, fraction in enumerate(fractions):
            sample = qpos[index].copy()
            mj.mj_integratePos(model, sample, tangent, fraction)
            interior[index, fraction_index] = query(sample)
    saved_minimum = float(saved.min())
    interior_minimum = float(interior.min())
    if saved_minimum <= interior_minimum:
        frame_index, pair_index = np.unravel_index(saved.argmin(), saved.shape)
        location = {
            "saved_frame": int(frame_index),
            "geoms": list(pairs[pair_index]),
        }
    else:
        interval, fraction_index, pair_index = np.unravel_index(
            interior.argmin(), interior.shape
        )
        location = {
            "interval_start_frame": int(interval),
            "interval_end_frame": int(interval + 1),
            "fraction": fractions[fraction_index],
            "geoms": list(pairs[pair_index]),
        }
    threshold, tolerance = 0.001, 1e-8
    checks = {
        "saved_cross_leg_geometry_clearance_at_least_1mm": bool(
            saved_minimum >= threshold - tolerance
        ),
        "sampled_interpolation_cross_leg_clearance_at_least_1mm": bool(
            interior_minimum >= threshold - tolerance
        ),
    }
    interval_minimum = np.minimum(
        interior.min(axis=(1, 2)),
        np.minimum(saved[:-1].min(axis=1), saved[1:].min(axis=1)),
    )
    return checks, {
        "method": "Independent mj_differentiatePos/mj_integratePos interpolation and direct mj_geomDistance queries for all nine cross-leg pairs",
        "saved_frame_count": len(qpos),
        "interval_count": len(qpos) - 1,
        "interior_sample_fractions": list(fractions),
        "evaluated_configuration_count": len(saved) + len(interior) * len(fractions),
        "geometry_distance_query_count": int(saved.size + interior.size),
        "distance_query_cap_m": distance_cap,
        "required_clearance_m": threshold,
        "clearance_tolerance_m": tolerance,
        "minimum_clearance_m": min(saved_minimum, interior_minimum),
        "minimum_location": location,
        "saved_configuration_minimum_clearance_m": saved_minimum,
        "interior_sample_minimum_clearance_m": interior_minimum,
        "interval_sampled_minimum_clearance_m": summary(interval_minimum),
        "interval_sampled_minimum_clearance_per_interval_m": interval_minimum.tolist(),
        "per_pair": [
            {
                "geoms": list(pair),
                "saved_configuration_distance_m": summary(saved[:, index]),
                "interior_sample_distance_m": summary(interior[:, :, index]),
            }
            for index, pair in enumerate(pairs)
        ],
        "note": "Only saved configurations and the three stated interior fractions are checked. This is not continuous swept-volume collision detection; other times, other geometry and dynamics are not established.",
    }


def _ground_and_contacts(model, qpos, root, threshold):
    robot_geoms = np.flatnonzero(model.body_rootid[model.geom_bodyid] == root)
    explicit = set(model.pair_geom1.tolist()) | set(model.pair_geom2.tolist())
    active = [
        int(gid)
        for gid in robot_geoms
        if model.geom_contype[gid] or model.geom_conaffinity[gid] or gid in explicit
    ]
    names = {model.geom(gid).name for gid in active}
    checks = {"exactly_six_active_robot_collision_primitives": names == COLLISION_NAMES}
    if not all(
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name) >= 0
        for name in COLLISION_NAMES
    ):
        raise ValueError("Missing skeleton17 foot/shin/thigh collision geometry")
    data = mj.MjData(model)
    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    clearances = {name: [] for name in sorted(COLLISION_NAMES)}
    foot_corners = {side: [] for side in ("l", "r")}
    foot_centers = {side: [] for side in ("l", "r")}
    pairs = {}
    self_frames = set()
    penetrating_frames = set()
    external_count = 0
    for index, frame in enumerate(qpos):
        data.qpos[:] = frame
        mj.mj_forward(model, data)
        floor_z = float(data.geom_xpos[floor_id, 2]) if floor_id >= 0 else 0.0
        if floor_id >= 0:
            if model.geom_type[floor_id] != int(
                mj.mjtGeom.mjGEOM_PLANE
            ) or not np.allclose(
                data.geom_xmat[floor_id].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-8
            ):
                raise ValueError("Ground checks expect a horizontal Z-up floor plane")
        for name in clearances:
            gid = model.geom(name).id
            rotation = data.geom_xmat[gid].reshape(3, 3)
            center = data.geom_xpos[gid]
            kind = model.geom_type[gid]
            if kind == int(mj.mjtGeom.mjGEOM_BOX):
                corners = (BOX_SIGNS * model.geom_size[gid]) @ rotation.T + center
                low = corners[:, 2].min()
                side = name.split("_")[1]
                foot_corners[side].append(corners.copy())
                foot_centers[side].append(center.copy())
            elif kind == int(mj.mjtGeom.mjGEOM_CAPSULE):
                radius, half_length = model.geom_size[gid, :2]
                low = center[2] - abs(rotation[2, 2]) * half_length - radius
            else:
                raise ValueError(f"Unsupported collision geometry type: {name}")
            clearances[name].append(float(low - floor_z))
        for contact in data.contact[: data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = model.geom_bodyid[[g1, g2]]
            if model.body_rootid[b1] != root or model.body_rootid[b2] != root:
                external_count += 1
                continue
            pair = tuple(sorted((model.geom(g1).name, model.geom(g2).name)))
            item = pairs.setdefault(pair, {"frames": set(), "distances": []})
            item["frames"].add(index)
            item["distances"].append(float(contact.dist))
            self_frames.add(index)
            if contact.dist < -1e-6:
                penetrating_frames.add(index)
    geometry = {
        name: {
            "active_in_mujoco_collision_filter": name in names,
            "clearance_m": summary(values),
            "frames_below_ground_gt_1mm": int(np.sum(np.asarray(values) < -0.001)),
        }
        for name, values in clearances.items()
    }
    support = np.minimum(clearances["foot_l_collision"], clearances["foot_r_collision"])
    checks["all_six_collision_primitives_above_ground"] = all(
        np.min(values) >= -1e-6 for values in clearances.values()
    )
    contact_report = {
        "contact_detection_disabled": bool(
            int(model.opt.disableflags) & int(mj.mjtDisableBit.mjDSBL_CONTACT)
        ),
        "self_contact_frame_count": len(self_frames),
        "self_penetration_gt_1um_frame_count": len(penetrating_frames),
        "external_contact_point_count": external_count,
        "pairs": [
            {
                "geoms": list(pair),
                "frame_count": len(item["frames"]),
                "frame_indices": sorted(item["frames"]),
                "signed_distance_m": summary(item["distances"]),
            }
            for pair, item in sorted(pairs.items())
        ],
        "note": "Fresh mj_forward contacts with actual MJCF masks, excludes and parent/weld filters. Positive distances can be margin contacts; this does not test the edited USD or contact forces.",
    }
    checks["mujoco_contact_detection_enabled"] = not contact_report[
        "contact_detection_disabled"
    ]
    checks["self_collision_no_penetration"] = not penetrating_frames
    return (
        checks,
        {
            "floor_z_m": floor_z,
            "floor_source": "MJCF floor plane"
            if floor_id >= 0
            else "virtual Z=0 plane; no floor geom in MJCF",
            "per_geom": geometry,
            "support_foot_minimum_clearance_m": summary(support),
            "support_foot_minimum_clearance_per_frame_m": support.tolist(),
            "stance_height_m": threshold,
        },
        contact_report,
        foot_corners,
        foot_centers,
    )


def _alignment_checks(metadata, root_z, support_clearance, fps):
    """Independently check saved root heights against the actual foot geometry."""
    frames = len(root_z)
    alignment = metadata.get("ground_alignment", {})
    shifts = _finite_vector(
        alignment.get("root_z_translation_m"), (frames,), "root Z shifts"
    )
    target = float(alignment.get("support_foot_box_clearance_m", 0.001))
    if not np.isfinite(target):
        raise ValueError("Invalid support foot clearance")
    mode = alignment.get("clearance_mode", "exact")
    if mode not in {"exact", "minimum"}:
        raise ValueError(f"Unsupported ground clearance mode: {mode}")
    actual = _finite_vector(support_clearance, (frames,), "support foot clearance")
    excess = actual - target
    tolerance = 1e-8
    clearance_ok = (
        np.all(abs(excess) < tolerance)
        if mode == "exact"
        else np.all(excess >= -tolerance)
    )
    checks = {
        "recorded_1mm_support_clearance": bool(
            alignment.get("enabled")
            and np.isclose(target, 0.001, atol=1e-12, rtol=0)
            and clearance_ok
        )
    }
    raw_required = root_z + target - actual
    extra = root_z - raw_required
    report = {
        "clearance_mode": mode,
        "target_support_foot_clearance_m": target,
        "clearance_tolerance_m": tolerance,
        "support_foot_excess_clearance_m": summary(excess),
        "support_foot_excess_clearance_per_frame_m": excess.tolist(),
        "frames_above_target_clearance": int(np.sum(excess > tolerance)),
        "independently_required_root_z_m": summary(raw_required),
        "independently_extra_root_height_m": summary(extra),
        "note": "Minimum clearance permits elevated feet; satisfying it does not establish a support contact.",
    }
    above_threshold = []
    for threshold in (target + tolerance, 0.003, 0.01, 0.02):
        mask = actual > threshold
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
        above_threshold.append(
            {
                "clearance_threshold_m": threshold,
                "both_feet_above_threshold_frame_count": int(mask.sum()),
                "sample_occupancy_seconds": float(mask.sum() / fps),
                "contiguous_sample_runs": [
                    {
                        "start_frame": int(start),
                        "end_frame_inclusive": int(end),
                        "sample_count": int(end - start + 1),
                        "first_to_last_sample_span_seconds": float((end - start) / fps),
                    }
                    for start, end in zip(starts, ends, strict=True)
                ],
            }
        )
    report["both_feet_clearance_above_threshold"] = above_threshold
    if "raw_required_root_z_m" in alignment:
        recorded_raw = _finite_vector(
            alignment["raw_required_root_z_m"], (frames,), "raw required root Z"
        )
        error = abs(recorded_raw - raw_required)
        checks["recorded_required_root_height_matches_fresh_geometry"] = bool(
            error.max() < tolerance
        )
        report["recorded_required_root_height_error_m"] = summary(error)
    if "extra_root_height_m" in alignment:
        recorded_extra = _finite_vector(
            alignment["extra_root_height_m"], (frames,), "extra root height"
        )
        error = abs(recorded_extra - extra)
        checks["recorded_extra_root_height_matches_fresh_geometry"] = bool(
            error.max() < tolerance
        )
        report["recorded_extra_root_height_error_m"] = summary(error)
    limit = metadata.get(
        "max_root_vertical_speed_m_s", alignment.get("max_root_vertical_speed_m_s")
    )
    if limit is not None:
        limit = float(limit)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("Invalid maximum root vertical speed")
        speed = abs(np.diff(root_z)) * fps
        checks["root_vertical_speed_within_recorded_limit"] = bool(
            np.all(speed <= limit + tolerance)
        )
        report.update(
            max_root_vertical_speed_m_s=limit,
            actual_abs_root_vertical_speed_m_s=summary(speed),
            root_vertical_speed_limit_violation_m_s=summary(
                np.maximum(speed - limit, 0)
            ),
        )
        if mode == "minimum" and "raw_required_root_z_m" in alignment:
            # Direct all-pairs definition is independent of the exporter's
            # forward/backward envelope construction.
            distance = abs(np.arange(frames)[:, None] - np.arange(frames)[None, :])
            envelope = np.max(raw_required[None, :] - distance * limit / fps, axis=1)
            error = abs(root_z - envelope)
            checks["root_height_is_minimum_speed_bounded_upper_envelope"] = bool(
                error.max() < tolerance
            )
            report["minimum_upper_envelope_error_m"] = summary(error)
    return checks, report, shifts


def _sliding(corners, centers, fps, floor_z, threshold):
    result = {}
    for side in ("l", "r"):
        points = np.asarray(corners[side])
        center = np.asarray(centers[side])
        low = points[:, :, 2] <= floor_z + threshold
        near_intervals = low[:-1] & low[1:]
        speed = np.linalg.norm(np.diff(points[:, :, :2], axis=0), axis=2) * fps
        center_speed = np.linalg.norm(np.diff(center[:, :2], axis=0), axis=1) * fps
        stance = low.any(axis=1)
        result[side] = {
            "candidate_stance_frame_count": int(stance.sum()),
            "near_floor_material_corner_speed_m_s": summary(speed[near_intervals]),
            "stance_box_center_horizontal_speed_m_s": summary(
                center_speed[stance[:-1] & stance[1:]]
            ),
        }
    return {
        "maximum_corner_height_above_ground_m": threshold,
        "feet": result,
        "note": "Consecutive near-floor material corners estimate sliding; rolling and contact switching also affect the values. No measured stance labels or contact-force truth.",
    }


def _source_tracking(reference_path, metadata, config, model, positions, rotations):
    names, source_pos, source_rot = source_reference_at_motion_times(
        reference_path, metadata, len(positions)
    )
    root_name = config["human_root_name"]
    root_positions = source_pos[:, names.index(root_name)]
    ratio = float(
        metadata.get("human_height_m", config["human_height_assumption"])
    ) / float(config["human_height_assumption"])
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Invalid human height ratio")
    scales = {
        name: float(value) * ratio
        for name, value in config["human_scale_table"].items()
    }
    translation = _finite_vector(
        metadata.get("world_translation_m", [0, 0, 0]), (3,), "world translation"
    )
    alignment = metadata.get("ground_alignment", {})
    shifts = (
        _finite_vector(
            alignment["root_z_translation_m"], (len(positions),), "root Z shifts"
        )
        if alignment.get("enabled")
        else np.zeros(len(positions))
    )
    runtime_offset = float(metadata.get("ground_offset_m", 0))
    if not np.isfinite(runtime_offset):
        raise ValueError("Invalid runtime ground offset")
    table = config["ik_match_table1"]
    other = config.get("ik_match_table2", {})
    offsets_match = all(
        body in table
        and entry[0] == table[body][0]
        and np.allclose(entry[3], table[body][3], atol=1e-12, rtol=0)
        and np.allclose(entry[4], table[body][4], atol=1e-12, rtol=0)
        for body, entry in other.items()
    )
    tasks = {}
    for body, entry in table.items():
        human, pos_cost, rot_cost, pos_offset, rot_offset = entry
        index, body_id = names.index(human), model.body(body).id
        target_rotation = Rotation.from_quat(
            source_rot[:, index], scalar_first=True
        ) * Rotation.from_quat(rot_offset, scalar_first=True)
        target_position = scales[root_name] * root_positions + scales[human] * (
            source_pos[:, index] - root_positions
        )
        target_position += target_rotation.apply(
            np.asarray(pos_offset) - [0, 0, config.get("ground_height", 0)]
        )
        target_position[:, 2] -= runtime_offset
        target_position += translation
        target_position, target_rotation = apply_heading_transform(
            metadata, target_position, target_rotation
        )
        shifted_target = target_position.copy()
        shifted_target[:, 2] += shifts
        angle_error = (
            target_rotation.inv()
            * Rotation.from_quat(rotations[:, body_id], scalar_first=True)
        ).magnitude()
        tasks[body] = {
            "source_body": human,
            "stage1_position_cost": pos_cost,
            "stage1_orientation_cost": rot_cost,
            "stage2_position_cost": other.get(body, [None, 0, 0])[1],
            "stage2_orientation_cost": other.get(body, [None, 0, 0])[2],
            "position_error_to_original_calibrated_target_m": summary(
                np.linalg.norm(positions[:, body_id] - target_position, axis=1)
            ),
            "position_error_after_same_export_root_z_shift_applied_to_target_m": summary(
                np.linalg.norm(positions[:, body_id] - shifted_target, axis=1)
            ),
            "orientation_error_degrees": summary(np.rad2deg(angle_error)),
        }
    return {"both_ik_stages_have_same_target_offsets": bool(offsets_match)}, {
        "task_count": len(tasks),
        "human_height_ratio": ratio,
        "world_translation_m": translation.tolist(),
        "heading_normalization": metadata.get("heading_normalization"),
        "ground_alignment_root_z_translation_m": summary(shifts),
        "per_body": tasks,
        "note": "Source FK is interpolated at recorded native source times; scale and local offsets reconstruct GMR targets independently. The original-target error includes export root-Z changes; the second position metric removes that common shift. Nonzero task errors are diagnostics, not a proof of dynamics or a format-check failure.",
    }


def validate(
    motion_path,
    xml_path,
    metadata_path,
    config_path,
    reference_path,
    *,
    expected_frames=340,
    expected_fps=120.0,
    stance_height=0.02,
):
    """Return format/constraint checks and quality diagnostics without mutation."""
    model = mj.MjModel.from_xml_path(str(xml_path))
    root, expected_body_ids, columns = _model_layout(model)
    metadata = json.loads(metadata_path.read_text())
    config = json.loads(config_path.read_text())
    with np.load(motion_path, allow_pickle=False) as archive:
        if set(archive.files) != set(FIELD_ORDER):
            raise ValueError("Expected exactly the six GMR fields")
        motion = {field: archive[field].copy() for field in archive.files}
        field_order = list(archive.files)
    frames = len(motion["root_pos"])
    if frames < 2:
        raise ValueError("Expected at least two output frames")
    names = motion["link_body_list"].tolist()
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("link_body_list must be a Unicode string array")
    body_ids = np.array([model.body(name).id for name in names])
    shapes = {
        "fps": (),
        "root_pos": (frames, 3),
        "root_rot": (frames, 4),
        "dof_pos": (frames, len(columns)),
        "local_body_pos": (frames, len(names), 3),
        "link_body_list": (len(names),),
    }
    for field, shape in shapes.items():
        values = motion[field]
        if values.shape != shape or values.dtype.hasobject:
            raise ValueError(f"Invalid {field} shape or object dtype")
        if field != "link_body_list" and not np.isfinite(values).all():
            raise ValueError(f"Nonfinite {field}")
    fps = float(motion["fps"])
    if fps <= 0:
        raise ValueError("FPS must be positive")
    qpos = np.column_stack(
        (motion["root_pos"], motion["root_rot"][:, [3, 0, 1, 2]], motion["dof_pos"])
    )
    checks = {
        "safe_finite_npz_shapes": True,
        "six_fields_in_official_order": field_order == FIELD_ORDER,
        "expected_21_hinges_85_bodies": len(columns) == 21
        and len(expected_body_ids) == 85,
        "body_names_match_entire_floating_subtree_in_model_order": np.array_equal(
            body_ids, expected_body_ids
        ),
        "metadata_link_body_list_matches": metadata.get("link_body_list") == names,
        "metadata_joint_columns_match_model": metadata.get("joint_columns") == columns,
        "metadata_declares_xyzw_and_meters": metadata.get("root_rotation_order")
        == "xyzw"
        and metadata.get("position_unit") == "meter",
    }
    pkl_path = motion_path.with_suffix(".pkl")
    if pkl_path.exists():
        with pkl_path.open("rb") as stream:
            pickled = ArrayUnpickler(stream).load()
        checks["pkl_matches_npz_exactly"] = bool(
            isinstance(pickled, dict)
            and list(pickled) == FIELD_ORDER
            and all(
                np.array_equal(np.asarray(pickled[field]), motion[field])
                for field in FIELD_ORDER
            )
        )
    else:
        checks["pkl_matches_npz_exactly"] = False
    time_checks, timing = _time_checks(
        metadata, frames, fps, expected_frames, expected_fps
    )
    checks.update(time_checks)
    hash_checks, hashes = _hash_checks(metadata, xml_path, config_path, reference_path)
    checks.update(hash_checks)
    norm_error = abs(np.linalg.norm(motion["root_rot"], axis=1) - 1)
    adjacent_dot = np.sum(motion["root_rot"][:-1] * motion["root_rot"][1:], axis=1)
    checks["root_quaternion_unit_norm"] = bool(norm_error.max() < 1e-6)
    checks["root_quaternion_sign_continuity"] = bool(adjacent_dot.min() >= -1e-12)
    joint_checks, joints = _joint_diagnostics(model, qpos, columns, fps, metadata)
    checks.update(joint_checks)
    sampled_checks, sampled_clearance = _sampled_cross_leg_clearance(model, qpos)
    checks.update(sampled_checks)
    data = mj.MjData(model)
    positions, rotations, local_positions = [], [], []
    for frame in qpos:
        data.qpos[:] = frame
        mj.mj_forward(model, data)
        positions.append(data.xpos.copy())
        rotations.append(data.xquat.copy())
        data.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
        mj.mj_forward(model, data)
        local_positions.append(data.xpos[body_ids].copy())
    positions, rotations, local_positions = map(
        np.asarray, (positions, rotations, local_positions)
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
    world_error = np.linalg.norm(reconstructed - positions[:, body_ids], axis=-1)
    checks["local_body_positions_match_fresh_fk"] = bool(local_error.max() < 1e-8)
    checks["world_reconstruction_matches_fresh_fk"] = bool(world_error.max() < 1e-8)
    ground_checks, ground, contacts, corners, centers = _ground_and_contacts(
        model, qpos, root, stance_height
    )
    checks.update(ground_checks)
    alignment_checks, alignment_report, shifts = _alignment_checks(
        metadata,
        motion["root_pos"][:, 2],
        ground["support_foot_minimum_clearance_per_frame_m"],
        fps,
    )
    checks.update(alignment_checks)
    tracking_checks, tracking = _source_tracking(
        reference_path, metadata, config, model, positions, rotations
    )
    checks.update(tracking_checks)
    warnings = []
    if contacts["self_penetration_gt_1um_frame_count"]:
        warnings.append(
            "MuJoCo detects robot self-penetration; inspect the contact pairs and frames."
        )
    if contacts["contact_detection_disabled"]:
        warnings.append(
            "MuJoCo contact detection is disabled; self-contact diagnostics are incomplete."
        )
    if abs(shifts).max() > 0.05:
        warnings.append(
            "Export ground alignment changes root Z by more than 5 cm; use the original-target error to assess this displacement."
        )
    strict_slide = _sliding(corners, centers, fps, ground["floor_z_m"], 0.003)
    for side, foot in strict_slide["feet"].items():
        mean = foot["near_floor_material_corner_speed_m_s"]["mean"]
        if mean is not None and mean > 0.05:
            warnings.append(
                f"{side} near-floor material corners move at mean {mean:.4f} m/s; kinematic slip remains."
            )
    displacement = np.diff(motion["root_pos"], axis=0)
    return {
        "motion": str(motion_path),
        "motion_sha256": sha256(motion_path),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "inputs": hashes,
        "model": {
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "internal_hinges": len(columns),
            "floating_subtree_bodies": len(expected_body_ids),
            "root_body": model.body(root).name,
            "mujoco_version": mj.__version__,
        },
        "frames": frames,
        "fps": fps,
        "sample_span_seconds": (frames - 1) / fps,
        "field_order": field_order,
        "field_shapes": {field: list(value.shape) for field, value in motion.items()},
        "checks": checks,
        "format_and_kinematic_constraints_pass": bool(all(checks.values())),
        "resampling": timing,
        "root_quaternion_norm_error": summary(norm_error),
        "adjacent_root_quaternion_dot": summary(adjacent_dot),
        "joint_diagnostics": joints,
        "local_body_fk_error_m": summary(local_error),
        "world_body_reconstruction_error_m": summary(world_error),
        "root_motion": {
            "net_displacement_xyz_m": (
                motion["root_pos"][-1] - motion["root_pos"][0]
            ).tolist(),
            "horizontal_speed_m_s": summary(
                np.linalg.norm(displacement[:, :2], axis=1) * fps
            ),
            "abs_vertical_speed_m_s": summary(abs(displacement[:, 2]) * fps),
        },
        "ground_geometry": ground,
        "ground_alignment": alignment_report,
        "ground_alignment_root_z_translation_m": summary(shifts),
        "self_collision_diagnostics": contacts,
        "sampled_cross_leg_clearance": sampled_clearance,
        "foot_sliding": _sliding(
            corners, centers, fps, ground["floor_z_m"], stance_height
        ),
        "foot_sliding_strict_3mm": strict_slide,
        "source_tracking": tracking,
        "warnings": warnings,
        "physical_validation": "No dynamics, inverse dynamics, PhysX, balance, actuation tracking or contact-force validation. Passing format/FK/limits does not imply a physically executable gait.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_xml", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--ik_config", type=Path, required=True)
    parser.add_argument("--source_reference", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected_frames", type=int, default=340)
    parser.add_argument("--expected_fps", type=float, default=120)
    parser.add_argument("--stance_height", type=float, default=0.02)
    args = parser.parse_args()
    if args.expected_frames < 2 or not all(
        np.isfinite(value) and value > 0
        for value in (args.expected_fps, args.stance_height)
    ):
        parser.error("Expected frames >= 2 and finite positive FPS/stance height")
    motion_path = args.motion.resolve(strict=True)
    report = validate(
        motion_path,
        args.robot_xml.resolve(strict=True),
        (args.metadata or motion_path.with_suffix(".json")).resolve(strict=True),
        args.ik_config.resolve(strict=True),
        args.source_reference.resolve(strict=True),
        expected_frames=args.expected_frames,
        expected_fps=args.expected_fps,
        stance_height=args.stance_height,
    )
    output = args.output or motion_path.with_name(motion_path.stem + "_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Saved {output}; format/constraints pass: {report['format_and_kinematic_constraints_pass']}"
    )
    for name, passed in report["checks"].items():
        if not passed:
            print(f"FAILED: {name}")
    for warning in report["warnings"]:
        print(warning)
    if not report["format_and_kinematic_constraints_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
