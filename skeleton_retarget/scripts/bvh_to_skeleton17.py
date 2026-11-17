"""Retarget LAFAN1 to the copied collision skeleton, with 17 independent axes.

The MJCF retains 21 scalar hinge coordinates: its four knee X/Z coordinates
are constrained to zero. Model-specific behavior stays in this workflow.
Run with ``python -m skeleton_retarget.scripts.bvh_to_skeleton17``.
"""

import argparse
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path

import mink
import mujoco as mj
import numpy as np
from tqdm import tqdm

from skeleton_retarget.retargeting.cli import (
    nonnegative_int,
    positive_float,
    positive_int,
)
from skeleton_retarget.retargeting.collision_ik import (
    CollisionAwareMotionRetargeting,
    minimum_clearance,
)
from skeleton_retarget.retargeting.heading import heading_transform
from skeleton_retarget.retargeting.motion import (
    inspect_model,
    local_body_positions,
    resample_qpos,
)
from skeleton_retarget.retargeting.paths import (
    CONFIGS_ROOT,
    DATA_ROOT,
    MODELS_ROOT,
    RETARGETING_ROOT,
    WALK_SELECTION,
)
from skeleton_retarget.retargeting.sources import read_bvh_fps
from general_motion_retargeting.utils.lafan1 import load_bvh_file

FOOT_BOXES = ("foot_l_collision", "foot_r_collision")
LOCKED_NAMES = {f"knee_{axis}_{side}" for axis in "xz" for side in "rl"}
MATCHED_BODIES = {
    "pelvis",
    "femur_r",
    "tibia_r",
    "calcn_r",
    "femur_l",
    "tibia_l",
    "calcn_l",
    "torso",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def inspect_collision_skeleton(model, config):
    """Reject stale full-body configs and unlocked/changed knee models."""
    body_ids, body_names, columns = inspect_model(model)
    if (model.nq, model.nv, model.nu, model.neq) != (28, 27, 21, 4):
        raise ValueError("Expected the copied 21-hinge/4-knee-lock collision model")
    active = np.flatnonzero(model.eq_active0)
    locked = set()
    for eq_id in active:
        if model.eq_type[eq_id] != int(mj.mjtEq.mjEQ_JOINT):
            raise ValueError("Only the four scalar knee locks are supported")
        if model.eq_obj2id[eq_id] != -1 or not np.allclose(model.eq_data[eq_id, :5], 0):
            raise ValueError("Expected zero-valued independent joint equalities")
        locked.add(model.joint(int(model.eq_obj1id[eq_id])).name)
    if locked != LOCKED_NAMES:
        raise ValueError(f"Unexpected active knee locks: {sorted(locked)}")
    for table in ("ik_match_table1", "ik_match_table2"):
        if set(config[table]) != MATCHED_BODIES:
            raise ValueError(f"{table} must track only pelvis, legs, feet and torso")
    for name in FOOT_BOXES:
        geom = model.geom(name)
        if model.geom_type[geom.id] != int(mj.mjtGeom.mjGEOM_BOX):
            raise ValueError(f"Support geometry {name} is not a box")
    for column in columns:
        column["locked_zero"] = column["name"] in locked
    return body_ids, body_names, columns, sorted(locked)


def align_support_height(model, qpos, fps, clearance=0.001, max_vertical_speed=0.5):
    """Use the new talus collision boxes, including all eight box corners.

    Take the smallest speed-bounded upper envelope of the required root height.
    This prevents penetration without the abrupt vertical jumps of exact
    per-frame snapping. Feet can be above the minimum; this is not a contact
    solver. Horizontal motion and all joint angles remain unchanged.
    """
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    geom_ids = [model.geom(name).id for name in FOOT_BOXES]
    data = mj.MjData(model)
    aligned = np.array(qpos, dtype=float, copy=True)
    required_height = []
    for index, frame in enumerate(qpos):
        data.qpos[:] = frame
        mj.mj_forward(model, data)
        minimum = min(
            (
                (signs * model.geom_size[geom_id])
                @ data.geom_xmat[geom_id].reshape(3, 3).T
                + data.geom_xpos[geom_id]
            )[:, 2].min()
            for geom_id in geom_ids
        )
        required_height.append(float(frame[2] + clearance - minimum))
    height = np.asarray(required_height).copy()
    step = max_vertical_speed / fps
    for index in range(1, len(height)):
        height[index] = max(height[index], height[index - 1] - step)
    for index in range(len(height) - 2, -1, -1):
        height[index] = max(height[index], height[index + 1] - step)
    aligned[:, 2] = height
    return (
        aligned,
        (height - qpos[:, 2]).tolist(),
        {
            "clearance_mode": "minimum",
            "max_root_vertical_speed_m_s": max_vertical_speed,
            "raw_required_root_z_m": required_height,
            "extra_root_height_m": (height - required_height).tolist(),
        },
    )


def check_joint_coordinates(model, qpos, columns, tolerance=1e-7):
    """Fail before writing an invalid reference; do not silently clip angles."""
    for column in columns:
        values = qpos[:, column["qpos_index"]]
        if column["locked_zero"]:
            if np.abs(values).max() > tolerance:
                raise ValueError(f"Knee lock violated: {column['name']}")
        elif model.jnt_limited[column["joint_id"]]:
            lo, hi = model.jnt_range[column["joint_id"]]
            if values.min() < lo - tolerance or values.max() > hi + tolerance:
                raise ValueError(f"Joint range violated: {column['name']}")


class OutputStepLimit(mink.Limit):
    """Bound each saved hinge increment against the previous output pose.

    The IK timestep describes an optimizer iteration. This limit instead uses
    the actual output sample interval, and stays fixed throughout each solve.
    """

    def __init__(self, model, max_increment):
        self.projection = np.eye(model.nv)[6:]
        self.max_increment = max_increment
        self.anchor = None

    def compute_qp_inequalities(self, configuration, dt):
        del dt
        current = configuration.q[7:]
        return mink.Constraint(
            G=np.vstack((self.projection, -self.projection)),
            h=np.r_[
                self.anchor[7:] + self.max_increment - current,
                current - self.anchor[7:] + self.max_increment,
            ],
        )


def repair_resampling_collisions(retargeter, qpos, fps, max_joint_speed=8.0):
    """Check every output pose; project unsafe interpolation with the same IK.

    Scalar interpolation between safe native poses can cross a curved collision
    boundary. The correction tracks joint angles with a posture task while the
    floating-root trajectory remains exactly as resampled.
    """
    model = retargeter.model
    minimums = np.array([minimum_clearance(model, q) for q in qpos])
    corrected = qpos.copy()
    retargeter.configuration.update(qpos[0])
    posture = mink.PostureTask(model, cost=20.0)
    step_limit = OutputStepLimit(model, max_joint_speed / fps)
    retargeter.ik_limits.append(step_limit)
    try:
        for index, target in enumerate(qpos[1:], start=1):
            step_limit.anchor = corrected[index - 1].copy()
            retargeter.interpolation_anchor = step_limit.anchor.copy()
            initial = retargeter.configuration.q.copy()
            initial[:7] = target[:7]
            retargeter.configuration.update(initial)
            posture.set_target(target)
            corrected[index] = retargeter.solve_tasks([posture], max_iterations=30)
            corrected[index, :7] = target[:7]
            retargeter.configuration.update(corrected[index])
    finally:
        retargeter.interpolation_anchor = None
        retargeter.ik_limits.remove(step_limit)
    after = np.array([minimum_clearance(model, q) for q in corrected])
    if after.min() < retargeter.safety_clearance - 1e-8:
        raise ValueError("120 Hz collision correction failed; no output written")
    return corrected, {
        "enabled": True,
        "method": "Posture IK with hard knee locks, joint/rate limits and nonlinear collision backtracking at output poses and quarter-interval samples; resampled root trajectory preserved",
        "max_joint_speed_rad_s": max_joint_speed,
        "max_actual_joint_speed_rad_s": float(
            np.abs(np.diff(corrected[:, 7:], axis=0)).max() * fps
        ),
        "minimum_clearance_before_m": float(minimums.min()),
        "minimum_clearance_after_m": float(after.min()),
        "unsafe_interpolated_pose_count": int(
            np.sum(minimums < retargeter.safety_clearance - 1e-8)
        ),
        "max_joint_correction_rad": float(np.abs(corrected[:, 7:] - qpos[:, 7:]).max()),
    }


def selection_for(source, start, stop):
    """Keep the previously reviewed source interval and its gait evidence."""
    if not WALK_SELECTION.is_file():
        return None
    for clip in json.loads(WALK_SELECTION.read_text())["clips"]:
        if (
            Path(clip["source_bvh"]).name == source.name
            and clip["start_frame"] == start
            and clip["stop_frame_exclusive"] == stop
        ):
            if clip["source_sha256"] != sha256(source):
                raise ValueError(
                    "The source BVH differs from the reviewed selection"
                )
            # Reuse source-interval evidence, not the old 31-model IK QC.
            source_selection = {
                key: value
                for key, value in clip.items()
                if key not in {"ik_qc", "selection_qc_notes"}
            }
            return {
                **source_selection,
                "plan": str(WALK_SELECTION),
                "qc_scope": "Source interval evidence reused; previous-model IK QC excluded. See this export's validation report for the new model.",
            }
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bvh_file", type=Path, default=DATA_ROOT / "raw/lafan1/walk3_subject3.bvh"
    )
    parser.add_argument(
        "--robot_xml",
        type=Path,
        default=MODELS_ROOT / "skeleton17_collision/human_skeleton.xml",
    )
    parser.add_argument(
        "--ik_config",
        type=Path,
        default=CONFIGS_ROOT / "skeleton17_collision/walk3_subject3.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "outputs/skeleton17_collision/walk/walk3_subject3_120hz",
    )
    parser.add_argument("--start_frame", type=nonnegative_int, default=2575)
    parser.add_argument("--max_frames", type=positive_int, default=86)
    parser.add_argument("--output_fps", type=positive_float, default=120.0)
    parser.add_argument("--speed", type=positive_float, default=1.0)
    parser.add_argument("--warmup_iterations", type=positive_int, default=10)
    parser.add_argument(
        "--align_heading",
        action="store_true",
        help="Rotate the complete clip about world Z so its initial root +X facing is world +X",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    source = args.bvh_file.expanduser().resolve(strict=True)
    xml = args.robot_xml.expanduser().resolve(strict=True)
    config_path = args.ik_config.expanduser().resolve(strict=True)
    prefix = args.output.expanduser().resolve()
    # Existing data must be deliberately selected, never overwritten by a default run.
    outputs = {ext: Path(f"{prefix}.{ext}") for ext in ("npz", "pkl", "json")}
    if any(path.exists() for path in outputs.values()):
        parser.error(
            "Output already exists; choose a new --output prefix to preserve it"
        )
    reference_path = prefix.parent / "references" / f"{source.stem}_source.npz"
    if reference_path.exists():
        parser.error("Source reference already exists; choose a new output directory")

    source_fps = read_bvh_fps(source)
    all_frames, human_height = load_bvh_file(str(source), format="lafan1")
    stop = args.start_frame + args.max_frames
    if args.start_frame >= len(all_frames) or stop > len(all_frames):
        parser.error("The requested source interval is outside the BVH")
    frames = all_frames[args.start_frame : stop]
    if len(frames) < 2:
        parser.error("At least two native frames are required for resampling")
    config = json.loads(config_path.read_text())
    model_manifest_path = xml.parent / "manifest.json"
    model_manifest = json.loads(model_manifest_path.read_text())
    calibration_path = config_path.parent / "references" / f"{source.stem}.json"
    calibration_record = {}
    if calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text())
        for name, path in (
            ("robot_xml", xml),
            ("source_bvh", source),
            ("ik_config", config_path),
        ):
            if calibration.get(name + "_sha256") != sha256(path):
                raise ValueError(
                    f"Calibration does not match {name}: {calibration_path}"
                )
        calibration_record = {
            "ik_calibration": str(calibration_path),
            "ik_calibration_sha256": sha256(calibration_path),
            "ik_calibration_scope": "Per-BVH static calibration against the shared collision MJCF; the model manifest retains its initial calibration provenance",
        }
    elif model_manifest.get("ik_config_sha256") != sha256(config_path):
        raise ValueError(
            "A new IK config requires its per-BVH references/<clip>.json calibration record"
        )
    retargeter = CollisionAwareMotionRetargeting(
        robot_xml_path=xml,
        ik_config_path=config_path,
        actual_human_height=human_height,
        use_equality_constraints=True,
        verbose=not args.quiet,
    )
    model = retargeter.model
    body_ids, body_names, columns, locked = inspect_collision_skeleton(model, config)
    collision_limit = retargeter.collision_limit
    key = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "retarget_reference")
    initial = model.key_qpos[key].copy() if key >= 0 else model.qpos0.copy()
    retargeter.update_targets(frames[0])
    initial[:3], initial[3:7] = retargeter.scaled_human_data[retargeter.human_root_name]
    retargeter.configuration.update(initial)
    for _ in range(args.warmup_iterations):
        retargeter.retarget(frames[0])
    native_qpos = np.stack(
        [
            retargeter.retarget(frame)
            for frame in tqdm(frames, desc="Skeleton17 native IK")
        ]
    )
    check_joint_coordinates(model, native_qpos, columns)
    qpos, resampling = resample_qpos(
        native_qpos, source_fps, args.output_fps, speed=args.speed
    )
    qpos, collision_repair = repair_resampling_collisions(
        retargeter, qpos, args.output_fps
    )
    translation = np.r_[-qpos[0, :2], 0.0]
    qpos[:, :3] += translation
    qpos, ground_shifts, ground_details = align_support_height(
        model, qpos, args.output_fps
    )
    check_joint_coordinates(model, qpos, columns)
    if not np.isfinite(qpos).all():
        raise ValueError("IK/resampling produced non-finite poses")

    heading_info = None
    if args.align_heading:
        positions, rotations, heading_info = heading_transform(
            qpos[:, :3], qpos[:, [4, 5, 6, 3]]
        )
        qpos[:, :3] = positions
        qpos[:, 3:7] = rotations[:, [3, 0, 1, 2]]

    motion = {
        "fps": args.output_fps,
        "root_pos": qpos[:, :3].copy(),
        "root_rot": qpos[:, [4, 5, 6, 3]].copy(),
        "dof_pos": qpos[:, 7:].copy(),
        "local_body_pos": local_body_positions(model, qpos, body_ids),
        "link_body_list": body_names,
    }
    if not np.isfinite(motion["local_body_pos"]).all():
        raise ValueError("FK produced non-finite positions")
    selection = selection_for(source, args.start_frame, stop)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    source_names = list(frames[0])
    np.savez_compressed(
        reference_path,
        body_names=np.asarray(source_names, dtype=str),
        positions=np.asarray(
            [[frame[name][0] for name in source_names] for frame in frames]
        ),
        rotations=np.asarray(
            [[frame[name][1] for name in source_names] for frame in frames]
        ),
        fps=source_fps,
        start_frame=args.start_frame,
    )
    metadata = {
        "target_model": "skeleton17_collision",
        "source_bvh": str(source),
        "source_bvh_sha256": sha256(source),
        "robot_xml": str(xml),
        "robot_xml_sha256": sha256(xml),
        "ik_config": str(config_path),
        "ik_config_sha256": sha256(config_path),
        **calibration_record,
        "model_manifest": str(model_manifest_path),
        "model_manifest_sha256": sha256(model_manifest_path),
        "model_assets": model_manifest["copied_dependencies"],
        "source_reference": str(reference_path),
        "source_reference_sha256": sha256(reference_path),
        "source_format": "bvh_lafan1",
        "fps": args.output_fps,
        "native_fps": source_fps,
        "source_fps": source_fps,
        "speed_factor": args.speed,
        "resampling": resampling,
        "resampling_collision_correction": collision_repair,
        "timing_note": "Native-rate IK then linear root/scalar interpolation and root SLERP, followed by recorded collision correction and bounded root-height alignment; no extrapolation; speed=1 retains source time scale",
        "runtime_versions": {
            name: importlib.metadata.version(name)
            for name in ("mink", "mujoco", "numpy", "scipy", "qpsolvers", "daqp")
        },
        "source_interval_duration_seconds": (len(frames) - 1) / source_fps,
        "saved_interval_duration_seconds": (len(qpos) - 1) / args.output_fps,
        "frames": len(qpos),
        "source_frames": len(frames),
        "start_frame": args.start_frame,
        "stop_frame_exclusive": stop,
        "human_height_m": human_height,
        "human_height_source": "lafan1_default",
        "use_equality_constraints": True,
        "warmup_iterations": args.warmup_iterations,
        "initial_reference_keyframe": "retarget_reference" if key >= 0 else None,
        "initial_floating_root_aligned_to_first_target": True,
        "world_translation_m": translation.tolist(),
        "ground_offset_m": 0.0,
        "ground_alignment": {
            "enabled": True,
            "method": "Minimum speed-bounded upper envelope of required root Z from all corners of the real talus foot boxes, flat z=0 floor",
            "foot_geom_names": list(FOOT_BOXES),
            "support_foot_box_clearance_m": 0.001,
            "root_z_translation_m": ground_shifts,
            "is_physical_contact_solver": False,
            **ground_details,
        },
        "root_rotation_order": "xyzw",
        "position_unit": "meter",
        "local_body_pos_frame": "floating root at zero position and identity rotation",
        "body_ids": body_ids.tolist(),
        "link_body_list": body_names,
        "joint_columns": columns,
        "internal_scalar_coordinates": len(columns),
        "independent_internal_dof_count": len(columns) - len(locked),
        "locked_zero_joint_names": locked,
        "active_joint_names_xml_order": [
            c["name"] for c in columns if not c["locked_zero"]
        ],
        "gait_selection": selection,
        "sampled_gait_cycles": (
            selection["gait_cycles"]
            * resampling["output_duration_seconds"]
            / resampling["source_duration_seconds"]
            if selection
            else None
        ),
        "solver": {
            "backend": retargeter.solver,
            "damping": retargeter.damping,
            "max_additional_iterations_per_stage": retargeter.max_iter,
            "velocity_limits": False,
            "collision_constraints": True,
            "collision_limit_gain_argument": collision_limit.gain,
            "collision_fraction_of_available_distance_per_ik_step": 0.5,
            "collision_gain_units_note": "Local dt compensation for installed Mink 1.3 collision-limit bound in a delta-q QP",
            "nonlinear_safety_clearance_m": retargeter.safety_clearance,
            "nonlinear_backtracking_statistics": retargeter.collision_statistics,
            "collision_minimum_distance_m": collision_limit.minimum_distance_from_collisions,
            "collision_detection_distance_m": collision_limit.collision_detection_distance,
            "collision_geom_pairs": [
                [model.geom(first).name, model.geom(second).name]
                for first, second in collision_limit.geom_id_pairs
            ],
        },
        "code_sha256": {
            str(path): sha256(path)
            for path in [
                Path(__file__).resolve(),
                RETARGETING_ROOT / "collision_ik.py",
                RETARGETING_ROOT / "motion_retarget.py",
                RETARGETING_ROOT / "motion.py",
                RETARGETING_ROOT / "sources.py",
            ]
        },
        "outputs": {ext: str(path) for ext, path in outputs.items()},
    }
    if heading_info is not None:
        metadata["heading_normalization"] = heading_info
        heading_path = RETARGETING_ROOT / "heading.py"
        metadata["code_sha256"][str(heading_path)] = sha256(heading_path)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with outputs["pkl"].open("wb") as stream:
        pickle.dump(motion, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(
        outputs["npz"],
        **{**motion, "link_body_list": np.asarray(body_names, dtype=str)},
    )
    write_json(outputs["json"], metadata)
    print(
        f"Saved {len(qpos)} frames at {args.output_fps:g} Hz; {len(columns)} scalar columns, {len(locked)} locked, {len(body_names)} bodies"
    )
    print(outputs["npz"])


if __name__ == "__main__":
    main()
