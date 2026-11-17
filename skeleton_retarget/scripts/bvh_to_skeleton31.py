"""Retarget BVH through the independent GMR core to six-field NPZ and PKL files.

Example (from the repository root):
    uv run python -m skeleton_retarget.scripts.bvh_to_skeleton31 --bvh_file data/walk.bvh \
        --robot_xml skeleton_retarget/model.xml --ik_config skeleton_retarget/ik.json \
        --format lafan1 --output skeleton_retarget/outputs/walk

Xsens input must use the 3ds Max BVH format with positions in centimeters.
--speed sets playback speed; 0.75 spreads the motion over more time. With
--output_fps, poses are resampled onto the requested uniform output grid.
"""

import argparse
import hashlib
import json
import pathlib
import pickle
import sys

import mujoco as mj
import numpy as np
from tqdm import tqdm

# Support both `python -m skeleton_retarget.scripts.bvh_to_skeleton31` and direct script use.
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from skeleton_retarget.retargeting.cli import nonnegative_int, positive_float, positive_int
from skeleton_retarget.retargeting.motion_retarget import GeneralMotionRetargeting
from skeleton_retarget.retargeting.motion import (
    inspect_model,
    local_body_positions,
    resample_qpos,
)
from skeleton_retarget.retargeting.sources import load_xsens_frames, read_bvh_fps
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def align_support_height(model, qpos, clearance=0.001):
    """Align this anatomical model's lowest foot-box corner to the ground.

    This translates only root Z; it does not enforce horizontal contact,
    friction, balance or contact forces. Call only for clips with floor support.
    """
    geom_ids = [model.geom(side + "_foot_box").id for side in ("l", "r")]
    if any(model.geom_type[i] != int(mj.mjtGeom.mjGEOM_BOX) for i in geom_ids):
        raise ValueError("Ground alignment requires l_foot_box/r_foot_box box geoms")
    floor_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    data = mj.MjData(model)
    shifts = []
    aligned = qpos.copy()
    for index, frame in enumerate(qpos):
        data.qpos[:] = frame
        mj.mj_forward(model, data)
        floor_z = data.geom_xpos[floor_id, 2] if floor_id >= 0 else 0.0
        min_z = min(
            (
                (signs * model.geom_size[geom_id])
                @ data.geom_xmat[geom_id].reshape(3, 3).T
                + data.geom_xpos[geom_id]
            )[:, 2].min()
            for geom_id in geom_ids
        )
        shift = floor_z + clearance - min_z
        aligned[index, 2] += shift
        shifts.append(float(shift))
    return aligned, shifts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh_file", type=pathlib.Path, required=True)
    parser.add_argument("--format", choices=("lafan1", "xsens"), default="lafan1")
    parser.add_argument("--robot_xml", type=pathlib.Path, required=True)
    parser.add_argument("--ik_config", type=pathlib.Path, required=True)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Output prefix, e.g. outputs/walk; writes .pkl, .npz and .json.",
    )
    parser.add_argument("--start_frame", type=nonnegative_int, default=0)
    parser.add_argument("--max_frames", type=positive_int)
    parser.add_argument(
        "--human_height",
        type=positive_float,
        help="Human height in meters; default: LAFAN1 1.75 m or Xsens neutral skeleton.",
    )
    parser.add_argument(
        "--use_equality_constraints",
        action="store_true",
        help="Enforce active MJCF equality constraints during IK.",
    )
    parser.add_argument(
        "--speed",
        type=positive_float,
        default=1.0,
        help="Playback speed factor; 0.75 slows motion, --output_fps fixes the output grid",
    )
    parser.add_argument(
        "--output_fps",
        type=positive_float,
        help="Resample poses to this FPS while preserving duration (e.g. 120)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Hide model name listings."
    )
    parser.add_argument(
        "--ground_align",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Translate root Z to put the lowest l/r_foot_box corner 1 mm above floor",
    )
    parser.add_argument(
        "--zero_xy", action="store_true", help="Start root horizontal position at (0,0)"
    )
    parser.add_argument(
        "--warmup_iterations",
        type=nonnegative_int,
        default=10,
        help="Fit the first source pose repeatedly before saving actual motion frames",
    )
    args = parser.parse_args()

    bvh_path = args.bvh_file.expanduser().resolve(strict=True)
    xml_path = args.robot_xml.expanduser().resolve(strict=True)
    config_path = args.ik_config.expanduser().resolve(strict=True)
    source_format = f"bvh_{args.format}"
    source_fps = read_bvh_fps(bvh_path)
    fps = args.output_fps if args.output_fps is not None else source_fps * args.speed
    if args.format == "xsens":
        frames, loader_height = load_xsens_frames(
            bvh_path, start_frame=args.start_frame, max_frames=args.max_frames
        )
        stop_frame = args.start_frame + len(frames)
    else:
        human_frames, loader_height = load_bvh_file(str(bvh_path), format="lafan1")
        if args.start_frame >= len(human_frames):
            parser.error(
                f"--start_frame {args.start_frame} is outside the motion's "
                f"{len(human_frames)} frames"
            )
        stop_frame = len(human_frames)
        if args.max_frames is not None:
            stop_frame = min(stop_frame, args.start_frame + args.max_frames)
        frames = human_frames[args.start_frame : stop_frame]
    human_height = args.human_height if args.human_height is not None else loader_height

    retargeter = GeneralMotionRetargeting(
        robot_xml_path=xml_path,
        ik_config_path=config_path,
        actual_human_height=human_height,
        use_equality_constraints=args.use_equality_constraints,
        verbose=not args.quiet,
    )
    model = retargeter.model
    body_ids, body_names, joint_columns = inspect_model(model)
    reference_key = mj.mj_name2id(model, mj.mjtObj.mjOBJ_KEY, "retarget_reference")
    initial_qpos = model.qpos0.copy()
    if reference_key >= 0:
        initial_qpos = model.key_qpos[reference_key].copy()
    # Match the floating root before fitting limbs. A 180-degree world-heading
    # mismatch can otherwise trap the arm solver in a poor bounded solution.
    retargeter.update_targets(frames[0])
    floating_joint = int(
        np.flatnonzero(model.jnt_type == int(mj.mjtJoint.mjJNT_FREE))[0]
    )
    align_initial_root = (
        model.body(retargeter.robot_root_name).id == model.jnt_bodyid[floating_joint]
        and retargeter.human_root_name in retargeter.scaled_human_data
    )
    if align_initial_root:
        position, rotation = retargeter.scaled_human_data[retargeter.human_root_name]
        initial_qpos[:3] = position
        initial_qpos[3:7] = rotation
    retargeter.configuration.update(initial_qpos)
    for _ in range(args.warmup_iterations):
        retargeter.retarget(frames[0])
    qpos = np.stack(
        [retargeter.retarget(frame) for frame in tqdm(frames, desc="Retargeting")]
    )
    resampling = {"enabled": False}
    if args.output_fps is not None:
        qpos, resampling = resample_qpos(
            qpos, source_fps, args.output_fps, speed=args.speed
        )
    translation = np.zeros(3)
    if args.zero_xy:
        translation[:2] = -qpos[0, :2]
        qpos[:, :3] += translation
    ground_shifts = []
    if args.ground_align:
        qpos, ground_shifts = align_support_height(model, qpos)
    if not np.isfinite(qpos).all():
        raise ValueError("Retargeting produced non-finite qpos values.")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1.0, atol=1e-6):
        raise ValueError("Retargeting produced unnormalized root quaternions.")

    motion = {
        "fps": fps,
        "root_pos": qpos[:, :3].copy(),
        "root_rot": qpos[:, [4, 5, 6, 3]].copy(),  # MuJoCo wxyz -> GMR xyzw
        "dof_pos": qpos[:, 7:].copy(),
        "local_body_pos": local_body_positions(model, qpos, body_ids),
        "link_body_list": body_names,
    }
    if not np.isfinite(motion["local_body_pos"]).all():
        raise ValueError("Forward kinematics produced non-finite body positions.")

    output_prefix = args.output.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pkl_path = pathlib.Path(f"{output_prefix}.pkl")
    npz_path = pathlib.Path(f"{output_prefix}.npz")
    manifest_path = pathlib.Path(f"{output_prefix}.json")
    with pkl_path.open("wb") as stream:
        pickle.dump(motion, stream, protocol=pickle.HIGHEST_PROTOCOL)
    np.savez_compressed(
        npz_path,
        **{**motion, "link_body_list": np.asarray(body_names, dtype=str)},
    )
    manifest = {
        "source_bvh": str(bvh_path),
        "source_bvh_sha256": hashlib.sha256(bvh_path.read_bytes()).hexdigest(),
        "robot_xml": str(xml_path),
        "robot_xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        "ik_config": str(config_path),
        "ik_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_format": source_format,
        "fps": fps,
        "source_fps": source_fps,
        "native_fps": source_fps,
        "speed_factor": args.speed,
        "timing_note": (
            "Poses resampled at output_fps with linear scalar joints/root position "
            "and root SLERP; original speed retained when speed_factor=1. "
            "The final fractional interval is omitted to avoid extrapolation."
            if args.output_fps is not None
            else "Saved FPS equals source FPS times speed_factor; pose samples "
            "are unchanged. Speed scaling does not turn the source into natively "
            "captured slow walking."
        ),
        "resampling": resampling,
        "source_interval_duration_seconds": (len(frames) - 1) / source_fps,
        "saved_interval_duration_seconds": (len(qpos) - 1) / fps,
        "frames": len(qpos),
        "source_frames": len(frames),
        "start_frame": args.start_frame,
        "stop_frame_exclusive": stop_frame,
        "human_height_m": human_height,
        "human_height_source": "override"
        if args.human_height is not None
        else ("neutral_skeleton" if args.format == "xsens" else "lafan1_default"),
        "use_equality_constraints": args.use_equality_constraints,
        "warmup_iterations": args.warmup_iterations,
        "initial_reference_keyframe": "retarget_reference"
        if reference_key >= 0
        else None,
        "initial_floating_root_aligned_to_first_target": bool(align_initial_root),
        "world_translation_m": translation.tolist(),
        "ground_alignment": {
            "enabled": args.ground_align,
            "method": "Per-frame root z translation from lowest foot-box corner",
            "support_foot_box_clearance_m": 0.001 if args.ground_align else None,
            "root_z_translation_m": ground_shifts,
            "is_physical_contact_solver": False,
        },
        "root_rotation_order": "xyzw",
        "position_unit": "meter",
        "local_body_pos_frame": "floating root at zero position and identity rotation",
        "body_ids": body_ids.tolist(),
        "link_body_list": body_names,
        "joint_columns": joint_columns,
        "outputs": {"pkl": str(pkl_path), "npz": str(npz_path)},
    }
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(f"Saved {len(qpos)} frames at {fps:.6g} FPS:")
    for path in (pkl_path, npz_path, manifest_path):
        print(path)


if __name__ == "__main__":
    main()
