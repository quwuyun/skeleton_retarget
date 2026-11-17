"""Retarget a local MimicKit humanoid motion to the user's skeleton.

MimicKit's three-channel joints are rotation vectors, even though its MJCF
lists three hinge axes. Do not feed these vectors to serial Euler joints.
"""

import argparse
import hashlib
import pathlib
import pickle
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from skeleton_retarget.retargeting.calibration import calibrate, json_write
from skeleton_retarget.retargeting.cli import nonnegative_int, positive_float, positive_int
from skeleton_retarget.retargeting.motion_retarget import GeneralMotionRetargeting
from skeleton_retarget.retargeting.motion import inspect_model, local_body_positions
from skeleton_retarget.retargeting.paths import DATA_ROOT, MODELS_ROOT

from .prepare_skeleton31 import target_reference
from .bvh_to_skeleton31 import align_support_height

DEFAULT_SOURCE = pathlib.Path("/home/hy/workspace/MimicKit")
MATCHES = {
    "pelvis": "pelvis",
    "femur_l": "left_thigh",
    "tibia_l": "left_shin",
    "calcn_l": "left_foot",
    "femur_r": "right_thigh",
    "tibia_r": "right_shin",
    "calcn_r": "right_foot",
    "torso": "torso",
    "humerus_l": "left_upper_arm",
    "ulna_l": "left_lower_arm",
    "hand_l": "left_hand",
    "humerus_r": "right_upper_arm",
    "ulna_r": "right_lower_arm",
    "hand_r": "right_hand",
}
# Reuse the copied calibration code with its standard human-node names.
HUMAN_NAMES = {
    "pelvis": "Hips",
    "left_thigh": "LeftUpLeg",
    "left_shin": "LeftLeg",
    "left_foot": "LeftFootMod",
    "right_thigh": "RightUpLeg",
    "right_shin": "RightLeg",
    "right_foot": "RightFootMod",
    "torso": "Spine2",
    "left_upper_arm": "LeftArm",
    "left_lower_arm": "LeftForeArm",
    "left_hand": "LeftHand",
    "right_upper_arm": "RightArm",
    "right_lower_arm": "RightForeArm",
    "right_hand": "RightHand",
}


class PlainMotionUnpickler(pickle.Unpickler):
    """The official example contains only a dictionary, lists and numbers."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")


def source_skeleton(xml_path):
    root = ET.parse(xml_path).getroot()
    if root.find("compiler").get("angle") != "radian":
        raise ValueError("Expected the MimicKit radian humanoid model")
    nodes = []
    dof_cursor = 0

    def visit(body, parent):
        nonlocal dof_cursor
        joints = body.findall("joint")
        if parent < 0:
            if body.find("freejoint") is None:
                raise ValueError("Source pelvis must have a freejoint")
        elif len(joints) not in (0, 1, 3):
            raise ValueError("Expected fixed, hinge or three-channel spherical joint")
        if len(joints) == 3:
            axes = np.array([np.fromstring(j.get("axis"), sep=" ") for j in joints])
            if not np.allclose(axes, np.eye(3)):
                raise ValueError("Spherical channel order must be x,y,z")
        node_id = len(nodes)
        nodes.append(
            {
                "name": body.get("name"),
                "parent": parent,
                "pos": np.fromstring(body.get("pos", "0 0 0"), sep=" "),
                "rotation": R.from_quat(
                    np.fromstring(body.get("quat", "1 0 0 0"), sep=" "),
                    scalar_first=True,
                ),
                "count": len(joints),
                "start": dof_cursor,
                "axis": None
                if len(joints) != 1
                else np.fromstring(joints[0].get("axis"), sep=" "),
            }
        )
        dof_cursor += len(joints)
        for child in body.findall("body"):
            visit(child, node_id)

    visit(root.find("worldbody/body"), -1)
    if dof_cursor != 28:
        raise ValueError(f"Expected MimicKit humanoid 28 channels, got {dof_cursor}")
    return nodes


def source_fk(frames, nodes):
    """MimicKit native frame: xyz, root rotvec, then joint rotvec/hinge."""
    poses = []
    rotations = []
    for node in nodes:
        parent = node["parent"]
        if parent < 0:
            positions = frames[:, :3].copy()
            rotation = R.from_rotvec(frames[:, 3:6])
        else:
            start = 6 + node["start"]
            count = node["count"]
            joint_rot = R.identity(len(frames))
            if count == 3:
                joint_rot = R.from_rotvec(frames[:, start : start + 3])
            elif count == 1:
                joint_rot = R.from_rotvec(frames[:, start, None] * node["axis"])
            positions = poses[parent] + rotations[parent].apply(node["pos"])
            rotation = rotations[parent] * node["rotation"] * joint_rot
        poses.append(positions)
        rotations.append(rotation)
    return np.stack(poses, axis=1), np.stack(
        [rotation.as_quat(scalar_first=True) for rotation in rotations], axis=1
    )


def human_frames(positions, rotations, nodes):
    indices = {node["name"]: i for i, node in enumerate(nodes)}
    return [
        {
            human: [position[indices[source]].copy(), rotation[indices[source]].copy()]
            for source, human in HUMAN_NAMES.items()
        }
        for position, rotation in zip(positions, rotations)
    ]


def load_motion(path, nodes, fix_neck=False):
    with path.open("rb") as stream:
        motion = PlainMotionUnpickler(stream).load()
    frames = np.asarray(motion["frames"], dtype=float)
    fps = float(motion["fps"])
    if frames.ndim != 2 or frames.shape[1] != 34 or len(frames) < 2:
        raise ValueError("Expected MimicKit humanoid frames shaped (N,34)")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid source FPS")
    repaired_columns = []
    if fix_neck:
        neck = next(node for node in nodes if node["name"] == "head")
        start = 6 + neck["start"]
        for column in range(start, start + neck["count"]):
            if not np.isfinite(frames[:, column]).all():
                frames[:, column] = 0.0
                repaired_columns.append(column)
    if not np.isfinite(frames).all():
        raise ValueError(
            "Motion contains nonfinite channels; --fix_neck fixes neck only"
        )
    return frames, fps, int(motion["loop_mode"]), repaired_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=pathlib.Path, default=DEFAULT_SOURCE)
    parser.add_argument("--motion_file", type=pathlib.Path)
    parser.add_argument("--source_xml", type=pathlib.Path)
    parser.add_argument(
        "--robot_xml",
        type=pathlib.Path,
        default=MODELS_ROOT / "skeleton31/base/human_skeleton.xml",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DATA_ROOT / "outputs/skeleton31/mimickit_walk_slow",
    )
    parser.add_argument("--start_frame", type=nonnegative_int, default=0)
    parser.add_argument("--max_frames", type=positive_int)
    parser.add_argument("--speed", type=positive_float, default=0.6)
    parser.add_argument("--fix_neck", action="store_true")
    parser.add_argument(
        "--ground_align",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align each frame's lowest foot-box corner to z=1 mm after IK",
    )
    args = parser.parse_args()
    source_xml = (
        args.source_xml or args.source_root / "data/assets/humanoid/humanoid.xml"
    ).resolve(strict=True)
    motion_path = (
        args.motion_file or args.source_root / "data/motions/humanoid/humanoid_walk.pkl"
    ).resolve(strict=True)
    robot_xml = args.robot_xml.resolve(strict=True)
    nodes = source_skeleton(source_xml)
    source_frames, source_fps, loop_mode, repaired = load_motion(
        motion_path, nodes, fix_neck=args.fix_neck
    )
    stop = len(source_frames)
    if args.start_frame >= stop:
        parser.error("start_frame is outside source motion")
    if args.max_frames is not None:
        stop = min(stop, args.start_frame + args.max_frames)
    selected = source_frames[args.start_frame : stop]

    # Give the bind pose the exact height of its box soles; no moving frame is
    # used for calibration. The XML is used only for source geometry/FK.
    source_model = mj.MjModel.from_xml_path(str(source_xml))
    source_data = mj.MjData(source_model)
    mj.mj_forward(source_model, source_data)
    sole_z = min(
        source_data.geom_xpos[source_model.geom(side + "_foot").id, 2]
        - source_model.geom_size[source_model.geom(side + "_foot").id, 2]
        for side in ("left", "right")
    )
    rest = np.zeros((1, 34))
    rest[0, 2] = -sole_z
    rest_pos, rest_rot = source_fk(rest, nodes)
    source_reference = human_frames(rest_pos, rest_rot, nodes)[0]
    model = mj.MjModel.from_xml_path(str(robot_xml))
    qref, target, yaw = target_reference(model, source_reference, "lafan1")
    config = calibrate(
        source_reference,
        target,
        {target: HUMAN_NAMES[source] for target, source in MATCHES.items()},
        1.75,
    )
    prefix = args.output.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    config_path = prefix.with_name(prefix.name + "_ik_config.json")
    json_write(config_path, config)
    json_write(
        prefix.with_name(prefix.name + "_reference.json"),
        {
            "source_xml": str(source_xml),
            "source_neutral_root_height_m": float(rest[0, 2]),
            "target_reference_qpos_wxyz": qref.tolist(),
            "target_reference_yaw_radians": yaw,
            "source_body_names": [node["name"] for node in nodes],
        },
    )
    retargeter = GeneralMotionRetargeting(
        robot_xml,
        config_path,
        use_equality_constraints=True,
        verbose=False,
    )
    # Settle at the calibrated reference before the first actual walking frame.
    retargeter.configuration.update(qref)
    for _ in range(3):
        retargeter.retarget(source_reference)
    source_pos, source_rot = source_fk(selected, nodes)
    poses = human_frames(source_pos, source_rot, nodes)
    qpos = np.stack(
        [retargeter.retarget(frame) for frame in tqdm(poses, desc="Walking IK")]
    )
    ground_shifts = []
    if args.ground_align:
        qpos, ground_shifts = align_support_height(model, qpos)
    body_ids, body_names, joint_columns = inspect_model(model)
    fps = source_fps * args.speed
    motion = {
        "fps": fps,
        "root_pos": qpos[:, :3].copy(),
        "root_rot": qpos[:, [4, 5, 6, 3]].copy(),
        "dof_pos": qpos[:, 7:].copy(),
        "local_body_pos": local_body_positions(model, qpos, body_ids),
        "link_body_list": body_names,
    }
    if not all(
        np.isfinite(motion[key]).all()
        for key in ("root_pos", "root_rot", "dof_pos", "local_body_pos")
    ):
        raise ValueError("Nonfinite retargeted motion")
    np.savez_compressed(
        f"{prefix}.npz", **{**motion, "link_body_list": np.array(body_names, dtype=str)}
    )
    with pathlib.Path(f"{prefix}.pkl").open("wb") as stream:
        pickle.dump(motion, stream, protocol=pickle.HIGHEST_PROTOCOL)
    json_write(
        pathlib.Path(f"{prefix}.json"),
        {
            "source_motion": str(motion_path),
            "source_xml": str(source_xml),
            "source_format": "MimicKit humanoid native rotvec",
            "source_motion_sha256": hashlib.sha256(
                motion_path.read_bytes()
            ).hexdigest(),
            "source_xml_sha256": hashlib.sha256(source_xml.read_bytes()).hexdigest(),
            "robot_xml": str(robot_xml),
            "robot_xml_sha256": hashlib.sha256(robot_xml.read_bytes()).hexdigest(),
            "ik_config": str(config_path),
            "source_fps": source_fps,
            "fps": fps,
            "speed": args.speed,
            "time_stretched": args.speed != 1.0,
            "frames": len(selected),
            "sample_span_seconds": (len(selected) - 1) / fps,
            "start_frame": args.start_frame,
            "stop_frame_exclusive": stop,
            "source_loop_mode": loop_mode,
            "neck_nonfinite_columns_set_to_zero": repaired,
            "root_rotation_order": "xyzw",
            "position_unit": "meter",
            "local_body_pos_frame": "floating root at zero and identity quaternion",
            "use_equality_constraints": True,
            "ground_alignment": {
                "enabled": args.ground_align,
                "method": "Per-frame root z translation from lowest foot-box corner",
                "support_foot_box_clearance_m": 0.001 if args.ground_align else None,
                "root_z_translation_m": ground_shifts,
                "is_physical_contact_solver": False,
            },
            "joint_columns": joint_columns,
            "body_ids": body_ids.tolist(),
            "link_body_list": body_names,
        },
    )
    print(f"Saved {prefix}.npz / .pkl / .json: {len(selected)} frames at {fps:g} Hz")


if __name__ == "__main__":
    main()
