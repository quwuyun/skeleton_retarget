"""Static source-reference construction and calibrated IK target offsets."""

import copy
import json

import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from general_motion_retargeting.utils.lafan_vendor.utils import quat_fk
from general_motion_retargeting.utils.xsens_vendor.BVHParser import BVHParser

UP_CONVERSION = R.from_euler("x", 90, degrees=True)
# These are requirements of each source skeleton, not a target-body mapping.
LAFAN1_REFERENCE_NODES = {
    "Hips",
    "LeftUpLeg",
    "LeftLeg",
    "RightUpLeg",
    "RightLeg",
    "Spine2",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
}
XSENS_REFERENCE_NODES = {
    "Hips",
    "LeftHip",
    "LeftKnee",
    "LeftAnkle",
    "RightHip",
    "RightKnee",
    "RightAnkle",
    "Chest4",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
}


def json_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def lafan1_source_reference(bvh_file):
    """Construct an anatomical T reference for LAFAN1's +X bone axes.

    BVH records offsets and animation rotations, but no separate bind-pose
    prerotations. In LAFAN1 most offsets point along local +X, so identity
    rotations put legs, spine and arms on the same line. Use explicit neutral
    rotations, with world +X forward, +Y left and +Z up, then run the same FK
    and Y-up conversion as the official loader. No animation sample is used.
    """
    animation = read_bvh(str(bvh_file))
    names = animation.bones
    required = LAFAN1_REFERENCE_NODES.copy()
    required.update(
        (
            "LeftFoot",
            "RightFoot",
            "LeftToe",
            "RightToe",
            "LeftShoulder",
            "RightShoulder",
        )
    )
    missing = required - set(names)
    if missing:
        raise ValueError(f"BVH does not have the LAFAN1 skeleton: {sorted(missing)}")
    quats = np.zeros((1, len(animation.bones), 4))
    quats[..., 0] = 1
    # Root local +X is superior, +Y anterior, +Z left. The world orientation
    # below agrees with the anatomical bone axes seen in the actual LAFAN1
    # sequence, up to arbitrary global yaw. Neutral hips reverse X and Z;
    # shoulders put +X outward while retaining the dataset's axial twist.
    root_world = R.from_matrix(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    rotations = {
        "Hips": UP_CONVERSION.inv() * root_world,
        "LeftUpLeg": R.from_euler("y", np.pi),
        "RightUpLeg": R.from_euler("y", np.pi),
        "LeftFoot": R.from_euler("z", np.pi / 2),
        "RightFoot": R.from_euler("z", np.pi / 2),
        "LeftShoulder": R.from_euler("y", -np.pi / 2) * R.from_euler("x", np.pi),
        "RightShoulder": R.from_euler("y", np.pi / 2) * R.from_euler("x", np.pi),
    }
    for name, rotation in rotations.items():
        quats[0, names.index(name)] = rotation.as_quat(scalar_first=True)
    positions = animation.offsets[None].copy()
    # The BVH ROOT OFFSET may duplicate an absolute animation position. It is
    # a translation origin, not a limb length or a rest-pose height.
    positions[0, names.index("Hips")] = 0
    global_quat, global_pos = quat_fk(quats, positions, animation.parents)
    result = {}
    for index, bone in enumerate(names):
        result[bone] = [
            UP_CONVERSION.apply(global_pos[0, index]) / 100,
            (
                UP_CONVERSION * R.from_quat(global_quat[0, index], scalar_first=True)
            ).as_quat(scalar_first=True),
        ]
    for side, sign in (("Left", 1), ("Right", -1)):
        hip = result[side + "UpLeg"][0]
        knee = result[side + "Leg"][0]
        ankle = result[side + "Foot"][0]
        toe = result[side + "Toe"][0]
        shoulder = result[side + "Arm"][0]
        wrist = result[side + "Hand"][0]
        if not hip[2] > knee[2] > ankle[2]:
            raise ValueError(f"The {side} neutral leg does not point downwards")
        if toe[0] <= ankle[0] or abs(toe[2] - ankle[2]) > 1e-5:
            raise ValueError(f"The {side} neutral foot does not point forward")
        if sign * (wrist[1] - shoulder[1]) <= 0 or abs(wrist[2] - shoulder[2]) > 1e-5:
            raise ValueError(f"The {side} neutral arm is not horizontal and outward")
    if result["Spine2"][0][2] <= result["Hips"][0][2]:
        raise ValueError("The neutral spine does not point upwards")
    for side in ("Left", "Right"):
        result[side + "FootMod"] = [
            result[side + "Foot"][0].copy(),
            result[side + "Toe"][1].copy(),
        ]
    # Feet sit at z=0; the root height follows the skeleton's actual lengths.
    shift = result["Hips"][0].copy()
    shift[2] = min(result["LeftFoot"][0][2], result["RightFoot"][0][2])
    for position, _ in result.values():
        position -= shift
    metadata = {
        "method": "canonical_lafan1_anatomical_T_pose",
        "uses_motion_frame": False,
        "world_axes": {"forward": "+X", "left": "+Y", "up": "+Z"},
        "root_offset_ignored_cm": animation.offsets[names.index("Hips")].tolist(),
        "nonidentity_local_rotations_wxyz": {
            name: rotation.as_quat(scalar_first=True).tolist()
            for name, rotation in rotations.items()
        },
        "reason": (
            "LAFAN1 BVH has no bind prerotations and uses local +X bone offsets; "
            "identity rotations are not an anatomical standing pose"
        ),
    }
    return result, metadata


def xsens_source_reference(bvh_file):
    """The 3ds Max Xsens BVH zero pose has both arms resting downwards."""
    parser = BVHParser(axis_order="zxy", scale=0.01)
    parser.parse(bvh_file.read_text(), start=0, end=1, reset_to_zero=False)
    offsets = np.asarray(parser.offsets, dtype=float)
    rotations = np.zeros((1, len(parser.names), 4))
    rotations[..., 0] = 1
    quats, positions = quat_fk(rotations, offsets[None], parser.parents)
    result = {
        name: [positions[0, i].copy(), quats[0, i].copy()]
        for i, name in enumerate(parser.names)
    }
    required = XSENS_REFERENCE_NODES | {
        "Head_end_site",
        "LeftToe_end_site",
        "RightToe_end_site",
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(
            f"BVH does not have the 3ds Max Xsens skeleton: {sorted(missing)}"
        )
    shift = result["Hips"][0].copy()
    shift[2] = min(result["LeftToe_end_site"][0][2], result["RightToe_end_site"][0][2])
    for position, _ in result.values():
        position -= shift
    height = float(result["Head_end_site"][0][2])
    return result, height


def calibrate(human_reference, robot_reference, matches, height_assumption):
    source_root = human_reference["Hips"][0]
    target_root = robot_reference["pelvis"][0]
    if source_root[2] <= 0:
        raise ValueError("The source reference has a non-positive standing root height")
    scales = {"Hips": float(target_root[2] / source_root[2])}
    entries1 = {}
    entries2 = {}
    # These are task costs, not hard priorities. Feet have the largest position
    # costs; stage two adds modest intermediate-joint position tracking.
    for body, human in matches.items():
        hp, hq = human_reference[human]
        rp, rq = robot_reference[body]
        if human != "Hips":
            source_length = np.linalg.norm(hp - source_root)
            target_length = np.linalg.norm(rp - target_root)
            if source_length < 1e-8:
                raise ValueError(f"Cannot determine scale for {human}")
            scales[human] = float(target_length / source_length)
        scaled_pos = scales["Hips"] * source_root + scales[human] * (hp - source_root)
        robot_rotation = R.from_quat(rq, scalar_first=True)
        offset_rotation = R.from_quat(hq, scalar_first=True).inv() * robot_rotation
        offset_position = robot_rotation.inv().apply(rp - scaled_pos)
        if body.startswith("calcn_"):
            costs1, costs2 = (200, 20), (200, 40)
        elif body == "pelvis":
            costs1, costs2 = (100, 15), (100, 10)
        elif body == "torso":
            costs1, costs2 = (0, 20), (3, 10)
        elif body.startswith("hand_"):
            costs1, costs2 = (10, 5), (15, 5)
        else:
            costs1, costs2 = (0, 10), (3, 5)
        offsets = [
            offset_position.tolist(),
            offset_rotation.as_quat(scalar_first=True).tolist(),
        ]
        entries1[body] = [human, *costs1, *copy.deepcopy(offsets)]
        entries2[body] = [human, *costs2, *copy.deepcopy(offsets)]
    return {
        "robot_root_name": "pelvis",
        "human_root_name": "Hips",
        "ground_height": 0.0,
        # Scales already use the source skeleton offsets. The loader should
        # supply this same static height, keeping the runtime height ratio one.
        "human_height_assumption": height_assumption,
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "human_scale_table": scales,
        "ik_match_table1": entries1,
        "ik_match_table2": entries2,
    }
