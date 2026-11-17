"""Constant world-Z heading changes for exported six-field motion.

The root local +X axis defines facing. One transform is applied to the complete
clip, so relative yaw, backward walking and turns retain their original signs.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def heading_transform(
    root_positions, root_rotations, *, target_heading=0.0, zero_xy=True
):
    """Return transformed root poses and their one constant rigid transform.

    Quaternions use xyzw. ``target_heading`` is measured in radians about world
    +Z. Only the first frame's horizontal facing is aligned; roll/pitch and
    subsequent changes in facing remain part of the motion.
    """
    positions = np.asarray(root_positions, dtype=float)
    quaternions = np.asarray(root_rotations, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
        raise ValueError("root_pos must have shape (T, 3), with T > 0")
    if quaternions.shape != (len(positions), 4):
        raise ValueError("root_rot must have shape (T, 4), in xyzw order")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("Root poses must be finite")
    if not np.isfinite(target_heading):
        raise ValueError("Target heading must be finite")
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("Root quaternions must be normalized")
    rotations = Rotation.from_quat(quaternions)
    forward = rotations[0].apply([1.0, 0.0, 0.0])
    if np.linalg.norm(forward[:2]) < 1e-6:
        raise ValueError("First-frame root +X axis is vertical; facing is undefined")
    original_heading = float(np.arctan2(forward[1], forward[0]))
    yaw = float(
        np.arctan2(
            np.sin(target_heading - original_heading),
            np.cos(target_heading - original_heading),
        )
    )
    world_rotation = Rotation.from_rotvec([0.0, 0.0, yaw])
    pivot = np.r_[positions[0, :2], 0.0]
    translation = -world_rotation.apply(pivot)
    if not zero_xy:
        translation += pivot
    result_pos = world_rotation.apply(positions) + translation
    result_rot = (world_rotation * rotations).as_quat()
    for index in range(1, len(result_rot)):
        if result_rot[index - 1] @ result_rot[index] < 0:
            result_rot[index] *= -1
    info = {
        "enabled": True,
        "method": "One constant world-Z yaw applied to the entire clip; root local +X defines facing",
        "reference_frame": 0,
        "heading_axis_root": [1.0, 0.0, 0.0],
        "original_heading_radians": original_heading,
        "target_heading_radians": float(target_heading),
        "yaw_rotation_radians": yaw,
        "world_rotation_xyzw": world_rotation.as_quat().tolist(),
        "world_translation_m": translation.tolist(),
        "first_root_xy_set_to_zero": bool(zero_xy),
        "preserves": "joint coordinates, local body positions, root height/tilt, relative turns, body-frame velocities and clip timing",
    }
    return result_pos, result_rot, info


def apply_heading_transform(metadata, positions, rotations):
    """Transform reconstructed pre-normalization IK targets for validation.

    ``rotations`` is a SciPy Rotation sequence. Source FK and calibration remain
    in their original frame; apply this only after constructing calibrated IK
    targets, including the original export's world translation.
    """
    info = metadata.get("heading_normalization", {})
    if not info.get("enabled"):
        return positions, rotations
    quaternion = np.asarray(info["world_rotation_xyzw"], dtype=float)
    translation = np.asarray(info["world_translation_m"], dtype=float)
    if (
        quaternion.shape != (4,)
        or translation.shape != (3,)
        or not np.isfinite(quaternion).all()
        or not np.isfinite(translation).all()
        or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-9, rtol=0)
        or not np.allclose(quaternion[:2], 0, atol=1e-12, rtol=0)
        or not np.isclose(translation[2], 0, atol=1e-12, rtol=0)
    ):
        raise ValueError("Invalid constant world-Z heading normalization transform")
    rotation = Rotation.from_quat(quaternion)
    return rotation.apply(positions) + translation, rotation * rotations
