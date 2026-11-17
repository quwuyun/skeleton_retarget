"""Six-field motion loading, scalar-joint layout, FK, and time resampling."""

import math

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def inspect_model(model):
    """Validate the root/joint layout used by the six-field motion format."""
    free_joints = np.flatnonzero(model.jnt_type == int(mj.mjtJoint.mjJNT_FREE))
    if len(free_joints) != 1:
        raise ValueError("The model must contain exactly one freejoint.")
    free_joint = int(free_joints[0])
    if int(model.jnt_qposadr[free_joint]) != 0:
        raise ValueError("The freejoint must occupy qpos[0:7].")

    joint_columns = []
    for joint_id in np.argsort(model.jnt_qposadr):
        joint_id = int(joint_id)
        if joint_id == free_joint:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (
            int(mj.mjtJoint.mjJNT_HINGE),
            int(mj.mjtJoint.mjJNT_SLIDE),
        ):
            name = model.joint(joint_id).name or f"joint_{joint_id}"
            raise ValueError(
                f"Joint {name!r} is not hinge/slide; ball joints cannot be "
                "exported as scalar joint positions by this script."
            )
        is_hinge = joint_type == int(mj.mjtJoint.mjJNT_HINGE)
        joint_columns.append(
            {
                "column": len(joint_columns),
                "name": model.joint(joint_id).name or f"joint_{joint_id}",
                "joint_id": joint_id,
                "qpos_index": int(model.jnt_qposadr[joint_id]),
                "type": "hinge" if is_hinge else "slide",
                "unit": "radian" if is_hinge else "meter",
            }
        )

    root_body = int(model.jnt_bodyid[free_joint])
    body_ids = np.flatnonzero(model.body_rootid == root_body)
    if not len(body_ids):
        raise ValueError("The floating root body has no robot subtree.")
    body_names = [
        model.body(int(body_id)).name or f"body_{body_id}" for body_id in body_ids
    ]
    return body_ids, body_names, joint_columns


def local_body_positions(model, qpos, body_ids):
    """Use MuJoCo FK with the floating root at the origin and identity rotation."""
    data = mj.MjData(model)
    positions = np.empty((len(qpos), len(body_ids), 3), dtype=np.float64)
    for frame_index, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        data.qpos[:7] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        mj.mj_forward(model, data)
        positions[frame_index] = data.xpos[body_ids]
    return positions


def resample_qpos(qpos, source_fps, output_fps, *, speed=1.0):
    """Interpolate floating-root poses and scalar joints onto a uniform grid.

    Root translation and scalar joints use linear interpolation; root rotation
    uses shortest-arc SLERP. The last output sample never extrapolates beyond
    the input clip. Body positions must be recomputed from the returned qpos.
    """
    qpos = np.asarray(qpos, dtype=float)
    if qpos.ndim != 2 or qpos.shape[0] < 2 or qpos.shape[1] < 7:
        raise ValueError("Resampling requires at least two floating-root qpos frames")
    if not np.isfinite(qpos).all():
        raise ValueError("Input qpos must be finite")
    for value in (source_fps, output_fps, speed):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Frame rates and speed must be finite and positive")
    if not np.allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1, atol=1e-6):
        raise ValueError("Input root quaternions must be normalized")
    native_times = np.arange(len(qpos), dtype=float) / source_fps
    duration = native_times[-1] / speed
    frame_count = int(np.floor(duration * output_fps + 1e-10)) + 1
    if frame_count < 2:
        raise ValueError("Selected motion is shorter than one output interval")
    output_times = np.arange(frame_count, dtype=float) / output_fps
    sample_times = np.minimum(output_times * speed, native_times[-1])
    result = np.empty((frame_count, qpos.shape[1]), dtype=float)
    for column in (*range(3), *range(7, qpos.shape[1])):
        result[:, column] = np.interp(sample_times, native_times, qpos[:, column])
    rotations = Rotation.from_quat(qpos[:, 3:7], scalar_first=True)
    result[:, 3:7] = Slerp(native_times, rotations)(sample_times).as_quat(
        scalar_first=True
    )
    # Equivalent q/-q representations should stay continuous for consumers.
    for index in range(1, len(result)):
        if result[index - 1, 3:7] @ result[index, 3:7] < 0:
            result[index, 3:7] *= -1
    return result, {
        "enabled": True,
        "method": "linear root translation/scalar joints, shortest-arc root SLERP; body FK recomputed",
        "stage": "after native-rate IK, before foot ground alignment",
        "input_frames": len(qpos),
        "input_fps": float(source_fps),
        "output_frames": len(result),
        "output_fps": float(output_fps),
        "speed_factor": float(speed),
        "source_duration_seconds": float(native_times[-1]),
        "output_duration_seconds": float(output_times[-1]),
        "unused_source_tail_seconds": float(native_times[-1] - sample_times[-1]),
        "source_sample_times_seconds": sample_times.tolist(),
        "output_sample_times_seconds": output_times.tolist(),
        "extrapolates": False,
    }


def load_motion(motion_path, model):
    """Validate NPZ timing/poses and convert disk xyzw into MuJoCo wxyz."""
    free_joints = np.flatnonzero(model.jnt_type == int(mj.mjtJoint.mjJNT_FREE))
    if len(free_joints) != 1 or model.jnt_qposadr[free_joints[0]] != 0:
        raise ValueError("The MJCF must have exactly one freejoint at qpos[0:7].")
    with np.load(motion_path, allow_pickle=False) as motion:
        required = {"fps", "root_pos", "root_rot", "dof_pos"}
        if missing := required.difference(motion.files):
            raise ValueError(f"Motion NPZ is missing fields: {sorted(missing)}")
        if motion["fps"].shape != ():
            raise ValueError("Motion fps must be a scalar.")
        fps = float(motion["fps"])
        root_pos = np.asarray(motion["root_pos"], dtype=float)
        root_rot = np.asarray(motion["root_rot"], dtype=float)
        dof_pos = np.asarray(motion["dof_pos"], dtype=float)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Motion fps must be finite and greater than zero.")
    if root_pos.ndim != 2 or root_pos.shape[1] != 3 or len(root_pos) == 0:
        raise ValueError("root_pos must have shape (T, 3) with T > 0.")
    frames = len(root_pos)
    if root_rot.shape != (frames, 4):
        raise ValueError("root_rot must have shape (T, 4), in xyzw order.")
    if dof_pos.shape != (frames, model.nq - 7):
        raise ValueError(
            f"dof_pos must have shape ({frames}, {model.nq - 7}) for this MJCF; "
            f"got {dof_pos.shape}."
        )
    qpos = np.column_stack((root_pos, root_rot[:, [3, 0, 1, 2]], dof_pos))
    if not np.isfinite(qpos).all():
        raise ValueError("Motion contains non-finite pose values.")
    if not np.allclose(np.linalg.norm(root_rot, axis=1), 1.0, atol=1e-6):
        raise ValueError("Motion root quaternions are not normalized.")
    return fps, qpos
