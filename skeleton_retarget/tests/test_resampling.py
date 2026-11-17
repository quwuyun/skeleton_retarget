"""Check real 120 Hz pose interpolation, timing, rotation and model FK."""

import mujoco as mj
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from skeleton_retarget.retargeting.motion import (
    inspect_model,
    local_body_positions,
    resample_qpos,
)
from skeleton_retarget.retargeting.paths import MODELS_ROOT
from skeleton_retarget.retargeting.validation import source_reference_at_motion_times


def floating_poses(frames, joints=2):
    qpos = np.zeros((frames, 7 + joints))
    qpos[:, 3] = 1.0
    return qpos


def rotation_error(actual_wxyz, expected):
    return (
        Rotation.from_quat(actual_wxyz, scalar_first=True).inv() * expected
    ).magnitude()


def test_linear_motion_and_exact_120_hz_grid_preserve_duration():
    native_times = np.arange(5) / 30.0
    qpos = floating_poses(5)
    origin = np.array([0.1, -0.2, 1.1])
    velocity = np.array([0.8, -0.1, 0.05])
    qpos[:, :3] = origin + native_times[:, None] * velocity
    qpos[:, 7:] = np.array([-0.4, 0.6]) + native_times[:, None] * [0.3, -0.8]
    original = qpos.copy()

    result, info = resample_qpos(qpos, 30, 120)

    output_times = np.arange(17) / 120.0
    assert result.shape == (17, 9), "120 Hz requires new interpolated samples"
    np.testing.assert_allclose(result[:, :3], origin + output_times[:, None] * velocity)
    np.testing.assert_allclose(
        result[:, 7:], [-0.4, 0.6] + output_times[:, None] * [0.3, -0.8]
    )
    np.testing.assert_array_equal(qpos, original)
    np.testing.assert_allclose(result[::4], original, atol=1e-15)
    np.testing.assert_allclose(info["output_sample_times_seconds"], output_times)
    np.testing.assert_allclose(np.diff(output_times), 1 / 120, atol=1e-16, rtol=0)
    assert info["source_duration_seconds"] == info["output_duration_seconds"]
    assert info["unused_source_tail_seconds"] == 0.0


def test_root_slerp_uses_shortest_arc_across_180_and_quaternion_signs():
    qpos = floating_poses(3)
    qpos[:, 3:7] = Rotation.from_euler(
        "z", np.array([170, 190, 210])[:, None], degrees=True
    ).as_quat(scalar_first=True)
    qpos[1, 3:7] *= -1

    result, info = resample_qpos(qpos, 30, 120)

    times = np.asarray(info["source_sample_times_seconds"])
    expected = Rotation.from_euler("z", (170 + times * 600)[:, None], degrees=True)
    np.testing.assert_allclose(rotation_error(result[:, 3:7], expected), 0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1, atol=1e-14)
    assert np.all(np.sum(result[1:, 3:7] * result[:-1, 3:7], axis=1) > 0)


def test_equivalent_q_and_minus_q_do_not_create_rotation():
    qpos = floating_poses(2)
    rotation = Rotation.from_rotvec([0.3, -0.7, 0.2])
    qpos[0, 3:7] = rotation.as_quat(scalar_first=True)
    qpos[1, 3:7] = -qpos[0, 3:7]

    result, _ = resample_qpos(qpos, 30, 120)

    np.testing.assert_allclose(rotation_error(result[:, 3:7], rotation), 0, atol=1e-14)


@pytest.mark.parametrize("speed", [0.6, 1.0, 1.3])
def test_noninteger_rate_floors_the_endpoint_without_extrapolation(speed):
    source_fps = 30.000300003
    qpos = floating_poses(7)
    qpos[:, 0] = np.arange(7) / source_fps

    result, info = resample_qpos(qpos, source_fps, 120, speed=speed)

    duration = 6 / source_fps
    count = int(np.floor(duration / speed * 120 + 1e-10)) + 1
    output_times = np.arange(count) / 120
    native_times = output_times * speed
    assert len(result) == count
    np.testing.assert_allclose(info["output_sample_times_seconds"], output_times)
    np.testing.assert_allclose(info["source_sample_times_seconds"], native_times)
    np.testing.assert_allclose(result[:, 0], native_times)
    assert native_times[-1] <= duration
    assert native_times[-1] + speed / 120 > duration
    assert 0 <= info["unused_source_tail_seconds"] < speed / 120
    assert info["extrapolates"] is False


@pytest.fixture
def real_model():
    path = MODELS_ROOT / "skeleton31/walks/walk1_subject1/human_skeleton.xml"
    return mj.MjModel.from_xml_path(str(path))


def test_linear_scalar_interpolation_preserves_real_model_limits_and_locks(real_model):
    model = real_model
    qpos = np.tile(model.qpos0, (2, 1))
    for jid in np.flatnonzero(model.jnt_limited):
        low, high = model.jnt_range[jid]
        qpos[:, model.jnt_qposadr[jid]] = [low + 0.1 * (high - low), high]
    locked = []
    for eid in np.flatnonzero(model.eq_active0):
        assert model.eq_type[eid] == int(mj.mjtEq.mjEQ_JOINT)
        assert model.eq_obj2id[eid] < 0
        index = model.jnt_qposadr[model.eq_obj1id[eid]]
        value = model.qpos0[index] + model.eq_data[eid, 0]
        qpos[:, index] = value
        locked.append((index, value))

    result, _ = resample_qpos(qpos, 30, 120)

    assert len(locked) == 8
    for jid in np.flatnonzero(model.jnt_limited):
        values = result[:, model.jnt_qposadr[jid]]
        low, high = model.jnt_range[jid]
        assert values.min() >= low - 1e-14
        assert values.max() <= high + 1e-14
    for index, value in locked:
        np.testing.assert_array_equal(result[:, index], value)


def test_body_fk_is_recomputed_after_interpolating_real_model_joints(real_model):
    model = real_model
    qpos = np.tile(model.qpos0, (2, 1))
    qpos[:, model.joint("hip_flexion_r").qposadr[0]] = [-0.2, 0.9]
    qpos[:, model.joint("knee_angle_r").qposadr[0]] = [0.1, 1.2]
    body_ids, _, _ = inspect_model(model)

    interpolated, _ = resample_qpos(qpos, 30, 120)
    local_positions = local_body_positions(model, interpolated, body_ids)

    expected = mj.MjData(model)
    expected.qpos[:] = interpolated[2]
    expected.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    mj.mj_forward(model, expected)
    np.testing.assert_allclose(local_positions[2], expected.xpos[body_ids], atol=1e-14)
    endpoints = local_body_positions(model, qpos, body_ids)
    linear_body_midpoint = endpoints.mean(axis=0)
    assert (
        np.max(np.linalg.norm(local_positions[2] - linear_body_midpoint, axis=1)) > 0.01
    )


def test_source_reference_uses_absolute_start_and_native_resampling_times(tmp_path):
    # The reference stores only frames 4000:4003 of the original capture.
    times = np.arange(3) / 30
    positions = np.zeros((3, 1, 3))
    positions[:, 0, 0] = 10 + 2 * times
    rotations = Rotation.from_euler(
        "z", np.array([170, 190, 210])[:, None], degrees=True
    ).as_quat(scalar_first=True)[:, None]
    rotations[1] *= -1
    reference = tmp_path / "source.npz"
    np.savez(
        reference,
        positions=positions,
        rotations=rotations,
        body_names=np.array(["Hips"]),
        fps=30.0,
        start_frame=4000,
    )
    sample_times = np.arange(5) / 120

    names, sampled_positions, sampled_rotations = source_reference_at_motion_times(
        reference,
        {
            "start_frame": 4001,
            "resampling": {
                "enabled": True,
                "source_sample_times_seconds": sample_times.tolist(),
            },
        },
        5,
    )

    assert names == ["Hips"]
    np.testing.assert_allclose(
        sampled_positions[:, 0, 0], 10 + 2 * (1 / 30 + sample_times)
    )
    expected = Rotation.from_euler(
        "z", (190 + 600 * sample_times)[:, None], degrees=True
    )
    np.testing.assert_allclose(
        rotation_error(sampled_rotations[:, 0], expected), 0, atol=1e-14
    )


@pytest.mark.parametrize("fps", [0.0, -1.0, np.nan, np.inf])
def test_rejects_invalid_output_frame_rates(fps):
    with pytest.raises(ValueError, match="finite and positive"):
        resample_qpos(floating_poses(2), 30, fps)
