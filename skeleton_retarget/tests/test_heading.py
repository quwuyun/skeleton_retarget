"""Heading changes preserve backward motion, turns, FK and original exports."""

import json

import mujoco as mj
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from skeleton_retarget.retargeting.heading import (
    apply_heading_transform,
    heading_transform,
)
from skeleton_retarget.scripts.normalize_heading import normalize_motion


@pytest.mark.parametrize("heading,direction", [(-np.pi / 2, 1), (np.pi / 2, -1)])
def test_forward_and_backward_become_opposite_world_x_directions(heading, direction):
    facing = Rotation.from_euler("z", [[heading], [heading]])
    positions = np.array([[3.0, -2.0, 0.9], [3.0, -3.0, 0.91]])
    result, rotations, info = heading_transform(positions, facing.as_quat())

    np.testing.assert_allclose(result[0], [0, 0, 0.9], atol=1e-14)
    np.testing.assert_allclose(result[1] - result[0], [direction, 0, 0.01], atol=1e-14)
    np.testing.assert_allclose(
        Rotation.from_quat(rotations).apply([[1, 0, 0]] * 2),
        [[1, 0, 0]] * 2,
        atol=1e-14,
    )
    np.testing.assert_array_equal(positions, [[3.0, -2.0, 0.9], [3.0, -3.0, 0.91]])
    assert info["reference_frame"] == 0


def test_constant_rotation_preserves_turns_tilt_and_world_fk():
    model = mj.MjModel.from_xml_string("""
    <mujoco><worldbody><body name="pelvis"><freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
      <body name="link" pos="0.2 0.1 -0.3"><joint axis="0 1 0"/>
        <geom type="capsule" size="0.05" fromto="0 0 0 0.3 0 -0.4" mass="1"/>
      </body></body></worldbody></mujoco>""")
    data = mj.MjData(model)
    root_positions = np.array([[3, 2, 1], [2.7, 2.1, 1.02], [2.5, 2.4, 1.0]])
    root_rotations = Rotation.from_euler(
        "ZYX", [[90, 10, -4], [120, 7, 3], [170, -5, 2]], degrees=True
    )
    result_pos, result_rot, info = heading_transform(
        root_positions, root_rotations.as_quat()
    )
    transform = Rotation.from_quat(info["world_rotation_xyzw"])
    translation = np.asarray(info["world_translation_m"])
    relative_before = root_rotations[0].inv() * root_rotations
    normalized_rotations = Rotation.from_quat(result_rot)
    relative_after = normalized_rotations[0].inv() * normalized_rotations
    np.testing.assert_allclose(
        (relative_before.inv() * relative_after).magnitude(), 0, atol=1e-14
    )
    for index in range(3):
        data.qpos[:] = [
            *root_positions[index],
            *root_rotations[index].as_quat(scalar_first=True),
            0.25 * index,
        ]
        mj.mj_forward(model, data)
        expected = transform.apply(data.xpos[1:].copy()) + translation
        data.qpos[:7] = [
            *result_pos[index],
            *normalized_rotations[index].as_quat(scalar_first=True),
        ]
        mj.mj_forward(model, data)
        np.testing.assert_allclose(data.xpos[1:], expected, atol=1e-14)
    np.testing.assert_allclose(result_pos[:, 2], root_positions[:, 2], atol=1e-14)


def test_output_preserves_fields_inputs_and_validator_target_transform(tmp_path):
    source = tmp_path / "original.npz"
    metadata_path = source.with_suffix(".json")
    motion = {
        "fps": np.asarray(120.0),
        "root_pos": np.array([[2.0, 3.0, 1.0], [2.0, 2.0, 1.01]]),
        "root_rot": Rotation.from_euler("z", [[-90], [-60]], degrees=True).as_quat(),
        "dof_pos": np.array([[0.1, -0.2], [0.2, -0.3]]),
        "local_body_pos": np.array([[[0, 0, 0], [0.2, 0.1, -0.5]]] * 2),
        "link_body_list": np.array(["pelvis", "foot"]),
    }
    np.savez_compressed(source, **motion)
    metadata_path.write_text(json.dumps({"frames": 2, "fps": 120.0}))
    original_npz, original_json = source.read_bytes(), metadata_path.read_bytes()
    outputs, _ = normalize_motion(source, tmp_path / "plus_x")
    metadata = json.loads(outputs["json"].read_text())
    with np.load(outputs["npz"], allow_pickle=False) as result:
        assert list(result) == list(motion)
        for key in ("fps", "dof_pos", "local_body_pos", "link_body_list"):
            np.testing.assert_array_equal(result[key], motion[key])
        predicted_pos, predicted_rot = apply_heading_transform(
            metadata, motion["root_pos"], Rotation.from_quat(motion["root_rot"])
        )
        np.testing.assert_allclose(predicted_pos, result["root_pos"])
        np.testing.assert_allclose(
            (predicted_rot.inv() * Rotation.from_quat(result["root_rot"])).magnitude(),
            0,
            atol=1e-14,
        )
    assert source.read_bytes() == original_npz
    assert metadata_path.read_bytes() == original_json
    with pytest.raises(FileExistsError):
        normalize_motion(source, tmp_path / "plus_x")
    # A second heading change composes with the original target transform.
    outputs2, _ = normalize_motion(
        outputs["npz"], tmp_path / "plus_y", target_heading=np.pi / 2
    )
    metadata2 = json.loads(outputs2["json"].read_text())
    predicted_pos, predicted_rot = apply_heading_transform(
        metadata2, motion["root_pos"], Rotation.from_quat(motion["root_rot"])
    )
    with np.load(outputs2["npz"], allow_pickle=False) as result:
        np.testing.assert_allclose(predicted_pos, result["root_pos"], atol=1e-14)
        np.testing.assert_allclose(
            (predicted_rot.inv() * Rotation.from_quat(result["root_rot"])).magnitude(),
            0,
            atol=1e-14,
        )


def test_vertical_first_root_forward_axis_has_no_defined_heading():
    with pytest.raises(ValueError, match="vertical"):
        heading_transform(
            [[0, 0, 1]], Rotation.from_euler("y", [[90]], degrees=True).as_quat()
        )
