import numpy as np
from scipy.spatial.transform import Rotation

from skeleton_retarget.scripts.mirror_skeleton17 import (
    mirror_joint_positions,
    mirror_name,
    mirror_quaternions_xyzw,
)


def test_mirror_name_swaps_bvh_and_mjcf_side_tokens():
    assert mirror_name("LeftUpLeg") == "RightUpLeg"
    assert mirror_name("hand_l__pisiform_lvs") == "hand_r__pisiform_rvs"
    assert mirror_name("pelvis__r_pelvis") == "pelvis__l_pelvis"
    assert mirror_name(mirror_name("hand_l__pisiform_lvs")) == "hand_l__pisiform_lvs"


def test_mirror_joint_positions_swaps_sides_and_reflects_axes():
    names = ["hip_x_r", "hip_x_l", "hip_y_r", "hip_y_l", "back_z"]
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    mirrored = mirror_joint_positions(values, names)
    np.testing.assert_array_equal(mirrored, [[-2.0, -1.0, 4.0, 3.0, -5.0]])
    np.testing.assert_array_equal(mirror_joint_positions(mirrored, names), values)


def test_mirror_quaternion_reverses_yaw_and_is_an_involution():
    source = Rotation.from_euler("z", [[0.3], [-0.8], [1.2]]).as_quat()
    mirrored = mirror_quaternions_xyzw(source)
    assert np.allclose(
        Rotation.from_quat(mirrored).as_euler("zxy")[:, 0],
        -Rotation.from_quat(source).as_euler("zxy")[:, 0],
    )
    np.testing.assert_allclose(mirror_quaternions_xyzw(mirrored), source)
