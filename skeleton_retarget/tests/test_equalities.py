"""Exercise optional MJCF equality constraints with the actual IK solver."""

import json
from pathlib import Path

import mink
import numpy as np
import pytest

from skeleton_retarget.retargeting.motion_retarget import GeneralMotionRetargeting


@pytest.fixture
def make_retargeter(tmp_path: Path):
    """Load a floating robot whose hinge can be locked by an equality."""

    def make(*, with_equalities=True, **kwargs):
        equality_xml = ""
        if with_equalities:
            # The inactive equality deliberately precedes the active one.
            equality_xml = """
              <equality>
                <weld name="inactive_weld" body1="pelvis" active="false"/>
                <joint name="hinge_lock" joint1="hinge"
                       polycoef="0 0 0 0 0" active="true"/>
              </equality>
            """
        xml_path = tmp_path / "robot.xml"
        xml_path.write_text(
            f"""
            <mujoco model="equality_test">
              <compiler angle="radian"/>
              <option timestep="0.01" gravity="0 0 0"/>
              <worldbody>
                <body name="pelvis" pos="0 0 0.8">
                  <freejoint name="floating_base"/>
                  <geom type="sphere" size="0.08" mass="1"
                        contype="0" conaffinity="0"/>
                  <body name="link" pos="0 0 0.2">
                    <joint name="hinge" type="hinge" axis="0 1 0"
                           limited="true" range="-1.5 1.5"/>
                    <geom type="capsule" size="0.03 0.1" mass="0.3"
                          contype="0" conaffinity="0"/>
                  </body>
                </body>
              </worldbody>
              {equality_xml}
            </mujoco>
            """
        )
        config_path = tmp_path / "ik.json"
        config_path.write_text(
            json.dumps(
                {
                    "robot_root_name": "pelvis",
                    "human_root_name": "pelvis",
                    "ground_height": 0.0,
                    "human_height_assumption": 1.8,
                    "human_scale_table": {"pelvis": 1.0},
                    "use_ik_match_table1": True,
                    "use_ik_match_table2": True,
                    # First emphasize position, then orientation, so that both
                    # stages exercise their iterative solves.
                    "ik_match_table1": {
                        "link": ["pelvis", 1.0, 0.01, [0, 0, 0], [1, 0, 0, 0]]
                    },
                    "ik_match_table2": {
                        "link": ["pelvis", 100.0, 10.0, [0, 0, 0], [1, 0, 0, 0]]
                    },
                }
            )
        )
        return GeneralMotionRetargeting(xml_path, config_path, verbose=False, **kwargs)

    return make


def human_frame():
    """A reachable pose that also drives an unconstrained hinge away from zero."""
    angle = 0.8
    return {
        "pelvis": (
            np.array([0.4, -0.1, 1.1]),
            np.array([np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]),
        )
    }


def test_active_equality_holds_at_every_ik_iteration(
    make_retargeter, monkeypatch: pytest.MonkeyPatch
):
    retargeter = make_retargeter(use_equality_constraints=True)
    configuration = retargeter.configuration
    hinge_index = int(retargeter.model.joint("hinge").qposadr[0])
    qpos = configuration.data.qpos.copy()
    qpos[hinge_index] = -0.35
    # Only equalities active when GMR was constructed should be selected.
    # Activating this weld afterwards must not pin the floating base to world.
    configuration.data.eq_active[retargeter.model.equality("inactive_weld").id] = True
    configuration.update(qpos)

    original_solve = mink.solve_ik
    original_integrate = configuration.integrate_inplace
    stages = []
    hinge_after_integration = []

    def record_solve(configuration, tasks, *args, **kwargs):
        stages.append(1 if tasks is retargeter.tasks1 else 2)
        return original_solve(configuration, tasks, *args, **kwargs)

    def record_integration(velocity, dt):
        original_integrate(velocity, dt)
        hinge_after_integration.append(configuration.data.qpos[hinge_index])

    monkeypatch.setattr(mink, "solve_ik", record_solve)
    monkeypatch.setattr(configuration, "integrate_inplace", record_integration)
    result = retargeter.retarget(human_frame())

    assert stages.count(1) > 1, "Stage 1 must exercise its iterative solve"
    assert stages.count(2) > 1, "Stage 2 must exercise its iterative solve"
    np.testing.assert_allclose(hinge_after_integration, 0.0, atol=1e-8)
    assert abs(result[hinge_index]) < 1e-8
    assert result[0] > 0.2, "The initially inactive weld must not pin the base"
    assert np.linalg.norm(retargeter.tasks2[0].compute_error(configuration)) < 0.01


def test_joint_equality_remains_opt_in(make_retargeter):
    """Existing callers can still fit poses without the MJCF joint lock."""
    retargeter = make_retargeter()
    result = retargeter.retarget(human_frame())
    hinge_index = int(retargeter.model.joint("hinge").qposadr[0])

    assert abs(result[hinge_index]) > 0.05
    assert (
        np.linalg.norm(retargeter.tasks2[0].compute_error(retargeter.configuration))
        < 0.01
    )


def test_opt_in_accepts_model_without_equalities(make_retargeter):
    """Enabling the option on an ordinary open chain remains usable."""
    retargeter = make_retargeter(with_equalities=False, use_equality_constraints=True)
    result = retargeter.retarget(human_frame())

    assert np.isfinite(result).all()
    assert (
        np.linalg.norm(retargeter.tasks2[0].compute_error(retargeter.configuration))
        < 0.01
    )
