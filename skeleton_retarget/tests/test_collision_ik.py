"""Exercise nonlinear geometry acceptance on the real copied skeleton."""

import unittest
from pathlib import Path

import mink
import mujoco as mj
import numpy as np

from skeleton_retarget.retargeting.collision_ik import (
    LOCKED_NAMES,
    CollisionAwareMotionRetargeting,
    minimum_clearance,
)

ROOT = Path(__file__).resolve().parents[2]


class CollisionIKTests(unittest.TestCase):
    def make_retargeter(self, **kwargs):
        return CollisionAwareMotionRetargeting(
            ROOT / "assets/skeleton17_collision/human_skeleton.xml",
            ROOT / "skeleton_retarget/retargeting/ik_configs/skeleton17_collision/walk3_subject3.json",
            verbose=False,
            **kwargs,
        )

    def inward_step(self, retargeter):
        velocity = np.zeros(retargeter.model.nv)
        for side, angle in (("r", 0.15), ("l", -0.15)):
            address = int(retargeter.model.joint("hip_x_" + side).dofadr[0])
            velocity[address] = angle / retargeter.model.opt.timestep
        return velocity

    def test_actual_penetrating_step_is_backtracked(self):
        retargeter = self.make_retargeter()
        velocity = self.inward_step(retargeter)
        candidate = retargeter.configuration.q.copy()
        dt = retargeter.model.opt.timestep
        mj.mj_integratePos(retargeter.model, candidate, velocity, dt)
        self.assertLess(minimum_clearance(retargeter.model, candidate), -0.1)
        self.assertTrue(retargeter._integrate_safely(velocity, dt))
        self.assertGreaterEqual(
            minimum_clearance(retargeter.model, retargeter.configuration.q), 0.001
        )
        self.assertLess(retargeter.collision_statistics["minimum_accepted_alpha"], 1.0)

    def test_exhausted_backtracking_keeps_previous_pose(self):
        retargeter = self.make_retargeter(max_backtracks=0)
        previous = retargeter.configuration.q.copy()
        self.assertFalse(
            retargeter._integrate_safely(
                self.inward_step(retargeter), retargeter.model.opt.timestep
            )
        )
        np.testing.assert_array_equal(previous, retargeter.configuration.q)
        self.assertEqual(retargeter.collision_statistics["rejected_steps"], 1)

    def test_public_posture_fit_keeps_geometry_and_knee_locks(self):
        retargeter = self.make_retargeter()
        target = retargeter.configuration.q.copy()
        for side, angle in (("r", 0.15), ("l", -0.15)):
            target[int(retargeter.model.joint("hip_x_" + side).qposadr[0])] = angle
        task = mink.PostureTask(retargeter.model, cost=1.0)
        task.set_target(target)
        result = retargeter.solve_tasks([task], max_iterations=20)
        self.assertGreaterEqual(minimum_clearance(retargeter.model, result), 0.001)
        for name in LOCKED_NAMES:
            self.assertEqual(result[int(retargeter.model.joint(name).qposadr[0])], 0.0)
        self.assertTrue(retargeter._joint_coordinates_valid(result))
        self.assertEqual(len(retargeter.collision_limit.geom_id_pairs), 9)

    def test_safe_endpoints_with_colliding_interior_are_rejected(self):
        retargeter = self.make_retargeter(max_backtracks=0)
        anchor = retargeter.model.qpos0.copy()
        candidate = anchor.copy()
        # A fixed regression pair: endpoints clear by 4.70/5.84 mm, while
        # their interpolated midpoint penetrates by about 3.68 mm.
        anchor[7:25] = [
            0.15622991,
            -0.36557147,
            -0.02091299,
            0,
            0.96673742,
            0,
            0.16928156,
            -0.39085628,
            -0.09960826,
            0.15562045,
            -0.00682817,
            0.26433684,
            0,
            0.38113630,
            0,
            0.06055626,
            -0.22780343,
            -0.03361269,
        ]
        candidate[7:25] = [
            0.12325193,
            -0.33134168,
            -0.03076850,
            0,
            0.91482388,
            0,
            0.15463095,
            -0.37444817,
            -0.09867345,
            0.17042250,
            -0.12953122,
            0.25160289,
            0,
            0.57424250,
            0,
            0.03304178,
            -0.23286215,
            -0.04435473,
        ]
        for endpoint in (anchor, candidate):
            self.assertGreater(minimum_clearance(retargeter.model, endpoint), 0.004)
        delta = np.empty(retargeter.model.nv)
        mj.mj_differentiatePos(retargeter.model, delta, 1.0, anchor, candidate)
        midpoint = anchor.copy()
        mj.mj_integratePos(retargeter.model, midpoint, delta, 0.5)
        self.assertLess(minimum_clearance(retargeter.model, midpoint), -0.003)
        retargeter.configuration.update(anchor)
        retargeter.interpolation_anchor = anchor
        self.assertFalse(retargeter._integrate_safely(delta, 1.0))
        np.testing.assert_array_equal(retargeter.configuration.q, anchor)
        self.assertEqual(
            retargeter.collision_statistics["anchor_rejected_candidates"], 1
        )


if __name__ == "__main__":
    unittest.main()
