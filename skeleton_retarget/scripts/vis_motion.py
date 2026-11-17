"""Play a six-field motion NPZ on its matching MJCF without a robot registry.

Example:
    uv run python -m skeleton_retarget play \
        --robot_xml MODEL.xml --motion MOTION.npz

--speed changes playback timing only. The NPZ file is never modified.
"""

import argparse
import pathlib
import time

import mujoco as mj

from skeleton_retarget.retargeting.cli import positive_float
from skeleton_retarget.retargeting.motion import load_motion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_xml", type=pathlib.Path, required=True)
    parser.add_argument("--motion", type=pathlib.Path, required=True)
    parser.add_argument("--speed", type=positive_float, default=1.0)
    parser.add_argument("--once", action="store_true", help="Exit after one playback.")
    args = parser.parse_args()

    xml_path = args.robot_xml.expanduser().resolve(strict=True)
    motion_path = args.motion.expanduser().resolve(strict=True)
    model = mj.MjModel.from_xml_path(str(xml_path))
    fps, qpos = load_motion(motion_path, model)
    data = mj.MjData(model)
    playback_fps = fps * args.speed

    # Import the GUI only after CLI parsing and data validation.
    import mujoco.viewer

    print(
        f"Playing {len(qpos)} frames, saved FPS={fps:.6g}, "
        f"playback speed={args.speed:.6g}. Close the viewer to stop."
    )
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 135.0
        start_time = time.monotonic()
        while viewer.is_running():
            elapsed_frame = int((time.monotonic() - start_time) * playback_fps)
            if args.once and elapsed_frame >= len(qpos):
                break
            frame_index = elapsed_frame % len(qpos)
            data.qpos[:] = qpos[frame_index]
            mj.mj_forward(model, data)
            with viewer.lock():
                viewer.cam.lookat[:] = qpos[frame_index, :3]
            viewer.sync()
            time.sleep(min(1.0 / playback_fps, 0.02))


if __name__ == "__main__":
    main()
