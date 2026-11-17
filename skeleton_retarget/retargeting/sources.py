"""BVH timing and source loaders, independent of a target model."""

import math
import pathlib
import re

import numpy as np

from general_motion_retargeting.utils.lafan_vendor.utils import quat_fk
from general_motion_retargeting.utils.xsens_vendor.BVHParser import BVHParser


def read_bvh_fps(path):
    """Keep the BVH time base; the existing LAFAN1 loader discards it."""
    with path.open() as stream:
        for line in stream:
            key, separator, value = line.strip().partition(":")
            if separator and key.strip() == "Frame Time":
                frame_time = float(value)
                if not math.isfinite(frame_time) or frame_time <= 0:
                    raise ValueError(f"Invalid BVH Frame Time in {path}: {value}")
                return 1.0 / frame_time
    raise ValueError(f"BVH has no Frame Time header: {path}")


def load_xsens_frames(bvh_file, start_frame=0, max_frames=None):
    """Load 3ds Max Xsens BVH without Qt, offset files or origin correction.

    Returns global human frames in meters/wxyz and the neutral skeleton's
    head-to-toe height, independent of the selected motion's body posture.
    """
    if start_frame < 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("start_frame must be >= 0 and max_frames must be > 0.")
    text = pathlib.Path(bvh_file).read_text()
    frame_match = re.search(r"^\s*Frames:\s*(\d+)", text, re.MULTILINE)
    if frame_match is None:
        raise ValueError(f"BVH has no Frames header: {bvh_file}")
    frame_count = int(frame_match.group(1))
    if start_frame >= frame_count:
        raise ValueError(
            f"start_frame {start_frame} is outside the motion's {frame_count} frames."
        )
    stop_frame = frame_count
    if max_frames is not None:
        stop_frame = min(stop_frame, start_frame + max_frames)

    bvh_parser = BVHParser(axis_order="zxy", scale=0.01)
    rotations, positions = bvh_parser.parse(
        text, start=start_frame, end=stop_frame, reset_to_zero=False
    )
    quats, positions, offsets, parents = bvh_parser._MOTION_data_post_processing(
        rotations, positions, reset_to_zero=False
    )
    global_rot, global_pos = quat_fk(quats, positions, parents)

    rest_quats = np.zeros((1, len(parents), 4))
    rest_quats[..., 0] = 1.0
    _, rest_pos = quat_fk(rest_quats, offsets[np.newaxis], parents)
    rest = dict(zip(bvh_parser.names, rest_pos[0]))
    human_height = float(
        rest["Head_end_site"][2]
        - min(rest["LeftToe_end_site"][2], rest["RightToe_end_site"][2])
    )
    if not math.isfinite(human_height) or human_height <= 0:
        raise ValueError(
            "The Xsens neutral skeleton has an invalid head-to-toe height."
        )

    frames = []
    for frame_rot, frame_pos in zip(global_rot, global_pos):
        frame = {
            name: (frame_pos[index], frame_rot[index])
            for index, name in enumerate(bvh_parser.names)
        }
        frame["LeftFootMod"] = frame["LeftAnkle"]
        frame["RightFootMod"] = frame["RightAnkle"]
        frames.append(frame)
    return frames, human_height
