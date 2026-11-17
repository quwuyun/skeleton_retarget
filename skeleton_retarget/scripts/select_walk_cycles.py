"""Select one or two complete native walking stride cycles from LAFAN1 BVH.

This module only reads motion data. The optional CLI writes a JSON plan; it
never retargets or changes the BVH. A gait cycle is one same-foot swing peak
to the next same-foot swing peak. The selected last peak is included.

API: select_walk_clip(source_path, preferred_cycles=2) -> dict.
CLI: python -m skeleton_retarget.scripts.select_walk_cycles FILE.bvh ... --output PLAN.json
"""

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from skeleton_retarget.retargeting.sources import read_bvh_fps
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def load_positions(source_path):
    """Return official-loader global positions in meters, without root shifts."""
    frames, _ = load_bvh_file(str(source_path), format="lafan1")
    names = list(frames[0])
    positions = np.asarray(
        [[frame[name][0] for name in names] for frame in frames], dtype=float
    )
    return names, positions


def swing_events(positions, names, fps):
    """Find robust same-foot swing peaks, merging heel/toe roll double humps."""
    events = {}
    for side in ("left", "right"):
        prefix = side.title()
        foot = positions[:, names.index(prefix + "Foot")]
        toe = positions[:, names.index(prefix + "Toe")]
        height = np.minimum(foot[:, 2], toe[:, 2])
        peaks, properties = find_peaks(
            uniform_filter1d(height, size=3),
            prominence=0.025,
            distance=round(fps * 0.85),
        )
        events[side] = [
            {
                "frame": int(frame),
                "side": side,
                "height_m": float(height[frame]),
                "prominence_m": float(prominence),
            }
            for frame, prominence in zip(peaks, properties["prominences"])
        ]
    return events


def window_stats(positions, names, fps, start, end_inclusive, events):
    """Calculate trajectory and walking evidence for an inclusive source span."""
    stop = end_inclusive + 1
    smooth_root = uniform_filter1d(
        positions[:, names.index("Hips"), :2], size=9, axis=0, mode="nearest"
    )
    velocity = np.gradient(smooth_root, 1.0 / fps, axis=0)[start:stop]
    root = smooth_root[start:stop]
    speed = np.linalg.norm(velocity, axis=1)
    direction = root[-1] - root[0]
    displacement = float(np.linalg.norm(direction))
    path = float(np.linalg.norm(np.diff(root, axis=0), axis=1).sum())
    heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    heading_smooth = uniform_filter1d(heading, size=round(fps), mode="nearest")
    cross_track = np.zeros(len(root))
    if displacement > 1e-8:
        unit = direction / displacement
        offset = root - root[0]
        cross_track = offset[:, 0] * unit[1] - offset[:, 1] * unit[0]

    feet = {}
    contacts = []
    for side in ("left", "right"):
        prefix = side.title()
        foot = positions[start:stop, names.index(prefix + "Foot")]
        toe = positions[start:stop, names.index(prefix + "Toe")]
        height = np.minimum(foot[:, 2], toe[:, 2])
        floor = float(np.percentile(height, 5))
        clearance = height - floor
        foot_speed = np.linalg.norm(np.gradient(foot, 1.0 / fps, axis=0), axis=1)
        contact = (clearance < 0.04) & (foot_speed < 0.7)
        contacts.append(contact)
        feet[side] = {
            "floor_estimate_z_m": floor,
            "lowest_foot_or_toe_z_range_m": [float(height.min()), float(height.max())],
            "clearance_p95_m": float(np.percentile(clearance, 95)),
            "estimated_contact_fraction": float(contact.mean()),
            "contact_foot_speed_mean_mps": float(foot_speed[contact].mean())
            if contact.any()
            else None,
        }
    interval_events = sorted(
        [
            event
            for side_events in events.values()
            for event in side_events
            if start <= event["frame"] <= end_inclusive
        ],
        key=lambda event: event["frame"],
    )
    alternation = (
        float(
            np.mean(
                [
                    a["side"] != b["side"]
                    for a, b in zip(interval_events, interval_events[1:])
                ]
            )
        )
        if len(interval_events) > 1
        else 0.0
    )
    return {
        "first_to_last_pose_duration_seconds": (end_inclusive - start) / fps,
        "horizontal_speed_mean_mps": float(speed.mean()),
        "horizontal_speed_std_mps": float(speed.std()),
        "horizontal_speed_cv": float(speed.std() / max(speed.mean(), 1e-8)),
        "horizontal_speed_percentiles_mps": dict(
            zip(
                ("p0", "p10", "p50", "p90", "p100"),
                np.percentile(speed, [0, 10, 50, 90, 100]).tolist(),
            )
        ),
        "fraction_speed_below_0_2_mps": float(np.mean(speed < 0.2)),
        "net_displacement_m": displacement,
        "path_length_m": path,
        "straightness_displacement_over_path": displacement / max(path, 1e-8),
        "cross_track_max_abs_m": float(np.max(np.abs(cross_track))),
        "travel_heading_mean_deg": float(
            np.rad2deg(np.angle(np.mean(np.exp(1j * heading))))
        ),
        "travel_heading_range_deg": float(np.rad2deg(np.ptp(heading))),
        "travel_heading_1sec_smoothed_range_deg": float(
            np.rad2deg(np.ptp(heading_smooth))
        ),
        "feet": feet,
        "estimated_double_support_fraction": float(np.mean(contacts[0] & contacts[1])),
        "estimated_neither_foot_in_contact_fraction": float(
            np.mean(~contacts[0] & ~contacts[1])
        ),
        "swing_events": interval_events,
        "swing_event_alternation_fraction": alternation,
    }


def select_walk_clip(source_path, preferred_cycles=2):
    """Select slow, steady, nearly straight complete stride cycles.

    Returned max_frames includes both same-foot boundary swing peaks. Thus
    end_frame_inclusive = start_frame + max_frames - 1 is exactly the final
    gait event, and stop_frame_exclusive = end_frame_inclusive + 1.
    """
    if preferred_cycles not in (1, 2):
        raise ValueError("preferred_cycles must be one or two.")
    source_path = pathlib.Path(source_path).expanduser().resolve(strict=True)
    fps = read_bvh_fps(source_path)
    names, positions = load_positions(source_path)
    events = swing_events(positions, names, fps)
    candidates = []
    # Prefer two complete cycles; use one when the two-cycle candidates fail.
    for cycles in [2, 1] if preferred_cycles == 2 else [1]:
        cycle_candidates = []
        for side, side_events in events.items():
            other_side = "right" if side == "left" else "left"
            for index in range(len(side_events) - cycles):
                bounds = side_events[index : index + cycles + 1]
                frames = np.asarray([event["frame"] for event in bounds])
                periods = np.diff(frames) / fps
                if periods.min() < 0.9 or periods.max() > 2.6:
                    continue
                if periods.std() / periods.mean() > 0.2:
                    continue
                # Exactly one opposite-foot swing peak in each stride.
                if any(
                    sum(a < event["frame"] < b for event in events[other_side]) != 1
                    for a, b in zip(frames, frames[1:])
                ):
                    continue
                start, end = int(frames[0]), int(frames[-1])
                stats = window_stats(positions, names, fps, start, end, events)
                speed = stats["horizontal_speed_mean_mps"]
                if not 0.4 <= speed <= 1.1:
                    continue
                if stats["horizontal_speed_percentiles_mps"]["p10"] < 0.3:
                    continue
                if stats["horizontal_speed_cv"] > 0.25:
                    continue
                if stats["straightness_displacement_over_path"] < 0.975:
                    continue
                if stats["travel_heading_1sec_smoothed_range_deg"] > 25:
                    continue
                if stats["swing_event_alternation_fraction"] < 0.999:
                    continue
                if stats["estimated_neither_foot_in_contact_fraction"] > 0.08:
                    continue
                if (
                    min(foot["clearance_p95_m"] for foot in stats["feet"].values())
                    < 0.025
                ):
                    continue
                # Native ~0.7 m/s walking wins, with strong consistency/turn penalties.
                score = (
                    abs(speed - 0.7)
                    + 1.5 * stats["horizontal_speed_cv"]
                    + 5.0 * (1.0 - stats["straightness_displacement_over_path"])
                    + stats["travel_heading_1sec_smoothed_range_deg"] / 150.0
                    + 0.5 * float(periods.std() / periods.mean())
                )
                cycle_candidates.append(
                    {
                        "start_frame": start,
                        "end_frame_inclusive": end,
                        "stop_frame_exclusive": end + 1,
                        "max_frames": end - start + 1,
                        "gait_cycles": cycles,
                        "boundary_foot": side,
                        "boundary_events": bounds,
                        "phase_events": bounds,
                        "stride_periods_seconds": periods.tolist(),
                        "selection_score": score,
                        "source_stats": stats,
                    }
                )
        if cycle_candidates:
            candidates = sorted(
                cycle_candidates, key=lambda candidate: candidate["selection_score"]
            )
            break
    if not candidates:
        raise ValueError(
            f"No steady alternating slow-walk cycle found in {source_path}"
        )
    best = candidates[0]
    return {
        "source_bvh": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_total_frames": len(positions),
        "source_fps": fps,
        "speed": 1.0,
        "native_timing": True,
        "selection_note": "Two complete steady stride cycles selected."
        if best["gait_cycles"] == 2
        else "One complete steady stride cycle selected because no two-cycle span passed all native slow/steady/straight/alternation checks."
        if preferred_cycles == 2
        else "One complete steady stride cycle requested.",
        "cycle_definition": "A stride cycle is one same-foot swing peak to the next same-foot swing peak, not a single left/right step.",
        "frame_bounds_note": "Both first and final swing-event frames are included; max_frames=end_frame_inclusive-start_frame+1.",
        **best,
        "valid_candidate_count": len(candidates),
        "alternative_candidates": candidates[1:4],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--cycles", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    selected = []
    for source in args.sources:
        clip = select_walk_clip(source, preferred_cycles=args.cycles)
        selected.append(clip)
        print(
            f"{source.name}: {clip['start_frame']}:{clip['stop_frame_exclusive']} "
            f"({clip['gait_cycles']} stride cycles, "
            f"{clip['source_stats']['horizontal_speed_mean_mps']:.3f} m/s)"
        )
    plan = {
        "target_fps": 120.0,
        "timing_note": "Preserve native duration, interpolate poses to actual 120 Hz; do not just relabel FPS or change speed.",
        "selection_method": "Official LAFAN1 FK in meters/+Z up; same-foot swing peaks at least 0.85s apart with prominence>=2.5cm, exactly one opposite swing per stride, native mean speed 0.4-1.1m/s, speed CV<=0.25, straightness>=0.975, smoothed heading range<=25deg, estimated flight<=8%.",
        "contact_note": "Foot contact is a kinematic heuristic: min(Foot_z,Toe_z) < clip 5th-percentile height+4cm and foot speed<0.7m/s; not measured ground forces.",
        "clips": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
