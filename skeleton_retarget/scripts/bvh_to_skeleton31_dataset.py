"""Export all selected LAFAN1 walking clips on the user's skeleton at 120 Hz."""

import argparse
import csv
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

import numpy as np

from skeleton_retarget.retargeting.calibration import json_write
from skeleton_retarget.retargeting.paths import (
    CONFIGS_ROOT,
    DATA_ROOT,
    MODELS_ROOT,
    REPO_ROOT,
    WALK_SELECTION,
    resolve_recorded_path,
)
from general_motion_retargeting.utils.lafan1 import load_bvh_file

from .validate_skeleton31 import validate

ROOT = REPO_ROOT


def run(command, log_path):
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        shlex.join(command) + "\n\n" + result.stdout + "\n" + result.stderr
    )
    if result.returncode:
        raise RuntimeError(f"Command failed; see {log_path}:\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plans",
        nargs="+",
        type=pathlib.Path,
        default=[WALK_SELECTION],
    )
    parser.add_argument(
        "--source_root",
        type=pathlib.Path,
        default=pathlib.Path("/home/hy/workspace/humanoid-exo_model/mjcf"),
    )
    parser.add_argument(
        "--template_xml",
        type=pathlib.Path,
        help="Use the same adapted skeleton31 template for every clip",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=DATA_ROOT / "outputs/skeleton31",
    )
    parser.add_argument(
        "--model_root",
        type=pathlib.Path,
        default=MODELS_ROOT / "skeleton31/walks",
        help="One calibrated model directory per source clip",
    )
    parser.add_argument(
        "--config_root",
        type=pathlib.Path,
        default=CONFIGS_ROOT / "skeleton31/walks",
        help="One calibrated IK JSON per source clip",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clips = [
        clip for plan in args.plans for clip in json.loads(plan.read_text())["clips"]
    ]
    clips.sort(key=lambda clip: pathlib.Path(clip["source_bvh"]).stem)
    stems = [pathlib.Path(clip["source_bvh"]).stem for clip in clips]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate motion names in plans")
    json_write(
        output_dir / "selection.json",
        {
            "target_fps": 120,
            "plans": [str(plan.resolve()) for plan in args.plans],
            "clips": clips,
        },
    )
    summary = []
    for clip in clips:
        source = resolve_recorded_path(clip["source_bvh"], strict=True)
        if hashlib.sha256(source.read_bytes()).hexdigest() != clip["source_sha256"]:
            raise ValueError(f"Source changed since cycle selection: {source}")
        stem = source.stem
        print(f"Converting {stem}: {clip['gait_cycles']} stride cycle(s)", flush=True)
        model_dir = args.model_root.resolve() / stem
        config = args.config_root.resolve() / f"{stem}.json"
        model = model_dir / "human_skeleton.xml"
        prefix = output_dir / f"{stem}_120hz"
        run(
            [
                sys.executable,
                "-m",
                "skeleton_retarget.scripts.prepare_skeleton31",
                "--source_root",
                str(args.source_root.resolve()),
                "--bvh_file",
                str(source),
                "--format",
                "lafan1",
                "--output_dir",
                str(model_dir),
                "--config_output",
                str(config),
                *(
                    ["--template_xml", str(args.template_xml.resolve())]
                    if args.template_xml is not None
                    else []
                ),
            ],
            output_dir / "logs" / f"{stem}_prepare.log",
        )
        run(
            [
                sys.executable,
                "-m",
                "skeleton_retarget.scripts.bvh_to_skeleton31",
                "--bvh_file",
                str(source),
                "--format",
                "lafan1",
                "--robot_xml",
                str(model),
                "--ik_config",
                str(config),
                "--start_frame",
                str(clip["start_frame"]),
                "--max_frames",
                str(clip["max_frames"]),
                "--output_fps",
                "120",
                "--speed",
                "1",
                "--use_equality_constraints",
                "--ground_align",
                "--zero_xy",
                "--quiet",
                "--output",
                str(prefix),
            ],
            output_dir / "logs" / f"{stem}_retarget.log",
        )
        metadata_path = prefix.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        metadata["gait_selection"] = clip
        metadata["sampled_gait_cycles"] = (
            clip["gait_cycles"]
            * metadata["saved_interval_duration_seconds"]
            / metadata["source_interval_duration_seconds"]
        )
        json_write(metadata_path, metadata)

        # Store only the selected native-rate source poses for reproducible QC.
        frames, _ = load_bvh_file(str(source), format="lafan1")
        frames = frames[clip["start_frame"] : clip["stop_frame_exclusive"]]
        names = list(frames[0])
        reference = output_dir / "references" / f"{stem}_source.npz"
        reference.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            reference,
            body_names=np.array(names, dtype=str),
            positions=np.array(
                [[frame[name][0] for name in names] for frame in frames]
            ),
            rotations=np.array(
                [[frame[name][1] for name in names] for frame in frames]
            ),
            fps=clip["source_fps"],
            start_frame=clip["start_frame"],
        )
        report = validate(
            prefix.with_suffix(".npz"),
            model,
            prefix.with_suffix(".pkl"),
            metadata_path,
            reference,
            0.02,
        )
        json_write(prefix.with_name(prefix.name + "_validation.json"), report)
        if not report["format_and_kinematic_constraints_pass"]:
            raise ValueError(f"Validation failed for {stem}")
        row = {
            "motion": stem,
            "fps": metadata["fps"],
            "frames": metadata["frames"],
            "duration_seconds": metadata["saved_interval_duration_seconds"],
            "selected_gait_cycles": clip["gait_cycles"],
            "sampled_gait_cycles": metadata["sampled_gait_cycles"],
            "source_start_frame": clip["start_frame"],
            "source_end_frame_inclusive": clip["end_frame_inclusive"],
            "target_speed_mean_m_s": report["root_motion"]["horizontal_speed_m_s"][
                "mean"
            ],
            "model": str(model),
            "npz": str(prefix.with_suffix(".npz")),
            "pkl": str(prefix.with_suffix(".pkl")),
            "validation_pass": True,
            "warnings": report["warnings"],
        }
        summary.append(row)
        print(
            f"Saved {metadata['frames']} frames at 120 Hz, "
            f"{row['duration_seconds']:.3f} s; checks passed",
            flush=True,
        )
    json_write(
        output_dir / "summary.json",
        {
            "output_fps": 120,
            "motion_count": len(summary),
            "resampling": "native IK -> root/scalar linear + root SLERP -> ground align -> body FK",
            "clips": summary,
        },
    )
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        columns = [key for key in summary[0] if key != "warnings"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    print(f"Completed {len(summary)} motions: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
