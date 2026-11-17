"""Retarget the reviewed walking intervals to the shared 17-DOF model.

Each BVH skeleton gets its own static IK calibration; model and mesh assets
are shared. Complete existing exports are independently validated and reused,
never overwritten. Failures remain visible in the batch summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from skeleton_retarget.retargeting.calibration import json_write
from skeleton_retarget.retargeting.cli import positive_int
from skeleton_retarget.retargeting.paths import (
    CONFIGS_ROOT,
    DATA_ROOT,
    MODELS_ROOT,
    REPO_ROOT,
    WALK_SELECTION,
    resolve_recorded_path,
)
from skeleton_retarget.retargeting.sources import read_bvh_fps
from skeleton_retarget.scripts.prepare_skeleton17 import calibrate_existing_model
from skeleton_retarget.scripts.validate_skeleton17 import validate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_clips(
    plans: list[Path], raw_root: Path, names: list[str] | None
) -> list[dict]:
    clips = []
    for plan in plans:
        for selected in json.loads(plan.read_text())["clips"]:
            stem = Path(selected["source_bvh"]).stem
            if names and stem not in names:
                continue
            source = (raw_root / f"{stem}.bvh").resolve(strict=True)
            if sha256(source) != selected["source_sha256"]:
                raise ValueError(
                    f"Source differs from the reviewed selection: {source}"
                )
            source_fps = read_bvh_fps(source)
            if not math.isclose(source_fps, selected["source_fps"], rel_tol=1e-10):
                raise ValueError(f"Source frame rate differs from the plan: {source}")
            if selected["max_frames"] != (
                selected["stop_frame_exclusive"] - selected["start_frame"]
            ):
                raise ValueError(f"Invalid source interval: {stem}")
            source_selection = {
                key: value
                for key, value in selected.items()
                if key not in {"ik_qc", "selection_qc_notes"}
            }
            clips.append(
                {
                    **source_selection,
                    "source_bvh": str(source),
                    "source_fps": source_fps,
                    "plan": str(plan),
                }
            )
    stems = [Path(clip["source_bvh"]).stem for clip in clips]
    if len(stems) != len(set(stems)):
        raise ValueError("Duplicate motion names in plans")
    missing = set(names or ()) - set(stems)
    if missing:
        raise ValueError(f"Clips not found in the plans: {sorted(missing)}")
    if not clips:
        raise ValueError("No selected clips")
    return sorted(clips, key=lambda clip: Path(clip["source_bvh"]).stem)


def export_clip(clip: dict, args: argparse.Namespace) -> dict:
    source = Path(clip["source_bvh"])
    stem = source.stem
    prefix = args.output_dir / f"{stem}_120hz"
    outputs = [prefix.with_suffix(f".{ext}") for ext in ("npz", "pkl", "json")]
    source_reference = args.output_dir / "references" / f"{stem}_source.npz"
    existing = any(path.exists() for path in (*outputs, source_reference))
    print(f"{'Checking existing' if existing else 'Retargeting'} {stem}", flush=True)
    if existing:
        if not all(path.is_file() for path in (*outputs, source_reference)):
            raise ValueError(
                f"Incomplete existing output; choose a new output directory: {prefix}"
            )
    else:
        config = args.config_root / f"{stem}.json"
        calibration_reference = args.config_root / "references" / f"{stem}.json"
        calibrate_existing_model(args.robot_xml, source, config, calibration_reference)
        command = [
            sys.executable,
            "-m",
            "skeleton_retarget.scripts.bvh_to_skeleton17",
            "--bvh_file",
            str(source),
            "--robot_xml",
            str(args.robot_xml),
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
            "--quiet",
            "--output",
            str(prefix),
        ]
        environment = os.environ.copy()
        environment.update(
            OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
        )
        result = subprocess.run(
            command, cwd=REPO_ROOT, text=True, capture_output=True, env=environment
        )
        log = args.report_dir / "logs" / f"{stem}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            shlex.join(command) + "\n\n" + result.stdout + "\n" + result.stderr
        )
        if result.returncode:
            raise RuntimeError(
                f"Retargeting failed; see {log}: {result.stderr[-1500:]}"
            )

    metadata = json.loads(outputs[2].read_text())
    expected_metadata = {
        "start_frame": clip["start_frame"],
        "stop_frame_exclusive": clip["stop_frame_exclusive"],
        "source_frames": clip["max_frames"],
        "source_bvh_sha256": clip["source_sha256"],
        "robot_xml_sha256": sha256(args.robot_xml),
        "source_fps": clip["source_fps"],
        "fps": 120,
        "speed_factor": 1,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Existing output does not match requested {key}: {prefix}"
            )
    config = resolve_recorded_path(metadata["ik_config"], strict=True)
    expected_frames = (
        math.floor((clip["max_frames"] - 1) / clip["source_fps"] * 120 + 1e-10) + 1
    )
    report = validate(
        outputs[0],
        args.robot_xml,
        outputs[2],
        config,
        source_reference,
        expected_frames=expected_frames,
        expected_fps=120,
    )
    # Preserve the previous single-clip validation evidence when reusing it.
    validation_path = prefix.with_name(prefix.name + "_validation.json")
    if not existing or not validation_path.exists():
        json_write(validation_path, report)
    batch_report = args.report_dir / "validation" / f"{stem}.json"
    json_write(batch_report, report)
    if not report["format_and_kinematic_constraints_pass"]:
        failed = [name for name, passed in report["checks"].items() if not passed]
        raise ValueError(
            f"Independent validation failed for {stem}: {failed}; see {batch_report}"
        )

    row = {
        "motion": stem,
        "status": "reused" if existing else "generated",
        "fps": report["fps"],
        "frames": report["frames"],
        "duration_seconds": report["sample_span_seconds"],
        "selected_gait_cycles": clip["gait_cycles"],
        "sampled_gait_cycles": metadata["sampled_gait_cycles"],
        "source_start_frame": clip["start_frame"],
        "source_end_frame_inclusive": clip["end_frame_inclusive"],
        "target_speed_mean_m_s": report["root_motion"]["horizontal_speed_m_s"]["mean"],
        "minimum_sampled_cross_leg_clearance_m": report["sampled_cross_leg_clearance"][
            "minimum_clearance_m"
        ],
        "model": str(args.robot_xml),
        "ik_config": str(config),
        "npz": str(outputs[0]),
        "pkl": str(outputs[1]),
        "validation_report": str(batch_report),
        "validation_pass": True,
        "warnings": report["warnings"],
    }
    print(
        f"Passed {stem}: {row['frames']} frames, {row['duration_seconds']:.3f} s",
        flush=True,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plans",
        type=Path,
        nargs="+",
        default=[WALK_SELECTION],
    )
    parser.add_argument("--raw_root", type=Path, default=DATA_ROOT / "raw/lafan1")
    parser.add_argument(
        "--robot_xml",
        type=Path,
        default=MODELS_ROOT / "skeleton17_collision/human_skeleton.xml",
    )
    parser.add_argument(
        "--config_root", type=Path, default=CONFIGS_ROOT / "skeleton17_collision"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DATA_ROOT / "outputs/skeleton17_collision/walk",
    )
    parser.add_argument(
        "--report_dir", type=Path, help="Default: OUTPUT_DIR/batch_reports"
    )
    parser.add_argument(
        "--clips", nargs="+", help="Optional motion stems from the reviewed plans"
    )
    parser.add_argument("--jobs", type=positive_int, default=2)
    args = parser.parse_args()
    for key in ("raw_root", "robot_xml", "config_root", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    args.plans = [path.expanduser().resolve(strict=True) for path in args.plans]
    args.report_dir = (
        (args.report_dir or args.output_dir / "batch_reports").expanduser().resolve()
    )
    clips = load_clips(args.plans, args.raw_root, args.clips)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(export_clip, clip, args): clip for clip in clips}
        for future in as_completed(futures):
            clip = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:
                stem = Path(clip["source_bvh"]).stem
                print(f"FAILED {stem}: {error}", file=sys.stderr, flush=True)
                rows.append(
                    {
                        "motion": stem,
                        "status": "failed",
                        "validation_pass": False,
                        "error": str(error),
                    }
                )
    rows.sort(key=lambda row: row["motion"])
    failures = [row for row in rows if not row["validation_pass"]]
    summary = {
        "target_model": "skeleton17_collision",
        "robot_xml": str(args.robot_xml),
        "robot_xml_sha256": sha256(args.robot_xml),
        "output_fps": 120,
        "plans": [str(plan) for plan in args.plans],
        "source_interval_policy": "Reuse reviewed source intervals containing one or two gait cycles; exclude old-model IK QC.",
        "model_policy": "One shared MJCF and Geometry directory; one static source-skeleton calibration per BVH.",
        "requested_motion_count": len(clips),
        "motion_count": len(rows) - len(failures),
        "generated_count": sum(row["status"] == "generated" for row in rows),
        "reused_count": sum(row["status"] == "reused" for row in rows),
        "failed_count": len(failures),
        "all_checks_pass": not failures,
        "clips": rows,
    }
    json_write(args.output_dir / "summary.json", summary)
    json_write(
        args.output_dir / "selection.json",
        {
            "target_model": "skeleton17_collision",
            "robot_xml_sha256": sha256(args.robot_xml),
            "target_fps": 120,
            "plans": [str(plan) for plan in args.plans],
            "qc_scope": "Reviewed source windows only; previous-model IK QC excluded. Current-model results are in batch_reports/validation/.",
            "clips": clips,
        },
    )
    columns = list(
        dict.fromkeys(key for row in rows for key in row if key != "warnings")
    )
    with (args.output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Completed: {summary['motion_count']}/{len(clips)} passed; {summary['generated_count']} generated, {summary['reused_count']} reused",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
