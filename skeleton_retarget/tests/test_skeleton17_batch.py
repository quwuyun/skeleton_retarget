"""Protect existing motions and source selection when resuming a batch."""

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from skeleton_retarget.scripts.bvh_to_skeleton17_dataset import (
    export_clip,
    load_clips,
)


def plan_with_clip(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "walk1_subject1.bvh"
    source.write_bytes(b"HIERARCHY\nMOTION\nFrames: 41\nFrame Time: 0.033333\n")
    clip = {
        "source_bvh": "/historical/path/walk1_subject1.bvh",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "start_frame": 10,
        "stop_frame_exclusive": 41,
        "max_frames": 31,
        "source_fps": 1 / 0.033333,
    }
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"clips": [clip]}))
    return raw, source, clip, plan


def test_plan_uses_canonical_raw_root_and_verifies_source_hash(tmp_path):
    raw, source, _, plan = plan_with_clip(tmp_path)
    clips = load_clips([plan], raw, [source.stem])
    assert clips[0]["source_bvh"] == str(source.resolve())
    source.write_bytes(b"modified input")
    with pytest.raises(ValueError, match="Source differs"):
        load_clips([plan], raw, None)


def test_duplicate_and_unknown_plan_clips_are_rejected(tmp_path):
    raw, _, _, plan = plan_with_clip(tmp_path)
    with pytest.raises(ValueError, match="Duplicate motion"):
        load_clips([plan, plan], raw, None)
    with pytest.raises(ValueError, match="not found"):
        load_clips([plan], raw, ["unknown_motion"])


def test_partial_existing_export_is_never_overwritten(tmp_path):
    raw, _, clip, _ = plan_with_clip(tmp_path)
    clip["source_bvh"] = str(raw / "walk1_subject1.bvh")
    output = tmp_path / "output"
    output.mkdir()
    motion = output / "walk1_subject1_120hz.npz"
    original = b"existing motion must remain untouched"
    motion.write_bytes(original)
    args = argparse.Namespace(output_dir=output)
    with pytest.raises(ValueError, match="Incomplete existing output"):
        export_clip(clip, args)
    assert motion.read_bytes() == original
    assert list(output.iterdir()) == [motion]


def test_complete_existing_export_with_wrong_interval_is_rejected(tmp_path):
    raw, _, clip, _ = plan_with_clip(tmp_path)
    clip["source_bvh"] = str(raw / "walk1_subject1.bvh")
    output = tmp_path / "output"
    output.mkdir()
    prefix = output / "walk1_subject1_120hz"
    for extension in ("npz", "pkl"):
        prefix.with_suffix(f".{extension}").write_bytes(b"keep unchanged")
    metadata = prefix.with_suffix(".json")
    metadata.write_text(json.dumps({"start_frame": 11}))
    references = output / "references"
    references.mkdir()
    (references / "walk1_subject1_source.npz").write_bytes(b"keep reference")
    model = tmp_path / "model.xml"
    model.write_bytes(b"model")
    original = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    args = argparse.Namespace(output_dir=output, robot_xml=model)
    with pytest.raises(ValueError, match="start_frame"):
        export_clip(clip, args)
    assert {
        path: path.read_bytes() for path in output.rglob("*") if path.is_file()
    } == original
