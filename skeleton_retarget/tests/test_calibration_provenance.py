"""Protect per-BVH calibration when multiple clips share one unchanged model."""

import json
import tempfile
from pathlib import Path

import pytest

from skeleton_retarget.retargeting.paths import (
    CONFIGS_ROOT,
    CUSTOM_ROOT,
    DATA_ROOT,
    MODELS_ROOT,
    REPO_ROOT,
)
from skeleton_retarget.retargeting.validation import sha256
from skeleton_retarget.scripts.prepare_skeleton17 import calibrate_existing_model
from skeleton_retarget.scripts.validate_skeleton17 import _hash_checks


@pytest.fixture
def calibrated():
    xml = (MODELS_ROOT / "skeleton17_collision/human_skeleton.xml").resolve()
    source = (DATA_ROOT / "raw/lafan1/walk1_subject1.bvh").resolve()
    with tempfile.TemporaryDirectory(dir=DATA_ROOT, prefix=".test_calibration_") as tmp:
        folder = Path(tmp)
        config = folder / "walk1_subject1.json"
        reference = folder / "references/walk1_subject1.json"
        calibrate_existing_model(xml, source, config, reference)
        yield xml, source, config, reference


def test_existing_calibration_is_reused_and_different_config_is_preserved(calibrated):
    xml, source, config, reference = calibrated
    expected = (CONFIGS_ROOT / "skeleton17_collision/walk1_subject1.json").read_text()
    assert json.loads(config.read_text()) == json.loads(expected)
    before = {p: p.read_bytes() for p in (config, reference)}
    calibrate_existing_model(xml, source, config, reference)
    assert before == {p: p.read_bytes() for p in before}
    edited = json.loads(config.read_text())
    edited["human_scale_table"]["Hips"] *= 1.1
    config.write_text(json.dumps(edited))
    changed = config.read_bytes()
    with pytest.raises(ValueError, match="Existing calibration differs"):
        calibrate_existing_model(xml, source, config, reference)
    assert config.read_bytes() == changed
    assert reference.read_bytes() == before[reference]


def test_shared_model_config_requires_matching_source_calibration(calibrated):
    xml, source, config, reference = calibrated
    source_fk = (
        DATA_ROOT
        / "outputs/skeleton17_collision/walk/references/walk3_subject3_source.npz"
    )
    manifest = xml.parent / "manifest.json"
    metadata = {
        "robot_xml": str(xml),
        "robot_xml_sha256": sha256(xml),
        "source_bvh": str(source),
        "source_bvh_sha256": sha256(source),
        "ik_config": str(config),
        "ik_config_sha256": sha256(config),
        "ik_calibration": str(reference),
        "ik_calibration_sha256": sha256(reference),
        "model_manifest": str(manifest),
        "model_manifest_sha256": sha256(manifest),
    }
    checks, _ = _hash_checks(metadata, xml, config, source_fk)
    assert all(checks.values()), checks
    # Refreshing the outer hash cannot make a reference for another BVH valid.
    edited = json.loads(reference.read_text())
    edited["source_bvh_sha256"] = "0" * 64
    reference.write_text(json.dumps(edited))
    metadata["ik_calibration_sha256"] = sha256(reference)
    checks, _ = _hash_checks(metadata, xml, config, source_fk)
    assert checks["metadata_ik_calibration_hash_matches"]
    assert not checks["calibration_source_bvh_hash_matches"]


def test_existing_calibration_with_historical_paths_is_reused(calibrated):
    xml, source, config, reference = calibrated
    record = json.loads(reference.read_text())
    record["robot_xml"] = str(
        CUSTOM_ROOT / "models/skeleton17_collision/human_skeleton.xml"
    )
    record["source_bvh"] = str(REPO_ROOT / "data/lafan1" / source.name)
    reference.write_text(json.dumps(record))
    before = {path: path.read_bytes() for path in (config, reference)}
    calibrate_existing_model(xml, source, config, reference)
    assert before == {path: path.read_bytes() for path in before}
