"""Recorded provenance must remain usable when the checkout moves."""

from pathlib import Path

import pytest

from skeleton_retarget.retargeting import paths


@pytest.fixture
def relocated_checkout(tmp_path, monkeypatch):
    root = tmp_path / "colleague-checkout"
    for name, value in (
        ("PROJECT_ROOT", root),
        ("REPO_ROOT", root),
        ("CUSTOM_ROOT", root),
        ("DATA_ROOT", root / "data"),
        ("MODELS_ROOT", root / "assets"),
        ("RETARGETING_ROOT", root / "skeleton_retarget/retargeting"),
        ("CONFIGS_ROOT", root / "skeleton_retarget/retargeting/ik_configs"),
    ):
        monkeypatch.setattr(paths, name, value)
    return root


@pytest.mark.parametrize(
    "recorded",
    [
        "/home/hy/workspace/skeleton_retarget/assets/skeleton17_collision/human_skeleton.xml",
        "/home/hy/workspace/GMR/custom_retarget/models/skeleton17_collision/human_skeleton.xml",
        "/home/hy/workspace/GMR/skeleton_retarget/models/skeleton17_collision/human_skeleton.xml",
    ],
)
def test_historical_model_paths_relocate(relocated_checkout, recorded):
    expected = relocated_checkout / "assets/skeleton17_collision/human_skeleton.xml"
    expected.parent.mkdir(parents=True)
    expected.touch()
    assert paths.resolve_recorded_path(recorded, strict=True) == expected


def test_output_layout_migrations_apply_after_checkout_relocation(relocated_checkout):
    old = "/home/hy/workspace/skeleton_retarget/data/outputs/skeleton17_collision"
    expected = relocated_checkout / "data/outputs/skeleton17_collision"
    assert paths.resolve_recorded_path(f"{old}/walk2_subject4_120hz.npz") == (
        expected / "walk/walk2_subject4_120hz.npz"
    )
    assert paths.resolve_recorded_path(
        f"{old}/long_segments_x_v02/mirrored/stand_start_left.npz"
    ) == expected / "long_mirror/stand_start_left.npz"


def test_current_relative_and_external_paths_are_preserved(relocated_checkout):
    model = relocated_checkout / "assets/skeleton17_collision/human_skeleton.xml"
    assert paths.resolve_recorded_path(model) == model
    assert paths.resolve_recorded_path("assets/skeleton17_collision/human_skeleton.xml") == model
    external = Path("/external-model/human_skeleton.xml")
    assert paths.resolve_recorded_path(external) == external
