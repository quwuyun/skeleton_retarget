"""Canonical locations for the independent retargeting workspace."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "skeleton_retarget"
# Keep REPO_ROOT/CUSTOM_ROOT as compatibility names for the migrated scripts.
REPO_ROOT = PROJECT_ROOT
CUSTOM_ROOT = PROJECT_ROOT
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "assets"
RETARGETING_ROOT = PACKAGE_ROOT / "retargeting"
CONFIGS_ROOT = RETARGETING_ROOT / "ik_configs"
WALK_SELECTION = DATA_ROOT / "selections/lafan1_walks.json"


def resolve_recorded_path(path: str | Path, *, strict: bool = False) -> Path:
    """Read historical provenance paths without directory aliases.

    Keep recorded JSON and its hashes intact. New exports use canonical paths;
    only references inside older records need these layout translations.
    Relative recorded paths are relative to this project.
    """
    path = Path(path).expanduser()
    # Existing manifests predate publishing this independent checkout. Map
    # their absolute provenance roots so a clone at another location can use
    # the bundled files without rewriting recorded JSON or its hashes.
    for legacy_root in (
        Path("/home/hy/workspace/skeleton_retarget"),
        Path("/home/hy/workspace/GMR/custom_retarget"),
        Path("/home/hy/workspace/GMR"),
    ):
        if path.is_absolute() and path.is_relative_to(legacy_root):
            path = REPO_ROOT / path.relative_to(legacy_root)
            break
    if not path.is_absolute():
        # Historical records sometimes included the old package directory.
        parts = path.parts
        if parts and parts[0] == "skeleton_retarget":
            path = REPO_ROOT.joinpath(*parts[1:])
        else:
            path = REPO_ROOT / path
    for old, current in (
        (CUSTOM_ROOT / "models", MODELS_ROOT),
        (CUSTOM_ROOT / "configs", CONFIGS_ROOT),
        (REPO_ROOT / "custom_retarget/models", MODELS_ROOT),
        (REPO_ROOT / "custom_retarget/configs", CONFIGS_ROOT),
        (REPO_ROOT / "custom_retarget/data", DATA_ROOT),
        (REPO_ROOT / "retargeting", RETARGETING_ROOT),
        (REPO_ROOT / "skeleton_retarget/models", MODELS_ROOT),
        (REPO_ROOT / "skeleton_retarget/configs", CONFIGS_ROOT),
        (MODELS_ROOT / "lafan1_walks", MODELS_ROOT / "skeleton31/walks"),
        (CONFIGS_ROOT / "lafan1_walks", CONFIGS_ROOT / "skeleton31/walks"),
        (REPO_ROOT / "data/lafan1", DATA_ROOT / "raw/lafan1"),
        (
            REPO_ROOT / "data/human_skeleton/walks_120hz",
            DATA_ROOT / "outputs/skeleton31",
        ),
        (
            REPO_ROOT / "data/human_skeleton17_collision",
            DATA_ROOT / "outputs/skeleton17_collision",
        ),
        (
            DATA_ROOT / "outputs/skeleton17_collision/long_segments_x_v02/mirrored",
            DATA_ROOT / "outputs/skeleton17_collision/long_mirror",
        ),
        (
            DATA_ROOT / "outputs/skeleton17_collision/long_segments_x_v02/long_forward_walk2_subject4",
            DATA_ROOT / "outputs/skeleton17_collision/long_segments_x_v02/long_forward_walk2_subject4_a",
        ),
    ):
        if path.is_relative_to(old):
            path = current / path.relative_to(old)

    # Retargeting exports originally lived directly in the model output root.
    # Keep those provenance paths usable after grouping ordinary walks.
    walk_root = DATA_ROOT / "outputs/skeleton17_collision"
    if path.is_relative_to(walk_root):
        relative = path.relative_to(walk_root)
        first = relative.parts[0] if relative.parts else ""
        if (
            first.startswith(
                ("walk1_subject", "walk2_subject", "walk3_subject", "walk4_subject")
            )
            or first in {"references", "preview", "batch_reports"}
            or first in {
                "summary.json",
                "summary.csv",
                "selection.json",
                "result_summary.json",
                "preservation_report.json",
            }
        ):
            path = walk_root / "walk" / relative
    return path.resolve(strict=strict)
