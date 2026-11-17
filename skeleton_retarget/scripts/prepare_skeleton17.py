"""Copy the collision skeleton and calibrate its eight LAFAN1 body targets.

The upstream collision XML and Geometry are read-only. The local copy retains
all 21 named hinges, four knee equality locks, actuators and collision geometry.
Only a retarget_reference keyframe is added; dependencies use relative paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from skeleton_retarget.retargeting.calibration import (
    calibrate,
    json_write,
    lafan1_source_reference,
)
from skeleton_retarget.retargeting.paths import (
    CONFIGS_ROOT,
    DATA_ROOT,
    MODELS_ROOT,
    REPO_ROOT,
    resolve_recorded_path,
)

SOURCE_XML = pathlib.Path(
    "/home/hy/workspace/humanoid-exo_model/mjcf/walk_skeleton_body_usd_collision.xml"
)
MODEL_ROOT = MODELS_ROOT / "skeleton17_collision"
CONFIG_ROOT = CONFIGS_ROOT / "skeleton17_collision"
DEFAULT_BVH = DATA_ROOT / "raw/lafan1/walk3_subject3.bvh"
BODY_MATCHES = {
    "pelvis": "Hips",
    "femur_l": "LeftUpLeg",
    "tibia_l": "LeftLeg",
    "calcn_l": "LeftFootMod",
    "femur_r": "RightUpLeg",
    "tibia_r": "RightLeg",
    "calcn_r": "RightFootMod",
    "torso": "Spine2",
}
EXPECTED_LOCKS = {"knee_x_r", "knee_z_r", "knee_x_l", "knee_z_l"}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_output(path: pathlib.Path) -> pathlib.Path:
    """Keep every generated artifact inside this checkout."""
    path = path.expanduser().resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError(f"Output must be below {REPO_ROOT}: {path}")
    return path


def inspect_model(model: mj.MjModel) -> dict:
    hinges = [
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE
    ]
    locked = [
        model.joint(int(model.eq_obj1id[i])).name
        for i in range(model.neq)
        if model.eq_active0[i]
    ]
    if (model.nq, model.nv, model.njnt, model.neq, model.nu) != (28, 27, 22, 4, 21):
        raise ValueError(
            "Expected the 21-hinge collision skeleton with four knee locks"
        )
    if set(locked) != EXPECTED_LOCKS or len(hinges) != 21:
        raise ValueError(f"Unexpected joints or equality locks: {locked}")
    if not np.allclose(model.qpos0[:7], [0, 0, 1, 1, 0, 0, 0], atol=1e-12):
        raise ValueError("Expected the identity pelvis frame at height 1 m")
    if not np.allclose(model.qpos0[7:], 0, atol=1e-12):
        raise ValueError("Expected zero hinge reference angles")
    for name in BODY_MATCHES:
        model.body(name)
    return {
        "nq": model.nq,
        "nv": model.nv,
        "joint_count": model.njnt,
        "internal_hinge_count": len(hinges),
        "independent_internal_dof_count": len(hinges) - len(locked),
        "equality_count": model.neq,
        "body_count_including_world": model.nbody,
        "geom_count": model.ngeom,
        "mesh_asset_count": model.nmesh,
        "site_count": model.nsite,
        "tendon_count": model.ntendon,
        "actuator_count": model.nu,
        "total_mass_kg": float(model.body_mass.sum()),
        "dof_names": hinges,
        "locked_joint_names": locked,
        "active_joint_names": [name for name in hinges if name not in EXPECTED_LOCKS],
    }


def target_reference(model: mj.MjModel) -> tuple[np.ndarray, dict]:
    """Use the new model's actual zero pose, with fixed arms and identity root."""
    data = mj.MjData(model)
    data.qpos[:] = model.qpos0
    mj.mj_forward(model, data)
    poses = {
        name: [
            data.xpos[model.body(name).id].copy(),
            data.xquat[model.body(name).id].copy(),
        ]
        for name in BODY_MATCHES
    }
    return data.qpos.copy(), poses


def calibration_errors(human: dict, robot: dict, config: dict) -> dict:
    """Check the same root-relative scaling and postrotation offsets used by IK."""
    root_position = human[config["human_root_name"]][0]
    scales = config["human_scale_table"]
    scaled_root = scales[config["human_root_name"]] * root_position
    errors = {}
    for body, entry in config["ik_match_table1"].items():
        source, _, _, pos_offset, rot_offset = entry
        position, quat = human[source]
        rotation = R.from_quat(quat, scalar_first=True) * R.from_quat(
            rot_offset, scalar_first=True
        )
        adjusted = (
            scaled_root
            + scales[source] * (position - root_position)
            + rotation.apply(pos_offset)
        )
        target_position, target_quat = robot[body]
        target_rotation = R.from_quat(target_quat, scalar_first=True)
        errors[body] = {
            "position_error_m": float(np.linalg.norm(adjusted - target_position)),
            "orientation_error_rad": float(
                (target_rotation.inv() * rotation).magnitude()
            ),
        }
    result = {
        "method": "Static adjusted source targets versus new-model qpos0 body FK",
        "per_body": errors,
        "max_position_error_m": max(e["position_error_m"] for e in errors.values()),
        "max_orientation_error_rad": max(
            e["orientation_error_rad"] for e in errors.values()
        ),
    }
    if (
        result["max_position_error_m"] > 1e-10
        or result["max_orientation_error_rad"] > 1e-10
    ):
        raise ValueError(f"Static calibration failed: {result}")
    return result


def calibration_for_bvh(model: mj.MjModel, bvh: pathlib.Path) -> tuple[dict, dict]:
    """Calibrate one source skeleton against the shared collision model."""
    metrics = inspect_model(model)
    human, source_metadata = lafan1_source_reference(bvh)
    qref, robot = target_reference(model)
    config = calibrate(human, robot, BODY_MATCHES, height_assumption=1.75)
    checks = calibration_errors(human, robot, config)
    reference = {
        "description": "LAFAN1 canonical anatomical reference calibrated to collision skeleton qpos0",
        "calibrated": True,
        "source_bvh": str(bvh),
        "source_bvh_sha256": sha256(bvh),
        "source_format": "lafan1",
        "source_height_assumption": 1.75,
        "source_reference_metadata": source_metadata,
        "source_reference_poses_wxyz": {
            name: [position.tolist(), quat.tolist()]
            for name, (position, quat) in human.items()
        },
        "target_reference_method": "Unmodified new model qpos0; identity pelvis rotation and fixed arms",
        "target_reference_qpos_wxyz": qref.tolist(),
        "target_body_poses_wxyz": {
            name: [position.tolist(), quat.tolist()]
            for name, (position, quat) in robot.items()
        },
        "body_matches": BODY_MATCHES,
        "dof_names": metrics["dof_names"],
        "locked_joint_names": metrics["locked_joint_names"],
        "active_joint_names": metrics["active_joint_names"],
        "calibration_checks": checks,
    }
    return config, reference


def calibrate_existing_model(
    robot_xml: pathlib.Path,
    bvh: pathlib.Path,
    config_path: pathlib.Path,
    reference_path: pathlib.Path,
) -> dict:
    """Write per-BVH calibration without copying or changing model assets.

    Matching existing calibration files are reused; differing files are never
    overwritten. Both paths are checked before either file is written.
    """
    robot_xml = robot_xml.expanduser().resolve(strict=True)
    bvh = bvh.expanduser().resolve(strict=True)
    config_path = local_output(config_path)
    reference_path = local_output(reference_path)
    if config_path == reference_path or {config_path, reference_path} & {
        robot_xml,
        bvh,
    }:
        raise ValueError(
            "Calibration outputs must be distinct from each other and inputs"
        )
    hashes = {path: sha256(path) for path in (robot_xml, bvh)}
    model = mj.MjModel.from_xml_path(str(robot_xml))
    config, reference = calibration_for_bvh(model, bvh)
    config_bytes = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()
    config_hash = (
        sha256(config_path)
        if config_path.is_file()
        else hashlib.sha256(config_bytes).hexdigest()
    )
    reference.update(
        robot_xml=str(robot_xml),
        robot_xml_sha256=hashes[robot_xml],
        ik_config=str(config_path),
        ik_config_sha256=config_hash,
    )
    for path, content in ((config_path, config), (reference_path, reference)):
        if path.exists():
            existing = json.loads(path.read_text())
            if path == reference_path:
                for field in ("robot_xml", "ik_config", "source_bvh"):
                    if field in existing:
                        existing[field] = str(resolve_recorded_path(existing[field]))
            if existing != content:
                raise ValueError(f"Existing calibration differs; choose new paths: {path}")
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError("An input changed during calibration")
    for path, content in ((config_path, config), (reference_path, reference)):
        if not path.exists():
            json_write(path, content)
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_xml", type=pathlib.Path, default=SOURCE_XML)
    parser.add_argument("--bvh_file", type=pathlib.Path, default=DEFAULT_BVH)
    parser.add_argument("--output_dir", type=pathlib.Path, default=MODEL_ROOT)
    parser.add_argument("--config_output", type=pathlib.Path)
    args = parser.parse_args()

    source = args.source_xml.expanduser().resolve(strict=True)
    bvh = args.bvh_file.expanduser().resolve(strict=True)
    output_dir = local_output(args.output_dir)
    config_path = local_output(args.config_output or CONFIG_ROOT / f"{bvh.stem}.json")
    output_xml = output_dir / "human_skeleton.xml"
    if source == output_xml:
        raise ValueError("The upstream model cannot be overwritten")

    original_hash = sha256(source)
    bvh_hash = sha256(bvh)
    source_model = mj.MjModel.from_xml_path(str(source))
    metrics = inspect_model(source_model)
    config, reference = calibration_for_bvh(source_model, bvh)
    qref = np.asarray(reference["target_reference_qpos_wxyz"])
    checks = reference["calibration_checks"]

    parser_xml = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(source, parser=parser_xml)
    root = tree.getroot()
    if root.find("include") is not None:
        raise ValueError("The collision source must be a single XML without includes")
    compiler = root.find("compiler")
    if compiler is not None and (compiler.get("meshdir") or compiler.get("assetdir")):
        raise ValueError("Resolve compiler meshdir/assetdir before copying this model")

    dependencies = []
    for asset in root.iter():
        filename = asset.get("file")
        if filename is None:
            continue
        source_asset = (source.parent / filename).resolve(strict=True)
        relative = pathlib.Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected a portable relative asset path: {filename}")
        target_asset = local_output(output_dir / relative)
        if target_asset == source_asset:
            raise ValueError(f"Cannot overwrite an upstream resource: {source_asset}")
        source_bytes = source_asset.read_bytes()
        digest = hashlib.sha256(source_bytes).hexdigest()
        target_asset.parent.mkdir(parents=True, exist_ok=True)
        target_asset.write_bytes(source_bytes)
        copied_hash = sha256(target_asset)
        if copied_hash != digest:
            raise ValueError(f"Resource copy failed: {target_asset}")
        dependencies.append(
            {
                "source": str(source_asset),
                "relative_path": relative.as_posix(),
                "source_sha256": digest,
                "copied_sha256": copied_hash,
            }
        )

    keys = root.find("keyframe")
    if keys is None:
        keys = ET.SubElement(root, "keyframe")
    for key in list(keys):
        if key.get("name") == "retarget_reference":
            keys.remove(key)
    ET.SubElement(
        keys,
        "key",
        name="retarget_reference",
        qpos=" ".join(format(float(value), ".17g") for value in qref),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    copied_model = mj.MjModel.from_xml_path(str(output_xml))
    if inspect_model(copied_model) != metrics:
        raise ValueError("Copied model topology or parameters differ from source")
    if not np.allclose(
        copied_model.key_qpos[copied_model.key("retarget_reference").id], qref
    ):
        raise ValueError("Copied reference keyframe differs from qpos0")
    if sha256(source) != original_hash or sha256(bvh) != bvh_hash:
        raise ValueError("An upstream input changed during preparation")
    for dependency in dependencies:
        if sha256(pathlib.Path(dependency["source"])) != dependency["source_sha256"]:
            raise ValueError(f"Upstream resource changed: {dependency['source']}")

    json_write(config_path, config)
    manifest = {
        **metrics,
        "source_xml": str(source),
        "source_xml_sha256": original_hash,
        "copied_xml": output_xml.name,
        "copied_xml_sha256": sha256(output_xml),
        "calibration_bvh": str(bvh),
        "calibration_bvh_sha256": bvh_hash,
        "calibration_format": "lafan1",
        "calibrated": True,
        "mujoco_version": mj.__version__,
        "external_mesh_dependencies": [],
        "copied_dependencies": dependencies,
        "dependency_file_count": len(
            {entry["relative_path"] for entry in dependencies}
        ),
        "requires_equality_constraints": True,
        "requires_ground_alignment_from_collision_boxes": True,
        "tracked_body_count": len(BODY_MATCHES),
        "ik_config": config_path.relative_to(REPO_ROOT).as_posix(),
        "ik_config_sha256": sha256(config_path),
        "calibration_checks": checks,
        "adaptations": [
            "Copy the collision XML and every referenced resource into the project checkout",
            "Preserve relative Geometry paths; no external runtime asset dependencies",
            "Keep joint axes, order, ranges, four knee locks, inertias, actuators and collision shapes unchanged",
            "Add retarget_reference keyframe equal to the new model qpos0",
            "Calibrate eight pelvis, leg and torso body targets; omit fixed arm targets",
        ],
    }
    json_write(output_dir / "reference.json", reference)
    json_write(output_dir / "manifest.json", manifest)
    print(f"Prepared {output_xml}")
    print(
        f"Copied {manifest['dependency_file_count']} resources; 21 hinges, four locks, 17 independent joints"
    )
    print(f"Calibrated eight bodies: {config_path}")
    print(
        "Static reference maximum error: "
        f"{checks['max_position_error_m']:.3g} m, "
        f"{checks['max_orientation_error_rad']:.3g} rad"
    )


if __name__ == "__main__":
    main()
