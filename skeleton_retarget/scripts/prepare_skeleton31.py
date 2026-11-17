"""Prepare a muscle-free anatomical skeleton and optional BVH IK calibration.

The original model and its Geometry directory remain read-only. Generated mesh
paths are relative to the output XML so bundled models remain portable.
"""

import argparse
import copy
import hashlib
import os
import pathlib
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R

from skeleton_retarget.retargeting.calibration import (
    calibrate,
    json_write,
    lafan1_source_reference,
    xsens_source_reference,
)
from skeleton_retarget.retargeting.paths import CONFIGS_ROOT, MODELS_ROOT

SOURCE_ROOT = pathlib.Path("/home/hy/workspace/humanoid-exo_model/mjcf")
OUTPUT_ROOT = MODELS_ROOT / "skeleton31/base"
CONFIG_ROOT = CONFIGS_ROOT / "skeleton31"
BODY_MATCHES = {
    "pelvis": "Hips",
    "femur_l": "LeftUpLeg",
    "tibia_l": "LeftLeg",
    "calcn_l": "LeftFootMod",
    "femur_r": "RightUpLeg",
    "tibia_r": "RightLeg",
    "calcn_r": "RightFootMod",
    "torso": "Spine2",
    "humerus_l": "LeftArm",
    "ulna_l": "LeftForeArm",
    "hand_l": "LeftHand",
    "humerus_r": "RightArm",
    "ulna_r": "RightForeArm",
    "hand_r": "RightHand",
}
XSENS_BODY_MATCHES = {
    "pelvis": "Hips",
    "femur_l": "LeftHip",
    "tibia_l": "LeftKnee",
    "calcn_l": "LeftAnkle",
    "femur_r": "RightHip",
    "tibia_r": "RightKnee",
    "calcn_r": "RightAnkle",
    "torso": "Chest4",
    "humerus_l": "LeftShoulder",
    "ulna_l": "LeftElbow",
    "hand_l": "LeftWrist",
    "humerus_r": "RightShoulder",
    "ulna_r": "RightElbow",
    "hand_r": "RightWrist",
}
ROOT_JOINTS = {
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
}
LOWER_LIMB_MUSCLES = {
    "addbrev",
    "addlong",
    "addmagDist",
    "addmagIsch",
    "addmagMid",
    "addmagProx",
    "bflh",
    "bfsh",
    "edl",
    "ehl",
    "fdl",
    "fhl",
    "gaslat",
    "gasmed",
    "glmax1",
    "glmax2",
    "glmax3",
    "glmed1",
    "glmed2",
    "glmed3",
    "glmin1",
    "glmin2",
    "glmin3",
    "grac",
    "iliacus",
    "perbrev",
    "perlong",
    "piri",
    "psoas",
    "recfem",
    "sart",
    "semimem",
    "semiten",
    "soleus",
    "tfl",
    "tibant",
    "tibpost",
    "vasint",
    "vaslat",
    "vasmed",
}


def object_references(root, object_type):
    """Find named references without counting an object's own name."""
    attributes = {
        "site": ("site", "site1", "site2", "refsite", "slidersite", "cranksite"),
        "tendon": ("tendon", "tendon1", "tendon2"),
    }[object_type]
    references = set()
    for element in root.iter():
        references.update(element.get(key) for key in attributes if element.get(key))
        for type_key, name_key in (("objtype", "objname"), ("reftype", "refname")):
            if element.get(type_key) == object_type and element.get(name_key):
                references.add(element.get(name_key))
    return references


def remove_lower_limb_muscles(root):
    """Remove muscle paths and their unshared attachment sites, keep landmarks."""
    muscle_names = {
        f"{muscle}_{side}" for muscle in LOWER_LIMB_MUSCLES for side in ("l", "r")
    }
    tendons = root.find("tendon")
    removed_tendons = []
    removed_path_sites = set()
    if tendons is not None:
        external = ET.Element("outside_tendons")
        for section in root:
            if section.tag != "tendon":
                external.append(copy.deepcopy(section))
        used_elsewhere = object_references(external, "tendon")
        for tendon in list(tendons):
            name = tendon.get("name", "")
            if name.removesuffix("_tendon") not in muscle_names:
                continue
            if name in used_elsewhere:
                raise ValueError(f"Muscle tendon has an external reference: {name}")
            removed_tendons.append(name)
            removed_path_sites.update(
                site.get("site") for site in tendon.iter("site") if site.get("site")
            )
            tendons.remove(tendon)
        if len(tendons) == 0:
            root.remove(tendons)

    # Include unused recfem/vastus attachment sites whose tendon paths were
    # commented out in the source. Preserve any site used by another object.
    still_referenced = object_references(root, "site")
    removed_sites = []
    preserved_sites = []
    for parent in root.iter():
        for site in list(parent.findall("site")):
            name = site.get("name", "")
            attachment_name, separator, point = name.rpartition("-P")
            muscle_site = name in removed_path_sites or (
                separator and point.isdigit() and attachment_name in muscle_names
            )
            if muscle_site and name not in still_referenced:
                removed_sites.append(name)
                parent.remove(site)
            elif name:
                preserved_sites.append(name)
    return {
        "removed_tendon_names": sorted(removed_tendons),
        "removed_muscle_site_names": sorted(removed_sites),
        "preserved_site_names": sorted(preserved_sites),
    }


def merge_scene(source_root):
    """Expand this model's scene include and coalesce repeated MJCF sections."""
    root = ET.Element("mujoco", model="human_skeleton_gmr")
    sections = {}
    source_files = []
    mesh_files = []

    def append_file(path):
        source_files.append(path.resolve())
        source = ET.parse(path).getroot()
        for child in source:
            if child.tag == "include":
                append_file(path.parent / child.attrib["file"])
                continue
            section = copy.deepcopy(child)
            for asset in section.iter():
                if "file" in asset.attrib:
                    asset_path = (path.parent / asset.attrib["file"]).resolve()
                    if not asset_path.is_file():
                        raise FileNotFoundError(asset_path)
                    asset.set("file", str(asset_path))
                    mesh_files.append(asset_path)
            if section.tag not in sections:
                sections[section.tag] = section
                root.append(section)
            else:
                sections[section.tag].attrib.update(section.attrib)
                sections[section.tag].extend(section)

    append_file(source_root / "scene_walking.xml")
    sections["option"].attrib.pop("collision", None)
    # Older collision="predefined" disabled automatically generated contacts.
    # IK does not step dynamics; retain the original explicit foot-floor pairs.
    sections["compiler"].set("angle", "radian")
    world = sections["worldbody"]
    for body in list(world.findall("body")):
        if body.get("mocap") == "true":
            world.remove(body)
    pelvis = world.find("body[@name='pelvis']")
    if pelvis is None:
        raise ValueError("The scene has no pelvis root body")
    old_roots = {joint.get("name") for joint in pelvis.findall("joint")}
    if old_roots != ROOT_JOINTS:
        raise ValueError(f"Unexpected pelvis root joints: {sorted(old_roots)}")
    for joint in list(pelvis.findall("joint")):
        pelvis.remove(joint)
    pelvis.insert(0, ET.Element("freejoint", name="root"))
    for joint in pelvis.iter("joint"):
        joint.set("limited", "true")
    equality = sections["equality"]
    for constraint in list(equality):
        if constraint.tag == "weld" and constraint.get("active") == "false":
            equality.remove(constraint)
    muscle_cleanup = remove_lower_limb_muscles(root)
    return root, source_files, sorted(set(mesh_files)), muscle_cleanup


def load_template(template_path):
    """Reuse an adapted skeleton if the historical muscle scene is unavailable."""
    template_path = template_path.expanduser().resolve(strict=True)
    root = ET.parse(template_path).getroot()
    if root.find("include") is not None:
        raise ValueError("The skeleton31 template must be an expanded MJCF")
    compiler = root.find("compiler")
    mesh_dir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_files = []
    for asset in root.findall("asset/mesh"):
        if "file" not in asset.attrib:
            continue
        mesh = (template_path.parent / mesh_dir / asset.get("file")).resolve(
            strict=True
        )
        asset.set("file", str(mesh))
        mesh_files.append(mesh)
    # Replace the calibration key; it does not define the model's rest qpos.
    for keyframe in root.findall("keyframe"):
        for key in list(keyframe):
            if key.get("name") == "retarget_reference":
                keyframe.remove(key)
    return (
        root,
        [template_path],
        sorted(set(mesh_files)),
        {
            "removed_tendon_names": [],
            "removed_muscle_site_names": [],
            "preserved_site_names": [site.get("name") for site in root.iter("site")],
            "note": "Reused a muscle-free skeleton31 template; no muscle removal was needed.",
        },
    )


def relative_mesh_paths(root, output_dir):
    """Serialize resolved mesh files relative to the generated MJCF directory."""
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    for mesh in root.findall("asset/mesh"):
        if "file" in mesh.attrib:
            mesh.set("file", os.path.relpath(mesh.get("file"), output_dir))


def target_reference(model, human_reference=None, source_format="lafan1"):
    """Match the anatomical reference stance and horizontal left/right axis."""
    data = mj.MjData(model)
    data.qpos[:] = model.qpos0
    if source_format == "lafan1":
        for side in ("r", "l"):
            data.qpos[model.joint("arm_add_" + side).qposadr] = -np.pi / 2
    yaw = 0.0
    if human_reference is not None:
        hip_names = (
            ("LeftUpLeg", "RightUpLeg")
            if source_format == "lafan1"
            else ("LeftHip", "RightHip")
        )
        left = human_reference[hip_names[0]][0] - human_reference[hip_names[1]][0]
        if np.linalg.norm(left[:2]) < 1e-6:
            raise ValueError(
                "The BVH reference has no horizontal left/right hip separation"
            )
        yaw = np.arctan2(left[1], left[0]) - np.pi / 2
        data.qpos[3:7] = (
            R.from_euler("z", yaw) * R.from_quat(model.qpos0[3:7], scalar_first=True)
        ).as_quat(scalar_first=True)
    mj.mj_forward(model, data)
    reference = {
        body: [
            data.xpos[model.body(body).id].copy(),
            data.xquat[model.body(body).id].copy(),
        ]
        for body in BODY_MATCHES
    }
    # The shoulder axes are mirrored: the same -90 degree anatomical abduction
    # opens both arms horizontally. Check world height instead of assuming it.
    if source_format == "lafan1":
        for side in ("r", "l"):
            wrist = reference["hand_" + side][0]
            shoulder = reference["humerus_" + side][0]
            if abs(wrist[2] - shoulder[2]) > 0.06:
                raise ValueError(f"The {side} reference arm is not horizontal")
    return data.qpos.copy(), reference, float(yaw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=pathlib.Path, default=SOURCE_ROOT)
    parser.add_argument(
        "--template_xml",
        type=pathlib.Path,
        help="Reuse an adapted skeleton31 MJCF; defaults to the bundled base when scene_walking.xml is absent",
    )
    parser.add_argument(
        "--bvh_file",
        type=pathlib.Path,
        help="Actual motion BVH; its hierarchy offsets provide static calibration",
    )
    parser.add_argument("--format", choices=("lafan1", "xsens"), default="lafan1")
    parser.add_argument("--output_dir", type=pathlib.Path, default=OUTPUT_ROOT)
    parser.add_argument("--config_output", type=pathlib.Path)
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Compile and report topology without writing any output",
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output = output_dir / "human_skeleton.xml"
    config_path = args.config_output or (
        CONFIG_ROOT / f"bvh_{args.format}_to_human_skeleton.json"
    )
    template = args.template_xml
    if template is None and not (source_root / "scene_walking.xml").is_file():
        template = MODELS_ROOT / "skeleton31/base/human_skeleton.xml"
    if template is None:
        model_xml, source_files, mesh_files, cleanup = merge_scene(source_root)
        preparation_mode = "historical_muscle_scene_adaptation"
    else:
        model_xml, source_files, mesh_files, cleanup = load_template(template)
        preparation_mode = "bundled_skeleton31_template_calibration"
        print(f"Using existing skeleton31 template: {template}")
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    # Validate using resolved mesh paths before serializing portable references.
    model = mj.MjModel.from_xml_string(ET.tostring(model_xml, encoding="unicode"))
    if (model.nq, model.nv, model.njnt, model.neq) != (38, 37, 32, 8):
        raise ValueError("Unexpected adapted topology; inspect the source model")
    if model.ntendon != 0 or model.nu != 0:
        raise ValueError("The expected muscle-free source has no tendons or actuators")
    matches = BODY_MATCHES if args.format == "lafan1" else XSENS_BODY_MATCHES
    height_assumption = 1.75
    human = None
    source_reference_metadata = None
    if args.bvh_file:
        if args.format == "lafan1":
            human, source_reference_metadata = lafan1_source_reference(args.bvh_file)
        else:
            human, height_assumption = xsens_source_reference(args.bvh_file)
            source_reference_metadata = {
                "method": "xsens_zero_rotation_offsets",
                "uses_motion_frame": False,
            }
    qref, robot, yaw = target_reference(model, human, args.format)
    keyframes = model_xml.find("keyframe")
    if keyframes is None:
        keyframes = ET.SubElement(model_xml, "keyframe")
    ET.SubElement(
        keyframes,
        "key",
        name="retarget_reference",
        qpos=" ".join(format(float(x), ".17g") for x in qref),
    )
    ET.indent(model_xml, space="  ")
    # Recompile the complete artifact, including its reference keyframe.
    model = mj.MjModel.from_xml_string(ET.tostring(model_xml, encoding="unicode"))
    joint_names = [model.joint(i).name for i in range(1, model.njnt)]
    locked = [model.joint(int(model.eq_obj1id[i])).name for i in range(model.neq)]
    description = (
        "Static anatomical BVH reference calibration; no motion frame is used"
        if human is not None
        else "Uncalibrated skeleton reference; supply a BVH before retargeting"
    )
    reference_data = {
        "description": description,
        "calibrated": human is not None,
        "source_bvh": str(args.bvh_file.resolve()) if args.bvh_file else None,
        "source_format": args.format,
        "source_height_assumption": height_assumption,
        "source_reference_metadata": source_reference_metadata,
        "source_bind_pose": None
        if human is None
        else {name: [p.tolist(), q.tolist()] for name, (p, q) in human.items()},
        "target_reference_qpos_wxyz": qref.tolist(),
        "target_reference_yaw_radians": yaw,
        "target_body_poses_wxyz": {
            name: [p.tolist(), q.tolist()] for name, (p, q) in robot.items()
        },
        "dof_names": joint_names,
        "locked_joint_names": locked,
    }
    manifest = {
        "preparation_mode": preparation_mode,
        "source_files_sha256": source_hashes,
        "external_mesh_dependencies": [str(path) for path in mesh_files],
        "mujoco_version": mj.__version__,
        "nq": model.nq,
        "nv": model.nv,
        "internal_hinge_count": len(joint_names),
        "equality_count": model.neq,
        "site_count": model.nsite,
        "tendon_count": model.ntendon,
        "actuator_count": model.nu,
        "muscle_cleanup": cleanup,
        "adaptations": [
            "Expand scene_walking.xml includes and use relative mesh references",
            "Remove obsolete option collision attribute unsupported by MuJoCo 3",
            "Replace six pelvis scalar joints with one free root joint",
            "Enable the original internal hinge joint ranges",
            "Remove four auxiliary mocap bodies and the inactive pelvis weld",
            "Remove lower limb muscle tendons and their exclusive attachment sites",
            "Preserve skeletal bodies, geometry, contact pairs and eight joint locks",
        ]
        if template is None
        else [
            "Preserve the adapted skeleton31 template; replace its static calibration keyframe"
        ],
        "calibrated": human is not None,
        "calibration_bvh": str(args.bvh_file.resolve()) if args.bvh_file else None,
        "calibration_format": args.format,
        "calibration_bvh_sha256": (
            hashlib.sha256(args.bvh_file.read_bytes()).hexdigest()
            if args.bvh_file
            else None
        ),
        "requires_equality_constraints": True,
    }
    config = (
        calibrate(human, robot, matches, height_assumption)
        if human is not None
        else None
    )
    # Treat the originals as read-only and detect concurrent edits while
    # preparing a reproducible derivative.
    for path in source_files:
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[str(path)]:
            raise ValueError(f"Source changed during preparation: {path}")
    if not args.inspect_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        relative_mesh_paths(model_xml, output_dir)
        ET.ElementTree(model_xml).write(output, encoding="utf-8", xml_declaration=True)
        json_write(output_dir / "reference.json", reference_data)
        json_write(output_dir / "manifest.json", manifest)
        if config is not None:
            json_write(config_path, config)
            print(f"Calibrated {args.format} IK: {config_path}")
        else:
            print("No BVH supplied: reference is uncalibrated; no IK JSON written.")
    print(
        f"{'Inspected' if args.inspect_only else 'Prepared'} {output}: "
        f"nq={model.nq}, nv={model.nv}, {len(joint_names)} internal hinges, "
        f"{model.neq} equality locks, {model.nsite} sites, {model.ntendon} tendons"
    )
    print(
        f"Removed {len(cleanup['removed_tendon_names'])} muscle tendons and "
        f"{len(cleanup['removed_muscle_site_names'])} exclusive attachment sites"
    )


if __name__ == "__main__":
    main()
