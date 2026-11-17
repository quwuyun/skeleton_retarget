"""Render a six-field motion using its matching MJCF, without physics steps.

Example (run from the repository root):
    uv run python -m skeleton_retarget.scripts.preview_motion \
        --robot_xml MODEL.xml --motion MOTION.npz --output preview/walk

Writes an eight-pose PNG contact sheet and a real-time MP4. EGL is selected
before importing MuJoCo unless MUJOCO_GL was explicitly set by the caller.
Only a temporary XML receives the visual floor/lights; source files and poses
are never changed. The floor is not a contact or dynamics validation.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw

from skeleton_retarget.retargeting.motion import load_motion


def visual_model(xml_path, temporary_dir, width, height):
    """Add rendering assets to a temporary copy, resolving external file paths."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.find("include") is not None:
        raise ValueError(
            "Preview requires a self-contained XML; expand includes first."
        )
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    asset_dir = compiler.get("assetdir", "")
    for tag, directory in (("mesh", "meshdir"), ("texture", "texturedir")):
        base = xml_path.parent / compiler.get(directory, asset_dir)
        for asset in root.findall(f"./asset/{tag}[@file]"):
            asset.set("file", str((base / asset.get("file")).resolve()))
        compiler.attrib.pop(directory, None)
    compiler.attrib.pop("assetdir", None)

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_settings = visual.find("global")
    if global_settings is None:
        global_settings = ET.SubElement(visual, "global")
    global_settings.set("offwidth", str(width))
    global_settings.set("offheight", str(height))
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.attrib.update(ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="preview_grid_texture",
        type="2d",
        builtin="checker",
        rgb1="0.17 0.21 0.25",
        rgb2="0.23 0.28 0.32",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="preview_grid_material",
        texture="preview_grid_texture",
        texrepeat="16 16",
        reflectance="0.1",
    )
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody.")
    # No joints or bodies are added, preserving qpos and body ordering exactly.
    ET.SubElement(
        worldbody,
        "geom",
        name="preview_visual_floor",
        type="plane",
        size="8 8 0.1",
        pos="0 0 0",
        material="preview_grid_material",
        contype="0",
        conaffinity="0",
        group="0",
        rgba="1 1 1 1",
    )
    ET.SubElement(
        worldbody,
        "light",
        name="preview_key_light",
        directional="true",
        pos="0 -3 5",
        dir="0 0 -1",
        diffuse="0.7 0.7 0.7",
    )
    target = Path(temporary_dir) / "preview_scene.xml"
    tree.write(target, encoding="utf-8", xml_declaration=True)
    return mj.MjModel.from_xml_path(str(target))


def camera_parameters(model, qpos):
    """Use actual mesh bounds and travel heading, not the model's default pose."""
    data = mj.MjData(model)
    geoms = np.flatnonzero(model.geom_bodyid != 0)
    bounds = []
    for frame in qpos[np.unique(np.linspace(0, len(qpos) - 1, 8).astype(int))]:
        data.qpos[:] = frame
        mj.mj_forward(model, data)
        rotations = data.geom_xmat[geoms].reshape(-1, 3, 3)
        centers = data.geom_xpos[geoms] + np.einsum(
            "gij,gj->gi", rotations, model.geom_aabb[geoms, :3]
        )
        extents = np.einsum("gij,gj->gi", np.abs(rotations), model.geom_aabb[geoms, 3:])
        bounds.extend((centers - extents - frame[:3], centers + extents - frame[:3]))
    bounds = np.concatenate(bounds)
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    height = max(float(upper[2] - lower[2]), 1.6)
    center_offset = (lower + upper) / 2
    displacement = qpos[-1, :2] - qpos[0, :2]
    if np.linalg.norm(displacement) < 0.01:
        # Fall back to the initial root's forward axis for stationary motions.
        data.qpos[:] = qpos[0]
        mj.mj_forward(model, data)
        forward = data.xmat[int(model.jnt_bodyid[0])].reshape(3, 3)[:, 0]
        displacement = forward[:2]
    heading = math.degrees(math.atan2(displacement[1], displacement[0]))
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.distance = height * 1.8
    camera.elevation = -12
    camera.azimuth = heading + 115
    return camera, center_offset, heading


def annotate(rgb, label):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 34), fill=(20, 25, 31))
    draw.text((12, 10), label, fill=(245, 245, 245))
    return image


def render_preview(args):
    xml_path = args.robot_xml.expanduser().resolve(strict=True)
    motion_path = args.motion.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.suffix.lower() in {".png", ".mp4"}:
        output = output.with_suffix("")
    if args.width < 320 or args.height < 240 or args.width % 2 or args.height % 2:
        raise ValueError("Width/height must be even and at least 320/240.")
    if not math.isfinite(args.video_fps) or args.video_fps <= 0:
        raise ValueError("video_fps must be finite and positive.")
    ffmpeg = shutil.which("ffmpeg")
    if not args.sheet_only and ffmpeg is None:
        raise RuntimeError("ffmpeg is unavailable; use --sheet_only for a PNG preview.")
    source_model = mj.MjModel.from_xml_path(str(xml_path))
    fps, qpos = load_motion(motion_path, source_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet_indices = np.unique(np.linspace(0, len(qpos) - 1, 8).round().astype(int))
    duration = (len(qpos) - 1) / fps
    # Select existing saved poses, retaining source timing to within one frame.
    video_times = np.arange(math.floor(duration * args.video_fps) + 1) / args.video_fps
    video_indices = np.minimum(np.rint(video_times * fps).astype(int), len(qpos) - 1)
    with tempfile.TemporaryDirectory(prefix="gmr_motion_preview_") as temporary:
        model = visual_model(xml_path, temporary, args.width, args.height)
        if model.nq != source_model.nq or model.nbody != source_model.nbody:
            raise ValueError(
                "Visual scene unexpectedly changed model coordinates/bodies."
            )
        camera, center_offset, heading = camera_parameters(model, qpos)
        data = mj.MjData(model)
        options = mj.MjvOption()
        options.geomgroup[3] = int(args.show_collision)
        options.sitegroup[:] = 0
        options.flags[mj.mjtVisFlag.mjVIS_INERTIA] = False
        with mj.Renderer(model, height=args.height, width=args.width) as renderer:

            def frame(index):
                data.qpos[:] = qpos[index]
                mj.mj_forward(model, data)
                camera.lookat[:] = qpos[index, :3] + center_offset
                renderer.update_scene(data, camera=camera, scene_option=options)
                return annotate(
                    renderer.render(),
                    f"{motion_path.stem} | frame {index}/{len(qpos) - 1} | "
                    f"t={index / fps:.3f}s | kinematic FK",
                )

            thumbnail_width = min(args.width, 640)
            thumbnail_height = round(args.height * thumbnail_width / args.width)
            sheet = Image.new(
                "RGB", (thumbnail_width * 4, thumbnail_height * 2), (20, 25, 31)
            )
            for cell, index in enumerate(sheet_indices):
                thumbnail = frame(int(index)).resize(
                    (thumbnail_width, thumbnail_height)
                )
                sheet.paste(
                    thumbnail,
                    ((cell % 4) * thumbnail_width, (cell // 4) * thumbnail_height),
                )
            sheet.save(output.with_suffix(".png"))
            print(f"Saved {output.with_suffix('.png')}", flush=True)

            if not args.sheet_only:
                command = [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{args.width}x{args.height}",
                    "-r",
                    str(args.video_fps),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output.with_suffix(".mp4")),
                ]
                with tempfile.TemporaryFile() as error_log:
                    encoder = subprocess.Popen(
                        command, stdin=subprocess.PIPE, stderr=error_log
                    )
                    try:
                        for index in video_indices:
                            encoder.stdin.write(np.asarray(frame(int(index))).tobytes())
                        encoder.stdin.close()
                        result = encoder.wait()
                    except BaseException:
                        encoder.kill()
                        encoder.wait()
                        raise
                    if result:
                        error_log.seek(0)
                        raise RuntimeError(error_log.read().decode(errors="replace"))
                print(f"Saved {output.with_suffix('.mp4')}", flush=True)
    metadata = {
        "robot_xml": str(xml_path),
        "motion": str(motion_path),
        "robot_xml_sha256": hashlib.sha256(xml_path.read_bytes()).hexdigest(),
        "motion_sha256": hashlib.sha256(motion_path.read_bytes()).hexdigest(),
        "source_fps": fps,
        "source_frames": len(qpos),
        "sample_span_seconds": duration,
        "model_nq": source_model.nq,
        "model_bodies_without_world": source_model.nbody - 1,
        "render_width": args.width,
        "render_height": args.height,
        "render_fps": args.video_fps,
        "render_frames": len(video_indices) if not args.sheet_only else 0,
        "sheet_source_frames": sheet_indices.tolist(),
        "camera_travel_heading_degrees": heading,
        "show_collision_primitives": args.show_collision,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "note": "Saved qpos replay with mj_forward only; temporary visual floor, no simulation or pose correction.",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_xml", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output prefix, or a .png/.mp4 filename.",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--video_fps", type=float, default=30.0)
    parser.add_argument(
        "--show_collision",
        action="store_true",
        help="Show translucent collision primitives.",
    )
    parser.add_argument(
        "--sheet_only",
        action="store_true",
        help="Write PNG/metadata without encoding MP4.",
    )
    render_preview(parser.parse_args())


if __name__ == "__main__":
    main()
