"""Model-independent validation helpers for locally generated motion."""

import hashlib
import pickle

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class ArrayUnpickler(pickle.Unpickler):
    """Allow only NumPy array reconstruction used by locally generated output."""

    def find_class(self, module, name):
        allowed = {
            ("numpy", "dtype"): np.dtype,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy._core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
            ("numpy.core.numeric", "_frombuffer"): np._core.numeric._frombuffer,
            (
                "numpy._core.multiarray",
                "_reconstruct",
            ): np._core.multiarray._reconstruct,
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported global: {module}.{name}")
        return allowed[module, name]

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("Persistent references are prohibited")


def summary(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not values.size:
        return {"count": 0, "min": None, "mean": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_reference_at_motion_times(reference_path, metadata, frames):
    """Evaluate source FK on the output's native source timeline, without IK."""
    with np.load(reference_path, allow_pickle=False) as reference:
        names = reference["body_names"].tolist()
        native_positions = reference["positions"].copy()
        native_rotations = reference["rotations"].copy()
        source_fps = float(reference["fps"])
        reference_start = (
            int(reference["start_frame"]) if "start_frame" in reference.files else 0
        )
    if (
        not np.isfinite(source_fps)
        or source_fps <= 0
        or len(native_positions) < 2
        or native_positions.shape != (len(native_positions), len(names), 3)
        or native_rotations.shape != (len(native_positions), len(names), 4)
        or not np.isfinite(native_positions).all()
        or not np.isfinite(native_rotations).all()
    ):
        raise ValueError("Invalid finite source FK reference shapes or frame rate")
    resampling = metadata.get("resampling", {})
    if resampling.get("enabled"):
        relative_times = np.asarray(
            resampling["source_sample_times_seconds"], dtype=float
        )
    else:
        relative_times = np.arange(frames, dtype=float) / source_fps
    if relative_times.shape != (frames,) or not np.isfinite(relative_times).all():
        raise ValueError("Invalid source sample times in resampling metadata")
    sample_times = (
        int(metadata.get("start_frame", 0)) - reference_start
    ) / source_fps + relative_times
    native_times = np.arange(len(native_positions), dtype=float) / source_fps
    if sample_times.min() < -1e-10 or sample_times.max() > native_times[-1] + 1e-10:
        raise ValueError("Source reference does not contain selected time interval")
    sample_times = np.clip(sample_times, 0, native_times[-1])
    columns = native_positions.reshape(len(native_positions), -1)
    positions = np.stack(
        [np.interp(sample_times, native_times, column) for column in columns.T],
        axis=1,
    ).reshape(frames, len(names), 3)
    rotations = np.stack(
        [
            Slerp(
                native_times,
                Rotation.from_quat(native_rotations[:, index], scalar_first=True),
            )(sample_times).as_quat(scalar_first=True)
            for index in range(len(names))
        ],
        axis=1,
    )
    return names, positions, rotations
