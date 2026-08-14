"""Shared readers and schema helpers for planner datasets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_PLANNER_DATASET = Path("robocasa/processed/composite_subtasks/annotations.jsonl")
DEFAULT_DATA_ROOT = Path("robocasa/datasets/v1.0/target/composite")


def _json_value(value: Any) -> Any:
    """Convert numpy/HDF5 scalar values into JSON-compatible Python values."""

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _records_from_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        for key in ("records", "samples", "data", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [dict(item) for item in nested if isinstance(item, Mapping)]
        return [dict(value)]
    raise ValueError("JSON dataset must contain an object or a list of objects")


def _read_hdf5(path: Path) -> list[dict[str, Any]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Reading HDF5 requires h5py. Install it with `python -m pip install h5py`."
        ) from exc

    def dataset_values(dataset: Any) -> Any:
        return _json_value(dataset[()])

    with h5py.File(path, "r") as handle:
        # Prefer a JSON/JSONL record dataset when present.
        for name in ("records", "samples", "data"):
            if name not in handle or not isinstance(handle[name], h5py.Dataset):
                continue
            value = dataset_values(handle[name])
            if isinstance(value, str):
                return _records_from_json(json.loads(value))
            if isinstance(value, list) and value and isinstance(value[0], str):
                return [dict(json.loads(item)) for item in value]

        datasets: dict[str, Any] = {}

        def visit(name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset):
                datasets[name] = dataset_values(node)

        handle.visititems(visit)
        if not datasets:
            return []

        # Columnar HDF5: datasets with the same first dimension become records.
        columns: dict[str, list[Any]] = {}
        lengths = []
        for name, values in datasets.items():
            if isinstance(values, list):
                columns[name] = values
                lengths.append(len(values))
        if not columns:
            return [{name: value for name, value in datasets.items()}]
        length = max(lengths)
        records = []
        for row in range(length):
            record = {}
            for name, values in columns.items():
                if row < len(values):
                    record[name] = values[row]
            records.append(record)
        return records


def load_records(path: Path) -> list[dict[str, Any]]:
    """Read JSON, JSONL, Parquet, or HDF5 into a list of record dictionaries."""

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = []
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                records.append(dict(value))
        return records
    if suffix == ".json":
        with path.open() as handle:
            return _records_from_json(json.load(handle))
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Reading Parquet requires pyarrow. Install it with `python -m pip install pyarrow`."
            ) from exc
        return [dict(row) for row in pq.read_table(path).to_pylist()]
    if suffix in {".h5", ".hdf5"}:
        return _read_hdf5(path)
    raise ValueError(f"Unsupported planner dataset format: {path}")


def choose_dataset_path(path: Path) -> Path:
    """Allow passing either a file or a directory containing a planner dataset."""

    if path.is_file():
        return path
    candidates = [
        path / "annotations.jsonl",
        path / "planner.jsonl",
        path / "samples.jsonl",
        path / "annotations.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    discovered = sorted(
        candidate
        for suffix in ("*.jsonl", "*.json", "*.parquet", "*.h5", "*.hdf5")
        for candidate in path.glob(suffix)
    )
    if len(discovered) == 1:
        return discovered[0]
    if not discovered:
        raise FileNotFoundError(f"No planner dataset file found below {path}")
    raise ValueError(f"Multiple planner dataset files found below {path}; pass the file explicitly")


def _first(record: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def episode_id(record: Mapping[str, Any]) -> str:
    sample_id = record.get("sample_id")
    if sample_id:
        parts = str(sample_id).split("/")
        for index, part in enumerate(parts):
            if part.startswith("episode_"):
                return "/".join(parts[: index + 1])
    value = _first(record, "episode_id", "episode", "episode_index")
    if value is not None:
        task = _first(record, "dataset_task", "task_name", "task")
        if task is not None:
            return f"{task}/episode_{int(value):06d}"
        return str(value)
    source = record.get("source")
    if isinstance(source, Mapping) and source.get("parquet_path"):
        return Path(str(source["parquet_path"])).stem
    return "<missing_episode_id>"


def global_task(record: Mapping[str, Any]) -> str:
    value = _first(record, "global_task", "task", "task_description", "instruction")
    return "" if value is None else str(value)


def _coerce_image_ref(value: Any, fallback_frame: Any = None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {"path": value, "frame_index": fallback_frame}
    if isinstance(value, Mapping):
        path = _first(value, "image", "image_path", "path", "video_path", "file")
        if path is None:
            return None
        return {
            "path": str(path),
            "frame_index": _first(value, "frame_index", "timestep", "frame")
            if _first(value, "frame_index", "timestep", "frame") is not None
            else fallback_frame,
        }
    return None


def image_references(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return image references while preserving the source camera names."""

    fallback_frame = _first(record, "frame_index", "timestep")
    raw = record.get("images")
    if isinstance(raw, Mapping):
        result = {}
        for key, value in raw.items():
            ref = _coerce_image_ref(value, fallback_frame)
            if ref is not None:
                result[str(key)] = ref
        return result

    observation = record.get("observation")
    result = {}
    if isinstance(observation, Mapping):
        for key, value in observation.items():
            ref = _coerce_image_ref(value, fallback_frame)
            if ref is not None:
                result[str(key)] = ref
    for key in ("front_image", "wrist_image", "front", "wrist"):
        ref = _coerce_image_ref(record.get(key), fallback_frame)
        if ref is not None:
            result[key] = ref
    return result


def _role_for_image(key: str) -> str | None:
    lowered = key.lower()
    if any(token in lowered for token in ("wrist", "eye_in_hand", "hand_camera")):
        return "wrist"
    if any(token in lowered for token in ("front", "agentview_left", "agentview_right", "agentview")):
        return "front"
    return None


def role_image_references(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    refs = image_references(record)
    roles: dict[str, dict[str, Any]] = {}
    front_key: str | None = None
    for key, ref in refs.items():
        role = _role_for_image(key)
        if role is not None and role not in roles:
            roles[role] = ref
            if role == "front":
                front_key = key
    # If a dataset uses arbitrary camera names, expose deterministic fallbacks.
    if "front" not in roles:
        for key in sorted(refs):
            if key not in {"wrist", "eye_in_hand"}:
                roles["front"] = refs[key]
                front_key = key
                break
    if "wrist" not in roles:
        for key in sorted(refs):
            if key != front_key and _role_for_image(key) == "wrist":
                roles["wrist"] = refs[key]
                break
    return roles


def planner_image_references(
    record: Mapping[str, Any], *, include_right_front: bool = False
) -> dict[str, dict[str, Any]]:
    """Return the ordered multi-camera views used by the VLM planner.

    The default Planner input intentionally matches the existing experiment:
    ``agentview_left + eye_in_hand``.  ``include_right_front=True`` can be used
    for a later three-view experiment without changing the current data path.
    """

    refs = image_references(record)
    roles: dict[str, dict[str, Any]] = {}
    fallback_front: dict[str, Any] | None = None

    for key, ref in refs.items():
        lowered = key.lower()
        if "agentview_left" in lowered or "front_left" in lowered:
            roles.setdefault("front_left", ref)
        elif include_right_front and ("agentview_right" in lowered or "front_right" in lowered):
            roles.setdefault("front_right", ref)
        elif any(token in lowered for token in ("wrist", "eye_in_hand", "hand_camera")):
            roles.setdefault("wrist", ref)
        elif "front" in lowered or "agentview" in lowered:
            fallback_front = fallback_front or ref

    # The current experiment uses the left agent camera as the front view.
    if "front_left" in roles:
        roles["front"] = roles.pop("front_left")
    elif fallback_front is not None:
        roles["front"] = fallback_front

    if "front" not in roles:
        old_roles = role_image_references(record)
        roles.setdefault("front", old_roles.get("front"))
        roles.setdefault("wrist", old_roles.get("wrist"))

    # Keep the visual token order stable across files and workers.
    ordered_roles = ("front", "front_right", "wrist")
    return {key: roles[key] for key in ordered_roles if roles.get(key) is not None}


def resolve_media_path(ref: Mapping[str, Any] | str, data_root: Path) -> Path | None:
    value = ref if isinstance(ref, str) else _first(ref, "path", "video_path", "image_path", "image")
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return data_root / path


def history(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = _first(record, "completed_subtask_history", "history", "completed_history")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    result = []
    for item in raw:
        if isinstance(item, Mapping):
            result.append(
                {
                    "instruction": str(_first(item, "instruction", "text", "task") or ""),
                    "skill": str(_first(item, "skill", "atomic_skill", "name") or ""),
                    "stage": str(_first(item, "stage") or ""),
                    **({"result": str(item["result"])} if "result" in item else {}),
                    **({"subtask_idx": item["subtask_idx"]} if "subtask_idx" in item else {}),
                }
            )
        else:
            result.append({"instruction": str(item), "skill": "", "stage": ""})
    return result


def target(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = _first(record, "target_subtask", "next_subtask", "target")
    if isinstance(raw, Mapping):
        result = dict(raw)
    else:
        result = {"instruction": raw if raw is not None else _first(record, "next_subtask_instruction", "target_instruction")}
    result.setdefault("instruction", "")
    result.setdefault("skill", _first(record, "next_subtask_name", "atomic_skill", "skill") or "")
    result.setdefault("stage", _first(record, "next_subtask_stage", "stage") or "")
    if "subtask_idx" not in result and "subtask_idx" in record:
        result["subtask_idx"] = record["subtask_idx"]
    if "frame_index" not in result:
        value = _first(record, "target_frame_index", "target_timestep", "transition_frame")
        if value is not None:
            result["frame_index"] = value
    return result


def observation_timestep(record: Mapping[str, Any]) -> int | float | None:
    value = _first(record, "observation_timestep", "observation_frame", "frame_index", "timestep")
    if value is None:
        source = record.get("source")
        if isinstance(source, Mapping):
            value = _first(source, "row_index", "frame_index", "timestep")
    return value


def target_timestep(record: Mapping[str, Any], target_record: Mapping[str, Any]) -> int | float | None:
    value = _first(target_record, "frame_index", "timestep", "transition_frame")
    if value is None:
        value = _first(record, "target_frame_index", "target_timestep", "transition_frame")
    if value is None:
        # Current boundary manifest observes the transition frame itself.
        value = observation_timestep(record)
    return value


def raw_previous_result(record: Mapping[str, Any]) -> tuple[str | None, bool]:
    value = _first(record, "previous_result", "prev_result", "execution_result")
    return (None if value is None else str(value), value is not None)


def derived_previous_result(record: Mapping[str, Any]) -> tuple[str, bool]:
    value, present = raw_previous_result(record)
    if present:
        return value or "unknown", False
    return ("success" if history(record) else "not_applicable"), True


def normalized_record(record: Mapping[str, Any]) -> dict[str, Any]:
    target_record = target(record)
    previous, derived = derived_previous_result(record)
    return {
        "episode_id": episode_id(record),
        "global_task": global_task(record),
        "images": image_references(record),
        "role_images": role_image_references(record),
        "history": history(record),
        "previous_result": previous,
        "previous_result_derived": derived,
        "target": target_record,
        "subtask_idx": record.get("subtask_idx"),
        "observation_timestep": observation_timestep(record),
        "target_timestep": target_timestep(record, target_record),
        "raw": record,
    }


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    yield from load_records(choose_dataset_path(path))
