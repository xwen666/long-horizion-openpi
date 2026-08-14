#!/usr/bin/env python3
"""Convert planner samples into Qwen3-VL SFT JSON messages.

The planner target is a semantic subtask, never an action or joint command.
For video-backed RoboCasa references, ``--image-mode frame`` materializes the
selected frame as a JPEG. ``--image-mode reference`` is useful for inspecting
the output but is not directly consumable by a standard Qwen image processor.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from planner_dataset_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLANNER_DATASET,
    choose_dataset_path,
    history,
    load_records,
    normalized_record,
    planner_image_references,
    resolve_media_path,
    target,
)


SYSTEM_PROMPT = """You are a high-level planner for a general-purpose robot.

Your task is to predict the next semantic subtask.

The user message contains two labeled camera views:
- agentview_left: front view
- eye_in_hand: wrist view

Use both camera views before identifying objects or left/right locations.

You do NOT predict:
- robot actions
- joint positions
- trajectories

You only output what the robot should do next."""


def _decode_frame(ref: dict[str, Any], data_root: Path) -> bytes:
    path = resolve_media_path(ref, data_root)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing image/video: {path}")
    frame_index = int(ref.get("frame_index") or 0)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        return path.read_bytes()
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index == frame_index:
                image = frame.to_image().convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=92)
                return buffer.getvalue()
    finally:
        container.close()
    raise IndexError(f"Frame {frame_index} unavailable in {path}")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "sample"


def _history_text(record: dict[str, Any]) -> str:
    entries = history(record)
    if not entries:
        return "None"
    lines = []
    for index, entry in enumerate(entries, start=1):
        skill = entry.get("skill") or "unknown_skill"
        stage = entry.get("stage") or "unknown_stage"
        result = entry.get("result") or "success"
        lines.append(f"{index}. [{skill}; {stage}; {result}] {entry.get('instruction', '')}")
    return "\n".join(lines)


def _prompt_text(record: dict[str, Any]) -> str:
    normalized = normalized_record(record)
    return f"""Global task:
{normalized['global_task']}

Completed subtasks:
{_history_text(record)}

Previous result:
{normalized['previous_result']}

Predict exactly one next subtask."""


def _image_content(
    *,
    role: str,
    ref: dict[str, Any],
    sample_index: int,
    record: dict[str, Any],
    data_root: Path,
    image_root: Path,
    image_mode: str,
) -> dict[str, Any]:
    if image_mode == "reference":
        path = resolve_media_path(ref, data_root)
        return {
            "type": "image",
            "image": {
                "video_path": str(path.resolve()) if path is not None else None,
                "frame_index": ref.get("frame_index", 0),
            },
        }
    sample_id = _safe_name(str(record.get("sample_id", f"sample_{sample_index:08d}")))
    output_path = image_root / f"{sample_index:08d}_{sample_id}_{role}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        data = _decode_frame(ref, data_root)
        output_path.write_bytes(data)
    return {"type": "image", "image": str(output_path.resolve())}


def _camera_label(role: str) -> str:
    return {
        "front_left": "agentview_left",
        "front_right": "agentview_right",
        "front": "agentview_left",
        "wrist": "eye_in_hand",
    }.get(role, role)


def convert(
    dataset_path: Path,
    data_root: Path,
    output_path: Path,
    image_mode: str,
    image_root: Path,
    max_samples: int | None,
) -> None:
    records = load_records(dataset_path)
    if max_samples is not None:
        records = records[:max_samples]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if image_mode == "frame":
        image_root.mkdir(parents=True, exist_ok=True)
    failures = []
    written = 0
    with output_path.open("w") as handle:
        for index, record in enumerate(records):
            try:
                normalized = normalized_record(record)
                target_record = target(record)
                mode = "done" if str(target_record.get("instruction", "")).strip().lower() in {"done", "task complete"} else "execute"
                user_content: list[dict[str, Any]] = []
                for role, ref in planner_image_references(record).items():
                    user_content.append({"type": "text", "text": f"Camera view: {_camera_label(role)}"})
                    user_content.append(
                        _image_content(
                            role=role,
                            ref=ref,
                            sample_index=index,
                            record=record,
                            data_root=data_root,
                            image_root=image_root,
                            image_mode=image_mode,
                        )
                    )
                user_content.append({"type": "text", "text": _prompt_text(record)})
                assistant = {
                    "mode": mode,
                    "instruction": str(target_record.get("instruction", "")),
                    "atomic_skill": str(target_record.get("skill", "")),
                    "stage": str(target_record.get("stage", "")),
                }
                handle.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
            except Exception as exc:
                failures.append({"index": index, "sample_id": record.get("sample_id"), "error": str(exc)})
                if len(failures) >= 100:
                    break
    summary = {
        "input": str(dataset_path.resolve()),
        "output": str(output_path.resolve()),
        "image_mode": image_mode,
        "written": written,
        "failed": len(failures),
        "failures": failures,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PLANNER_DATASET)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("robocasa/processed/composite_subtasks/qwen3vl.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("robocasa/processed/composite_subtasks/qwen3vl_images"))
    parser.add_argument("--image-mode", choices=("frame", "reference"), default="frame")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    convert(
        choose_dataset_path(args.dataset),
        args.data_root,
        args.output,
        args.image_mode,
        args.image_root,
        args.max_samples,
    )


if __name__ == "__main__":
    main()
