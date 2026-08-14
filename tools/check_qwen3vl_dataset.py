#!/usr/bin/env python3
"""Validate a Qwen3-VL JSONL file against the source planner manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from planner_dataset_utils import (
    DEFAULT_PLANNER_DATASET,
    choose_dataset_path,
    history,
    load_records,
    normalized_record,
    target,
)


REQUIRED_ASSISTANT_KEYS = {"mode", "instruction", "atomic_skill", "stage"}
FORBIDDEN_ASSISTANT_KEYS = {"action", "actions", "joint", "joint_state", "trajectory", "action_chunk"}


def _assistant_json(message: Any) -> dict[str, Any] | None:
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _prompt_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text")


def _report_issue(issues: list[dict[str, Any]], kind: str, index: int, detail: str) -> None:
    if len(issues) < 500:
        issues.append({"index": index, "type": kind, "detail": detail})


def check(
    source_path: Path,
    qwen_path: Path,
    image_root: Path,
    report_path: Path,
    verify_images: bool,
) -> dict[str, Any]:
    source = load_records(source_path)
    qwen = load_records(qwen_path)
    counts = Counter()
    issues: list[dict[str, Any]] = []
    checked_images: set[str] = set()
    corrupt_images: set[str] = set()

    if len(source) != len(qwen):
        counts["sample_count_mismatch"] += 1
        _report_issue(issues, "sample_count_mismatch", -1, f"source={len(source)}, qwen={len(qwen)}")

    for index, (source_record, qwen_record) in enumerate(zip(source, qwen)):
        source_item = normalized_record(source_record)
        source_target = target(source_record)
        sample_id = source_record.get("sample_id", str(index))
        if not isinstance(qwen_record, dict) or not isinstance(qwen_record.get("messages"), list):
            counts["invalid_messages"] += 1
            _report_issue(issues, "invalid_messages", index, str(sample_id))
            continue
        messages = qwen_record["messages"]
        roles = [message.get("role") for message in messages if isinstance(message, dict)]
        if roles != ["system", "user", "assistant"]:
            counts["message_role_error"] += 1
            _report_issue(issues, "message_role_error", index, str(roles))
        user = messages[1] if len(messages) > 1 and isinstance(messages[1], dict) else {}
        user_content = user.get("content")
        if not isinstance(user_content, list):
            counts["invalid_user_content"] += 1
            _report_issue(issues, "invalid_user_content", index, str(sample_id))
            continue
        image_items = [item for item in user_content if isinstance(item, dict) and item.get("type") == "image"]
        # New RoboCasa Planner samples use left/right agent views plus wrist;
        # accept legacy two-view files so old experiments remain checkable.
        if len(image_items) not in {2, 3}:
            counts["image_count_error"] += 1
            _report_issue(issues, "image_count_error", index, f"count={len(image_items)}")
        for item in image_items:
            path_value = item.get("image")
            if not isinstance(path_value, str):
                counts["non_string_image_path"] += 1
                _report_issue(issues, "non_string_image_path", index, str(path_value))
                continue
            path = Path(path_value)
            if not path.is_absolute():
                path = image_root / path
            if not path.exists() or path.stat().st_size == 0:
                counts["missing_image"] += 1
                _report_issue(issues, "missing_image", index, str(path))
                continue
            checked_images.add(str(path))
            if verify_images and str(path) not in corrupt_images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as exc:
                    corrupt_images.add(str(path))
                    counts["corrupt_image"] += 1
                    _report_issue(issues, "corrupt_image", index, f"{path}: {exc}")

        prompt = _prompt_text(user_content)
        if source_item["global_task"] not in prompt:
            counts["global_task_not_in_prompt"] += 1
            _report_issue(issues, "global_task_not_in_prompt", index, str(sample_id))
        for entry in history(source_record):
            if str(entry.get("instruction", "")) not in prompt:
                counts["history_not_in_prompt"] += 1
                _report_issue(issues, "history_not_in_prompt", index, str(sample_id))
                break
        if source_item["previous_result"] not in prompt:
            counts["previous_result_not_in_prompt"] += 1
            _report_issue(issues, "previous_result_not_in_prompt", index, str(sample_id))

        assistant = _assistant_json(messages[2] if len(messages) > 2 else None)
        if assistant is None:
            counts["invalid_assistant_json"] += 1
            _report_issue(issues, "invalid_assistant_json", index, str(sample_id))
            continue
        missing_keys = REQUIRED_ASSISTANT_KEYS - set(assistant)
        if missing_keys:
            counts["assistant_missing_keys"] += 1
            _report_issue(issues, "assistant_missing_keys", index, str(sorted(missing_keys)))
        forbidden = FORBIDDEN_ASSISTANT_KEYS & set(assistant)
        if forbidden:
            counts["low_level_action_fields"] += 1
            _report_issue(issues, "low_level_action_fields", index, str(sorted(forbidden)))
        expected = {
            "mode": "done" if str(source_target.get("instruction", "")).strip().lower() in {"done", "task complete"} else "execute",
            "instruction": str(source_target.get("instruction", "")),
            "atomic_skill": str(source_target.get("skill", "")),
            "stage": str(source_target.get("stage", "")),
        }
        for key, value in expected.items():
            if assistant.get(key) != value:
                counts[f"target_{key}_mismatch"] += 1
                _report_issue(issues, f"target_{key}_mismatch", index, f"expected={value!r}, actual={assistant.get(key)!r}")

    report = {
        "source": str(source_path.resolve()),
        "qwen_dataset": str(qwen_path.resolve()),
        "image_root": str(image_root.resolve()),
        "source_samples": len(source),
        "qwen_samples": len(qwen),
        "checked_samples": min(len(source), len(qwen)),
        "checked_unique_images": len(checked_images),
        "verify_images": verify_images,
        "issue_counts": dict(counts),
        "issues": issues,
        "status": "pass" if not counts else "fail",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote Qwen3-VL quality report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("robocasa/processed/composite_subtasks/splits/train.jsonl"))
    parser.add_argument("--qwen", type=Path, default=Path("robocasa/processed/composite_subtasks/qwen3vl_train.jsonl"))
    parser.add_argument("--image-root", type=Path, default=Path("robocasa/processed/composite_subtasks/qwen3vl_images/train"))
    parser.add_argument("--output", type=Path, default=Path("outputs/qwen3vl_train_quality_report.json"))
    parser.add_argument("--skip-image-decode", action="store_true")
    args = parser.parse_args()
    check(
        choose_dataset_path(args.source),
        choose_dataset_path(args.qwen),
        args.image_root,
        args.output,
        verify_images=not args.skip_image_decode,
    )


if __name__ == "__main__":
    main()
