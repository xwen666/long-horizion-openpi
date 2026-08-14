#!/usr/bin/env python3
"""Run structural and leakage checks on a planner dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from planner_dataset_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLANNER_DATASET,
    choose_dataset_path,
    episode_id,
    global_task,
    history,
    image_references,
    load_records,
    normalized_record,
    resolve_media_path,
    target,
)


def _add_issue(issues: list[dict[str, Any]], kind: str, sample_id: str, detail: str) -> None:
    if len(issues) < 500:
        issues.append({"type": kind, "sample_id": sample_id, "detail": detail})


def _source_segments(path: Path) -> tuple[list[int], dict[int, int]] | None:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=["subtask_idx", "frame_index"])
    except Exception:
        return None
    indices = table["subtask_idx"].to_pylist()
    frames = table["frame_index"].to_pylist()
    sequence: list[int] = []
    first_frame: dict[int, int] = {}
    previous = None
    for index, frame in zip(indices, frames):
        index = int(index)
        if index == previous:
            continue
        sequence.append(index)
        first_frame[index] = int(frame)
        previous = index
    return sequence, first_frame


def check(dataset_path: Path, data_root: Path, output_path: Path) -> dict[str, Any]:
    records = load_records(dataset_path)
    normalized = [normalized_record(record) for record in records]
    issues: list[dict[str, Any]] = []
    counts = Counter()
    by_episode: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for record, item in zip(records, normalized):
        by_episode[item["episode_id"]].append((record, item))

    source_cache: dict[str, tuple[list[int], dict[int, int]] | None] = {}
    sequence_failures = []
    boundary_failures = []
    for episode, items in by_episode.items():
        items.sort(key=lambda pair: (pair[1]["observation_timestep"] is None, pair[1]["observation_timestep"] or 0))
        sample_indices = [item["target"].get("subtask_idx") for _, item in items]
        sample_indices = [int(value) for value in sample_indices if value is not None]
        if len(sample_indices) != len(set(sample_indices)):
            counts["duplicate_subtask_samples"] += 1
            _add_issue(issues, "duplicate_subtask_sample", episode, str(sample_indices))

        previous_indices = []
        for record, item in items:
            sample_id = str(record.get("sample_id", episode))
            target_record = item["target"]
            target_index = target_record.get("subtask_idx")
            history_indices = [entry["subtask_idx"] for entry in item["history"] if "subtask_idx" in entry]
            if history_indices and target_index is not None and history_indices != sample_indices[: sample_indices.index(int(target_index))]:
                counts["history_sequence_mismatch"] += 1
                _add_issue(issues, "history_sequence_mismatch", sample_id, f"history={history_indices}, target_indices={sample_indices}")
            instruction = str(target_record.get("instruction", "") or "").strip()
            if not instruction:
                counts["empty_target"] += 1
                _add_issue(issues, "empty_target", sample_id, "target instruction is empty")
            if any(instruction == str(entry.get("instruction", "")).strip() for entry in item["history"]):
                # RoboCasa legitimately repeats navigation instructions at
                # different subtask_idx values. This is a warning, not a
                # structural failure, as long as the subtask indices differ.
                counts["repeated_target_instruction"] += 1
                _add_issue(issues, "warning_repeated_target_instruction", sample_id, instruction)
            if not global_task(record).strip():
                counts["empty_global_task"] += 1
                _add_issue(issues, "empty_global_task", sample_id, "global task is empty")
            obs_t = item["observation_timestep"]
            target_t = item["target_timestep"]
            if obs_t is not None and target_t is not None and float(obs_t) > float(target_t):
                counts["image_leakage"] += 1
                _add_issue(issues, "image_leakage", sample_id, f"observation={obs_t}, target={target_t}")
            for camera, ref in image_references(record).items():
                path = resolve_media_path(ref, data_root)
                if path is None or not path.exists():
                    counts["invalid_image_path"] += 1
                    _add_issue(issues, "invalid_image_path", sample_id, f"{camera}: {path}")

            source = record.get("source")
            if isinstance(source, dict) and source.get("parquet_path"):
                source_path = data_root / str(source["parquet_path"])
                cache_key = str(source_path)
                if cache_key not in source_cache:
                    source_cache[cache_key] = _source_segments(source_path) if source_path.exists() else None
                source_info = source_cache[cache_key]
                if source_info is None:
                    counts["missing_source_parquet"] += 1
                    _add_issue(issues, "missing_source_parquet", sample_id, str(source_path))
                else:
                    source_sequence, first_frames = source_info
                    if target_index is not None and int(target_index) in first_frames:
                        expected_frame = first_frames[int(target_index)]
                        if item["observation_timestep"] is not None and int(item["observation_timestep"]) != expected_frame:
                            counts["wrong_boundary_frame"] += 1
                            boundary_failures.append(sample_id)
                            _add_issue(issues, "wrong_boundary_frame", sample_id, f"expected={expected_frame}, actual={item['observation_timestep']}")

        source_paths = [record.get("source", {}).get("parquet_path") for record, _ in items if isinstance(record.get("source"), dict)]
        if source_paths:
            source_path = data_root / str(source_paths[0])
            source_info = source_cache.get(str(source_path))
            if source_info is None and source_path.exists():
                source_info = _source_segments(source_path)
                source_cache[str(source_path)] = source_info
            if source_info is not None:
                source_sequence, _ = source_info
                if sample_indices not in (source_sequence, source_sequence[:-1]):
                    counts["sequence_inconsistency"] += 1
                    sequence_failures.append(episode)
                    _add_issue(issues, "sequence_inconsistency", episode, f"source={source_sequence}, samples={sample_indices}")

    valid_samples = len(records) - sum(
        counts[key] for key in ("empty_target", "empty_global_task", "image_leakage", "invalid_image_path", "wrong_boundary_frame")
    )
    report = {
        "dataset_path": str(dataset_path.resolve()),
        "data_root": str(data_root.resolve()),
        "samples": len(records),
        "episodes": len(by_episode),
        "valid_samples_lower_bound": max(0, valid_samples),
        "checks": {
            "boundary_correctness": {
                "status": "pass" if not boundary_failures else "fail",
                "wrong_boundary_frames": len(boundary_failures),
            },
            "empty_target": {"status": "pass" if not counts["empty_target"] else "fail", "count": counts["empty_target"]},
            "history_leakage": {"status": "warning" if counts["repeated_target_instruction"] else "pass", "count": counts["repeated_target_instruction"]},
            "image_leakage": {"status": "pass" if not counts["image_leakage"] else "fail", "count": counts["image_leakage"]},
            "sequence_consistency": {"status": "pass" if not sequence_failures else "fail", "episodes_with_errors": len(sequence_failures)},
            "invalid_image_path": {"status": "pass" if not counts["invalid_image_path"] else "fail", "count": counts["invalid_image_path"]},
        },
        "issue_counts": dict(counts),
        "issues": issues,
        "notes": [
            "The current manifest uses the first frame of each subtask_idx segment as the observation and the active segment as the target instruction.",
            "The final terminal/task-complete segment is omitted by the default converter, hence source_sequence[:-1] is accepted.",
            "A raw previous_result field is not present in the current manifest; success/not_applicable is derived by the shared normalizer.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote quality report: {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PLANNER_DATASET)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/data_quality_report.json"))
    args = parser.parse_args()
    check(choose_dataset_path(args.dataset), args.data_root, args.output)


if __name__ == "__main__":
    main()
