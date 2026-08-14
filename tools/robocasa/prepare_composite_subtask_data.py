#!/usr/bin/env python3
"""Inspect and convert RoboCasa composite-task annotations.

RoboCasa's LeRobot files store annotation text as integer indices.  The indices
are resolved through ``meta/tasks.jsonl``.  This tool keeps that indirection
explicit and produces a JSONL manifest whose images are references to source
video frames, rather than copying millions of image files.

Examples:

    # Print the real schema, decoded annotations, and subtask transitions.
    python tools/robocasa/prepare_composite_subtask_data.py inspect \
        --task ArrangeBreadBasket --episode 0

    # Convert all 32 downloaded composite tasks.
    python tools/robocasa/prepare_composite_subtask_data.py convert \
        --output-dir robocasa/processed/composite_subtasks

The default conversion emits one sample at the first frame of every
``subtask_idx`` segment.  A sample has the form:

    global task + current multi-view image + completed subtask history
        -> next subtask instruction

Use ``--sample-mode every-frame`` only when a frame-level manifest is really
needed; the current dataset contains about 12.7 million frames.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LOGGER = logging.getLogger("robocasa_subtasks")

TASK_DESCRIPTION_KEY = "annotation.human.task_description"
TASK_NAME_KEY = "annotation.human.task_name"
SUBTASK_KEY = "annotation.human.subtask"
SUBTASK_NAME_KEY = "annotation.human.subtask_name"
SUBTASK_STAGE_KEY = "annotation.human.subtask_stage"
SUBTASK_INDEX_KEY = "subtask_idx"
FRAME_INDEX_KEY = "frame_index"

ANNOTATION_KEYS = (
    TASK_DESCRIPTION_KEY,
    TASK_NAME_KEY,
    SUBTASK_KEY,
    SUBTASK_NAME_KEY,
    SUBTASK_STAGE_KEY,
    SUBTASK_INDEX_KEY,
)

TERMINAL_TEXT = {"done", "task complete", "complete", "finished"}


def _load_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the active venv
        raise RuntimeError(
            "pyarrow is required. Install it in the active environment with "
            "`python -m pip install pyarrow`."
        ) from exc
    return pq


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _task_map(dataset_dir: Path) -> dict[int, str]:
    rows = _read_jsonl(dataset_dir / "meta" / "tasks.jsonl")
    return {int(row["task_index"]): str(row["task"]) for row in rows}


def _load_info(dataset_dir: Path) -> dict[str, Any]:
    with (dataset_dir / "meta" / "info.json").open() as f:
        return json.load(f)


def _find_dataset_dir(data_root: Path, task: str) -> Path:
    candidates = sorted(data_root.glob(f"{task}/*/lerobot"))
    if not candidates:
        raise FileNotFoundError(
            f"No LeRobot dataset found for task {task!r} below {data_root}"
        )
    if len(candidates) > 1:
        LOGGER.warning("Multiple versions found for %s; using %s", task, candidates[-1])
    return candidates[-1]


def _all_dataset_dirs(data_root: Path, tasks: Iterable[str] | None) -> list[tuple[str, Path]]:
    selected = set(tasks) if tasks else None
    result: list[tuple[str, Path]] = []
    for task_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        if selected is not None and task_dir.name not in selected:
            continue
        candidates = sorted(task_dir.glob("*/lerobot"))
        if not candidates:
            LOGGER.warning("Skipping %s: no lerobot directory", task_dir)
            continue
        if len(candidates) > 1:
            LOGGER.warning("Multiple versions found for %s; using %s", task_dir.name, candidates[-1])
        result.append((task_dir.name, candidates[-1]))
    if selected:
        found = {task for task, _ in result}
        missing = sorted(selected - found)
        if missing:
            raise FileNotFoundError(f"Requested composite task(s) not found: {', '.join(missing)}")
    return result


def _episode_index(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.parquet$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def _episode_files(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "data").glob("**/episode_*.parquet"), key=_episode_index)


def _decode(task_map: dict[int, str], values: dict[str, Any], key: str, row: int) -> str:
    raw = int(values[key][row])
    return task_map.get(raw, f"<missing task_index={raw}>")


def _video_keys(info: dict[str, Any]) -> list[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if key.startswith("observation.images.") and feature.get("dtype") == "video"
    ]


def _video_path(dataset_dir: Path, info: dict[str, Any], image_key: str, episode: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    relative = template.format(
        episode_chunk=episode // chunk_size,
        episode_index=episode,
        video_key=image_key,
    )
    path = dataset_dir / relative
    if not path.exists():
        raise FileNotFoundError(f"Missing video for {image_key}, episode {episode}: {path}")
    return path


def _relative_path(path: Path, root: Path, absolute: bool) -> str:
    if absolute:
        return str(path.resolve())
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_terminal(instruction: str, name: str, stage: str) -> bool:
    return (
        instruction.strip().lower() in TERMINAL_TEXT
        or name.strip().lower() in TERMINAL_TEXT
        or stage.strip().lower() in TERMINAL_TEXT
    )


def _read_annotation_table(path: Path) -> dict[str, list[Any]]:
    pq = _load_pyarrow()
    columns = [FRAME_INDEX_KEY, *ANNOTATION_KEYS]
    table = pq.read_table(path, columns=columns)
    return {column: table[column].to_pylist() for column in columns}


def _segment_rows(values: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Return the first row of each consecutive subtask_idx segment.

    ``subtask_idx`` is the ordering signal.  The text and primitive task fields
    are taken from the first row of that segment, so the generated label is
    tied to the exact image at the transition.
    """

    result: list[dict[str, Any]] = []
    previous_idx: int | None = None
    for row in range(len(values[SUBTASK_INDEX_KEY])):
        index = int(values[SUBTASK_INDEX_KEY][row])
        if index == previous_idx:
            continue
        result.append(
            {
                "row": row,
                "subtask_idx": index,
                "subtask": int(values[SUBTASK_KEY][row]),
                "subtask_name": int(values[SUBTASK_NAME_KEY][row]),
                "subtask_stage": int(values[SUBTASK_STAGE_KEY][row]),
            }
        )
        previous_idx = index
    return result


def _completed_history_entry(
    task_map: dict[int, str], values: dict[str, list[Any]], segment: dict[str, Any]
) -> dict[str, Any]:
    """Describe a segment known to be complete because the next segment began."""

    row = segment["row"]
    return {
        "subtask_idx": segment["subtask_idx"],
        "instruction": _decode(task_map, values, SUBTASK_KEY, row),
        "skill": _decode(task_map, values, SUBTASK_NAME_KEY, row),
        "stage": _decode(task_map, values, SUBTASK_STAGE_KEY, row),
        "result": "success",
    }


def _sample_from_segment(
    *,
    task: str,
    dataset_dir: Path,
    data_root: Path,
    info: dict[str, Any],
    task_map: dict[int, str],
    values: dict[str, list[Any]],
    parquet_path: Path,
    segment: dict[str, Any],
    history: list[dict[str, Any]],
    image_keys: list[str],
    absolute_paths: bool,
    include_terminal: bool,
) -> dict[str, Any] | None:
    row = segment["row"]
    global_task = _decode(task_map, values, TASK_DESCRIPTION_KEY, row)
    task_name = _decode(task_map, values, TASK_NAME_KEY, row)
    instruction = _decode(task_map, values, SUBTASK_KEY, row)
    primitive_name = _decode(task_map, values, SUBTASK_NAME_KEY, row)
    stage = _decode(task_map, values, SUBTASK_STAGE_KEY, row)
    terminal = _is_terminal(instruction, primitive_name, stage)
    if terminal and not include_terminal:
        return None

    episode = _episode_index(parquet_path)
    frame = int(values[FRAME_INDEX_KEY][row])
    images = {}
    for image_key in image_keys:
        video = _video_path(dataset_dir, info, image_key, episode)
        images[image_key] = {
            "video_path": _relative_path(video, data_root, absolute_paths),
            "frame_index": frame,
        }

    return {
        "sample_id": f"{task}/episode_{episode:06d}/subtask_{segment['subtask_idx']:03d}",
        "dataset_task": task,
        "global_task": global_task,
        "task_name": task_name,
        "episode_index": episode,
        "frame_index": frame,
        "subtask_idx": segment["subtask_idx"],
        "next_subtask_instruction": instruction,
        "next_subtask_name": primitive_name,
        "next_subtask_stage": stage,
        "completed_subtask_history": history,
        "images": images,
        "source": {
            "parquet_path": _relative_path(parquet_path, data_root, absolute_paths),
            "row_index": row,
        },
        "terminal": terminal,
    }


def inspect_dataset(data_root: Path, task: str | None, episode: int, max_transitions: int) -> None:
    if task is None:
        tasks = _all_dataset_dirs(data_root, None)
        if not tasks:
            raise FileNotFoundError(f"No composite LeRobot datasets below {data_root}")
        task = tasks[0][0]
    dataset_dir = _find_dataset_dir(data_root, task)
    info = _load_info(dataset_dir)
    task_map = _task_map(dataset_dir)
    files = _episode_files(dataset_dir)
    matching = [path for path in files if _episode_index(path) == episode]
    if not matching:
        raise FileNotFoundError(f"Episode {episode} not found in {dataset_dir}")
    parquet_path = matching[0]
    values = _read_annotation_table(parquet_path)

    print(f"dataset_dir: {dataset_dir}")
    print(f"episode: {episode}")
    print(f"rows: {len(values[FRAME_INDEX_KEY])}")
    print(f"image_keys: {_video_keys(info)}")
    print("annotation columns:")
    for key in ANNOTATION_KEYS:
        feature = info.get("features", {}).get(key, {})
        print(f"  {key}: dtype={feature.get('dtype')} shape={feature.get('shape')}")

    print("decoded task index map used by this episode:")
    used_indices = set()
    for key in (TASK_DESCRIPTION_KEY, TASK_NAME_KEY, SUBTASK_KEY, SUBTASK_NAME_KEY, SUBTASK_STAGE_KEY):
        used_indices.update(int(value) for value in values[key])
    for index in sorted(used_indices):
        print(f"  {index}: {task_map.get(index, '<missing>')}")

    print("subtask transitions:")
    segments = _segment_rows(values)
    for segment in segments[:max_transitions]:
        row = segment["row"]
        print(
            "  "
            f"frame={int(values[FRAME_INDEX_KEY][row])} "
            f"subtask_idx={segment['subtask_idx']} "
            f"subtask={_decode(task_map, values, SUBTASK_KEY, row)!r} "
            f"subtask_name={_decode(task_map, values, SUBTASK_NAME_KEY, row)!r} "
            f"stage={_decode(task_map, values, SUBTASK_STAGE_KEY, row)!r}"
        )
    if len(segments) > max_transitions:
        print(f"  ... {len(segments) - max_transitions} more transitions")


def convert(
    data_root: Path,
    output_dir: Path,
    tasks: list[str] | None,
    sample_mode: str,
    include_terminal: bool,
    absolute_paths: bool,
    max_episodes: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "annotations.jsonl"
    summary_path = output_dir / "summary.json"
    datasets = _all_dataset_dirs(data_root, tasks)
    if not datasets:
        raise FileNotFoundError(f"No composite LeRobot datasets below {data_root}")

    stats: Counter[str] = Counter()
    task_summary: dict[str, dict[str, Any]] = {}
    total_samples = 0
    with manifest_path.open("w") as manifest:
        for task, dataset_dir in datasets:
            info = _load_info(dataset_dir)
            task_map = _task_map(dataset_dir)
            image_keys = _video_keys(info)
            episode_files = _episode_files(dataset_dir)
            if max_episodes is not None:
                episode_files = episode_files[:max_episodes]
            task_stat: Counter[str] = Counter()
            task_samples = 0
            for episode_number, parquet_path in enumerate(episode_files, start=1):
                values = _read_annotation_table(parquet_path)
                segments = _segment_rows(values)
                history: list[dict[str, Any]] = []
                emitted_for_episode = 0
                if sample_mode == "every-frame":
                    # Compute the completed history once per segment, then reuse it
                    # for every frame in that segment.
                    segment_histories: list[list[dict[str, Any]]] = []
                    segment_history: list[dict[str, Any]] = []
                    for segment in segments:
                        segment_histories.append(list(segment_history))
                        row = segment["row"]
                        instruction = _decode(task_map, values, SUBTASK_KEY, row)
                        primitive = _decode(task_map, values, SUBTASK_NAME_KEY, row)
                        stage = _decode(task_map, values, SUBTASK_STAGE_KEY, row)
                        if not _is_terminal(instruction, primitive, stage):
                            segment_history.append(_completed_history_entry(task_map, values, segment))

                    # For every frame, select the segment containing that frame.
                    segment_number = 0
                    for row in range(len(values[SUBTASK_INDEX_KEY])):
                        while (
                            segment_number + 1 < len(segments)
                            and segments[segment_number + 1]["row"] <= row
                        ):
                            segment_number += 1
                        segment = segments[segment_number]
                        current = {
                            "row": row,
                            "subtask_idx": segment["subtask_idx"],
                            "subtask": segment["subtask"],
                            "subtask_name": segment["subtask_name"],
                            "subtask_stage": segment["subtask_stage"],
                        }
                        sample = _sample_from_segment(
                            task=task,
                            dataset_dir=dataset_dir,
                            data_root=data_root,
                            info=info,
                            task_map=task_map,
                            values=values,
                            parquet_path=parquet_path,
                            segment=current,
                            history=segment_histories[segment_number],
                            image_keys=image_keys,
                            absolute_paths=absolute_paths,
                            include_terminal=include_terminal,
                        )
                        if sample is not None:
                            sample["sample_id"] = f"{task}/episode_{_episode_index(parquet_path):06d}/frame_{int(values[FRAME_INDEX_KEY][row]):06d}"
                            manifest.write(json.dumps(sample, ensure_ascii=False) + "\n")
                            emitted_for_episode += 1
                else:
                    for segment in segments:
                        sample = _sample_from_segment(
                            task=task,
                            dataset_dir=dataset_dir,
                            data_root=data_root,
                            info=info,
                            task_map=task_map,
                            values=values,
                            parquet_path=parquet_path,
                            segment=segment,
                            history=history,
                            image_keys=image_keys,
                            absolute_paths=absolute_paths,
                            include_terminal=include_terminal,
                        )
                        instruction = _decode(task_map, values, SUBTASK_KEY, segment["row"])
                        primitive = _decode(task_map, values, SUBTASK_NAME_KEY, segment["row"])
                        stage = _decode(task_map, values, SUBTASK_STAGE_KEY, segment["row"])
                        if sample is not None:
                            manifest.write(json.dumps(sample, ensure_ascii=False) + "\n")
                            emitted_for_episode += 1
                        if not _is_terminal(instruction, primitive, stage):
                            history.append(_completed_history_entry(task_map, values, segment))
                task_samples += emitted_for_episode
                task_stat["episodes"] += 1
                task_stat["samples"] += emitted_for_episode
                task_stat["frames"] += len(values[FRAME_INDEX_KEY])
                if episode_number % 50 == 0 or episode_number == len(episode_files):
                    LOGGER.info("%s: %d/%d episodes", task, episode_number, len(episode_files))
            task_summary[task] = {
                "dataset_dir": str(dataset_dir),
                "episodes": task_stat["episodes"],
                "frames": task_stat["frames"],
                "samples": task_samples,
                "image_keys": image_keys,
            }
            stats.update(task_stat)
            total_samples += task_samples

    summary = {
        "data_root": str(data_root.resolve()),
        "output_manifest": str(manifest_path.resolve()),
        "sample_mode": sample_mode,
        "include_terminal": include_terminal,
        "absolute_paths": absolute_paths,
        "num_tasks": len(datasets),
        "total_episodes": stats["episodes"],
        "total_frames": stats["frames"],
        "total_samples": total_samples,
        "tasks": task_summary,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    LOGGER.info("Wrote %d samples to %s", total_samples, manifest_path)
    LOGGER.info("Wrote summary to %s", summary_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inspect", "convert"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("robocasa/datasets/v1.0/target/composite"),
        help="Directory containing the 32 composite task directories.",
    )
    parser.add_argument("--task", help="Task name for inspect mode, e.g. ArrangeBreadBasket.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index for inspect mode.")
    parser.add_argument("--max-transitions", type=int, default=100)
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Subset of task directory names for convert mode; default is all tasks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("robocasa/processed/composite_subtasks"),
    )
    parser.add_argument(
        "--sample-mode",
        choices=("subtask-start", "every-frame"),
        default="subtask-start",
        help="Emit one sample per subtask transition, or one sample per frame.",
    )
    parser.add_argument(
        "--include-terminal",
        action="store_true",
        help="Also emit the terminal 'done/task complete' segment.",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Store absolute source paths instead of paths relative to --data-root.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Conversion smoke test: process at most this many episodes per task.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Composite data root does not exist: {data_root}")
    if args.mode == "inspect":
        inspect_dataset(data_root, args.task, args.episode, args.max_transitions)
    else:
        convert(
            data_root=data_root,
            output_dir=args.output_dir.resolve(),
            tasks=args.tasks,
            sample_mode=args.sample_mode,
            include_terminal=args.include_terminal,
            absolute_paths=args.absolute_paths,
            max_episodes=args.max_episodes,
        )


if __name__ == "__main__":
    main()
