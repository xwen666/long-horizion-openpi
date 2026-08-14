#!/usr/bin/env python3
"""Evaluate the Planner with its own predicted subtask history.

This is a history-closed-loop evaluation with recorded observations: images are
still taken from the held-out boundary frames, but after each prediction the
next prompt receives the model's previous JSON instead of the ground-truth
``Completed subtasks`` history.  It isolates history error accumulation from
low-level execution and observation drift.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from evaluate_qwen3vl_planner import _move_inputs, _parse_json
from planner_dataset_utils import episode_id, load_records
from train_qwen3vl_planner import _materialize_images


FIELDS = ("mode", "instruction", "atomic_skill", "stage")


def _history_text(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return "None"
    lines = []
    for index, entry in enumerate(entries, start=1):
        skill = entry.get("skill") or "unknown_skill"
        stage = entry.get("stage") or "unknown_stage"
        result = entry.get("result") or "success"
        lines.append(f"{index}. [{skill}; {stage}; {result}] {entry.get('instruction', '')}")
    return "\n".join(lines)


def _prompt_text(source: dict[str, Any], predicted_history: list[dict[str, Any]]) -> str:
    global_task = source.get("global_task") or source.get("task") or ""
    previous_result = "success" if predicted_history else "not_applicable"
    return f"""Global task:
{global_task}

Completed subtasks:
{_history_text(predicted_history)}

Previous result:
{previous_result}

Predict exactly one next subtask."""


def _with_history(messages: list[dict[str, Any]], prompt_text: str) -> list[dict[str, Any]]:
    result = copy.deepcopy(messages)
    user = result[1]
    content = user.get("content", [])
    for item in reversed(content):
        if isinstance(item, dict) and item.get("type") == "text":
            text = str(item.get("text", ""))
            if not text.startswith("Camera view:"):
                item["text"] = prompt_text
                return result
    raise ValueError("Could not find the Planner prompt text in the Qwen record")


def _predict(model: Any, processor: Any, messages: list[dict[str, Any]], device: torch.device, max_new_tokens: int) -> tuple[dict[str, Any] | None, str]:
    prompt = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    prompt = _move_inputs(prompt, device)
    with torch.inference_mode():
        generated = model.generate(**prompt, max_new_tokens=max_new_tokens, do_sample=False)
    generated_tokens = generated[:, prompt["input_ids"].shape[-1] :]
    raw = processor.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return _parse_json(raw), raw


def evaluate(
    model_path: Path,
    adapter_path: Path,
    qwen_path: Path,
    source_path: Path,
    output_path: Path,
    predictions_path: Path,
    device_name: str,
    max_episodes: int | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    source_records = load_records(source_path)
    qwen_records = load_records(qwen_path)
    if len(source_records) != len(qwen_records):
        raise ValueError(f"Source/Qwen count mismatch: {len(source_records)} vs {len(qwen_records)}")

    grouped: defaultdict[str, list[tuple[int, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for index, (source, qwen) in enumerate(zip(source_records, qwen_records)):
        grouped[episode_id(source)].append((index, source, qwen))
    episode_keys = sorted(grouped)
    if max_episodes is not None:
        episode_keys = episode_keys[:max_episodes]

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    model.config.use_cache = True

    totals = Counter()
    field_correct = Counter()
    episode_results: dict[str, list[bool]] = {}
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w") as prediction_file:
        for episode_key in episode_keys:
            records = sorted(grouped[episode_key], key=lambda item: (int(item[1].get("subtask_idx", 0)), item[0]))
            predicted_history: list[dict[str, Any]] = []
            results = []
            for index, source, qwen in records:
                messages = _materialize_images(
                    _with_history(qwen["messages"][:-1], _prompt_text(source, predicted_history))
                    + [qwen["messages"][-1]]
                )
                predicted, raw_output = _predict(
                    model,
                    processor,
                    messages[:-1],
                    device,
                    max_new_tokens,
                )
                expected = _parse_json(qwen["messages"][-1].get("content"))
                exact = predicted is not None and expected is not None and predicted == expected
                results.append(bool(exact))
                totals["samples"] += 1
                totals["valid_json"] += int(predicted is not None)
                totals["exact_json"] += int(exact)
                for field in FIELDS:
                    field_correct[field] += int(
                        predicted is not None and predicted.get(field) == (expected or {}).get(field)
                    )
                prediction_file.write(
                    json.dumps(
                        {
                            "index": index,
                            "sample_id": source.get("sample_id"),
                            "history_before_prediction": predicted_history,
                            "expected": expected,
                            "predicted": predicted,
                            "raw_output": raw_output,
                            "exact": bool(exact),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                # In this isolated evaluator, assume the low-level executor
                # verifies each predicted subtask successfully.  This lets us
                # measure self-conditioned history; real deployment must append
                # only verifier-confirmed subtasks.
                if predicted is not None and predicted.get("mode") != "done":
                    predicted_history.append(
                        {
                            "instruction": str(predicted.get("instruction", "")),
                            "skill": str(predicted.get("atomic_skill", "")),
                            "stage": str(predicted.get("stage", "")),
                            "result": "success",
                        }
                    )
            episode_results[episode_key] = results
            print(
                f"episode {episode_key}: {sum(results)}/{len(results)} exact, "
                f"history_len={len(predicted_history)}",
                flush=True,
            )

    totals["episodes"] = len(episode_results)
    totals["episode_exact"] = sum(all(results) for results in episode_results.values())
    metrics = {
        "mode": "predicted_history_recorded_observations",
        "model": str(model_path.resolve()),
        "adapter": str(adapter_path.resolve()),
        "qwen_dataset": str(qwen_path.resolve()),
        "source_dataset": str(source_path.resolve()),
        "samples": totals["samples"],
        "valid_json": totals["valid_json"],
        "json_parse_rate": totals["valid_json"] / max(1, totals["samples"]),
        "exact_json": totals["exact_json"],
        "exact_json_accuracy": totals["exact_json"] / max(1, totals["samples"]),
        "episodes": totals["episodes"],
        "episode_exact": totals["episode_exact"],
        "episode_exact_accuracy": totals["episode_exact"] / max(1, totals["episodes"]),
        "field_accuracy": {
            field: field_correct[field] / max(1, totals["samples"]) for field in FIELDS
        },
        "predictions": str(predictions_path.resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    evaluate(
        args.model,
        args.adapter,
        args.qwen,
        args.source,
        args.output,
        args.predictions_output,
        args.device,
        args.max_episodes,
        args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
