#!/usr/bin/env python3
"""Evaluate a trained Qwen3-VL Planner adapter on the held-out JSONL split."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from planner_dataset_utils import episode_id, load_records
from train_qwen3vl_planner import _materialize_images


FIELDS = ("mode", "instruction", "atomic_skill", "stage")


def _parse_json(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().strip().split())


def _move_inputs(inputs: Any, device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


def evaluate(
    model_path: Path,
    adapter_path: Path,
    qwen_path: Path,
    source_path: Path,
    output_path: Path,
    predictions_path: Path,
    device_name: str,
    max_samples: int | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    source_records = load_records(source_path)
    qwen_records = load_records(qwen_path)
    if len(source_records) != len(qwen_records):
        raise ValueError(f"Source/Qwen count mismatch: {len(source_records)} vs {len(qwen_records)}")
    count = min(len(source_records), max_samples or len(source_records))

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
    episode_results: defaultdict[str, list[bool]] = defaultdict(list)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w") as prediction_file:
        for index in range(count):
            source_record = source_records[index]
            qwen_record = qwen_records[index]
            messages = _materialize_images(qwen_record["messages"])
            prompt = processor.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt = _move_inputs(prompt, device)
            with torch.inference_mode():
                generated = model.generate(
                    **prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            generated_tokens = generated[:, prompt["input_ids"].shape[-1] :]
            raw_output = processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            expected = _parse_json(qwen_record["messages"][-1].get("content"))
            predicted = _parse_json(raw_output)
            totals["samples"] += 1
            if predicted is not None:
                totals["valid_json"] += 1
            sample_field_matches = {}
            for field in FIELDS:
                matches = predicted is not None and predicted.get(field) == (expected or {}).get(field)
                sample_field_matches[field] = bool(matches)
                field_correct[field] += int(matches)
            exact = predicted is not None and expected is not None and predicted == expected
            instruction_match = predicted is not None and expected is not None and _normalize_text(
                predicted.get("instruction")
            ) == _normalize_text(expected.get("instruction"))
            totals["exact_json"] += int(exact)
            totals["instruction_normalized_exact"] += int(instruction_match)
            key = episode_id(source_record)
            episode_results[key].append(bool(exact))
            prediction_file.write(
                json.dumps(
                    {
                        "index": index,
                        "sample_id": source_record.get("sample_id"),
                        "expected": expected,
                        "predicted": predicted,
                        "raw_output": raw_output,
                        "field_matches": sample_field_matches,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (index + 1) % 10 == 0:
                print(f"evaluated {index + 1}/{count}", flush=True)

    totals["episodes"] = len(episode_results)
    totals["episode_exact"] = sum(all(results) for results in episode_results.values())
    metrics = {
        "model": str(model_path.resolve()),
        "adapter": str(adapter_path.resolve()),
        "qwen_dataset": str(qwen_path.resolve()),
        "source_dataset": str(source_path.resolve()),
        "samples": totals["samples"],
        "valid_json": totals["valid_json"],
        "json_parse_rate": totals["valid_json"] / max(1, totals["samples"]),
        "exact_json": totals["exact_json"],
        "exact_json_accuracy": totals["exact_json"] / max(1, totals["samples"]),
        "instruction_normalized_exact": totals["instruction_normalized_exact"],
        "instruction_normalized_accuracy": totals["instruction_normalized_exact"] / max(1, totals["samples"]),
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
    parser.add_argument("--max-samples", type=int)
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
        args.max_samples,
        args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
