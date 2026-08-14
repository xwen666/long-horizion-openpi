#!/usr/bin/env python3
"""LoRA fine-tuning for the RoboCasa Qwen3-VL semantic planner.

The JSONL files are produced by ``convert_to_qwen3vl_format.py``.  The
assistant response is supervised as JSON; robot actions are intentionally not
part of this training target.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


LANGUAGE_LAYER_PREFIX = "model.language_model.layers."
LORA_PROJECTIONS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def _load_records(path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if max_samples is not None and len(records) >= max_samples:
                    break
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    return records


def _materialize_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace JSON image paths with detached PIL images for the processor."""

    result = copy.deepcopy(messages)
    for message in result:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") != "image":
                continue
            image_path = item.get("image")
            if not isinstance(image_path, str):
                raise TypeError(f"Expected a local image path, got {image_path!r}")
            with Image.open(image_path) as image:
                item["image"] = image.convert("RGB")
    return result


class PlannerDataset(Dataset):
    def __init__(self, path: Path, max_samples: int | None = None):
        self.records = _load_records(path, max_samples)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class PlannerCollator:
    def __init__(self, processor: Any):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = processor.tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("The Qwen tokenizer has neither pad_token_id nor eos_token_id")

    def _encode(self, record: dict[str, Any]) -> dict[str, torch.Tensor]:
        messages = _materialize_images(record["messages"])
        prompt_messages = messages[:-1]
        full = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]
        prompt_length = prompt["input_ids"].shape[-1]
        if prompt_length > input_ids.shape[-1]:
            raise ValueError("Prompt token length is longer than full conversation")

        labels = input_ids.clone()
        labels[:prompt_length] = -100
        encoded: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key, value in full.items():
            if key in encoded or not isinstance(value, torch.Tensor):
                continue
            if key not in {
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
            } and value.ndim > 0 and value.shape[0] == 1:
                value = value[0]
            encoded[key] = value
        return encoded

    @staticmethod
    def _pad_1d(values: list[torch.Tensor], padding_value: int) -> torch.Tensor:
        width = max(value.shape[0] for value in values)
        result = values[0].new_full((len(values), width), padding_value)
        for index, value in enumerate(values):
            result[index, : value.shape[0]] = value
        return result

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode(record) for record in records]
        batch: dict[str, torch.Tensor] = {
            "input_ids": self._pad_1d(
                [item["input_ids"] for item in encoded], self.pad_token_id
            ),
            "attention_mask": self._pad_1d(
                [item["attention_mask"] for item in encoded], 0
            ),
            "labels": self._pad_1d([item["labels"] for item in encoded], -100),
        }

        remaining_keys = set().union(*(item.keys() for item in encoded)) - set(batch)
        for key in remaining_keys:
            values = [item[key] for item in encoded if key in item]
            if len(values) != len(encoded):
                raise ValueError(f"Missing {key!r} in part of the batch")
            if key in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}:
                batch[key] = torch.cat(values, dim=0)
            elif all(value.ndim == 1 for value in values):
                batch[key] = self._pad_1d(values, 0)
            else:
                try:
                    batch[key] = torch.stack(values)
                except RuntimeError as exc:
                    raise ValueError(f"Cannot collate processor field {key!r}") from exc
        return batch


def _load_model(model_path: Path, lora_rank: int, lora_alpha: int, lora_dropout: float):
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Qwen training requires a recent transformers, peft, and accelerate. "
            "Install them in the dedicated Qwen environment first."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    target_modules = []
    for name, module in model.named_modules():
        if not name.startswith(LANGUAGE_LAYER_PREFIX):
            continue
        if name.rsplit(".", 1)[-1] in LORA_PROJECTIONS and isinstance(module, torch.nn.Linear):
            target_modules.append(name)
    if not target_modules:
        raise RuntimeError("Could not find language-side Qwen3-VL LoRA target modules")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(f"Language-side LoRA targets: {len(target_modules)} modules")
    return model, processor


def _training_args_kwargs(args: argparse.Namespace, num_train_samples: int) -> dict[str, Any]:
    from transformers import TrainingArguments

    supported = inspect.signature(TrainingArguments).parameters
    values: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_batch_size,
        "per_device_eval_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 3,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "ddp_find_unused_parameters": False,
        "report_to": [],
        "save_safetensors": True,
    }
    if "warmup_ratio" in supported:
        values["warmup_ratio"] = 0.03
    elif "warmup_steps" in supported:
        # transformers 5.x removed warmup_ratio. Preserve the previous 3% schedule
        # by converting it to optimizer steps using the launched world size.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        batches_per_epoch = math.ceil(
            num_train_samples / (args.per_device_batch_size * world_size)
        )
        optimizer_steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
        total_optimizer_steps = math.ceil(optimizer_steps_per_epoch * args.num_train_epochs)
        values["warmup_steps"] = max(1, round(0.03 * total_optimizer_steps))
    if "eval_strategy" in supported:
        values["eval_strategy"] = "steps"
    else:
        values["evaluation_strategy"] = "steps"
    if "gradient_checkpointing_kwargs" in supported:
        values["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    # TrainingArguments has removed a few legacy kwargs across transformers
    # releases. Keep the script usable with both the older dedicated env and
    # transformers 5.x without passing unsupported optional fields.
    return {name: value for name, value in values.items() if name in supported}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--resume-from-checkpoint", type=str)
    args = parser.parse_args()

    from transformers import Trainer, TrainingArguments

    model, processor = _load_model(
        args.model, args.lora_rank, args.lora_alpha, args.lora_dropout
    )
    train_dataset = PlannerDataset(args.train, args.max_train_samples)
    validation_dataset = PlannerDataset(args.validation, args.max_validation_samples)
    collator = PlannerCollator(processor)
    training_args = TrainingArguments(**_training_args_kwargs(args, len(train_dataset)))
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": collator,
    }
    if "processing_class" in inspect.signature(Trainer).parameters:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor.tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
