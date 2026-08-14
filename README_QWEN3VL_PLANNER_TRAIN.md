# RoboCasa Qwen3-VL Planner LoRA 训练

本文训练的是高层语义 Planner，不是 OpenPI 动作策略。Planner 输入当前图像、全局任务和已完成子任务 history，输出一个 JSON 子任务：

```json
{
  "mode": "execute",
  "instruction": "place the bread in the basket",
  "atomic_skill": "PickPlaceCabinetToCounter",
  "stage": "place"
}
```

基础模型使用本地权重：

```text
/cc/openpi_wam/models/Qwen3-VL-8B-Instruct
```

## 1. 数据状态

三个数据集已经生成并通过完整校验：

```text
train:  robocasa/processed/composite_subtasks/qwen3vl_train.jsonl  (67530)
val:    robocasa/processed/composite_subtasks/qwen3vl_val.jsonl    (8423)
test:   robocasa/processed/composite_subtasks/qwen3vl_test.jsonl   (8448)
```

质量报告位于：

```text
outputs/qwen3vl_train_quality_report.json
outputs/qwen3vl_val_quality_report.json
outputs/qwen3vl_test_quality_report.json
```

## 2. 环境

不要升级 OpenPI 的 `.venv`，因为它固定使用 `transformers==4.53.2`。Qwen3-VL 使用独立环境：

```bash
cd /cc/openpi_wam
source .venv_qwen3vl/bin/activate
```

环境已经安装了：

```text
transformers 5.15.0
peft 0.20.0
accelerate 1.14.0
torch 2.7.1
```

如需重新安装并使用本地反向代理：

```bash
export HTTP_PROXY=http://127.0.0.1:17897
export HTTPS_PROXY=http://127.0.0.1:17897
uv pip install --python .venv_qwen3vl/bin/python \
  torch==2.7.1 torchvision==0.22.1 'transformers>=4.57.0' \
  peft accelerate datasets pillow
```

训练脚本使用 `local_files_only=True`，不会重新下载本地 Qwen 权重。

## 3. 单样本检查

训练前可以检查 processor 和模型 forward：

```bash
.venv_qwen3vl/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/cc/openpi_wam")
from tools.train_qwen3vl_planner import _load_model, PlannerDataset, PlannerCollator
import torch

model, processor = _load_model(
    Path("/cc/openpi_wam/models/Qwen3-VL-8B-Instruct"),
    16, 32, 0.05,
)
record = PlannerDataset(
    Path("/cc/openpi_wam/robocasa/processed/composite_subtasks/qwen3vl_train.jsonl"),
    max_samples=1,
)[0]
batch = PlannerCollator(processor)([record])
with torch.no_grad():
    output = model(**batch)
print("loss=", float(output.loss))
PY
```

脚本只对 `model.language_model.layers.*` 的 attention/MLP 投影添加 LoRA，视觉编码器保持冻结。

## 4. 正式训练

下面命令使用两张 GPU，每卡 batch size 为 1，梯度累积 16，因此全局 batch size 为 32：

```bash
cd /cc/openpi_wam
source .venv_qwen3vl/bin/activate

CUDA_VISIBLE_DEVICES=1,2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 \
  tools/train_qwen3vl_planner.py \
  --model /cc/openpi_wam/models/Qwen3-VL-8B-Instruct \
  --train /cc/openpi_wam/robocasa/processed/composite_subtasks/qwen3vl_train.jsonl \
  --validation /cc/openpi_wam/robocasa/processed/composite_subtasks/qwen3vl_val.jsonl \
  --output-dir /cc/openpi_wam/checkpoints/qwen3vl_planner/robocasa_lora \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --num-train-epochs 2 \
  --learning-rate 2e-5 \
  --save-steps 500 \
  --eval-steps 500 \
  --logging-steps 10
```

当前每个 epoch 约为 `67530 / 32 = 2110` 个 optimizer steps，2 个 epoch 约 `4220` steps。

正式运行前应确认选中的 GPU 有足够空闲显存。不要终止其他任务来腾显存；如果显存不足，保持每卡 batch size 为 1，优先增加梯度累积，或换用空闲 GPU。

## 5. 续训

```bash
CUDA_VISIBLE_DEVICES=1,2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 \
  tools/train_qwen3vl_planner.py \
  --model /cc/openpi_wam/models/Qwen3-VL-8B-Instruct \
  --train /cc/openpi_wam/robocasa/processed/composite_subtasks/qwen3vl_train.jsonl \
  --validation /cc/openpi_wam/robocasa/processed/composite_subtasks/qwen3vl_val.jsonl \
  --output-dir /cc/openpi_wam/checkpoints/qwen3vl_planner/robocasa_lora \
  --resume-from-checkpoint /cc/openpi_wam/checkpoints/qwen3vl_planner/robocasa_lora/checkpoint-XXXX
```

## 6. 训练后评估

测试集只在模型和超参数确定后使用。评估时统计：

- assistant JSON 是否可解析
- `mode`、`instruction`、`atomic_skill`、`stage` 的字段准确率
- 完整 JSON exact match
- 语义指令归一化后的准确率
- 多步 episode 中连续预测正确的比例

Planner 输出之后，再由 Template Renderer 转为底层 VLA 的固定 prompt；Qwen Planner 本身不输出 action、joint state 或 action chunk。
