# Cosmos Predict / WAM Cache 与 pi0.5 训练流程

这份文档记录当前 `pi05_cosmos` 的 WorldPilot 对齐训练流程：

```text
当前 observation o_t
  -> frozen Cosmos Predict / WAM forward
  -> 读取固定 future-image latent slot
  -> 保存为 .npz cache
  -> pi0.5 训练时读取 cache 做 latent steering
```

## 目标

使用冻结的 Cosmos Predict / WAM 为每个相机视角生成一个未来状态 latent，
然后训练带 latent steering 的 pi0.5。

当前约束：

- action chunk size 为 `K=16`。
- 每个样本只使用 `observation[t+16]` 作为 future state target。
- 不生成或选择多个 future steps。
- 不使用 `floor(pixel_offset / 4)` 从自然视频时间 latent 中选择 `t+16`。
- Cosmos Policy 的 latent-frame injection 形式是：

```text
[o_t, o_t, o_t, o_t, o_{t+16}, o_{t+16}, o_{t+16}, o_{t+16}]
```

经过 temporal compression 后，当前图像和未来图像分别对应独立的 latent frame。
WorldPilot 只读取固定的 future-image latent slot。本设置中该 slot 是 `1`。

## 当前 OpenPI 配置

训练配置名：

```text
pi05_cosmos
```

模型配置：

```text
action_dim: 14
action_horizon: 16
cosmos_latent_dim: 15360
num_cosmos_latent_tokens: 2
max_future_steps: 17
max_views: 2
```

数据和动作配置：

```text
原始 dataset state/actions: 12D
prepend_zero_dims: true
模型内部 state/actions: 14D
use_delta_joint_actions: false
output_drop_first_n_dims: 2
```

含义：

- 数据集原始动作是 12 维。
- 训练前会在前面补 2 维 0，变成 14 维。
- 模型内部训练 14 维 absolute actions。
- 不再做 delta action。
- policy 输出时丢掉前 2 维，只给执行端 12 维动作。

norm stats 路径：

```text
/cc/openpi_wam/assets/pi05_absolute/absolute/norm_stats.json
```

训练默认读取的 Cosmos cache 路径：

```text
/cc/openpi_wam/cosmos_cache/move_aloha_bottles_box_basket_openpi_predict_video2world_k16_future_slot
```

## Cosmos Predict Checkpoints

2B post-trained 主模型 checkpoint：

```text
/cc/openpi_wam/cosmos_checkpoints/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt
```

Wan2.1 tokenizer / VAE：

```text
/cc/openpi_wam/cosmos_checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth
```

注意：完整 `predict_video2world` 路径可能还会用到本地的 Reason/Qwen 模型缓存。

## 生成 NPZ Cache

cache 生成要在 Cosmos Predict 环境里运行。

先进入 Cosmos repo：

```bash
cd /cc/openpi_wam/cosmos-predict2.5
export CUDA_HOME="$PWD/.venv/lib/python3.13/site-packages/nvidia/cu13"
export FFMPEG6_HOME="$PWD/.ffmpeg6"
export LD_LIBRARY_PATH="$FFMPEG6_HOME/lib:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

这里必须把 `$PWD/.ffmpeg6/lib` 加到 `LD_LIBRARY_PATH`。否则 `decord`
会找不到 `libavformat.so.60`，报错类似：

```text
OSError: libavformat.so.60: cannot open shared object file
```

先跑 smoke test，只生成 2 个样本：

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python /cc/openpi_wam/tools/build_cosmos_latent_cache.py \
  --dataset-dir /cc/openpi_wam/datasets/move_aloha_bottles_box_basket_openpi \
  --output-dir /cc/openpi_wam/cosmos_cache/move_aloha_bottles_box_basket_openpi_predict_video2world_k16_future_slot \
  --views image,wrist_image \
  --dataset-reader local \
  --target-step 16 \
  --future-slot-index 1 \
  --cosmos-latent-source predict_video2world \
  --cosmos-num-steps 10 \
  --cosmos-guidance 7.0 \
  --cosmos-checkpoint /cc/openpi_wam/cosmos_checkpoints/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt \
  --max-samples 2
```

如果 smoke test 成功，去掉 `--max-samples 2` 跑全量：

```bash
CUDA_VISIBLE_DEVICES=7 .venv/bin/python /cc/openpi_wam/tools/build_cosmos_latent_cache.py \
  --dataset-dir /cc/openpi_wam/datasets/move_aloha_bottles_box_basket_openpi \
  --output-dir /cc/openpi_wam/cosmos_cache/move_aloha_bottles_box_basket_openpi_predict_video2world_k16_future_slot \
  --views image,wrist_image \
  --dataset-reader local \
  --target-step 16 \
  --future-slot-index 1 \
  --cosmos-latent-source predict_video2world \
  --cosmos-num-steps 10 \
  --cosmos-guidance 7.0 \
  --cosmos-checkpoint /cc/openpi_wam/cosmos_checkpoints/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt
```

脚本默认会跳过已经存在且 shape 正确的 `.npz` 文件，所以中断后可以继续跑。
如果你改了 `--cosmos-num-steps`、`--cosmos-guidance` 或其他生成参数，需要重新生成，
可以加：

```bash
--overwrite-existing
```

## Cache 文件内容

每个 dataset frame 会生成一个文件：

```text
episode_000000/frame_000000.npz
```

期望内容：

```text
latent:   (2, 15360) float32
mask:     (2,) bool
time_ids: (2,) int32, 期望为 [16, 16]
view_ids: (2,) int32, 期望为 [0, 1]
```

两个 latent token 分别是：

```text
view 0: image 的 future slot
view 1: wrist_image 的 future slot
```

检查任意一个 cache 文件：

```bash
cd /cc/openpi_wam
.venv/bin/python - <<'PY'
import numpy as np
from pathlib import Path

cache_root = Path("/cc/openpi_wam/cosmos_cache/move_aloha_bottles_box_basket_openpi_predict_video2world_k16_future_slot")
path = next(cache_root.glob("episode_*/frame_*.npz"))
print(path)
with np.load(path) as z:
    print("latent", z["latent"].shape, z["latent"].dtype)
    print("time_ids", z["time_ids"].tolist())
    print("view_ids", z["view_ids"].tolist())
    print("mask", z["mask"].tolist())
PY
```

期望输出类似：

```text
latent (2, 15360) float32
time_ids [16, 16]
view_ids [0, 1]
mask [True, True]
```

## 正式路径和 Debug 路径

正式 WorldPilot 对齐路径：

```text
--cosmos-latent-source predict_video2world
future_latent = CosmosPredict(o_t, instruction)
读取 fixed future slot index 1
```

这个路径不会读取 dataset 的真实未来图像作为输入。

Debug / oracle VAE 路径：

```text
--cosmos-latent-source future_vae
编码 [o_t x4, o_{t+16} x4]
读取 fixed future slot index 1
```

`future_vae` 会读取 dataset 中真实的 `o_{t+16}`，所以只能用于调试 cache 管线和 shape，
不能作为最终 WorldPilot 对齐训练 cache。

## 训练 pi0.5

训练要回到 OpenPI 环境：

```bash
cd /cc/openpi_wam
XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python scripts/train.py pi05_cosmos \
  --exp_name=move_aloha_absolute_cosmos_k16
```

训练阶段不会在线运行 Cosmos。训练只通过下面的 transform 读取已经生成好的 `.npz`：

```text
src/openpi/transforms.py::AttachCosmosLatent
```

所以训练前必须先生成完整 cache。

## 一致性要求

cache 生成和未来真实部署时，这些参数必须保持一致：

```text
cosmos_latent_source: predict_video2world
target_step: 16
future_slot_index: 1
cosmos_num_steps: 10
cosmos_guidance: 7.0
views: image,wrist_image
```

如果你修改了 `--cosmos-num-steps` 或 `--cosmos-guidance`，需要重新生成 cache，
并且真实部署时也要使用相同设置。

## 常用检查

检查当前 `pi05_cosmos` 配置：

```bash
cd /cc/openpi_wam
.venv/bin/python - <<'PY'
from openpi.training import config as cfg
c = cfg.get_config("pi05_cosmos")
print("action_horizon", c.model.action_horizon)
print("action_dim", c.model.action_dim)
print("num_cosmos_latent_tokens", c.model.num_cosmos_latent_tokens)
print("cosmos_latent_dim", c.model.cosmos_latent_dim)
print("cache_dir", c.data.cosmos_latent_cache_dir)
PY
```

期望：

```text
action_horizon 16
action_dim 14
num_cosmos_latent_tokens 2
cosmos_latent_dim 15360
```

如果训练报错：

```text
Cosmos latent cache file not found
```

说明 `.npz` cache 还没生成，或者 `OPENPI_COSMOS_PLUG_CACHE_DIR` 指向了错误目录。

如果训练报错 norm stats 缺失，检查：

```text
/cc/openpi_wam/assets/pi05_absolute/absolute/norm_stats.json
```
