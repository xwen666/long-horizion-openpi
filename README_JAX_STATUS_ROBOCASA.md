# JAX pi0.5 状态头训练

这部分是标准 OpenPI JAX/NNX 训练路径，模型为 `Pi0Config(pi05=True)`。
RoboCasa 状态头不依赖 Cosmos、WAM latent、WAM action prior 或 Cosmos checkpoint。
Cosmos/WAM 仍然可以作为另一个独立的外接条件分支，但不会进入本配置。

## 1. 数据和归一化

默认数据目录：

```text
/cc/openpi_wam/robocasa/datasets/v1.0/target/composite
/cc/openpi_wam/robocasa/processed/composite_subtasks/splits
```

配置会自动发现每个任务下面的日期目录，例如：

```text
<task>/<date>/lerobot
```

先计算训练集的 state/action 统计：

```bash
cd /cc/openpi_wam
source .venv/bin/activate
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python tools/compute_robocasa_norm_stats.py \
  --config-name pi05_robocasa_status
```

输出文件：

```text
/cc/openpi_wam/assets/pi05_robocasa_status/robocasa_composite_status/norm_stats.json
```

当前统计的原始维度是 state=16、action=12。模型输入的 `action_dim=32`，
数据变换会在归一化之后把 state/action padding 到 32 维。

## 2. JAX 状态训练

配置名称：

```text
pi05_robocasa_status
```

两张卡示例：

```bash
cd /cc/openpi_wam
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python scripts/train.py pi05_robocasa_status \
  --exp-name=robocasa_status_jax \
  --fsdp-devices=2 \
  --batch-size=16 \
  --num-workers=4
```

四张卡时把 `CUDA_VISIBLE_DEVICES` 改成四张卡、`--fsdp-devices=4`，并保证
`--batch-size` 能被可见 GPU 数量整除。

继续已有实验：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/train.py pi05_robocasa_status \
  --exp-name=robocasa_status_jax \
  --fsdp-devices=2 \
  --batch-size=16 \
  --num-workers=4 \
  --resume
```

## 3. VLA + status 联合训练

如果希望像 RLT 一样让 π0.5 VLA 和状态 token 一起训练，使用配置：

```text
pi05_robocasa_joint
```

两张卡示例：

```bash
cd /cc/openpi_wam
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python scripts/train.py pi05_robocasa_joint \
  --exp-name=robocasa_vla_status_joint \
  --fsdp-devices=2 \
  --batch-size=8 \
  --num-workers=4
```

联合训练的目标为：

```text
total_loss = action_loss + status_loss_weight * done_loss
```

其中：

- `action_loss` 使用 RoboCasa 的真实 12 个 action 维度；padding 到 32 维的后 20 维被 mask，不参与 loss。
- `done_loss` 训练 `StatusEncoder` 和 `DoneHead` 判断当前 subtask 是否完成。
- π0.5 VLM、SigLIP、action expert 和动作投影层都会接收联合损失的梯度。
- Cosmos/WAM 不在这条配置和训练路径中。

联合训练要使用新的实验目录，不要对 status-only 实验直接 `--resume`：

```text
pi05_robocasa_status  ->  只训练状态分支
pi05_robocasa_joint   ->  π0.5 VLA + 状态分支联合训练
```

## 4. 训练目标和冻结范围

默认 `status_only_trainable=True`：

```text
pi0.5 image + prompt
        -> frozen pi0.5 prefix/VLM
        -> StatusEncoder
        -> DoneHead
        -> done_loss
```

状态训练时不会构造 noisy action，也不会调用 action expert。只有
`status_encoder/*` 和 `done_head/*` 参数参与梯度更新。训练日志会输出：

```text
loss
done_loss
done_accuracy
done_positive_accuracy
done_negative_accuracy
grad_norm/status_encoder
grad_norm/done_head
update_norm/status_encoder
update_norm/done_head
param_norm/status_encoder
param_norm/done_head
```

如果以后需要联合训练 action，可以把配置中的 `status_only_trainable` 改成
`False`；这时总损失变为：

```text
action_loss + status_loss_weight * done_loss
```

## 5. 检查

运行状态头 CPU smoke test：

```bash
cd /cc/openpi_wam
source .venv/bin/activate
JAX_PLATFORMS=cpu python tools/test_jax_status_branch.py
```

检查完整配置：

```bash
python - <<'PY'
from openpi.training import config

c = config.get_config("pi05_robocasa_joint")
print(c.name)
print(type(c.model).__name__, c.model.enable_status_head, c.model.status_only_trainable)
print(c.data.create(c.assets_dirs, c.model).norm_stats.keys())
PY
```

训练入口必须使用 `scripts/train.py`。不要使用旧的
`scripts/train_pytorch.py` 来训练这个配置。
