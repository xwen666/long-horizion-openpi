# 远程推理

OpenPI 的策略服务入口是 [`scripts/serve_policy.py`](../scripts/serve_policy.py)，客户端和服务端可通过 websocket 分离部署。Cosmos/WAM 实时 worker 的路径由 `OPENPI_WAM_ROOT`、`COSMOS_REPO` 和 `COSMOS_PYTHON` 控制。

LIBERO + WAM 的完整命令见 [`workflows.md`](workflows.md) 和 [`README_LIBERO_WAM_EVAL.md`](../README_LIBERO_WAM_EVAL.md)。
