# Docker

Docker 相关文件位于 [`scripts/docker`](../scripts/docker)。常用入口包括：

- `serve_policy.Dockerfile`：构建推理服务镜像。
- `compose.yml`：启动策略服务。
- `install_docker_ubuntu22.sh` 和 `install_nvidia_container_toolkit.sh`：主机安装辅助脚本。

训练通常需要访问本机数据集、checkpoint 和 GPU；建议优先使用仓库的 uv 虚拟环境。Cosmos/WAM 运行时依赖与 OpenPI 依赖分开，详见 [`repository_layout.md`](repository_layout.md)。
