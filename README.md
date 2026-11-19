# Behavior Prompting Policy 复现

本项目包含 BPP 的 LIBERO 评测复现。安装、评测命令和运行记录见 [bpp/README.md](bpp/README.md)。

BPP 官方源码自身包含 `deps/icrt` 依赖；它由 BPP 环境管理，与已移除的独立 ICRT 复现无关。

[E4-Chain · Latent Residual Behavior Transfer](e4_chain/README.md) 使用官方 Chain 训练视图学习可直接解码动作的 RGB 行为 latent。
