# Behavior Prompting Policy：LIBERO / LIBERO-Gen 复现

对应 [real-stanford/behavior_prompting](https://github.com/real-stanford/behavior_prompting)，目标是加载官方 BPP ckpt，执行官方 LIBERO 闭环 rollout。

## 当前状态

**标准 LIBERO smoke 端到端已通过；Spatial Combination 已接入，待专用资源；尚未完成全量评测。**

2026-11-10 已完成源码、独立环境、官方 BPP 权重、CLIP 初始化权重和 smoke 任务演示的准备。修复不录视频时的结果整理错误后，在 RTX A6000（`--gpu 5`）上完成一次完整 smoke 运行，退出码为 0，train/test 各一个短回合得分均为 0。24 步 smoke 用于验证调用流程，不能作为正式成功率。成功记录：`logs/smoke_20261110T111644_168819Z/invocation.json`；指标见同目录 `rollout.log`。

该次 `--gpu 0` 选中了 RTX PRO 6000 Blackwell（`sm_120`），与默认 PyTorch 2.7.1+cu118 构建不兼容。现已增加独立的 `sm120` 环境配置，尚待在可联网且能访问 GPU 的终端安装和验证。具体原始报错日志：`logs/smoke_20261110T105912_187678Z/rollout.log`。

当前命令执行沙箱仍无法访问 GPU 驱动和本机代理，实际 CUDA 验证需在用户的 VS Code 终端执行。`logs/status.json` 和早期 `*-attempt.log` 保留的是安装前的历史状态。

## 目录和版本

- `src/`：官方源码，目标 commit `ec29e62dde999ac3423df7d99a773c1dd28b480e`；主仓库和固定版本的 ICRT/LIBERO 子模块已就绪，`bash bpp/setup.sh source` 已通过。环境快照已补齐漏存的三行空行，与该 commit 的压缩包原文件逐字节一致。
- `env/bpp/`：本项目内 conda 环境的安装目标。沿用官方 Python 3.10.20、Torch 2.7.1 / torchvision 0.22.1（CUDA 11.8）等版本。
- `env/bpp-sm120/`：Blackwell 的独立安装目标，使用 Python 3.10.20、Torch 2.8.0 / torchvision 0.23.0（CUDA 12.8）。通过 `--env sm120` 选择；其余依赖、源码、权重和数据与默认环境共用配置，环境内的软件包分别安装。
- `upstream/environment.yml`：从上述 commit 读取的原始环境清单，保留 MIT License。实际安装将 conda 与 pip 阶段分开，并单独选择 Torch 的 CUDA wheel index。
- `patches/`：固定上游 commit 之上的本地修复。当前补丁仅在不录视频时用任务名生成辅助输出名称，避免结果统计阶段调用 `basename(None)`。源码准备和评测入口会自动应用补丁；遇到冲突会停止，已应用的补丁不会重复修改。补丁文件名和 SHA256 记录在 `logs/source-patches.json` 以及新运行的 `invocation.json` 中。无视频/有视频的奖励统计回归检查见 `logs/no-video-regression.json`。
- `checkpoints/libero_behavior_prompting.ckpt`：官方 `austinpatel/bp_libero` 权重下载目标。下载前查询并记录 Hugging Face revision，按发布元数据校验大小与 SHA256，未完成文件为 `.part`。
- `datasets/libero/`：HDF5 prompt 数据。标准 LIBERO 版本固定在 `config/sources.json`；LIBERO-Gen 首次读取官方元数据后将 revision 固定到独立 manifest。
- `config/libero/config.yaml`：项目内配置，不修改 `~/.libero`。assets 指向 XVLA 的已有文件；BDDL 和 init states 使用 BPP 自带版本。
- `.cache/`、`logs/`：依赖缓存、临时文件、模型缓存、rollout 缓存及日志，均在此项目内。

复用 XVLA 的资源文件；BPP 的 Python 3.10 依赖及定制 robomimic 安装到独立环境，不混入 XVLA 的 Python 3.8 环境。若 XVLA 已有标准 LIBERO HDF5 文件，`scripts/configure.py` 会建立只读使用的文件链接；下载校验不通过时会停止，不覆写链接目标。

## 网络和 GPU 可用后的命令

从 `icl/` 执行：

```bash
# 安装源码与环境，下载官方 ckpt，以及一个 smoke 任务的完整演示文件
bash bpp/setup.sh

# 单任务、1 个 train 初始化 + 1 个 test 初始化、24 步策略控制
bash bpp/run_eval.sh --mode smoke --gpu 5

# 保存 train/test 各一个回合的视频
bash bpp/run_eval.sh --mode smoke --gpu 5 --save-video

# 下载所有 130 个任务的演示（官方数据约 100 GB）后进行完整评测
bash bpp/setup.sh data-full
bash bpp/run_eval.sh --mode full --gpu 5 --seed 0
```

独立恢复阶段：`bash bpp/setup.sh source`、`env`、`checkpoint`、`data`、`data-full`。安装成功后保存 `logs/requirements-installed.txt` 和 `logs/conda-explicit.txt`。

Smoke 默认不录视频。添加 `--save-video` 后，MP4 保存在本次输出目录的 `media/` 下，文件名分别以 `train_` 和 `test_` 开头；这会重新运行两个回合，之前没有录制的运行不能补出视频。可用 `--output bpp/logs/smoke_video_run1` 指定新的输出目录。完整评测默认沿用上游的录像数量，添加 `--save-video` 时改为每个任务录制 train/test 各一个回合。

### Blackwell / sm_120 独立环境

固定源码的 `environment.yml` 给出了 Torch 2.8.0、torchvision 0.23.0 和 CUDA 12.8 的替代组合；这里使用其官方 CUDA 12.8 wheels，匹配 [PyTorch 安装版本表](https://pytorch.org/get-started/previous-versions/#v280)。原来的 `env/bpp` 保留供 A6000 使用。

```bash
# 只安装新的 Python 环境；复用已有源码、权重、数据和缓存
bash bpp/setup.sh env --env sm120

# 当前映射 --gpu 0 为物理 GPU 6：RTX PRO 6000 Blackwell
bash bpp/run_eval.sh --env sm120 --mode smoke --gpu 0 --save-video

# 也支持 LIBERO-Gen；仍须具备相应 benchmark 的专用权重和数据
bash bpp/run_eval.sh --env sm120 --benchmark liberogen_spatial_combination \
  --mode smoke --gpu 0 --save-video

# 预览安装计划或评测命令，无需 GPU，也不会下载
python3 bpp/scripts/environment.py --env sm120 --dry-run
bash bpp/run_eval.sh --env sm120 --gpu 0 --dry-run
```

`setup.sh` 和 `run_eval.sh` 都支持 `--env default|sm120`；也可设置 `BPP_ENV=sm120`，显式参数优先。默认 profile 仍是 `default`。安装过程用精确的 Torch/torchvision CUDA 构建约束后续依赖解析，防止被替换为其他版本；安装结束后检查依赖并记录版本。`sm120` 的安装记录保存在 `logs/env-sm120/`，每次评测的 `invocation.json` 也记录选择的环境和 Python 前缀。

当前已通过双环境路径选择、安装参数和现有 benchmark 的离线回归检查；实际安装尝试被命令执行沙箱的网络限制阻止（`logs/environment-sm120-attempt.log`），因此新环境尚未安装成功，也未运行 Blackwell rollout。请在能正常使用代理和 GPU 的 VS Code 终端执行上述安装及 smoke 命令；启动器会用真实 CUDA 运算确认选中显卡和构建兼容。

### LIBERO-Gen Spatial Combination

已接入 `--benchmark liberogen_spatial_combination`。复用现有源码、Python 环境、CLIP 缓存和仿真 assets；BDDL 与 init states 也已随源码提供。默认不指定 benchmark 时仍运行标准 LIBERO。

**现有 `checkpoints/libero_behavior_prompting.ckpt` 是标准 LIBERO 权重。** 已读取其内部配置：`task.name=libero_all`，训练 split 为 `libero_10/libero_90/libero_goal/libero_object/libero_spatial`，unseen split 为空。Spatial Combination 使用专用 `liberogen_spatial_combination_behavior_prompting.ckpt`；启动器在使用 GPU 前核对 checkpoint 的任务名，混用时会退出。

官方资源：[文档](https://github.com/real-stanford/behavior_prompting/blob/main/docs/liberogen.md)、[专用权重](https://huggingface.co/austinpatel/bp_liberogen_spatial_combination)、[HDF5 演示](https://huggingface.co/datasets/austinpatel/libero_gen_spatial_combination_hdf5)。固定版本源码中的权重仓库链接少了 `bp_` 前缀，启动器采用当前官方文档的地址。首次获取元数据后将 revision 和 size/SHA256 保存在独立 manifest，后续复用，不重复查询或下载已有匹配资源。

```bash
# 预览命令；不会加载模型或下载资源
bash bpp/run_eval.sh --benchmark liberogen_spatial_combination --dry-run

# 离线检查本地权重与数据；缺少资源时明确退出，不使用 GPU
bash bpp/run_eval.sh --benchmark liberogen_spatial_combination --check-only

# 已有专用权重/数据时可原地使用，不复制或下载
bash bpp/run_eval.sh --benchmark liberogen_spatial_combination --check-only \
  --checkpoint /path/to/liberogen_spatial_combination_behavior_prompting.ckpt \
  --data-root /path/to/demonstration_data
```

数据根目录下面应直接包含 `libero_spatial_selected_combinations_inverse_view`（训练时见过的组合）和 `libero_spatial_selected_combinations_view`（未见组合）。Smoke 只运行一个未见组合、一个测试初始化、24 步；`--save-video` 可保存这个回合。Full 保留官方 seen/unseen 评测配置与 550 步上限，未见组合成功率才是组合泛化的关注指标。

如需准备缺失资源，只执行对应阶段，无需重新运行源码和环境安装：

```bash
# 仅记录官方文件元数据，不下载权重或演示数据
bash bpp/setup.sh checkpoint --benchmark liberogen_spatial_combination --metadata-only
bash bpp/setup.sh data-full --benchmark liberogen_spatial_combination --metadata-only

# 使用已记录的元数据校验并链接已有资源；完全离线，缺少时只报告
bash bpp/setup.sh checkpoint --benchmark liberogen_spatial_combination \
  --reuse-from /path/to/existing/checkpoints --local-only
bash bpp/setup.sh data-full --benchmark liberogen_spatial_combination \
  --reuse-from /path/to/demonstration_data --local-only

# 确有缺失时，去掉 --local-only 才会下载缺失项；data 只准备一个 smoke 任务
# 项目和用户默认 HF 缓存会自动检查；--reuse-from 可多次传入

# 所需资源齐全并确认 GPU 空闲后运行
bash bpp/run_eval.sh --benchmark liberogen_spatial_combination --mode smoke --gpu 5
bash bpp/run_eval.sh --benchmark liberogen_spatial_combination --mode full --gpu 5
```

Full 根据官方 HDF5 文件目录核对完整性，不按源码 BDDL 数量判断（源码包含未成功生成演示的额外任务）。使用 `--data-root` 原地评测时，也需先记录 `data-full --metadata-only` 的完整目录。每次 rollout 保存独立 LIBERO 配置及 checkpoint/benchmark/data-root 信息；不会更改已有数据的路径。

当前仅完成离线配置与入口验证，未下载 Spatial Combination 的专用权重或数据，尚未执行此 benchmark 的 GPU rollout。离线回归检查：`bpp/env/bpp/bin/python -m unittest discover -s bpp/tests -v`。

### DrawAnything-Sim

已接入 `--benchmark drawanything_sim`，对应官方 [drawanything_sim 文档](https://github.com/real-stanford/behavior_prompting/blob/main/docs/drawanything_sim.md)：纯 2D 绘画环境（pygame/pymunk 渲染，无需 MuJoCo/EGL），复用同一源码与环境。入口为 `draw_policy_dunetp` + `+rollout=draw`，与 LIBERO 的 modifier/BDDL 流程无关。

资源（官方仓库都是 `austinpatel/drawanything_sim`，模型库与数据集库同名）：

- `checkpoints/drawanything_sim_behavior_prompting.ckpt`：behavior prompting 策略权重（1.4 GB）。官方还提供 goal image 与 ICRT 基线权重，启动器只拉取 prompting 版本。
- `datasets/drawanything/`：`*.zarr.zip` 按官方 `unzip_zarr.py` 解压为同目录的目录存储。`data` 只准备 `eval_handmade.zarr`（50 个手绘任务，~800 MB）；`data-full` 额外准备官方训练集 `procedural_2000_10.zarr`（~41 GB 压缩包，seen 任务 rollout 或训练用）。

```bash
setproxy
bash bpp/setup.sh checkpoint --benchmark drawanything_sim
bash bpp/setup.sh data --benchmark drawanything_sim        # 仅 eval_handmade
bash bpp/setup.sh data-full --benchmark drawanything_sim   # 追加 41GB 训练集（可选）

# smoke：1 个未见手绘任务、4 个 test 环境、120 步，不渲染注意力图
bash bpp/run_eval.sh --benchmark drawanything_sim --mode smoke --gpu 5 --save-video

# 离线核对权重任务名（应为 draw）与本地输入
bash bpp/run_eval.sh --benchmark drawanything_sim --check-only
bash bpp/run_eval.sh --benchmark drawanything_sim --dry-run

# 完整评测：eval_handmade 全部 50 个未见任务，官方 n_test=22(+n_train=10) 共 32 个
# 环境、400 步上限、注意力图可视化；训练集存在时追加官方 10 个 seen 任务的 rollout
bash bpp/run_eval.sh --benchmark drawanything_sim --mode full --gpu 5 --save-video
```

环境构造时需要访问 Hugging Face 下载 timm `vit_base_patch16_clip_224.openai` 初始化权重（已缓存在 `bpp/.cache/huggingface`）；在本机透明代理会把 TLS 换成自签证书的网络下，运行前请先 `setproxy`，否则模型构造阶段会因证书校验失败退出。

指标为像素级 Chamfer 距离的相反数（`reward = -compute_draw_distance`），日志里的 `mean_scores/test/all` 越接近 0 越好；rollout 视频在输出目录 `media/`，注意力图合并在 `vis/`。2026-11-13：smoke 与未见集评测已在 RTX A6000（`--gpu 6`）上端到端通过。完整评测 51 个手绘任务、每任务 32 个未见初始化、400 步上限，`eval/mean_scores/test/all = -3.62`（简单字母约 -1.2，多段复杂图形最差 -17.4）；smoke 输出见 `logs/drawanything_sim_smoke_20261113T111925_407508Z/`，完整评测见 `logs/drawanything_sim_full_20261113T112021_507673Z/`。

### 多种 GPU 共存时的选择

映射记录来自 `/data3/huangxinagyun/project/WorldForge/AGENTS.md`。当前默认环境中，`--gpu` 直接设置 `CUDA_VISIBLE_DEVICES`，不等同于 `nvidia-smi` 的物理编号：

| `nvidia-smi` 编号 | GPU 型号 | `--gpu` 值 |
| ---: | --- | ---: |
| 0 | RTX A6000 | 5 |
| 1 | RTX A6000 | 6 |
| 2 | RTX 6000D | 1 |
| 3 | RTX 6000D | 2 |
| 4 | RTX 6000D | 3 |
| 5 | RTX 6000D | 4 |
| 6 | RTX PRO 6000 Blackwell Server Edition | 0 |

先查看空闲显存，选择一张 A6000（`--gpu 5` 或 `--gpu 6`），也可以直接传 UUID：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv
bash bpp/run_eval.sh --mode smoke --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43
```

上面 UUID 是当前服务器检测到的一张 A6000，也可替换为另一张空闲 A6000 的 UUID。UUID 可避免混淆 CUDA 顺序与其他工具的编号；若改变 `CUDA_DEVICE_ORDER` 或驱动环境，应重新核对映射。更换显卡直接运行 `run_eval.sh`，会复用已下载权重和生成的缓存。

启动器会先在选中的卡上运行小型 CUDA 乘加和矩阵乘法，检查真实内核能否执行；`torch.cuda.is_available()` 为真并不足以证明架构兼容。检查失败会在加载模型前退出并打印显卡型号、架构和 PyTorch 构建信息。

### Git TLS 连接中断

`gnutls_handshake() failed` 发生在 Git 下载阶段，后续的 `Setup incomplete` 是它造成的缺失项列表。

现在源码获取会先使用 HTTP/1.1 浅拉取固定 commit，失败重试三次，再用 curl 下载 GitHub 官方 commit 压缩包。Git 检出的子模块版本从该 commit 的 gitlink 获取；压缩包模式从 GitHub 上该完整 commit 的 `deps/` 页面读取子模块链接，并将完整 SHA 缓存在 `.cache/source/submodules-<commit>.json`，不再调用 GitHub REST API。XVLA 的 LIBERO 仓库若已有同一 commit，会直接复用其 Git 对象进行本地检出。源码压缩包不含 `.git`，会另存版本和来源记录，评测入口同时支持两种来源。

若 Git 持续 TLS 失败，在可以联网且 CUDA 可用的终端执行下面这条命令，直接使用官方压缩包传输，并且仅在安装成功后启动评测：

```bash
cd /data2/huangxiangyun/project/icl
setproxy
BPP_SOURCE_METHOD=archive bash bpp/setup.sh && bash bpp/run_eval.sh --mode smoke --gpu 5
```

没有关闭 TLS 证书验证。压缩包解压与本地仓库复用的测试记录在 `logs/source-fetch-tests.json`；实际远程下载和模型运行仍受上述环境阻碍。

### GitHub API 返回 403 rate limit exceeded

旧版 archive 流程需要匿名调用 `api.github.com` 查询子模块 SHA，可能耗尽代理出口 IP 共享的每小时 60 次额度。当前流程已移除这项 API 查询，可直接重跑上面的安装命令；已有主仓库会复用。首次仍需访问 GitHub 普通仓库页面和缺失源码的官方压缩包地址，因此不保证免于所有网络限制。若页面缺少唯一的完整子模块 SHA，会停止并报错，不会改用 `main` 或未经核对的版本。

无需源码、环境或 GPU 即可预览命令：

```bash
bash bpp/run_eval.sh --mode smoke --dry-run
bash bpp/run_eval.sh --mode full --dry-run
```

## 评测行为

入口直接执行上游 `behavior_prompting/train_network/train.py`，使用 `libero_policy_dunetp`、`task=libero_all`、`+modifiers=libero/libero_all`、`+rollout=libero`，由上游 `RolloutPolicyWorkspace` 加载权重并执行闭环控制。模型结构与 prompting 配置沿用官方配置，没有替换为语言条件策略。

Smoke 仅选 `libero_spatial` 中一个指定任务，缩短控制步数并减少环境数，验证 prompt、动作预测和环境交互；**其成功率不可作为论文指标**。Full 保留官方任务和 rollout 配置；数据文件数量不足时会提前报错，避免空任务列表被误认为成功完成。

每次运行创建独立输出目录，包含 `invocation.json`、`rollout.log` 和上游 Hydra/rollout 产物。W&B 使用离线模式。默认视觉编码器配置含预训练 CLIP ViT，首次模型构造可能额外下载其初始化权重到项目缓存。

`src`、环境、ckpt、数据未就绪时会明确以非零状态退出，不会生成虚假模型结果。完整模型加载、EGL 渲染、任务成功率仍需在实际可联网且 CUDA 可用的环境中验证。

参考：[官方 LIBERO 文档](https://github.com/real-stanford/behavior_prompting/blob/main/docs/libero.md)、[BPP 模型](https://huggingface.co/austinpatel/bp_libero)、[LIBERO 数据](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)。
