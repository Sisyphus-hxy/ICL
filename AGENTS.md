# 服务器 GPU 选择

## Codex 沙箱内运行 GPU 命令

Codex 的 `workspace-write` 沙箱可能不暴露 `/dev/nvidia*`：此时沙箱内的
`nvidia-smi` 会报无法连接驱动，但不能据此断定服务器 GPU 不可用。
本项目已验证，同一条 `nvidia-smi` 命令在沙箱外可以正常列出 GPU。

- GPU 检查命令若因设备或驱动不可见而失败，立即使用 `exec_command` 的
  `sandbox_permissions: "require_escalated"` 和简短的 `justification`，在沙箱外重跑；
  不要只报告“Codex 不能访问 GPU”。
- 需要实际运行 CUDA/BPP 时，也对相应命令申请沙箱外执行。批准后在同一执行方式下
  运行启动器的 CUDA 运算预检查和评测；沙箱内的预检查失败不算模型运行失败。
- 如果沙箱外仍失败，再检查驱动、设备节点、CUDA 环境与选卡映射；如果提权未获批准，
  明确说明是执行权限限制。无需为了单次 GPU 任务要求用户永久开启 Full access。

本服务器混用多种显卡，`nvidia-smi` 物理编号与默认 CUDA 枚举顺序不同。
`bpp/run_eval.sh --gpu VALUE` 会直接设置 `CUDA_VISIBLE_DEVICES=VALUE`；选中的单张卡在进程内称为 `cuda:0`。
不要把 `nvidia-smi` 编号或 `/proc/driver/nvidia/gpus` 的 minor 编号直接当作 `--gpu` 值。

映射来源：`/data3/huangxinagyun/project/WorldForge/AGENTS.md`。当前默认环境记录如下：

| `nvidia-smi` 物理编号 | GPU 型号 | `CUDA_VISIBLE_DEVICES` / BPP `--gpu` 值 |
| ---: | --- | ---: |
| 0 | NVIDIA RTX A6000 | 5 |
| 1 | NVIDIA RTX A6000 | 6 |
| 2 | NVIDIA RTX 6000D | 1 |
| 3 | NVIDIA RTX 6000D | 2 |
| 4 | NVIDIA RTX 6000D | 3 |
| 5 | NVIDIA RTX 6000D | 4 |
| 6 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 0 |

- 遵循用户指定的显卡；启动前查看型号、UUID 和空闲显存：
  `nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total --format=csv`。
- 需要准确指定某张卡时，优先使用 GPU UUID；`--gpu GPU-<uuid>` 也受支持。
  若修改 `CUDA_DEVICE_ORDER` 或驱动环境，重新核对上表。
- BPP 默认环境 `bpp/env/bpp` 为 PyTorch 2.7.1+cu118，使用 RTX A6000（`--gpu 5` 或 `--gpu 6`）。
  `--gpu 0` 对应 Blackwell `sm_120`，已确认与这个 cu118 构建不兼容。
- Blackwell 使用独立的 `bpp/env/bpp-sm120` 环境，配置为 PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128。
  安装命令：`bash bpp/setup.sh env --env sm120`；评测命令：
  `bash bpp/run_eval.sh --env sm120 --mode smoke --gpu 0 --save-video`。
  该 profile 的配置与入口已通过离线检查，但当前沙箱的网络限制阻止了实际安装，尚未验证 sm120 GPU rollout。
  不能仅凭 `torch.cuda.is_available()` 判断架构兼容，需通过启动器的真实 CUDA 运算检查。
- 首选试跑物理 GPU 0 的 RTX A6000：
  `bash bpp/run_eval.sh --mode smoke --gpu 5`。
  空闲状态随时变化，后续运行仍需检查。
- BPP 启动器会先运行小型 CUDA 运算检查；若执行环境无法访问 GPU 驱动，明确报告该限制，不将预检查失败当作模型运行结果。
- 启动后可用 `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv`
  将进程 PID 与 GPU UUID 对应，确认实际选卡。

# 实验输出目录规范

`bpp/experiments/**` 下所有脚本产出一律按层级目录组织，禁止把实验参数全部编码进
超长文件名。参考实现：`procedural_constraints/data/{minimal,zero_shot,c0_baseline}`。

## 结构

```
<output_root>/
  <task>/                              # 实验任务/条件名，一层一个维度
    <task级汇总图>.png                  # 跨 layout/seed 的汇总图放任务层
    layout_XX/                         # 实验条件文件夹（布局、材质等逐层嵌套）
      seedY/                           # seed 作为子文件夹；无 seed 维度则省略
        <procedure>.{hdf5,mp4}         # 文件名只用 procedure/角色，参数全在路径里
        <procedure>_trajectory.csv
        <procedure>_contacts.json
        <procedure>_statistics.csv
        pair.png                       # 同层成对对比图，统一叫 pair.png
        teacher_forced.{csv,npz}
  # 顶层只放整个实验的汇总与元数据：
  invocation.json  model_loading.json  *_summary.{csv,json}  REPORT.md
  statistics.csv   manifest.json       config.json           initial_state.npz
  scene.xml        *_validation.json   rollout_results.csv   stage_results.csv
```

## 命名规则

- 目录即参数：task、layout、seed、procedure 等维度逐层做文件夹；
  叶子文件名只保留角色（`narrow` / `wide` / `normal` / `gentle` / `original`），
  不再出现 `rollout_<task>_query.._source.._<proc>_seed..` 这类全参数拼接名。
- 同一 leaf 目录内的一组产物共享前缀 + 固定后缀：
  `<name>.hdf5`、`<name>.mp4`、`<name>_trajectory.csv`、`<name>_contacts.json`、
  `<name>_statistics.csv`。
- 汇总表/元数据/hdf5 attrs 里的 `name` 字段 = 相对路径（不含扩展名），如
  `obstacle_clearance/layout_00/seed0/narrow`。这样 `output / f"{name}_trajectory.csv"`
  直接命中文件，行记录自定位；同时保证 `name` 在表里唯一。
- 画图、读轨迹等按行定位文件时，一律从 `name` 或 (task_id, layout_id, seed,
  procedure_id) 列重建路径，不要再拼平铺文件名。
- Recorder/落盘类约定接收 `(叶子目录, 文件前缀)`；目录由 `mkdir(parents=True,
  exist_ok=True)` 创建，调用方不手工拼路径。

## 迁移既有输出

- 文件按旧文件名解析（regex）移动到对应层级；顶层汇总文件原位不动。
- 同步更新 CSV 的 `name` 列、hdf5 `attrs["name"]`、validation/report JSON 里
  的 `name` 字段为新相对路径，否则 analyze/plot 阶段按 `name` 找文件会失配。
- 改完跑一次 `py_compile` + 路径命中自检（按新规则重建路径，确认文件存在）。

## 待整理

- `bpp/experiments/procedural_demos/outputs/`：仍是平铺长命名
  （`obstacle_l{N}_{material}_s{seed}_{procedure}.*`、`cup_layout{N}_{state}_{proc}.*`、
  `vis_*.png`、`traj_*.png`、`manifest.csv`），应按本规范改为
  `<task>/<condition>/<seed>/<procedure>.*` 结构并同步修改其生成脚本。
