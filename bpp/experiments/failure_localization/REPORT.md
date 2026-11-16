# Native Baseline Regression & Failure Localization

日期：2026-11-11。范围：诊断 `procedural_constraints` 的 C0 / zero_shot 大量失败，
未训练模型、未修改权重、未下载资源。

## 结论摘要

1. **Native baseline 完全复现**：`native_regression` 用同一 checkpoint、同一 prompt HDF5、
   同一 query states（demo_0–9）、同一 diffusion seeds、同一推理参数重跑 20 条 rollout，
   每条结果与历史 `paired10_controlled` **逐位一致**（sim_state sha256、agentview sha256、
   成功/步数全同）：original **7/10**，curved 5/10。评测管线没有回归。
2. **C0 不是原生基线**：C0 只有 *prompt* 是原生 demo_0；query 场景是改造后的 OOD 场景
   （碗 -9.5cm、盒 +4.4cm、搬运距离 15cm→30cm、额外可见橙色柱子），且全部 20 条共用
   同一个 canonical 初始状态，没有 12 步 open-gripper warmup。
3. **0/20 collision-free 部分被高估，但非误统计主因**：原始 `collision` 标志统计的是
   "除 gripper-bowl、robot-floor、非 transport 期 obj 接触外的一切接触"。按
   `unintended_collision`（障碍物/ distractor/goal-box/抓取窗口外 table 接触/
   transport 期 bowl-table 拖蹭）重分类：C0 为 16/20 真实非预期碰撞
   （11 条涉及新增障碍物），仅 4/20 为良性 finger-table 擦碰；zero_shot 37/40 为真实
   非预期碰撞（17 条为搬运中的 bowl↔post 撞击）。抓握接触没有被误计为失败。
4. **失败主因是环境分布变化 + prompt/scene 几何不匹配**，不是评测实现问题：
   - `obstacle_clearance`：策略固有的 +x overshoot（原生 seed0 也到 x≈+0.032）使
     gripper 在第 ~36 步撞上位于 x=-0.02 的柱子（10/10 全部如此），之后卡死或游荡
     到 goal-box 区域，bowl 从未抓起 → 0/10 grasp。
   - `gentle_transport`：柱子在 y=0.26–0.34 不在路径上 → grasp 10/10；失败集中在
     30cm 搬运/放置段（4/10 goal），与原生 ~7/10 的差距在方向和量级上符合 OOD 退化。
   - `zero_shot`：grasp 20/20（prompt 与场景几何一致），但搬运中 carried bowl
     撞柱（17/20 条 obstacle↔bowl/gripper）→ 0/40，说明模型并未采纳演示的绕行路径。

---

## 1. Native Baseline Regression（任务一）

历史配置：`experiments/native_prompt_swap/outputs/paired10_controlled/evaluation_config.json`
— checkpoint `liberogen_spatial_combination_behavior_prompting.ckpt`（严格加载、官方
`libero_policy_dunetp` 配置）、prompt `original_replayed_demo.hdf5`、query = source
demo_0–9 初始状态、diffusion seed = query index、12 步 open-gripper warmup、550 步上限、
exec horizon 12/16、20 Hz。

复现命令（A6000，UUID 选卡，EGL device 0）：

```bash
bash bpp/experiments/native_prompt_swap/run.sh evaluate \
  --output bpp/experiments/native_prompt_swap/outputs/native_regression \
  --gpu GPU-e0914423-1b44-589b-fdb3-0360e8f79b43 --egl-device 0 \
  --trials 10 --max-steps 550
```

结果：`outputs/native_regression/rollout_results.json`；逐 seed 对比见
`failure_localization/native_baseline_regression.csv`。

| seed | original 历史 | original 复现 | curved 历史 | curved 复现 |
|---:|:---:|:---:|:---:|:---:|
| 0 | ✗ 550 | ✗ 550 | ✗ 550 | ✗ 550 |
| 1 | ✓ 143 | ✓ 143 | ✓ 142 | ✓ 142 |
| 2 | ✓ 126 | ✓ 126 | ✓ 127 | ✓ 127 |
| 3 | ✓ 140 | ✓ 140 | ✓ 141 | ✓ 141 |
| 4 | ✓ 134 | ✓ 134 | ✓ 134 | ✓ 134 |
| 5 | ✗ 550 | ✗ 550 | ✗ 550 | ✗ 550 |
| 6 | ✓ 115 | ✓ 115 | ✓ 115 | ✓ 115 |
| 7 | ✓ 125 | ✓ 125 | ✗ 550 | ✗ 550 |
| 8 | ✓ 196 | ✓ 196 | ✗ 550 | ✗ 550 |
| 9 | ✗ 550 | ✗ 550 | ✗ 550 | ✗ 550 |

`prompt_diagnostics.json` 与历史逐字节一致；`initial_identity`（sim_state sha256 +
首帧 RGB sha256）20/20 全同。**7/10 可复现，推理/控制/预处理路径无漂移**，
无需继续逐项排查 normalization —— 同一份 `inference.py` 的 `load_prompt` /
`query_observations` / `RotationTransformer` 在三个实验间共用。

## 2. Scene Initialization Diff（任务二）

完整表：`failure_localization/config_diff.csv`。关键差异：

| 维度 | Native (7/10) | C0 (4/20) | Zero-shot (0/40) |
|---|---|---|---|
| BDDL/场景 | 原生 | 原生 + 注入 mocap 橙柱 (r25mm×h180mm) | 同 C0 |
| 碗初始 XY | x=-0.075, y∈[-0.013,0.013]，每 demo 不同 | (-0.170, 0.0073) 固定 | 同 C0 |
| 目标盒 XY | ≈(0.086, 0.018) | (0.130, 0.025) | 同 C0 |
| 搬运距离 | ~0.15 m | ~0.30 m | ~0.30 m |
| 障碍物 | 无 | obstacle_clearance: x=-0.02,y=0.005–0.025 在接近/搬运走廊内；gentle: y=0.26–0.34 路径外 | 同 C0 |
| Query 初始状态 | 10 个不同 demo states | 1 个 canonical state × 5 柱位 × 2 seed | 同 C0 |
| Prompt | 原生 demo_0（重捕获版） | 原生 demo_0（源 HDF5 直读） | 生成演示（布局 (i+1)%5，慢速 700+ 步专家） |
| Prompt↔Query 几何 | 匹配 | **不匹配**（prompt 原生几何，query 重定位+柱） | 部分匹配（同重定位，柱位不同） |
| Warmup | 12 步 open-gripper | 无；首帧历史为单帧复制 | 同 C0 |
| 步数上限/执行域 | 550 / 12 | 800 / 12 | 800 / 12 |

首帧对比（`failure_localization/frames/`）：C0/zero_shot 首帧中橙色柱子位于
agentview 中央、碗在更左侧、盒子更远；与 prompt 中的原生场景明显不同。

**C0 没有使用原生 LIBERO query**：`query_identities.json` 中 20 条 rollout 的
`sim_state_sha256` 完全相同（`957465dd…`），来自 `minimal/initial_state.npz`
（= demo_0 状态 + 碗 x→-0.170、盒→(0.130,0.025) 的改造状态），而非 demo_0–9 的
各自初始状态。唯一用原生的是 prompt（`load_prompt` 读取源 HDF5 的 `data/demo_0`）。
另注：C0 用的是源 HDF5 直接读取版 prompt —— 历史上该版本在 `paired10` 中为 6/10；
重捕获版（RGB 与源差异最大 216/255、ee_pos 差 3.3mm）为 7/10。

## 3. Collision Audit（任务三）

输出：`failure_localization/collision_audit.csv`（每 rollout 一行：首次接触的
步数/阶段/物体对 + 分类计数 + `any_contact`/`unintended_collision`/`benign_only`）。

分类规则：检测器已排除 gripper↔bowl（正常抓取）与 robot↔floor；
`unintended` = 涉及 `procedure_obstacle`、distractor（plate_1/flat_stove）、
cookies_1 goal-box、抓取窗口（<120 步）之外的 robot-table、或 transport 阶段的
bowl-table 拖蹭。良性 = 仅 grasp 窗口内 finger-table 擦碰或非 transport bowl-table 静置。

**C0（20 条）**：
- `any_contact` 20/20；`unintended_collision` 16/20；benign-only 4/20；无接触 0。
- obstacle_clearance 10/10 全部为 gripper finger/hand/link5/6 ↔ obstacle，
  **首次接触集中在第 36–40 步（grasp 接近期的 +x overshoot）**，且为数千次持续按压。
- gentle 失败条另有 finger↔plate_1（按压 distractor）与 1 条 link5↔obstacle；
  成功条中只有 step 55–58 的 finger-table 擦碰（benign）。1 条成功条（layout_04/seed0）
  在 transport 段发生 bowl↔flat_stove 真碰撞但任务仍成功。
- 结论：**0/20 collision-free 不是把正常抓取接触误算**（gripper-bowl 本就被排除），
  但 "collision-free" 指标混入了良性擦碰 —— 4/20 实为 benign-only。
  不过 16/20 的真实非预期碰撞是实质性的。

**Zero-shot（40 条）**：`unintended` 37/40；3 条无接触（与 REPORT 的
collision-free 2/20+1/20 一致）。obstacle_clearance 17/20 为搬运中
carried bowl / gripper hand ↔ post（首次接触多在 transport 第 131–570 步）；
gentle 多为 finger-table 晚期按压（step ~600+，抓取已丢失后磨桌）与 transport
bowl-table 拖蹭。**碰撞统计是真实的**。

**Stage-detector 说明**：`phase` 标签由 `bowl_z ≤ init+3.5cm → "grasp"` 等阈值决定；
bowl 被举起后又被撞落/推倒时阶段回退为 "grasp"，此时 bowl↔obstacle 接触记为
grasp 阶段 —— 是"重新抓取/失手"语义而非初次接近。`failure_stage` 由里程碑推导，
不受该标签歧义影响；但按阶段统计接触时需注意此语义。

## 4. 失败机制（逐问题回答）

**Q1 为何之前 7/10、现在 C0 很差？**
评测实现无回归（native 逐位复现）。差异在实验设计：C0 的 query 场景 OOD ——
物体移位使搬运距离翻倍且画面中出现训练分布外的橙色立柱；prompt 仍是原生几何。
直接机制分两段：
- obstacle_clearance：原生策略在接近碗时本就有 +x overshoot（native seed0 eef 到
  +0.032；C0 gentle 同场景下也 overshoot 到 -0.008 仍可回抓）。obstacle_clearance
  的柱子 (x∈[-0.045,+0.005], y∈[-0.02,+0.05], z 0.90–1.08) 正好堵住这段 overshoot
  → finger 第 ~36 步撞柱后持续推压（最多 12,676 次接触事件），bowl 始终未被抓起。
- gentle_transport：柱不在路径上，grasp 全部成功；失败在 30cm 的运输/放置段
  （4/10），符合"同一 prompt + 更远目标"的 OOD 退化，而非实现缺陷。

**Q2 C0 是否真正使用原生 LIBERO Query？** 否。仅 prompt 原生；query 状态、场景几何、
初始观测均来自改造场景；另缺 12 步 warmup（首帧历史为单帧复制）且步数上限为 800。

**Q3 0/20 collision-free 是否误统计？** 不是误统计：检测器已排除正常 gripper-bowl
接触；但指标偏严 —— C0 有 4/20 仅含良性擦碰（其中 3 条任务成功）。若按
`unintended_collision` 口径，C0 = 16/20 碰撞，zero_shot = 37/40 碰撞，
**碰撞本身是真实物理事件**（finger 撞柱、carried bowl 撞柱、按压 distractor/桌面）。

**Q4 失败主因？** 环境分布变化（重定位 + 可见障碍物的 OOD 场景）与
prompt/scene 几何不匹配是主因；模型行为上表现为按 prompt 几何运动的倾向
（overshoot 方向与 prompt 原生接近路径一致）。未排除项：
- SOURCE 直读 prompt（历史 6/10 版）与重捕获 prompt 的 1-seed 差异对 C0 的影响未单独测；
- 缺 warmup / 单帧历史对首步预测的独立影响未隔离；
- C0 gentle 4/10 与原生 ~7/10 的差是否超出小样本噪声未量化；
- "OOD 场景导致 overshoot" 与 "overshoot 是本有行为、只是被柱子惩罚" 之间，
  现有证据偏后者（native/gentle 均见 overshoot），但无法完全分离视觉 OOD 的贡献 ——
  这正是后续 controlled experiment 需要隔离的变量。

## 附：产物清单

- `failure_localization/native_baseline_regression.csv` — 20 条回归对比（全同）。
- `failure_localization/config_diff.csv` — Native/C0/Zero-shot 配置差异表。
- `failure_localization/collision_audit.csv` — 60 条 rollout 接触分类审计。
- `failure_localization/frames/` — 首帧/末帧/碰撞时刻 PNG 对比。
- 回归 rollout 目录：`experiments/native_prompt_swap/outputs/native_regression/`
  （含每条 rollout mp4 与 trajectory CSV）。
- 失败案例视频（既有）：`procedural_constraints/data/c0_baseline/obstacle_clearance/*/seed*/original.mp4`
  （finger 撞柱后卡死）；`procedural_constraints/data/zero_shot/obstacle_clearance/*/seed*/{narrow,wide}.mp4`
  （carried bowl 撞柱）。
