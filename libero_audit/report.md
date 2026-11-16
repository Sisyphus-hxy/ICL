# LIBERO / LIBERO-Gen HDF5 demonstration 数据审计

本报告由 `libero_audit/audit.py` 生成。原始 HDF5 以只读方式打开；RGB montage 逐帧读取，没有把整段图像一次性载入内存。

## 数据定位与范围

- 扫描根目录：`/data2/huangxiangyun/project/icl`。发现 216 个可识别的 robomimic 风格 `data/demo_*` HDF5（另有 0 个无法按该 schema 读取）。
- 其中 2 个文件有至少 5 条 episode，作为多示范互补性主分析对象；214 个仅有单条 episode 的 canonical HDF5 记录在 inventory 中，但不能构造同 task 的 support/query 对。当前主分析对应两份 `bpp/datasets/libero/` 数据，各 50 episodes。
- `.cache/rollout/` 下还存在 Zarr rollout，但本次任务限定 HDF5，因此未混入 HDF5 统计。

### 主分析文件

| family | task | episodes | instruction |
|---|---|---:|---|
| LIBERO | `LIBERO__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo` | 50 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| LIBERO-Gen/custom | `LIBERO-Gen_custom__pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box_demo` | 50 | pick the akita black bowl from table center and place it on the cookies box |

`LIBERO-Gen/custom` 是基于路径包含 `selected_combinations_view` 与 episode-level `seed` 的推断；文件没有显式 `LIBERO-Gen` tag，因而报告不把这一点当作观测事实。`LIBERO` 则有 `data.attrs[tag]=libero-v1` 的直接证据。

## Schema 与真实统计

两份主数据均为 HDF5 → `data` → `demo_i`；每条 demo 有 `actions`、`dones`、`rewards`、`states`、`robot_states` 和 `obs`。`obs` 的真实字段包括 `agentview_rgb`、`eye_in_hand_rgb`、`ee_pos`、`ee_ori`、`ee_states`、`gripper_states`、`joint_states`。动作均为 7 维；生成变体另外有 7 维 `abs_actions`。

| task | length min/p25/median/p75/max (mean±std) | cameras | state | object pose |
|---|---|---|---:|---|
| `LIBERO__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo` | 75.0/89.0/95.0/107.2/168.0 (101.4±20.0) | agentview_rgb, eye_in_hand_rgb | 92 (time+qpos+qvel) | akita_black_bowl_1, akita_black_bowl_2, cookies_1, glazed_rim_porcelain_ramekin_1, plate_1 |
| `LIBERO-Gen_custom__pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box_demo` | 114.0/132.0/142.0/149.8/190.0 (142.3±15.2) | agentview_rgb, eye_in_hand_rgb | 92 (time+qpos+qvel) | akita_black_bowl_1, akita_black_bowl_2, cookies_1, glazed_rim_porcelain_ramekin_1, plate_1 |

`states` 的 92 维在 XML joint 计数下可解析为 `time + qpos(48) + qvel(43)`；free joint 的物体位姿使用 qpos 中的 xyz+quaternion 计算。这个映射是由数据中的 model XML 和维数一致性得到的真实结构检查，不需要视觉模型。

RGB 帧按 `data.attrs['macros_image_convention']='opengl'`（OpenGL 底左原点）存储，直接可视化是上下颠倒的；本报告的 montage 已做 `flipud` 还原为自然视角。任何把 RGB 喂给模型的管线都必须同样处理这个翻转，否则等于在颠倒画面上训练。

## 同 task demonstrations 的差异

下表为所有 demo pair 的数值统计：state/EEF 是首帧或末帧的 L2 距离；动作和 EEF trajectory 先各自重采样到 32 个相对时间点，再计算每时间点 L2 的均值；长度差为绝对 step 数。

| task | init state pair mean | final state pair mean | action trajectory pair mean | EEF trajectory pair mean | length diff mean/p95/max |
|---|---:|---:|---:|---:|---:|
| `LIBERO__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo` | 0.609 | 1.718 | 0.836 | 0.059 | 20.0/70.8/93.0 |
| `LIBERO-Gen_custom__pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box_demo` | 0.139 | 1.007 | 0.388 | 0.051 | 16.9/44.0/76.0 |

物体位姿变化的均值/分位数已写入 `summary.csv` 和 `pairwise_stats.csv` 的 JSON 字段；`episode_stats.csv` 保存每条 episode 的初末位置、旋转角、EEF/关节/robot state 变化。数值是状态和动作数据的直接统计，未使用大型视觉模型。

## Montage

每个主任务随机（固定 seed=0）抽取最多 5 条 episode，每条从头到尾均匀取 8 帧。每个可用 RGB 相机各生成一张 montage，文件位于 `montages/`；抽样 episode、精确 frame index 和是否应用 `flipud` 记录在 `montages/index.csv`。

- `agentview_rgb`: `montages/LIBERO__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo__agentview_rgb.png`
- `eye_in_hand_rgb`: `montages/LIBERO__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo__eye_in_hand_rgb.png`
- `agentview_rgb`: `montages/LIBERO-Gen_custom__pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box_demo__agentview_rgb.png`
- `eye_in_hand_rgb`: `montages/LIBERO-Gen_custom__pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box_demo__eye_in_hand_rgb.png`

## Support-query pairs

对每个 task，N 条 support 与另一条 query 独立抽取、且 query 不进入 support。50 条 episode 使 N=1/2/4/8 全部可行；若固定 query，剩余可选 support 组合数分别为 C(49,N)。

| task | N=1 | N=2 | N=4 | N=8 | query count (按 N) |
|---|---|---|---|---|---|
| `LIBERO__pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo` | yes | yes | yes | yes | 49/49/49/49 |
| `LIBERO-Gen_custom__pick_up_the_black_bowl_from_table_center_and_place_it_on_the_cookie_box_demo` | yes | yes | yes | yes | 49/49/49/49 |

这表示数据层面可以做 support-query pair；它不保证 support 一定包含 query 所需的互补信息，互补性需要在严格 episode-level split 上评测。

## Task 可推断性与数据捷径风险

以下是基于字段和场景结构的审计判断，属于推测/风险分析，不是训练分类器得到的准确率：

- 语言指令直接存于 `data.attrs['problem_info']['language_instruction']`，并且 task 文件名、BDDL 名称也携带目标物体和动作。若 predictive study 输入 instruction，模型可能只靠语言完成 task identity 判断而忽略 support。
- `agentview_rgb` 和 `eye_in_hand_rgb` 同时存在；初始帧可见物体颜色、几何和相对布局。若 query 目标/场景在视觉上足够显著，模型可能只依赖当前 observation。
- 原始 `states` 含可解析的物体 free-joint 位姿，`joint_states`/`ee_states` 含机器人状态。若把这些状态直接提供给模型，任务和进度可能被状态捷径编码。
- 两个主文件各自只对应一个 task；因此当前仓库不能凭这两份数据估计跨 task 的 instruction/observation-only baseline，也不能证明 support 对 query 的增益。应显式加入 no-support、language-only、initial-observation-only 和 state-only 对照。

## 离线 predictive study 适用性与划分建议

判断：适合做小规模、同 task、episode-level 的离线 predictive study 原型，尤其适合检验动作/轨迹预测中 support 数量和示范差异的影响；不适合作为广泛 LIBERO/多 task 泛化结论的数据集。当前只有 2 个 task、每 task 50 条 demo，且不同文件的生成协议不同，需在论文中明确数据来源。

建议划分：对每个 task 按 episode ID 做 30/10/10 的 train/val/test（或 35/7/8），所有帧和由同一 episode 生成的窗口必须留在同一 split；support 与 query 只能从同一 split 内抽取，禁止跨 split 配对。生成变体可按 seed 分层，LIBERO-v1 没有 seed 字段则按 demo ID 固定随机划分并保存 manifest。需要跨 task 泛化时只能做 leave-one-task-out 探索，当前两 task 规模过小，不能作为稳定泛化结论。

## 输出文件

- `summary.csv`: 每个主 task 的 schema、轨迹分布、相机、动作维度、物体位姿和 support-query 可行性。
- `episode_stats.csv`: 每条 demo 的长度、状态/动作/EEF/物体变化和奖励字段。
- `pairwise_stats.csv`: 同 task demo pair 的差异汇总。
- `inventory.csv`: 扫描到的所有 canonical HDF5 及其 episode 数量，便于复核排除的单 episode rollout。
- `montages/`: 相机对比图及 `montages/index.csv`。

