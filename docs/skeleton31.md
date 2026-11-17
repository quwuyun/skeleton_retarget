# 31 关节骨架：LAFAN1 行走流程

这里保留原无肌肉骨架的 12 条 120 Hz 行走结果与复现入口。当前训练使用的 17 个活动轴 collision 模型，请先看 [自用工作区说明](../README.md)；两套模型的关节定义不同，不能混用 NPZ。

所有命令从项目根目录执行，使用项目的 `uv` 环境。

## 文件对应关系

| 内容 | 相对项目根目录的位置 |
| --- | --- |
| 原始 BVH | `data/raw/lafan1/` |
| 两套模型共用的已选步态窗口 | `data/selections/lafan1_walks.json` |
| 每条动作的标定模型 | `assets/skeleton31/walks/<动作名>/human_skeleton.xml` |
| 每条动作的 IK 配置 | `skeleton_retarget/retargeting/ik_configs/skeleton31/walks/<动作名>.json` |
| 正式 NPZ、PKL、元数据与验证报告 | `data/outputs/skeleton31/` |
| 选段源姿态，用于误差核验 | 输出目录中的 `references/` |

每份模型目录还有 `reference.json`、`manifest.json`。各动作的物理骨架相同，但源人体骨段比例和参考标定不同，模型 keyframe 也有微小朝向差异；**不能把这些模型或配置合并为一个文件**。`assets/skeleton31/walks/` 是使用相对网格路径的可移植模型，适用于新的重定向输出。现有 `data/outputs/skeleton31/` 元数据仍绑定较早的 `assets/skeleton31/<动作名>/` 模型快照；对这些旧结果做严格验证时，必须使用其 `.json` 记录的模型文件和哈希，不能直接换成 `/walks/` 中的 XML。

## 先回放和检查现有结果

```bash
uv run python -m skeleton_retarget.scripts.vis_motion \
  --robot_xml assets/skeleton31/walks/walk3_subject3/human_skeleton.xml \
  --motion data/outputs/skeleton31/walk3_subject3_120hz.npz
```

切换动作时同时替换模型路径和 NPZ 文件名。播放器默认回到起点循环，`--once` 仅播一次，`--speed 0.5` 仅改变显示速度。片段按步态相位截取，未强制首尾姿态完全相同，也未累积循环位移。

重新检查时可把新报告写到临时位置，保留原验证证据：

```bash
uv run python -m skeleton_retarget.scripts.validate_skeleton31 \
  --robot_xml assets/skeleton31/walk3_subject3/human_skeleton.xml \
  --motion data/outputs/skeleton31/walk3_subject3_120hz.npz \
  --source_reference data/outputs/skeleton31/references/walk3_subject3_source.npz \
  --output /tmp/walk3_subject3_skeleton31_validation.json
```

验证器默认读取同名 PKL、元数据 JSON，检查六字段、时间网格、来源哈希、关节限位与锁、四元数、FK、脚盒高度，并报告跟踪误差与滑步诊断。

上面的验证命令针对当前目录保留的历史模型快照。该旧 XML 的 mesh 路径记录了原工作区绝对路径；跨机器使用时，优先在新的输出目录用 `assets/skeleton31/walks/` 模板重新导出，再验证新生成的元数据。新导出的操作示例见[根目录 README](../README.md)。

## 单条重新导出

优先复用保存的模型和 IK 配置，避免外部源 MJCF 后续改动影响旧骨架。下面复现 `walk1_subject1` 的已选窗口，另存结果：

```bash
uv run python -m skeleton_retarget.scripts.bvh_to_skeleton31 \
  --format lafan1 \
  --bvh_file data/raw/lafan1/walk1_subject1.bvh \
  --robot_xml assets/skeleton31/walks/walk1_subject1/human_skeleton.xml \
  --ik_config skeleton_retarget/retargeting/ik_configs/skeleton31/walks/walk1_subject1.json \
  --start_frame 4095 --max_frames 100 \
  --output_fps 120 --speed 1 \
  --use_equality_constraints --ground_align --zero_xy --quiet \
  --output data/outputs/skeleton31_replay/walk1_subject1_120hz
```

`start_frame` 为零基源帧下标，`max_frames` 为源帧数；其余动作的窗口从共享清单 `data/selections/lafan1_walks.json` 读取。`output_fps` 控制重采样频率，`speed` 控制时间缩放。单条入口不会另外生成源参考，可用原正式结果的同窗口 `references/` 验证。

若要重新选择窗口：

```bash
uv run python -m skeleton_retarget.scripts.select_walk_cycles \
  data/raw/lafan1/walk1_subject1.bvh \
  --cycles 2 --output /tmp/new_selection.json
```

自动选择器会在两周期候选不足时尝试一周期。新候选还需经过目标模型 IK、限位诊断和回放检查；已有 12 条窗口包含人工复核后的调整，应以共享清单为准。批处理仍支持通过 `--plans` 传入其他独立选段文件。

## 已有 12 条动作

| 动作 | 帧数 | 采样跨度 / 秒 | 源步态周期 |
| --- | ---: | ---: | ---: |
| walk1_subject1 | 396 | 3.292 | 2 |
| walk1_subject2 | 120 | 0.992 | 1 |
| walk1_subject5 | 120 | 0.992 | 1 |
| walk2_subject1 | 256 | 2.125 | 2 |
| walk2_subject3 | 388 | 3.225 | 2 |
| walk2_subject4 | 388 | 3.225 | 2 |
| walk3_subject1 | 340 | 2.825 | 2 |
| walk3_subject2 | 284 | 2.358 | 2 |
| walk3_subject3 | 340 | 2.825 | 2 |
| walk3_subject4 | 244 | 2.025 | 2 |
| walk3_subject5 | 140 | 1.158 | 1 |
| walk4_subject1 | 240 | 1.992 | 2 |

共 3256 帧，9 条双周期、3 条单周期。一个周期指同一只脚再次到达相同相位，包含左右两步；低抬脚的 `walk3_subject3` 使用同脚向前摆动峰确定相位。源骨盆均速约 0.39–1.39 m/s，目标根均速约 0.47–1.66 m/s，均保持原速；需要慢走可先看 `walk3_subject3`。`summary.csv/json` 和 `selection.json` 保存准确统计。

源采样约 30.0003 Hz。IK 先在原生时间线上求解，再对根平移和标量关节线性插值、根四元数做最短弧 SLERP，最后重新计算脚盒高度与 body FK。120 Hz 时间网格不外推，末尾不足一帧的部分会略去；准确跨度为 `(帧数-1)/120`，实际覆盖周期略少于表中整数。

## 输出格式与使用范围

NPZ 使用 `np.load(path, allow_pickle=False)` 读取，PKL 保存相同六字段。

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `fps` | 标量 | 120 Hz |
| `root_pos` | `(T,3)` | 根世界位置，米 |
| `root_rot` | `(T,4)` | 根世界旋转，`xyzw` |
| `dof_pos` | `(T,31)` | 内部 hinge 角，弧度，含 8 个锁零坐标 |
| `local_body_pos` | `(T,20,3)` | 根置零平移、单位旋转时的模型 FK 位置 |
| `link_body_list` | `(20,)` | body 名称及排列顺序 |

列顺序见同名 JSON 的 `joint_columns`；MuJoCo qpos 的根四元数为 `wxyz`，播放器已转换。重采样后的逐帧根 Z 调整使最低脚盒角点离地 1 mm，首帧根 XY 平移到原点；修正量记录在元数据中。

保存报告确认了格式与运动学约束。动作仍有跟踪误差和滑步：足部最大姿态误差约 25.46°，共同高度修正后的足目标位置误差最大约 2.36 cm；低位脚盒材料角点的水平平均速度约 0.023–0.576 m/s。高度对齐不包含接触力、摩擦、平衡或控制求解。新 USD 应使用匹配的 17 轴流程，并在 Isaac 中另行检查关节映射与 FK。

模型制备、标定原理与批量复现的依赖说明见 [31 关节模型说明](skeleton31_model_notes.md)。
