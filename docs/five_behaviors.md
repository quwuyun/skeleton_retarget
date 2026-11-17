# 初始面朝 +X 的五类动作

本集合面向当前 `skeleton17_collision` 模型，包含前进、后退、中立直立静站、完整左转、完整右转。输出仍为官方六字段 NPZ/PKL、120 Hz、原速；21 个 hinge 列中四列锁零，实际 17 个活动坐标。官方 GMR 文件、旧动作及对应模型/配置保持原样。

所有命令从 `/home/hy/workspace/skeleton_retarget` 执行。

## 文件、来源与窗口

- 模型：`assets/skeleton17_collision/human_skeleton.xml`。
- 原始数据：`data/raw/lafan1/`。
- 固定选段及源哈希：[lafan1_five_behaviors.json](../data/selections/lafan1_five_behaviors.json)。
- 输出：[five_behaviors_x/](../data/outputs/skeleton17_collision/five_behaviors_x/)，汇总见 [summary.json](../data/outputs/skeleton17_collision/five_behaviors_x/summary.json)。
- 每条文件前缀：`five_behaviors_x/<动作名>/<动作名>_120hz`，配套 `.npz/.pkl/.json` 和 `_validation.json`。

下表源帧均为零基、左闭右开 `[start,stop)`；时长是 `(输出帧数−1)/120` 的首末采样跨度。

| 动作名 | BVH 源帧 | 输出帧 | 时长 | 输出根朝向变化 | 输出净 XY 位移（米） |
| --- | --- | ---: | ---: | ---: | --- |
| `forward_walk2_subject4` | `[2763,2861)` | 388 | 3.225 s | −2.43° | `(+2.779,−0.031)` |
| `backward_walk3_subject4` | `[1866,1928)` | 244 | 2.025 s | −3.93° | `(−1.689,−0.218)` |
| `idle_neutral_standing` | 当前 MJCF 中立位 | 240 | 1.992 s | 0° | `(0,0)` |
| `turn_left_walk1_subject1` | `[590,745)` | 616 | 5.125 s | +95.53° | `(+1.748,+2.454)` |
| `turn_right_walk1_subject1` | `[995,1135)` | 556 | 4.625 s | −97.27° | `(+1.044,−2.362)` |

前进和后退使用已经验证过的历史窗口；只变换旧结果的根位姿，不再求解 IK。左右转使用 `walk1_subject1` 的既有标定 JSON 重新求解，源参考放在各片段自己的 `references/` 中，避免同一 BVH 的不同窗口互相覆盖。静站不使用 mocap 关节姿态，直接从当前 MJCF 的中立位生成。

转弯使用 `walk1_subject1` 中约 90° 的完整事件，包含转弯前后的稳定朝向区间；5.125 s 和 4.625 s 用于保证覆盖完整过程。此前行走扫描报告中的短转向候选用于识别“存在转弯”，不能直接当成完整事件窗口。静站将当前 MJCF 的 21 个标量关节坐标全部设为 0，根四元数设成直立的单位四元数，固定根位置，并按两只脚碰撞盒最低点设置根高，使脚底离地约 1 mm。原 mocap 自然站立文件 `idle_walk1_subject2` 保留在本目录，不再用作这条中立静站。

此处将所有关节角乘以 0 是可行的，因为此 MJCF 零位对应髋、膝、踝中立，双膝伸直、两腿并排；但**不能只改 `dof_pos` 后保留原根姿态/高度**。原 mocap 站姿带有约 20–32° 膝屈曲和明显髋姿态差异，原根高度也会使零位脚底穿地。新静站另外重设了直立根姿态和经几何校准的根高度，身体 FK 同步重算。输出 240 帧完全重复同一个姿态，因此根和关节速度都为零。

## 朝向转换改变什么

当前模型骨盆局部坐标以 +X 为前、+Y 为左、+Z 为上。先由首帧根四元数求出局部 +X 在世界 XY 平面的方向，再为整段确定**同一个**世界 Z 轴旋转 `Rz`。保存的根数据按以下规则变换：

```text
p'[t] = Rz · p[t] + translation_xy
q'[t] = qz ⊗ q[t]
```

`translation_xy` 令首帧 XY 为零，Z 不变。根四元数在文件中仍为 `xyzw`。首帧朝向 +X 不等于首帧四元数必须为单位旋转：原有 roll/pitch 继续保留。

| 数据 | 转换结果 |
| --- | --- |
| `root_pos` / `root_rot` | 世界 XY 平移和整段固定 yaw 变换 |
| `dof_pos` / `local_body_pos` / `link_body_list` / `fps` | 与对应未归一化结果相同；已有前后退结果逐元素一致 |
| 关节速度、关节正负方向 | 不变；本格式未另存速度，后续差分结果不受固定根朝向变换影响 |
| 世界线速度、世界角速度、世界 body 轨迹 | 应与根数据采用同一旋转；由新位姿差分/FK 重新计算即可 |
| 根/heading 坐标下的速度及相对转向 | 保持不变 |

因此前进主要沿世界 +X，后退主要沿世界 −X，二者首帧都面朝 +X。原数据仍有小幅横向漂移与 yaw 变化，没有强制拉直轨迹。左转为绕世界 +Z 的正 yaw，右转为负 yaw；随后身体会朝向新的世界方向，不能逐帧把朝向改回 +X，否则会抹掉转弯。

同名前缀 JSON 的 `heading_normalization` 记录旋转、平移及原朝向；前后退还记录父 NPZ/JSON 路径和哈希。`behavior_selection` 记录来源选段及清单哈希。源参考保持原坐标，验证器会对标定后的目标应用同一朝向变换，避免把合法的整段旋转误判为跟踪误差。

## 生成或复验集合

```bash
uv run python -m skeleton_retarget skeleton17 behaviors
```

命令使用固定清单及现有配套模型/IK 配置。新中立静站从模型零位生成；已有且与当前模型和选段清单一致的结果会复验复用。不匹配或不完整时会报错。指定新的 `--output_dir` 可另行构建一份。

需要在新目录重新生成时：

```bash
uv run python -m skeleton_retarget skeleton17 behaviors \
  --selection data/selections/lafan1_five_behaviors.json \
  --output_dir data/outputs/skeleton17_collision/five_behaviors_x_v02
```

普通 `skeleton17 batch` 仍使用原先的 12 条行走清单，不会自动加入这五条。新窗口应另存选段清单，使用 `behaviors --selection` 指定，不要改历史选段来混淆来源。

## 回放、预览与验证

回放完整左转，播放一次：

```bash
uv run python -m skeleton_retarget play \
  --robot_xml assets/skeleton17_collision/human_skeleton.xml \
  --motion data/outputs/skeleton17_collision/five_behaviors_x/turn_left_walk1_subject1/turn_left_walk1_subject1_120hz.npz \
  --once
```

无窗口预览，也可去掉 `--sheet_only` 输出 MP4：

```bash
uv run python -m skeleton_retarget preview \
  --robot_xml assets/skeleton17_collision/human_skeleton.xml \
  --motion data/outputs/skeleton17_collision/five_behaviors_x/turn_left_walk1_subject1/turn_left_walk1_subject1_120hz.npz \
  --output /tmp/gmr_turn_left_x --show_collision --sheet_only
```

独立验证左转时须填写实际的 616 帧：

```bash
uv run python -m skeleton_retarget skeleton17 validate \
  --robot_xml assets/skeleton17_collision/human_skeleton.xml \
  --motion data/outputs/skeleton17_collision/five_behaviors_x/turn_left_walk1_subject1/turn_left_walk1_subject1_120hz.npz \
  --ik_config skeleton_retarget/retargeting/ik_configs/skeleton17_collision/walk1_subject1.json \
  --source_reference data/outputs/skeleton17_collision/five_behaviors_x/turn_left_walk1_subject1/references/walk1_subject1_source.npz \
  --expected_frames 616 --expected_fps 120 \
  --output /tmp/gmr_turn_left_x_validation.json
```

前进/后退的验证继续使用原来对应 BVH 的 `references/`，以各自 JSON 的 `source_reference` 为准。`behaviors` 自动验证五条，额外检查首帧 +X/XY 零点、前后退方向、左右转符号及前后退关节数据保持一致；中立静站检查所有关节坐标为零、根姿态/位置恒定、脚底高度、两脚前后对齐与跨腿碰撞间隙，并核对 NPZ/PKL/FK。

中立静站使用目标模型的碰撞盒进行几何检查，运动学回放不会执行力矩控制或重力下的平衡验证。原自然站立结果及说明仍可在 `idle_walk1_subject2/` 查阅。

预览视频和八帧拼图见输出目录的 `preview/`。坐标轨迹、完整转弯源窗口及旧文件保留证据见 [本次检查报告](../reports/lafan1_five_behaviors_2026-11-15/README.md)。

## 单独转换其他已有结果

对六字段 NPZ 做独立朝向转换，默认首帧 +X 且 XY 清零；不会重跑 IK：

```bash
uv run python -m skeleton_retarget normalize-heading \
  --motion data/outputs/skeleton17_collision/walk/walk2_subject4_120hz.npz \
  --output data/outputs/skeleton17_collision/walk/heading_trial01/walk2_subject4_x_120hz
```

输入 NPZ 的同名 JSON 需保留，输出前缀必须是新文件。可用 `--target_heading_deg 90` 改为首帧面朝世界 +Y，`--keep_xy` 保留首帧 XY。工具也可处理 31 模型的六字段结果，但模型的根局部 +X 必须确实代表所需前向。

在 17 单条重定向命令中追加 `--align_heading`，可以直接输出初始 +X 的结果；不加时保持旧流程的世界朝向。两种方式都将变换写入元数据，供对应验证器使用。

## 训练使用

五类数据已使用同一个 17 模型和同一种 NPZ schema。Isaac/USD 侧仍应按关节名映射，正确处理根四元数顺序，并由最终使用的位姿计算速度。保留 JSON 以追溯模型、来源与列顺序。

完整转弯不是首尾闭合的周期动作。若训练加载器直接从末帧跳回首帧，会产生根位置与朝向跳变；应在片段结束时重置，或显式累计首尾的刚体变换后连接。即使直行，也不能把根世界位置直接首尾拼接而不处理位移。速度奖励可以调节控制目标，但不会自动改变参考动作的速度分布。
