# 17 自由度 collision 骨架：LAFAN1 行走重定向

本流程适配当前训练模型，核心计算在 `skeleton_retarget/retargeting/`，命令入口平铺于 `skeleton_retarget/scripts/`。31 关节全身模型及其 12 条正式动作独立保留，见 [31 关节流程](skeleton31.md)。目录总览和常用命令见[工作区 README](../README.md)。使用的上游模型为：

```text
/home/hy/workspace/humanoid-exo_model/mjcf/walk_skeleton_body_usd_collision.xml
```

本项目内的自包含副本为 `assets/skeleton17_collision/human_skeleton.xml`，同目录 `Geometry/` 存放 81 个复制资源，`manifest.json` 记录来源和 SHA256。模型物理定义、关节顺序/方向/限位、驱动、碰撞体和四个膝锁保持一致；仅加入等于新模型 qpos0 的 `retarget_reference` keyframe。没有引入下肢肌肉。

## 批量行走结果（2026-11-13）

已把共享清单 `data/selections/lafan1_walks.json` 中的 12 条行走重定向到本模型，共 3256 帧，均为 120 Hz、原速、约 1–2 个源步态周期。该清单与 31 模型共用，保存源帧窗口、周期和来源哈希。输出位于 `data/outputs/skeleton17_collision/walk/`，具体清单和均速见该目录的 `summary.csv/json`。不同片段速度不同，慢走优先看 `walk3_subject3`。

```bash
# 完整结果复验复用，缺失结果才新生成；不覆盖已有动作。
uv run python -m skeleton_retarget skeleton17 batch \
  --output_dir data/outputs/skeleton17_collision/walk

# 将全批另存至新的实验目录。
uv run python -m skeleton_retarget skeleton17 batch \
  --output_dir data/outputs/skeleton17_collision/walk/rebuild01
```

全批共用一份模型/Geometry。不同 BVH 各自产生静态 IK 标定和参考记录；JSON 的生成原理见 [IK 配置说明](ik_configuration.md)。模型 `manifest.json` 保留最初 `walk3_subject3` 的复制/标定历史；其余动作元数据中的 `ik_calibration` 指向各自新标定，分别核验源 BVH、模型和配置哈希。

`batch_reports/validation/` 是最新逐条复验；每个动作同名 `_validation.json` 是首次导出检查。这里的通过指格式和运动学约束，仍需结合脚滑、跟踪误差与双脚离地诊断评估动作质量。原慢走的详细说明如下。

## walk3_subject3 数据

正式文件为 `data/outputs/skeleton17_collision/walk/walk3_subject3_120hz.npz`，同名 `.pkl` 和 `.json` 分别是相同内容的 PKL 和详细元数据。

- BVH：`data/raw/lafan1/walk3_subject3.bvh`。
- 源区间：零基索引 `2575:2661`，即第 2575–2660 帧，共 86 帧，沿用之前挑选的约 2 个慢走步态周期。
- 源帧率：`30.000300003 Hz`。保持源时间速度，没有额外慢放。
- 输出：`120 Hz`、340 帧，时间戳为 `arange(340)/120`，首末采样跨度 `2.825 s`。
- 源首末跨度为 `2.833305 s`；末尾不足一个输出周期的约 `8.305 ms` 不外推。
- 调参中间版本已在目录整理时清理；正式动作及验证证据保持原样。

## 独立执行命令

使用项目通过 `uv sync --group dev` 创建的环境，并从项目根目录执行命令。实际 MuJoCo、Mink 和数值库版本由 `uv.lock` 固定，输出元数据会记录运行版本。

已有模型可直接重定向。需要从外部模型重新复制并标定时，写入一个新版本目录；后续重定向、验证同步指定此 XML/配置：

```bash
cd /home/hy/workspace/skeleton_retarget
uv run python -m skeleton_retarget skeleton17 prepare \
  --output_dir assets/skeleton17_collision/v2 \
  --config_output skeleton_retarget/retargeting/ik_configs/skeleton17_collision/v2/walk3_subject3.json
```

重定向。默认正式前缀已存在时会拒绝覆盖，重新实验应指定新的输出目录：

```bash
uv run python -m skeleton_retarget skeleton17 retarget \
  --bvh_file data/raw/lafan1/walk3_subject3.bvh \
  --start_frame 2575 --max_frames 86 --output_fps 120 \
  --output data/outputs/skeleton17_collision/rerun01/walk3_subject3_120hz
```

下面的验证、预览和回放命令读取已生成的正式结果（验证报告另写到 `/tmp`）：

```bash
uv run python -m skeleton_retarget skeleton17 validate \
  --robot_xml assets/skeleton17_collision/human_skeleton.xml \
  --ik_config skeleton_retarget/retargeting/ik_configs/skeleton17_collision/walk3_subject3.json \
  --motion data/outputs/skeleton17_collision/walk/walk3_subject3_120hz.npz \
  --source_reference data/outputs/skeleton17_collision/walk/references/walk3_subject3_source.npz \
  --output /tmp/gmr_walk3_validation.json

uv run python -m skeleton_retarget preview \
  --robot_xml assets/skeleton17_collision/human_skeleton.xml \
  --motion data/outputs/skeleton17_collision/walk/walk3_subject3_120hz.npz \
  --output /tmp/gmr_walk3_preview

uv run python -m skeleton_retarget play \
  --robot_xml assets/skeleton17_collision/human_skeleton.xml \
  --motion data/outputs/skeleton17_collision/walk/walk3_subject3_120hz.npz
```

交互回放需要桌面显示；预览脚本默认使用 EGL，生成 8 帧 PNG 和原速 30 fps MP4，仅用 `mj_forward` 回放保存的姿态。临时地板用于显示，不改变模型文件或动作。

## 处理原理和本次适配

1. LAFAN1 BVH 做 FK，统一为米和 Z 向上；按官方约定，FootMod 位置来自 Foot，旋转来自 Toe。
2. 用新模型真实零位重新计算长度缩放、位置偏移和四元数偏移。仅追踪骨盆、左右大腿/小腿/后足和躯干共 8 个 body，固定手臂不设动作目标。
3. 原生帧率下运行 GMR 两阶段加权 IK，同时加入关节限位和四个膝关节零位硬等式。
4. 左右腿的 thigh/shin/foot 共 9 对几何加入碰撞约束。每次 IK 更新后重新计算实际几何距离，不安全时减半步长。当前 Mink 1.3.0 的碰撞上界与 Δq QP 有 dt 系数差异，因此在这个独立适配器中局部补偿，不修改依赖或官方代码。
5. 插值到 120 Hz：根平移和标量关节线性插值，根旋转 SLERP，不外推。安全的原生端点不能保证中间安全，因此再用姿态 IK 修正，并限制相邻保存帧关节速度不超过 `8 rad/s`；每个相邻区间的 1/4、1/2、3/4 处也检查跨腿几何间隙。该离散检查不构成连续扫掠体积证明。
6. 水平根位置从零起步。根据两只真实 talus 脚盒全部角点计算避免穿地所需的根高度，再取最大垂直速度 `0.5 m/s` 的最小上包络，保持脚盒至少离地 1 mm。这个高度处理会允许短暂离地，不能当成物理接触求解。
7. 从最终关节角重新 FK，写出官方六字段。独立验证器核对字段、顺序、哈希、限位、膝锁、FK、地面和自碰撞，并量化源目标误差、滑动和速度。

需要避碰的原因：新脚部方盒包络覆盖足部网格并留每面约 3 mm 余量。原来的精确足部目标在部分帧会使这两个保守包络重叠，因此实际使用了“允许跟踪误差来满足新碰撞模型”的方案。碰撞盒相交不等于骨骼网格必然相交。

## NPZ 内容和顺序

所有位置单位为米，关节角为弧度。`np.load(path, allow_pickle=False)` 可直接读取。

| 字段顺序 | 字段 | 形状 | 含义 |
|---|---|---|---|
| 1 | `fps` | 标量 | 120.0 |
| 2 | `root_pos` | `(340, 3)` | 世界系根位置 XYZ |
| 3 | `root_rot` | `(340, 4)` | 世界系根四元数 **xyzw** |
| 4 | `dof_pos` | `(340, 21)` | 新 MJCF 的 21 个标量 hinge，按 qpos 顺序 |
| 5 | `local_body_pos` | `(340, 85, 3)` | 根置零平移和单位旋转后，各 body 的 FK 位置 |
| 6 | `link_body_list` | `(85,)` | 与位置第二维逐一对应的 body 名 |

21 列 `dof_pos` 顺序：

```text
0  hip_x_r        1  hip_y_r        2  hip_z_r
3  knee_x_r       4  knee_y_r       5  knee_z_r
6  ankle_x_r      7  ankle_y_r      8  ankle_z_r
9  hip_x_l       10  hip_y_l       11  hip_z_l
12 knee_x_l      13  knee_y_l      14  knee_z_l
15 ankle_x_l     16  ankle_y_l     17  ankle_z_l
18 back_x        19  back_y        20  back_z
```

其中 `[3, 5, 12, 14]` 是锁零的膝 X/Z 坐标。去掉它们得到 17 个活动列，**训练时仍应按 USD 实际 joint 名重排**，不能假定 USD 与这个数组顺序一致。85 个 body 包含固定的网格子 body；它们不是 85 个活动关节。旧模型的 31 列关节和 20 个 body 顺序不能直接用于这里。

用 MuJoCo 恢复姿态时：

```python
motion = np.load(
    "data/outputs/skeleton17_collision/walk/walk3_subject3_120hz.npz"
)
data.qpos[:3] = motion["root_pos"][i]
data.qpos[3:7] = motion["root_rot"][i, [3, 0, 1, 2]]  # xyzw -> wxyz
data.qpos[7:] = motion["dof_pos"][i]
mujoco.mj_forward(model, data)
```

## 检查范围

详细结果在 `walk/` 中的 `walk3_subject3_120hz_validation.json`；`preservation_report.json` 核对原文件未改动。重采样、膝锁以及新避碰回退的回归测试可运行：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest skeleton_retarget/tests -q
```

动作是运动学参考，尚未做动力学跟踪、平衡或 Isaac Sim/PhysX 测试。快速摆腿、为避碰产生的跟踪误差、脚底滑动和短暂离地均应结合报告/视频评估；这里的速度上限也不是执行器能力证明。此次没有修改 USD 或训练环境。
