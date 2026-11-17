# 31 关节无肌肉骨架：模型与标定

本文说明旧 12 条行走动作使用的骨架。当前 collision 训练模型的流程见 [工作区 README](../README.md)；旧动作操作入口见 [31 关节流程](skeleton31.md)。

## 模型来源和改动

原始来源为 `/home/hy/workspace/humanoid-exo_model/mjcf/scene_walking.xml` 及其包含的 `human.xml`。制备脚本只读源文件，输出到 `assets/skeleton31/`；官方 GMR 源码和外部 MJCF 保持原样。

- 移除 72 条下肢肌肉 tendon、270 个肌肉专用 site；其中 248 个来自活动肌肉路径，22 个为已注释路径的遗留附着点。
- 保留骨骼 body、惯性、内部关节、网格、碰撞、足底与地面 contact pair，以及 8 个 `l_foot_box-P1..P4`、`r_foot_box-P1..P4` 标记。
- 把骨盆的 3 个平移和 3 个转动关节替换为一个 freejoint，启用源文件已有的内部关节限位。
- 保留双侧距下关节、跖趾关节、腕屈伸、腕偏转的 8 个 joint equality 锁；移除辅助 mocap body 与原先不活动的骨盆 weld。

最终 `nq=38`、`nv=37`，含 31 个内部 hinge、8 个 equality、8 个 site、0 个 tendon、0 个 actuator。8 个锁需要显式交给 IK 求解器，命令中使用 `--use_equality_constraints`；单独保留 XML equality 不会让普通运动学 IK 自动遵守它。

`assets/skeleton31/base/` 是未对具体 BVH 标定的基础副本。`assets/skeleton31/walks/<动作名>/` 与 `skeleton_retarget/retargeting/ik_configs/skeleton31/walks/<动作名>.json` 是正式动作对应的独立标定文件。它们的物理骨架一致，但人体比例、参考旋转与 keyframe 不完全相同，不应直接去重合并。

项目副本已把源模型使用的网格复制到 `assets/skeleton31/Geometry/`，各动作 XML 改为相对路径，因此回放和重定向不再依赖外部 Geometry。`manifest.json` 仍记录原始 XML 和标定来源，便于追溯。

## LAFAN1 标定原理

LAFAN1 的腿、脊柱、手臂 offset 大多沿局部 +X，BVH 不单独编码 bind pose 预旋转，全零旋转不是站立 T pose。共享 [标定代码](../skeleton_retarget/retargeting/calibration.py) 根据实际层级长度构造静态解剖学参考：世界 +X 向前、+Y 向左、+Z 向上；髋绕局部 Y 转 180° 使腿向下，左右肩分别绕 Y 转 ∓90° 再绕 X 转 180° 使手臂展开，脚绕 Z 转 90° 使脚趾向前。忽略 BVH 根 OFFSET 的任意绝对平移，并把参考足部放在 z=0。

标定据此计算骨段缩放与目标 body 的局部位置、旋转偏移，不把动作首帧当成 T pose。实际行走旋转全部来自 BVH。IK 使用模型内的 `retarget_reference` keyframe 初始化，再对齐首帧根位置和朝向并预热求解。

每条 `reference.json` 保存参考构造、关节顺序、参考 qpos（根旋转 `wxyz`）、映射 body 位姿和源人体参数。历史键名 `source_bind_pose` 的实际语义由 `source_reference_metadata.method=canonical_lafan1_anatomical_T_pose` 定义。

初始数值标定证据集中保存在 [skeleton31_initial_calibration.json](../reports/skeleton31_initial_calibration.json)：静态参考的位置残差约 `2e-17 m`、旋转残差约 `1.4e-16 rad`。肩部在 T pose 有轴重合，约 `1e-4 rad` 的等价关节变化不影响 body 位姿。

## 重新准备模型

已有动作优先使用保存的标定模型和配置。外部 `scene_walking.xml` 或 `human.xml` 可能已为训练修改，复现旧骨架时推荐明确指定保存的 31 关节模板：

```bash
uv run python -m skeleton_retarget.scripts.prepare_skeleton31 \
  --template_xml assets/skeleton31/base/human_skeleton.xml \
  --bvh_file data/raw/lafan1/walk1_subject1.bvh \
  --format lafan1 \
  --output_dir assets/skeleton31/experiments/walk1_subject1 \
  --config_output skeleton_retarget/retargeting/ik_configs/skeleton31/experiments/walk1_subject1.json
```

模板方式保留物理结构，重新构造该 BVH 的参考 keyframe 与 IK 标定。若不指定模板，脚本尝试从 `--source_root` 的 `scene_walking.xml` 展开并去肌肉；该源文件缺失时回退到保存的 31 关节基础模板。源文件仍存在但拓扑已修改时不能依赖自动回退，应显式使用模板或走新模型工作流。

不传 `--bvh_file` 时，输出 `reference.json` 标为 `calibrated=false`，不生成 IK 配置。`--inspect_only` 仅编译检查、不写文件。`--format xsens` 只适用于项目原有的 3ds Max Xsens BVH 约定，不代表支持任意 BVH 骨架。

## 批量复现入口

从项目根目录运行，结果、模型、配置全部另存：

```bash
uv run python -m skeleton_retarget.scripts.bvh_to_skeleton31_dataset \
  --template_xml assets/skeleton31/base/human_skeleton.xml \
  --plans data/selections/lafan1_walks.json \
  --model_root assets/skeleton31/rebuild \
  --config_root skeleton_retarget/retargeting/ik_configs/skeleton31/rebuild \
  --output_dir data/outputs/skeleton31_rebuild
```

批量流程逐条执行标定、原生帧率 IK、120 Hz 重采样、脚底对齐、FK、导出及验证，运行日志写入新结果目录的 `logs/`。两套模型默认共用 `data/selections/lafan1_walks.json`；输入 BVH 的 SHA256 必须与选段记录匹配。`--plans` 保持支持一次传入一个或多个独立选段文件。

批量制备也遵循上面的源模型与模板策略；示例显式指定保存的模板，避免外部源模型改动影响复现。`--output_dir` 只改变动作结果目录，要隔离模型和配置须同时指定 `--model_root`、`--config_root`。需要只复现动作、保持原标定文件时，使用 [单条重导出命令](skeleton31.md#单条重新导出)。
