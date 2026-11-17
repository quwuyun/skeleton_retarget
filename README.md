# skeleton_retarget

基于 GMR 的 LAFAN1 人体动作重定向工作区，包含两套无肌肉 MuJoCo 骨架：`skeleton17_collision`（17 个活动自由度，另有 4 个锁零膝坐标）和 `skeleton31`。项目保留上游 GMR 为 `third_party` 依赖；本地脚本负责模型适配、选段、批处理、验证和回放。以下命令均从项目根目录运行。

## 环境和数据

需要 Git、[uv](https://docs.astral.sh/uv/) 和 Python 3.10–3.13（建议使用项目锁定环境中的 3.13）。先 clone 与项目锁文件匹配的 GMR：

```bash
mkdir -p third_party
git clone https://github.com/YanjieZe/GMR.git third_party/GMR
git -C third_party/GMR checkout --detach 25f56db79ac411258b377b6d4ccba0c8ff3372ac
uv python install 3.13
uv sync --group dev
uv run python -c "import mujoco, mink, general_motion_retargeting, skeleton_retarget; print('environment ok')"
```

从 [Ubisoft LAFAN1 仓库](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)下载 `lafan1/lafan1.zip`，把 BVH 解压到 `data/raw/lafan1/`：

```bash
mkdir -p data/raw/lafan1
unzip -jn ~/Downloads/lafan1.zip '*.bvh' -d data/raw/lafan1
```

## 重定向流程

先用命令帮助确认可选参数：

```bash
uv run python -m skeleton_retarget --help
uv run python -m skeleton_retarget skeleton17 --help
uv run python -m skeleton_retarget skeleton17 retarget --help
uv run python -m skeleton_retarget skeleton31 batch --help
```

### 17 自由度模型

普通 12 条 walk 依照已审核的帧窗口批量重定向到 120 Hz。默认输出目录为 `data/outputs/skeleton17_collision/walk/`，NPZ、PKL、JSON 和关联报告都写入该目录。改用其他选段或输出位置时再指定 `--plans`、`--output_dir`：

```bash
uv run python -m skeleton_retarget skeleton17 batch
```

单条处理可复用项目模型和动作 IK 配置；`start_frame` 是零基帧，`max_frames` 是源帧数。输出前缀应使用新路径，避免覆盖已有结果：

```bash
uv run python -m skeleton_retarget skeleton17 retarget \
  --bvh_file data/raw/lafan1/walk3_subject3.bvh \
  --robot_xml assets/skeleton17_collision/human_skeleton.xml \
  --ik_config skeleton_retarget/retargeting/ik_configs/skeleton17_collision/walk3_subject3.json \
  --start_frame 2575 --max_frames 86 --output_fps 120 \
  --output data/outputs/skeleton17_collision/walk/manual/walk3_subject3_120hz
```

生成五类动作（静止、前进、后退、左转、右转），或按另一份已审核清单生成长片段：

```bash
uv run python -m skeleton_retarget skeleton17 behaviors

uv run python -m skeleton_retarget skeleton17 behaviors \
  --selection data/selections/lafan1_long_segments_v02.json \
  --output_dir data/outputs/skeleton17_collision/long_segments_x_v02
```

对长片段生成左右镜像副本。默认读取 `long_segments_x_v02/`，并把结果保存到平级 `long_mirror/`；原动作保留。镜像约定和检查范围见[镜像说明](data/outputs/skeleton17_collision/long_mirror/README.md)：

```bash
uv run python -m skeleton_retarget skeleton17 mirror
```

`normalize-heading` 可将已有六字段 NPZ 的首帧朝向旋到世界 +X，并默认把首帧根 XY 移到原点；输入保留，输出使用新前缀：

```bash
uv run python -m skeleton_retarget normalize-heading \
  --motion data/outputs/skeleton17_collision/walk/walk2_subject4_120hz.npz \
  --output data/outputs/skeleton17_collision/walk/walk2_subject4_x_120hz
```

### 31 关节模型

批处理使用 31 关节模板，为每条源 BVH 保存单独的标定模型和 IK 配置：

```bash
uv run python -m skeleton_retarget skeleton31 batch \
  --plans data/selections/lafan1_walks.json \
  --template_xml assets/skeleton31/base/human_skeleton.xml \
  --model_root assets/skeleton31/rebuild01 \
  --config_root skeleton_retarget/retargeting/ik_configs/skeleton31/rebuild01 \
  --output_dir data/outputs/skeleton31/rebuild01
```

单条 LAFAN1 BVH 使用 `skeleton31 retarget`；若模型包含 equality 锁，加入 `--use_equality_constraints`。单条输入可指定帧窗口、输出频率和高度/XY 对齐，详见该命令的 `--help`。`skeleton31 mimickit` 是另一个入口，仅用于本地已有 MimicKit 数据，不是 LAFAN1 流程。

## 验证和回放

导出后先用对应模型重新计算 FK 并运行验证，再做 MuJoCo 可视化：

```bash
uv run python -m skeleton_retarget skeleton17 validate \
  --robot_xml assets/skeleton17_collision/human_skeleton.xml \
  --motion data/outputs/skeleton17_collision/walk/walk3_subject3_120hz.npz \
  --ik_config skeleton_retarget/retargeting/ik_configs/skeleton17_collision/walk3_subject3.json \
  --source_reference data/outputs/skeleton17_collision/walk/references/walk3_subject3_source.npz \
  --expected_fps 120 --output /tmp/walk3_validation.json

uv run python -m skeleton_retarget play \
  --robot_xml assets/skeleton17_collision/human_skeleton.xml \
  --motion data/outputs/skeleton17_collision/walk/walk3_subject3_120hz.npz

uv run python -m skeleton_retarget preview \
  --robot_xml assets/skeleton17_collision/human_skeleton.xml \
  --motion data/outputs/skeleton17_collision/walk/walk3_subject3_120hz.npz \
  --output /tmp/walk3_preview
```

`play` 需要图形桌面。`preview` 生成关键帧 PNG 和 MP4，添加 `--sheet_only` 可只生成 PNG。两者只回放保存的运动学姿态，不做动力学、接触或控制验证。NPZ 是 GMR 六字段格式；17/31 模型的关节排列不同，不可混用模型或 IK 配置。

## 输出目录

| 路径 | 内容 |
| --- | --- |
| `data/outputs/skeleton17_collision/walk/` | 普通 walk 的 NPZ/PKL/JSON、源参考、预览、批次报告及汇总 |
| `data/outputs/skeleton17_collision/five_behaviors_x/` | 初始面朝 +X 的五类行为和预览 |
| `data/outputs/skeleton17_collision/long_segments_x_v02/` | 长前进、起步、暂停、停止等片段 |
| `data/outputs/skeleton17_collision/long_mirror/` | 上述长片段的左右镜像副本 |
| `data/outputs/skeleton31/` | 31 关节模型的导出结果 |

## 核心文件

| 路径 | 职责 |
| --- | --- |
| `pyproject.toml`, `uv.lock` | 项目依赖、GMR 本地包引用和可复现锁定 |
| `third_party/GMR/` | 独立 clone 的上游重定向库，按指定提交使用 |
| `skeleton_retarget/__main__.py` | `skeleton17`、`skeleton31`、回放、预览和朝向转换命令路由 |
| `skeleton_retarget/scripts/` | 单条/批量重定向、模型准备、镜像、朝向变换、验证及预览脚本 |
| `skeleton_retarget/retargeting/` | GMR 适配、标定、IK、数据读写和验证实现 |
| `assets/` | 项目自用 MJCF、网格及模型清单 |
| `skeleton_retarget/retargeting/ik_configs/` | 按模型和源动作保存的 IK 配置及参考标定 |
| `data/selections/` | 已审核选段和行为清单，含源文件哈希、帧窗口 |
| `docs/`, `reports/` | 操作说明、模型记录和数据质量检查报告 |

检查项与详细模型说明见 [17 自由度流程](docs/skeleton17.md)、[31 关节流程](docs/skeleton31.md)、[五类行为](docs/five_behaviors.md)、[IK 配置](docs/ik_configuration.md)。CLI 完整参数均可通过对应命令追加 `--help` 查询。运行测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest skeleton_retarget/tests -q
uv run ruff check skeleton_retarget
```
