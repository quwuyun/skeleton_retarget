# 2026-11-12 新训练骨架与已有 GMR 动作兼容性分析

结论：现有 31 列 NPZ 不能直接裁剪或改名作为新 17 自由度训练模型的参考。已有 BVH、选段计划、120 Hz 时间设置可保留，但需要针对新目标模型重新做受约束 IK，或者从旧动作 FK 目标做一次目标模型 IK 投影；这两种方式都需要生成新模型的数据。当前只做分析，没有重新重定向或修改 MJCF、USD、已有 NPZ。

## 1. 检查对象

- 新 MJCF：`/home/hy/workspace/humanoid-exo_model/mjcf/walk_skeleton_body_usd_collision.xml`。
- 新 USD：`/home/hy/workspace/humanoid-exo_model/usd/walk_skeleton_body_usd_collision/walk_skeleton_body_usd_collision.usd`。
- 旧目标模型：每条动作元数据所指的 `skeleton_retarget/models/lafan1_walks/<动作名>/human_skeleton.xml`。
- 旧动作：`data/human_skeleton/walks_120hz/*_120hz.npz`，12 条、3256 帧、120 Hz；读入时核验了对应旧 XML 的 SHA256。
- 数值工具：GMR `.venv` 中 MuJoCo + SciPy；仅调用 FK，不做 IK 或物理仿真。USD 使用 Isaac 自带 pxr 静态检查，没有启动 PhysX。

新 XML SHA256：`54eb2a8dadc3a1208007ba4352674986b9452569a54fabf4adc29986cfc82d0a`。

## 2. 哪些修改影响姿态

| 修改 | 核对结果 | 对已有数据的影响 |
| --- | --- | --- |
| 骨长、骨架20个body层级 | 换基后各body偏移误差不超过1.11e-16 m | 没有缩放身体，无需重新选数据 |
| 质量与惯量 | 合并固定mesh子body后总质量84.872134479472 kg保持，惯量换基误差不超过4.44e-15 | 不要求因质量变化重做姿态；仍需控制验证 |
| 固定mesh子body | 新body含world共86；旧21 | 不能按新body枚举索引使用旧20项，按名称选择 |
| 根与所有body框架 | 旧局部X前/Y上/Z右，新X前/Y左/Z上 | 根四元数与local_body_pos语义需要转换 |
| 髋与back | 旋转顺序由旧Y-X-Z换成新X-Y-Z，部分轴反号 | 不能简单换列，必须有限角重参数化或IK |
| 膝 | 旧斜轴偏离新纯Y约4.05°；新X/Z轴锁0 | 无法用单标量复制保持全部骨段姿态 |
| 踝 | 旧单斜轴变新XYZ三个轴 | 可以矩阵分解表示旧局部旋转，不能只拷到ankle_y |
| 肩、肘、前臂、腕 | 新模型固定；back仍可动 | 旧手臂角可不用于训练，但旧上肢body轨迹必须弃用或按新FK重算 |
| 旧subtalar/mtp和腕锁 | 旧8列均接近0 | 这部分可以删除，不是本次不兼容的主因 |
| 关节限位 | 新back分别±25/45/35°，髋也变化 | 部分旧动作超出新范围，单纯转换不能解决 |
| 脚底碰撞 | 小薄足盒改成talus上的整足大盒 | 旧root_z和地面对齐结果不能沿用 |

当前 MuJoCo 为 nq=28、nv=27、21 hinge、21 motor、4条膝X/Z equality；17是独立内部运动轴数。USD修复版为17个未锁转动轴。新MJCF向量不能直接当成17列，MuJoCo需要保留四个锁轴的零值；Isaac侧按运行时joint_names映射。

令 C=Rx(+90°)，新根朝向应为 R_new=R_old C^-1；根位置可先沿用，后续重新处理脚底高度。该式是根局部框架变更，不是把世界轨迹旋转90°；单纯xyzw/wxyz重排也不能完成它。

## 3. 全部3256帧的FK迁移诊断

比较对象是迁移后新模型与旧目标模型的同名calcn原点世界位置，并非相对原BVH的跟踪误差。所有方案均已正确转换根朝向。表中是每条动作、两脚、全部帧的最大位置偏差。

- A：仅重排/取反髋与腰，膝角复制，踝仅填Y，其余新轴填0。
- B：精确转换髋/腰/踝的有限旋转，膝用最接近旧斜轴旋转的纯Y旋转，膝X/Z保持0。
- C：在B基础上用踝XYZ补偿膝造成的足部世界朝向误差，保留膝锁；脚位置仍会变化。
- 对照：允许新膝X/Z也运动时，可以按矩阵分解复现旧下肢与躯干FK，位置误差最高1.44e-15 m。这仅用于验证换基和几何一致，违反新膝锁，不能作为训练动作。

| 动作 | 帧数 | A脚位置最大偏差/cm | B脚位置最大偏差/cm | C脚位置最大偏差/cm | B新脚盒最低Z/cm |
| --- | --- | --- | --- | --- | --- |
| walk1_subject1 | 396 | 10.94 | 3.86 | 3.01 | -2.98 |
| walk1_subject2 | 120 | 8.58 | 4.70 | 3.75 | -2.78 |
| walk1_subject5 | 120 | 10.58 | 3.49 | 2.76 | -1.41 |
| walk2_subject1 | 256 | 9.36 | 4.83 | 3.88 | -2.32 |
| walk2_subject3 | 388 | 6.09 | 4.21 | 3.35 | -3.13 |
| walk2_subject4 | 388 | 3.90 | 4.05 | 3.18 | -2.96 |
| walk3_subject1 | 340 | 10.98 | 4.37 | 3.46 | -2.37 |
| walk3_subject2 | 284 | 11.05 | 4.60 | 3.66 | -2.29 |
| walk3_subject3 | 340 | 9.37 | 3.48 | 2.71 | -3.74 |
| walk3_subject4 | 244 | 4.62 | 2.82 | 2.20 | -2.40 |
| walk3_subject5 | 140 | 12.96 | 5.37 | 4.35 | -3.04 |
| walk4_subject1 | 240 | 17.63 | 4.73 | 3.77 | -1.63 |

B方案另有约4.64–6.58°的每片段最大足部朝向变化，来源是膝斜轴无法由新纯Y轴表达。C可消除该足部朝向差，但无法直接消除脚位置变化。上述方案均未做全腿IK，数字不是最佳IK误差的下界；它们用于证明简单数据裁剪或局部角转换尚未得到一致参考。

慢走walk3_subject3：A最大脚位差9.37 cm；B为3.48 cm；C为2.71 cm。B下新脚盒最低Z=-3.74 cm；该340帧每帧至少一只脚盒低于地面。即使对照方案完全保留旧下肢几何位姿，新脚盒也会穿地，说明不只是膝轴误差，碰撞盒本身也改变了离地条件。

全部片段B方案下，每帧至少一个脚盒角点低于地面。若仅靠根Z平移使最低角点达到1 mm，慢走样本需每帧额外抬高约0.82–3.84 cm；但这会改变根轨迹，不能代替整腿IK和接触检查。

## 4. 新限位不接受的旧动作

以下为B/C中未裁剪的精确髋/腰旋转转换结果；两方案髋和腰相同。不得把直接clamp之后的动作视为原动作完全保留。

| 动作 | 超限关节 | 帧数 | 最大超限角度/度 |
| --- | --- | --- | --- |
| walk1_subject5 | hip_x_r | 7 | 4.769 |
| walk1_subject5 | hip_z_r | 7 | 1.850 |
| walk1_subject5 | hip_z_l | 24 | 11.579 |
| walk2_subject3 | hip_x_r | 2 | 0.296 |
| walk3_subject5 | hip_x_r | 1 | 0.083 |
| walk3_subject5 | back_y | 49 | 6.743 |
| walk4_subject1 | hip_z_l | 44 | 11.437 |
| walk4_subject1 | back_y | 240 | 24.551 |

特别是walk4_subject1，back_y约61.52–69.55°，240帧全部超过新±45°；walk3_subject5的back_y最高约51.74°，49/140帧超限。walk3_subject3本次角转换未出现硬限位超限，但仍有膝轴和脚底差异。

## 5. USD静态检查

- `usd/`下主资产是修复版：Z-up、米，根参考为identity，17个未锁轴，四膝X/Z显式锁且无活动drive。85个body参考框架与XML最大矩阵元素差约8.38e-8，关节闭合误差约5.22e-8。
- 当前这份主USD把四个大腿/小腿collisions子树设为active=False，实际只保留两脚盒碰撞；源XML仍有六个碰撞体。这影响训练接触设置，应以实际加载USD为准，本轮未改动。
- `mjcf/walk_skeleton_body_usd_collision/`是另一个界面导出副本，没有经过同等root/锁/drive修复，不要混用；不能把修复版的17DOF结论直接套给它。
- 多轴D6有限混合角与MuJoCo连续XYZ hinge的运行时对应，本轮没有在PhysX中验证；参考生成后应至少抽查多轴非零静态帧，而非只看零位。
- 仅MJCF→USD转换本身不必导致重新重定向；本次要求新目标IK的原因是源训练MJCF运动学和约束已改变。

## 6. 训练接口的额外适配

只读核查当前isaaclab_exo_rl仓库：humanoid_amp默认配置仍使用Humanoid28、action_space=28；新17轴资产仅定义，尚未接入该环境。其NPZ loader要求dof_names/body_names、关节位置和速度、body世界位姿和线/角速度等训练schema，不能直接读取GMR六字段。PKL loader也明确拒绝GMR布局。

已有按名称选关节能力，因此“多余关节列可忽略”作为接口原则成立；前提是参考已经对应新模型的关节语义和FK。当前还存在Humanoid28旧关节名、torso作为reset根而新模型根为pelvis、wxyz与GMR磁盘xyzw的区别，这些都需随新资产接入调整。

## 7. 推荐后续工作范围

1. 保留12份原始BVH、最终选段计划、现有NPZ，以及120 Hz、speed=1、1–2周期设置，不必重新下载或从头选段。
2. 在独立目录复制适配新collision MJCF的准备器/IK配置：根参考identity，关节由新XML枚举，显式遵守四个膝锁，删除不可实现的手臂追踪任务。现有prepare_model.py硬编码31hinge和arm_add，不能直接换路径。
3. 优先针对walk3_subject3的小窗口重新求目标IK，可复用源BVH世界位姿；也可把旧NPZ按旧模型FK得到的目标作为新IK输入，但需接受继承旧跟踪误差。
4. 按新真实talus碰撞盒检查足底高度；原ground_align依赖旧8个site，新模型无site，需要独立适配。
5. 重新计算新模型body FK、body朝向和速度字段，按训练schema导出；按名称映射到USD实际17DOF，抽查多轴姿态FK，之后再扩展整批。
6. 不用纯截断角度来掩盖walk4_subject1等片段的限位问题；视新IK结果选择修正或替换源窗口。

## 8. 本地依据

- `skeleton_retarget/models/lafan1_walks/walk3_subject3/human_skeleton.xml`：旧髋膝踝约第14行起、腰及上肢约第83行起。
- `/home/hy/workspace/humanoid-exo_model/mjcf/walk_skeleton_body_usd_collision.xml`：根54/75行、右腿80行、左腿131行、back201行、四锁551行起。
- `/home/hy/workspace/humanoid-exo_model/mjcf/README_isaaclab_models.md`：模型、坐标、限位、导入与USD路径说明。
- `/home/hy/workspace/humanoid-exo_model/scripts/import_split_mjcf.py`：关节框架、D6轴锁、drive角度单位、root API修复。
- `/home/hy/workspace/humanoid-exo_model/scripts/isaaclab_humanoid_cfg.py`：17轴过滤、pelvis初始化及驱动配置。
- `/home/hy/workspace/isaaclab_exo_rl/source/isaaclab_exo_rl/isaaclab_exo_rl/tasks/direct/humanoid_amp/motions/motion_loader.py`：31行起为NPZ必需字段，229行附近为名称查列。
- 同目录`motion_loader_pkl.py`：104行附近拒绝GMR schema；`humanoid_amp_env.py`：345行附近reset与torso根假设。
- 本目录`fk_comparison.json`：逐片段、逐body误差、限位、碰撞角点高度、原文件哈希。
- 本目录`audit_fk.py`：只读数值分析方法。脚本原默认JSON输出到`/tmp/gmr_model_compatibility_20261112.json`。
