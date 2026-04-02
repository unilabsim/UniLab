# G1 SAC 训练性能修复日志

**基准版本：** `d5d9ae4`（2026-03-31）  
**涉及文件：** `src/unilab/envs/locomotion/g1/joystick_sac.py` · `conf/offpolicy/reward/g1_sac_mujoco.yaml` · `conf/offpolicy/task/g1_sac.yaml`

---

## 背景

`d5d9ae4` 修复了对称增强模块的观测维度错误，但 G1 SAC（MuJoCo 后端）训练性能仍严重退化，机器人无法正常行走。根本原因是 SAC 相关的观测缩放、奖励权重及超参数均未对齐参考实现，多个配置问题叠加导致训练发散。

---

## 改动说明

### 1. `src/unilab/envs/locomotion/g1/joystick_sac.py`

#### 新增 `_compute_obs` 覆盖，加入观测缩放

```python
# 修改前（继承自 PPO，无缩放）
actor = [gyro, -gravity, diff, dof_vel, last_actions, command, gait_phase]
privileged = linvel

# 修改后
actor = [gyro * 0.25, -gravity, diff, dof_vel * 0.05, last_actions, command, gait_phase]
privileged = linvel * 2.0
```

**原因：** PPO 父类的观测使用原始量纲，各维度数值范围差异大（gyro 可达 ±10 rad/s，dof_vel 可达 ±30 rad/s），输入方差不一致会减慢神经网络收敛。缩放后 gyro 和 dof_vel 的典型幅值压缩到 ±1 附近，linvel 放大 2 倍使速度误差信号更显著，对齐参考实现。

#### 修正 `PenaltyCurriculum` 参数

| 参数 | 修改前 | 修改后 |
|---|---|---|
| `initial_scale` / `min_scale` | 0.1 | 0.5 |
| `level_down_threshold` | 50 步 | 150 步 |
| `level_up_threshold` | 500 步 | 750 步 |
| `degree` | 0.002 | 0.001 |

**原因：** `initial_scale=0.1` 意味着训练初期惩罚项权重只有最终值的 10%，策略在几乎无约束的环境下习得的行为（如乱摆四肢）在惩罚权重提升后被迫大幅修正，造成训练不稳定。提升至 0.5 使惩罚信号从一开始就有足够强度指导策略方向。阈值放宽（150/750）和 `degree` 减小使课程推进更平缓，避免难度跳变。

---

### 2. `conf/offpolicy/reward/g1_sac_mujoco.yaml`

```yaml
# 修改前
penalty_feet_ori: -25.0
# 修改后
penalty_feet_ori: -5.0
```

**原因：** 脚部姿态惩罚权重（25.0）为速度跟踪奖励（2.0）的 12.5 倍。训练早期机器人尚未学会行走，脚部姿态不可避免地偏离理想值，导致该惩罚项持续主导总奖励，速度跟踪信号被完全淹没。降至 5.0 后各奖励项量级趋于平衡。

---

### 3. `conf/offpolicy/task/g1_sac.yaml`

```yaml
# 修改前
alpha_init: 0.01
# 修改后
alpha_init: 0.001
```

**原因：** `alpha` 是 SAC 的熵正则化系数，控制策略的探索程度。`target_entropy_ratio=0.0` 表示目标熵为零（策略趋于确定性），但 `alpha_init=0.01` 的初始值过高，早期训练阶段熵损失项过强，策略倾向于保持随机性而非学习有效行为。降低 10 倍后初始探索强度更温和，与零目标熵的设定协调一致。

---

## 架构说明

本次修复保持了 `d5d9ae4` 引入的非对称 Actor-Critic 架构不变：

```
G1JoystickPPO (joystick.py)
  obs_groups_spec: {"obs": 98, "privileged": 3}
  _compute_obs:    actor = [gyro, -gravity, diff, dof_vel, actions, cmd, phase]
                   privileged = linvel

G1WalkTaskMjSAC (joystick_sac.py) 继承 G1JoystickPPO，覆盖 _compute_obs：
  actor obs:      98 维，含观测缩放，不含 linvel（可直接部署至真机）
  privileged obs:  3 维，linvel×2.0（仅 critic 训练时使用）
```

对称增强模块（`symmetry.py`）按 98 维 actor obs 构建，不含 linvel，行为与 `d5d9ae4` 一致。
