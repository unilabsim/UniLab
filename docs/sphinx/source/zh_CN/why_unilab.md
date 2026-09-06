# 为什么选择 UniLab？

机器人强化学习依赖一条既足够忠实、又足够高效的仿真闭环。如今这条闭环已经不是
“一种方案适用所有任务”：运动控制、接触丰富的操作、可变形物体和部署验证，可能
需要不同的物理 solver。团队的硬件也各不相同——NVIDIA CUDA 很重要，但并不是唯一
的工作站或 learner 目标。

多数 RL stack 把 simulator 作为 stack 中心。一旦更换 simulator、solver 或 worker 模型，
变化就会蔓延到任务代码、训练命令和实验基础设施。UniLab 填补的是另一个空白：让
机器人任务和 RL 工作流保持稳定，同时允许更换物理实现和学习硬件。

> 任务定义一次，使用适合工作的 solver 和硬件。

## 设计承诺

UniLab 围绕五项承诺构建：

1. **稳定的面向任务界面。** observation、action、reward、termination、command、
   event 和 curriculum 都是可复用的任务组件。更换后端不应要求复制一套任务生命周期。
2. **solver 无关的执行方式。** 物理实现可以驻留 GPU、运行在 CPU/native，或作为
   external worker；无需先重写成 CUDA simulator，就能参与同一套机器人 RL 工作流。
3. **与硬件解耦的学习过程。** 仿真和学习是分开的 runtime 关注点。learner 可以使用
   CUDA、ROCm、MPS 或 XPU，而 solver 使用适合自己的执行模型。除 Linux 外，Windows
   和 Apple Silicon macOS 也有文档化路径。
4. **先有证据，再谈一致性。** backend 名称本身不是支持承诺。[支持矩阵](5-reference/5-support_matrix)
   记录每个 backend/task 组合属于 registered、configured、tested、benchmarked 还是
   recommended。
5. **为 replay-based off-policy 训练提供专门的加速路径。** SAC 及相关方法可以复用
   历史经验，使仿真数据采集与 learner update 能够重叠，而不必在每次更新时汇合。UniLab
   的 FastSAC/FlashSAC 在代表性评估配置上报告了 **3–10 倍端到端训练效率提升**；硬件、
   任务和测量范围请参阅[论文](https://arxiv.org/abs/2605.30313)。这是 runtime 优化，
   不是新的 SAC objective。

## 为什么现在需要这样的边界

[NVIDIA 关于 Newton 与工业机器人](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/)
的技术博客描述了 Newton + Isaac Lab 的流程：替换 simulation backend 时，task 定义、
PPO loop、observation 和 reward 保持不变。这是行业中对“任务/物理边界”的一个实例，
也说明这不是 UniLab 的自定义偏好。

同一篇文章还展示了为什么需要这条边界：Newton 组合刚体与可变形 solver，并将 MuJoCo
Warp 与 VBD/MPM 耦合，用于接触丰富的任务。solver 选择越来越由任务保真度驱动。GPU
驻留路径在合适硬件上可以非常快——文章报告 MuJoCo Warp 在 RTX PRO 6000 Blackwell
上相对 MJX 达到 locomotion 252 倍、manipulation 475 倍加速——但这不意味着每个有用
的 solver 都必须先 CUDA 化，机器人训练才能使用。

UniLab 是位于 physics engine 演进之上的 RL 基础设施/runtime 层。当仓库具备对应配置和
验证证据时，它可以连接 MuJoCo、Motrix、Newton 或其他注册 adapter。UniLab 不声称
每个 solver 或 task 组合都已经达到生产就绪。

## 范围

UniLab 提供机器人 RL 所需的 task、environment、配置、adapter 和实验工作流。物理引擎
由 [`unisim-core`](https://github.com/unilabsim/unisim) 提供；算法和异步 runner 组件由
[`unilab-rl`](https://github.com/unilabsim/unilab_rl) 提供。这样的分工让机器人专属仓库
可以复用任务 recipe，而不必同时维护所有物理引擎和 learner 实现。

UniLab 不是新的通用 physics engine，不保证跨后端数值等价，也不承诺每个平台都支持每个
task。安装细节、后端限制和 contract 行为请使用对应的用户指南与开发者指南。

## 对比

选择哪种方案，取决于你希望什么保持稳定：

| 框架 | 主要优化目标 | 更适合 |
| --- | --- | --- |
| **UniLab** | 在异构 solver 和硬件之间保持稳定的 task/RL 工作流 | 需要比较、迁移或长期维护多种物理/runtime 路径的团队 |
| [mjlab](https://github.com/mujocolab/mjlab) | 轻量、可检查的 MuJoCo Warp 与 manager API | 想要最短 NVIDIA GPU 路径的 MuJoCo 用户；跨 simulator 可移植性是明确的非目标 |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | 完整的 manager-based 机器人 stack 与 Isaac 生态 | 需要 Isaac Sim、Omniverse 或其集成工具链的项目 |
| [Newton](https://github.com/newton-physics/newton) | 基于 GPU、支持多刚体/可变形 solver 和 OpenUSD 的物理平台 | 需要 Newton solver 组合、contact model 或可微分仿真的项目 |
| [MuJoCo Playground](https://playground.mujoco.org/) | 最少抽象和快速实验 | 一次性原型，以及希望贴近 simulator 编写环境的项目 |

当 simulator 是可能变化的工程决策时，UniLab 更合适。如果 MuJoCo 本身就是固定的仿真器
选择，mjlab 或 MuJoCo Playground 可能更直接。如果首要需求是 Newton 或 Isaac Lab 的
物理和生态，则应从它们开始；当存在经过验证的 adapter 与 task owner 时，UniLab 可以
位于这些选择之上。

## 从证据开始

- [运行第一次 demo](1-getting_started/1-quick_demo)
- [选择物理后端](2-user_guide/3-backends/0-index)
- [查看支持矩阵](5-reference/5-support_matrix)
- [学习 manager-based API](4-developer_guide/1-architecture/6-manager_based_api)
- [阅读 sim-to-sim 指南](3-deployment/2-sim_to_sim/1-backend_swap)
