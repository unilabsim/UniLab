# Why UniLab?

Robot reinforcement learning depends on a simulation loop that is both
faithful enough for the task and fast enough for iteration. That loop is no
longer one-size-fits-all: locomotion, contact-rich manipulation, deformables,
and deployment validation can favor different physics solvers. Teams also have
different hardware—NVIDIA CUDA is important, but it is not the only workstation
or learner target.

Most RL stacks make the simulator the center of the stack. A change of
simulator, solver, or worker model then leaks into task code, training commands,
and experiment infrastructure. UniLab fills a different gap: it keeps the
robot task and RL workflow stable while the physics implementation and learning
hardware can change.

> Define the task once. Use the solver and hardware that fit the job.

## Design philosophy

UniLab is built around five commitments.

1. **A stable task-facing surface.** Observations, actions, rewards,
   terminations, commands, events, and curricula are reusable task components.
   A backend change should not require a second copy of the task's lifecycle.
2. **Solver-neutral execution.** A physics implementation may be GPU-resident,
   CPU/native, or an external worker. It can participate in the same robot-RL
   workflow without first being rewritten as a CUDA simulator.
3. **Hardware-neutral learning.** Simulation and learning are separate runtime
   concerns. The learner can use CUDA, ROCm, MPS, or XPU while the solver uses
   the execution model appropriate to it. Windows and Apple Silicon macOS are
   documented targets alongside Linux.
4. **Evidence before parity claims.** A backend name is not a support promise.
   The [support matrix](5-reference/5-support_matrix) records whether a
   backend/task combination is registered, configured, tested, benchmarked, or
   recommended.

5. **Replay-based off-policy training gets its own acceleration path.** SAC and
   related methods reuse experience, allowing simulation data collection and
   learner updates to overlap instead of meeting at every update. UniLab's
   FastSAC/FlashSAC results report **3–10× end-to-end training-efficiency gains**
   on representative evaluated configurations; see the [paper](https://arxiv.org/abs/2605.30313)
   for the hardware, tasks, and measurement scope. This is a runtime
   optimization, not a new SAC objective.

## Why this boundary matters now

The [NVIDIA Technical Blog on Newton and industrial robotics](https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/)
describes a Newton + Isaac Lab workflow in which the task definition, PPO loop,
observations, and rewards stay the same while the simulation backend changes.
That is an industry example of the task/physics boundary UniLab makes explicit.

The same article shows why the boundary matters: Newton combines rigid-body and
deformable solvers and couples MuJoCo Warp with VBD/MPM for contact-rich tasks.
Solver choice is increasingly driven by task fidelity. GPU-resident paths can
be extremely fast on the right hardware—the article reports 252× locomotion
and 475× manipulation speedups for MuJoCo Warp versus MJX on an RTX PRO 6000
Blackwell system—but that does not make CUDA-porting every useful solver a
prerequisite for robot training.

UniLab is the RL infrastructure/runtime layer above that physics-engine evolution. It
can connect MuJoCo, Motrix, Newton, or another registered adapter when the
repository has the corresponding configuration and validation evidence. It
does not claim that every solver or task combination is already production
ready.

## Scope

UniLab provides the task, environment, configuration, adapter, and experiment
workflow for robot RL. Physics engines are supplied by
[`unisim-core`](https://github.com/unilabsim/unisim); algorithms and asynchronous
runner components are supplied by
[`unilab-rl`](https://github.com/unilabsim/unilab_rl). This separation lets a
robot-specific repository reuse task recipes without taking ownership of every
physics engine or learner implementation.

UniLab is not a new universal physics engine, a guarantee of cross-backend
numerical equivalence, or a promise that every platform supports every task.
For installation details, backend limitations, and contract-level behavior,
use the linked user and developer guides.

## Comparison

The right choice depends on what you want to keep stable.

| Framework | Primary optimization | Best fit |
| --- | --- | --- |
| **UniLab** | Stable task/RL workflow across heterogeneous solvers and hardware | Teams comparing, migrating, or operating more than one physics/runtime path |
| [mjlab](https://github.com/mujocolab/mjlab) | Lightweight, inspectable MuJoCo Warp with a manager API | MuJoCo users who want the shortest NVIDIA GPU path; cross-simulator portability is intentionally out of scope |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | Comprehensive manager-based robotics stack and Isaac ecosystem | Projects that need Isaac Sim, Omniverse, or its integrated tooling |
| [Newton](https://github.com/newton-physics/newton) | GPU physics platform with multiple rigid/deformable solvers and OpenUSD | Projects that need Newton's solver composition, contact models, or differentiable simulation |
| [MuJoCo Playground](https://playground.mujoco.org/) | Minimal abstractions and quick experiments | One-off prototypes and environments authored close to the simulator |

UniLab is a good fit when the simulator is an engineering decision that may
change. mjlab or MuJoCo Playground may be a better fit when MuJoCo itself is the
fixed simulator choice. Newton or Isaac Lab may be the better starting point when
their physics and ecosystem are the primary requirement. UniLab can sit above
those choices when a validated adapter and task owner are available.

## Start with evidence

- [Run the first demo](1-getting_started/1-quick_demo)
- [Choose a backend](2-user_guide/3-backends/0-index)
- [Read the support matrix](5-reference/5-support_matrix)
- [Learn the manager-based API](4-developer_guide/1-architecture/6-manager_based_api)
- [Follow the sim-to-sim guide](3-deployment/2-sim_to_sim/1-backend_swap)
