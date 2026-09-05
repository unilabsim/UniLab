# MuJoCo 后端

MuJoCo 是已提交 owner 配置中的默认后端路径。其 Python 依赖为官方
`mujoco` 包（`~=3.11.0`，默认版本由已提交的 `uv.lock` 精确钉住）加 `mujoco-uni-runtime`（见 `pyproject.toml`），适配层位于
`unisim.backend.mujoco` 下。

## 何时使用

- 你想要 PPO、APPO、off-policy SAC/TD3 或 FlashSAC 的默认训练路线。
- task owner 仅以 `src/unilab/conf/.../<task>/mujoco.yaml` 形式存在。
- 你需要 MuJoCo 专有工具，例如 `scripts/play_viser.py`，或从 MuJoCo XML/MJB
  模型导出场景。

## 命令

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task go1_joystick_flat --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

回放模式由 `unisim.backend.base` 中的 backend contract 解析。
MuJoCo 在 `unisim.backend.mujoco.backend` 中声明对物理状态回放的支持；
`auto` 回放会录制视频，而不是打开 Motrix 原生交互式渲染器。

## 切换 MuJoCo 版本

pyproject 约束 `mujoco~=3.11.0`；默认版本由已提交的 `uv.lock` 精确钉住，
uv 的 prefer-locked 语义保证普通 relock 不会漂移。默认安装路径使用
`mujoco-uni-runtime` 的预编译 wheel（绑定 `mujoco==3.11.0`），不需要编译器。
支持窗口为 `>=3.5,<3.12`；切换到 wheel 绑定以外的任何版本都必须走源码重建
路径。`mujoco-uni-runtime` 的原生扩展会记录编译时的 `mujoco` 版本，且拒绝在
其它版本下加载，因此切换版本 = 重钉 `mujoco` + 从源码重编扩展（需要 C++17
工具链和 Python 开发头文件；缺少编译器时，该目标的 `check-cxx-toolchain`
预检会立即失败并打印各平台的安装命令）：

```bash
make mujoco MJ=3.10.0
```

该目标依次执行 `uv lock --upgrade-package mujoco==3.10.0`、清除 uv 对
`mujoco-uni-runtime` 的构建缓存（缓存无法感知扩展对 mujoco 版本的依赖）、
并强制在本环境内重新编译同步（`--no-binary-package mujoco-uni-runtime
--reinstall-package mujoco-uni-runtime`）。不用 Makefile 时的等价命令：

```bash
uv lock --upgrade-package mujoco==3.10.0
uv cache clean mujoco-uni-runtime
uv sync --extra mujoco --extra motrix --no-binary-package mujoco-uni-runtime --reinstall-package mujoco-uni-runtime
```

如果省略清缓存或强制重装，uv 可能复用按旧版本 mujoco 编译的扩展，
import 时会以版本 watchdog 错误失败（fail-closed，不会静默出错行为）。
切回默认预编译 wheel 路径：运行 `make mujoco MJ=3.11.0`，或
`git restore -- uv.lock` 后重新执行 `uv sync --extra mujoco`。
