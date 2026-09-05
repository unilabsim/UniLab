# MuJoCo Backend

MuJoCo is the default backend path in the committed owner configs. The Python
dependencies are the official `mujoco` package (`~=3.11.0`, with the exact
default version pinned by the committed `uv.lock`) plus
`mujoco-uni-runtime` in `pyproject.toml`, and the adapter lives
under `unisim.backend.mujoco`.

## When To Use It

- You want the default training route for PPO, APPO, off-policy SAC/TD3, or
  FlashSAC.
- The task owner exists only as `src/unilab/conf/.../<task>/mujoco.yaml`.
- You need MuJoCo-specific tooling such as `scripts/play_viser.py` or scene
  export from a MuJoCo XML/MJB model.

## Commands

```bash
uv run train --algo ppo --task go2_joystick_flat --sim mujoco
uv run train --algo appo --task go1_joystick_flat --sim mujoco training.no_play=true
uv run train --algo sac --task g1_walk_flat --sim mujoco
```

Playback mode is resolved by the backend contract in
`unisim.backend.base`. MuJoCo reports physics-state playback support
in `unisim.backend.mujoco.backend`; `auto` playback records video
rather than opening the Motrix native interactive renderer.

## Switching MuJoCo Versions

pyproject constrains `mujoco~=3.11.0`; the committed `uv.lock` pins the exact
default version, and uv's prefer-locked semantics keep ordinary relocks from
drifting. The default install path uses the prebuilt `mujoco-uni-runtime`
wheel, which binds `mujoco==3.11.0` — no compiler is needed. The support
window is `>=3.5,<3.12`; switching to any version other than the wheel's
binding always takes the source-rebuild path. The
`mujoco-uni-runtime` native extension records its build-time `mujoco`
version and refuses to load against any other version,
so switching versions means installing the requested `mujoco` and rebuilding
the extension from source (a C++17 toolchain and Python development headers
are required; the target's `check-cxx-toolchain` preflight fails fast with
per-platform install commands when no compiler is found):

```bash
make mujoco MJ=3.10.0
```

Because the `~=` bound can never be re-locked to another line, the target
operates on the environment directly (`uv pip`, without touching `uv.lock`):
it installs `mujoco==3.10.0` plus the runtime's build requirements
(`pybind11`, `wheel`, `setuptools`), clears uv's build
cache for `mujoco-uni-runtime` (the cache cannot see that the extension
depends on the mujoco version), and forces an in-env sdist rebuild of the
runtime.
Without the Makefile shortcut, the equivalent is:

```bash
uv pip install "mujoco==3.10.0" pybind11 wheel setuptools
uv cache clean mujoco-uni-runtime
uv pip install --force-reinstall --no-deps --no-build-isolation \
  --no-binary mujoco-uni-runtime "mujoco-uni-runtime==0.5.0"
```

Skipping the cache clean or the forced reinstall lets uv reuse a cached
extension built against the previous mujoco version, which then fails to
import with a version-watchdog error (fail-closed, never a silent behavior
change). The override is environment-local: to return to the default
prebuilt-wheel path, run `uv sync --extra mujoco --reinstall-package
mujoco-uni-runtime`. The `--reinstall-package` flag is required — a plain
`uv sync` restores mujoco but keeps the locally rebuilt extension, which
then fails to load against the reverted mujoco.
