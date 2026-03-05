"""Off-policy collector for SAC and TD3.

Collects (obs, action, reward, next_obs, done) transitions using the current
actor policy.  Runs in a subprocess; writes to a SharedReplayBuffer.

Data flow:
    env.step(actions_np) → state (numpy-like) → np.asarray() → replay buffer (numpy)
    obs (numpy-like) → np.asarray → torch.from_numpy → actor → actions (torch) → numpy → env
"""

import torch
import numpy as np
import pkgutil
import importlib


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _layer_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat * weight + bias


class _NumpySACActor:
    def __init__(self, action_scale: np.ndarray, action_bias: np.ndarray, use_layer_norm: bool = True):
        self.use_layer_norm = bool(use_layer_norm)
        self.action_scale = action_scale.astype(np.float32, copy=True)
        self.action_bias = action_bias.astype(np.float32, copy=True)
        self.params: dict[str, np.ndarray] = {}

    @classmethod
    def from_state_dict(cls, state_dict: dict, use_layer_norm: bool = True) -> "_NumpySACActor":
        action_scale = state_dict["action_scale"].detach().cpu().numpy()
        action_bias = state_dict["action_bias"].detach().cpu().numpy()
        obj = cls(action_scale=action_scale, action_bias=action_bias, use_layer_norm=use_layer_norm)
        obj.update_from_state_dict(state_dict)
        return obj

    def update_from_state_dict(self, state_dict: dict) -> None:
        self.params = {
            name: tensor.detach().cpu().numpy().astype(np.float32, copy=True)
            for name, tensor in state_dict.items()
        }

    def _linear(self, x: np.ndarray, prefix: str) -> np.ndarray:
        w = self.params[f"{prefix}.weight"]
        b = self.params[f"{prefix}.bias"]
        return x @ w.T + b

    def _maybe_ln(self, x: np.ndarray, prefix: str) -> np.ndarray:
        if not self.use_layer_norm:
            return x
        w_name = f"{prefix}.weight"
        b_name = f"{prefix}.bias"
        if w_name not in self.params or b_name not in self.params:
            return x
        return _layer_norm_np(x, self.params[w_name], self.params[b_name])

    def explore(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        x = obs.astype(np.float32, copy=False)
        x = self._linear(x, "net.0")
        x = self._maybe_ln(x, "net.1")
        x = _silu_np(x)

        x = self._linear(x, "net.3")
        x = self._maybe_ln(x, "net.4")
        x = _silu_np(x)

        x = self._linear(x, "net.6")
        x = self._maybe_ln(x, "net.7")
        x = _silu_np(x)

        mean = self._linear(x, "fc_mu")
        mean = np.clip(mean, -10.0, 10.0)
        mean = np.nan_to_num(mean, nan=0.0)

        log_std = self._linear(x, "fc_logstd")
        log_std = np.tanh(log_std)
        log_std = -5.0 + 0.5 * (0.0 - (-5.0)) * (log_std + 1.0)
        log_std = np.nan_to_num(log_std, nan=-5.0)

        if deterministic:
            raw = mean
        else:
            std = np.exp(log_std)
            noise = np.random.normal(loc=0.0, scale=1.0, size=mean.shape).astype(np.float32)
            raw = mean + std * noise
        actions = np.tanh(raw) * self.action_scale + self.action_bias
        return actions.astype(np.float32, copy=False)


def ensure_registries():
    """Import all env modules so they are registered."""
    try:
        import unilab.envs.locomotion
        package = unilab.envs.locomotion
        if hasattr(package, "__path__"):
            for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
                try:
                    importlib.import_module(name)
                except Exception:
                    pass
    except ImportError:
        pass


def _build_actor(algo_type, obs_dim, action_dim, actor_hidden_dim, use_layer_norm, device):
    """Build the correct actor model based on algorithm type."""
    if algo_type == "sac":
        from unilab.algos.torch.fast_sac.learner import SACActor
        return SACActor(obs_dim=obs_dim, action_dim=action_dim,
                        hidden_dim=actor_hidden_dim, use_layer_norm=use_layer_norm,
                        device=device)
    elif algo_type == "td3":
        from unilab.algos.torch.fast_td3.learner import TD3Actor
        return TD3Actor(obs_dim=obs_dim, action_dim=action_dim,
                        hidden_dim=actor_hidden_dim, use_layer_norm=use_layer_norm,
                        device=device)
    else:
        raise ValueError(f"Unknown algo_type: {algo_type}")


def off_policy_collector_fn(
    stop_event,
    env_name: str,
    env_cfg_overrides: dict,
    num_envs: int,
    shm_buffer_name: str,
    buffer_capacity: int,
    obs_dim: int,
    action_dim: int,
    weight_sync_name: str,
    weight_param_shapes: dict,
    algo_type: str = "sac",
    actor_hidden_dim: int = 512,
    use_layer_norm: bool = True,
    collector_device: str = "cpu",
    exploration_noise: float = 0.1,
    warmup_steps: int = 5000,
    metrics_queue=None,
    buffer_lock=None,
    weight_sync_lock=None,
    collector_infer_backend: str = "torch",
    **kwargs,
):
    """Entry point for the off-policy collector subprocess."""
    import traceback
    try:
        _run_collector(
            stop_event=stop_event,
            env_name=env_name, env_cfg_overrides=env_cfg_overrides,
            num_envs=num_envs, shm_buffer_name=shm_buffer_name,
            buffer_capacity=buffer_capacity, obs_dim=obs_dim, action_dim=action_dim,
            weight_sync_name=weight_sync_name, weight_param_shapes=weight_param_shapes,
            algo_type=algo_type, actor_hidden_dim=actor_hidden_dim,
            use_layer_norm=use_layer_norm, collector_device=collector_device,
            exploration_noise=exploration_noise, warmup_steps=warmup_steps,
            metrics_queue=metrics_queue, buffer_lock=buffer_lock,
            weight_sync_lock=weight_sync_lock,
            collector_infer_backend=collector_infer_backend,
        )
    except Exception as e:
        traceback.print_exc()
        if metrics_queue is not None:
            try:
                metrics_queue.put_nowait({"error": str(e)})
            except Exception:
                pass


def _run_collector(
    stop_event,
    env_name, env_cfg_overrides, num_envs,
    shm_buffer_name, buffer_capacity, obs_dim, action_dim,
    weight_sync_name, weight_param_shapes,
    algo_type, actor_hidden_dim, use_layer_norm, collector_device,
    exploration_noise, warmup_steps, metrics_queue, buffer_lock,
    weight_sync_lock, collector_infer_backend,
):
    from unilab.ipc import SharedReplayBuffer, SharedWeightSync
    from unilab.envs import registry

    ensure_registries()

    # Connect to shared memory
    replay_buffer = SharedReplayBuffer(
        buffer_capacity, obs_dim, action_dim, create=False, shm_name=shm_buffer_name,
        lock=buffer_lock,
    )
    weight_sync = SharedWeightSync(
        weight_param_shapes, create=False, shm_name=weight_sync_name, lock=weight_sync_lock
    )

    # Create environment - use numpy backend for PyTorch algorithms
    env = registry.make(env_name, num_envs=num_envs, sim_backend="mujoco")

    infer_backend = str(collector_infer_backend).strip().lower()
    if infer_backend not in ("torch", "numpy"):
        infer_backend = "torch"

    # Build actor
    model_device = collector_device if infer_backend == "torch" else "cpu"
    actor = _build_actor(algo_type, obs_dim, action_dim, actor_hidden_dim, use_layer_norm, model_device)
    actor.eval()
    numpy_actor = None

    # Load initial weights
    sd = dict(actor.state_dict())
    weight_sync.read_weights_into(sd)
    actor.load_state_dict(sd)
    if infer_backend == "numpy" and algo_type == "sac":
        numpy_actor = _NumpySACActor.from_state_dict(sd, use_layer_norm=use_layer_norm)
    local_weight_version = weight_sync.version

    total_steps = 0
    ep_rewards = []
    ep_lengths = []
    current_ep_rewards = np.zeros(num_envs, dtype=np.float32)
    current_ep_lengths = np.zeros(num_envs, dtype=np.int32)
    from collections import defaultdict
    ep_reward_components = defaultdict(list)
    timing_accum_ms = defaultdict(float)
    timing_count = 0
    policy_timing_accum_ms = defaultdict(float)
    policy_timing_count = 0
    done_count_window = 0
    timeout_count_window = 0
    terminated_count_window = 0

    # Initial step to get first observation
    actions_np = np.zeros((num_envs, action_dim), dtype=np.float32)
    state = env.step(actions_np)
    obs_np = np.asarray(state.obs, dtype=np.float32)
    max_episode_steps = getattr(getattr(env, "cfg", None), "max_episode_steps", None)
    if max_episode_steps is not None and int(max_episode_steps) > 0:
        step_offsets = np.random.randint(0, int(max_episode_steps), size=(num_envs,), dtype=np.uint32)
        if hasattr(env, "state") and env.state is not None and isinstance(getattr(env.state, "info", None), dict):
            if "steps" in env.state.info:
                env.state.info["steps"][:] = step_offsets
        if isinstance(getattr(state, "info", None), dict) and "steps" in state.info:
            state.info["steps"][:] = step_offsets
    import time as _time
    _last_log_time = _time.time()

    # Collection loop
    while not stop_event.is_set():
        # Check for weight updates
        if weight_sync.version > local_weight_version:
            sync_t0 = _time.perf_counter()
            sd = dict(actor.state_dict())
            local_weight_version = weight_sync.read_weights_into(sd)
            actor.load_state_dict(sd)
            if numpy_actor is not None:
                numpy_actor.update_from_state_dict(sd)
            policy_timing_accum_ms["weight_sync_ms"] += (_time.perf_counter() - sync_t0) * 1000.0

        # Select action
        if total_steps < warmup_steps:
            actions_np = np.random.uniform(-1, 1, (num_envs, action_dim)).astype(np.float32)
        else:
            infer_t0 = _time.perf_counter()
            if infer_backend == "numpy" and numpy_actor is not None and algo_type == "sac":
                numpy_t0 = _time.perf_counter()
                actions_np = numpy_actor.explore(obs_np, deterministic=False)
                policy_timing_accum_ms["numpy_infer_ms"] += (_time.perf_counter() - numpy_t0) * 1000.0
                policy_timing_accum_ms["h2d_ms"] += 0.0
                policy_timing_accum_ms["d2h_ms"] += 0.0
                policy_timing_accum_ms["torch_infer_ms"] += 0.0
            else:
                with torch.no_grad():
                    h2d_t0 = _time.perf_counter()
                    obs_torch = torch.from_numpy(obs_np).to(collector_device)
                    policy_timing_accum_ms["h2d_ms"] += (_time.perf_counter() - h2d_t0) * 1000.0

                    infer_t1 = _time.perf_counter()
                    if algo_type == "sac":
                        actions_torch = actor.explore(obs_torch)
                    elif algo_type == "td3":
                        actions_torch = actor(obs_torch)
                        noise = torch.randn_like(actions_torch) * exploration_noise
                        actions_torch = (actions_torch + noise).clamp(-1, 1)
                    else:
                        actions_torch = torch.zeros((num_envs, action_dim), device=collector_device)
                    policy_timing_accum_ms["torch_infer_ms"] += (_time.perf_counter() - infer_t1) * 1000.0

                    d2h_t0 = _time.perf_counter()
                    actions_np = actions_torch.cpu().numpy()
                    policy_timing_accum_ms["d2h_ms"] += (_time.perf_counter() - d2h_t0) * 1000.0
                policy_timing_accum_ms["numpy_infer_ms"] += 0.0

            policy_timing_accum_ms["policy_total_ms"] += (_time.perf_counter() - infer_t0) * 1000.0
            policy_timing_count += 1

        # Step environment
        state = env.step(actions_np)

        timing_info = state.info.get("timing", {}) if hasattr(state, "info") else {}
        if timing_info:
            for key in ("env_step_total_ms", "step_core_ms", "update_state_ms", "reset_done_ms"):
                if key in timing_info:
                    timing_accum_ms[key] += float(timing_info[key])
            timing_count += 1

        # Extract data as numpy
        next_obs_np = np.asarray(state.obs, dtype=np.float32)
        rewards_np = np.asarray(state.reward, dtype=np.float32).ravel()
        terminated_np = np.asarray(state.terminated, dtype=np.float32).ravel() if state.terminated is not None else np.zeros(num_envs, dtype=np.float32)
        truncated_np = np.asarray(state.truncated, dtype=np.float32).ravel() if state.truncated is not None else np.zeros(num_envs, dtype=np.float32)
        combined_dones = np.clip(terminated_np + truncated_np, 0, 1)
        done_mask_np = combined_dones > 0.5
        timeout_mask_np = truncated_np > 0.5
        terminated_mask_np = np.logical_and(terminated_np > 0.5, ~timeout_mask_np)

        done_count_window += int(np.count_nonzero(done_mask_np))
        timeout_count_window += int(np.count_nonzero(timeout_mask_np))
        terminated_count_window += int(np.count_nonzero(terminated_mask_np))

        # Handle true terminal observations
        if "_final_observation" in state.info:
            has_final = state.info["_final_observation"]
            has_final_np = np.asarray(has_final, dtype=bool)
            if np.any(has_final_np):
                final_obs_np = np.asarray(state.info["final_observation"], dtype=np.float32)
                next_obs_np[has_final_np] = final_obs_np[has_final_np]

        # Write to replay buffer
        replay_buffer.add_batch(obs_np, actions_np, rewards_np, next_obs_np, terminated_np, truncated_np)

        # Track episode rewards - vectorized
        current_ep_rewards += rewards_np
        current_ep_lengths += 1
        reset_mask = combined_dones > 0.5
        reset_indices = np.where(reset_mask)[0]
        if len(reset_indices) > 0:
            ep_rewards.extend(current_ep_rewards[reset_indices].tolist())
            ep_lengths.extend(current_ep_lengths[reset_indices].tolist())
            current_ep_rewards[reset_indices] = 0.0
            current_ep_lengths[reset_indices] = 0

        obs_np = next_obs_np
        total_steps += num_envs

        # Progress log every 2 seconds
        now = _time.time()
        if now - _last_log_time > 2.0:
            _last_log_time = now
            phase = "warmup" if total_steps < warmup_steps else "policy"
            mean_r = np.mean(ep_rewards[-50:]) if ep_rewards else 0.0

        # Extract reward components from env info
        log_info = state.info.get("log", {})
        if log_info:
            for k, v in log_info.items():
                if k.startswith("reward/"):
                    ep_reward_components[k].append(v)

        # Send metrics periodically
        if metrics_queue is not None and total_steps % (num_envs * 10) == 0 and ep_rewards:
            import statistics
            try:
                msg = {
                    "total_steps": total_steps,
                    "mean_ep_reward": statistics.mean(ep_rewards[-100:]),
                    "mean_ep_length": statistics.mean(ep_lengths[-100:]) if ep_lengths else 0.0,
                    "buffer_size": replay_buffer.size,
                }
                # Add mean reward components
                if ep_reward_components:
                    components_mean = {}
                    for k, vals in ep_reward_components.items():
                        if vals:
                            components_mean[k] = statistics.mean(vals)
                    msg["reward_components"] = components_mean
                    ep_reward_components.clear()  # reset after sending

                if timing_count > 0:
                    msg["collector_timing_ms"] = {
                        k: (v / timing_count) for k, v in timing_accum_ms.items()
                    }
                    timing_accum_ms.clear()
                    timing_count = 0

                if policy_timing_count > 0:
                    msg["collector_policy_timing_ms"] = {
                        k: (v / policy_timing_count) for k, v in policy_timing_accum_ms.items()
                    }
                    msg["collector_infer_backend"] = infer_backend
                    policy_timing_accum_ms.clear()
                    policy_timing_count = 0

                if done_count_window > 0:
                    msg["timeout_rate"] = timeout_count_window / done_count_window
                    msg["terminated_rate"] = terminated_count_window / done_count_window
                    done_count_window = 0
                    timeout_count_window = 0
                    terminated_count_window = 0

                metrics_queue.put_nowait(msg)
            except Exception:
                pass

    replay_buffer.close()
    weight_sync.close()
