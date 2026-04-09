import math
import os
import sys
from typing import Any

import imageio


def _resolve_gl_backend() -> str:
    """Pick a valid MUJOCO_GL backend for the current platform."""
    current = os.environ.get("MUJOCO_GL", "")
    safe_values = {"glfw", "osmesa", "disabled"}

    if sys.platform == "darwin":
        return current if current in safe_values else "glfw"

    if current in safe_values:
        return current

    try:
        import ctypes

        ctypes.CDLL("libEGL.so.1")
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
        return "egl"
    except OSError:
        return "glfw"


# Must be set *before* importing mujoco (it reads the var at import time).
os.environ["MUJOCO_GL"] = _resolve_gl_backend()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402


def get_grid_offsets(num_envs, spacing=1.0):
    rows = int(math.ceil(math.sqrt(num_envs)))
    cols = int(math.ceil(num_envs / rows))
    offsets = np.zeros((num_envs, 2))
    for i in range(num_envs):
        r = i // cols
        c = i % cols
        offsets[i, 0] = r * spacing
        offsets[i, 1] = c * spacing
    return offsets


# Worker global context
_worker_ctx: dict[str, Any] = {}


def _load_motrix_batch_renderer():
    """Load the optional Motrix batch renderer lazily.

    When motrixsim is not installed, render_many should continue to use the
    original MuJoCo rendering path without any behavior change.
    """
    try:
        from unilab.utils.motrix_batch_renderer import (
            MOTRIX_BATCH_RENDERER_AVAILABLE,
            MotrixBatchRenderer,
        )
    except Exception:
        return None

    if not MOTRIX_BATCH_RENDERER_AVAILABLE:
        return None

    return MotrixBatchRenderer


def _close_worker():
    """Explicitly close the renderer in the worker context."""
    if "renderer" in _worker_ctx:
        _worker_ctx["renderer"].close()


def init_worker(model_path, shape):
    """Initialize MuJoCo context for worker process."""
    import atexit

    _worker_ctx["model"] = mujoco.MjModel.from_xml_path(model_path)
    _worker_ctx["model"].vis.global_.offwidth = 3840
    _worker_ctx["model"].vis.global_.offheight = 2160

    _worker_ctx["data"] = mujoco.MjData(_worker_ctx["model"])
    _worker_ctx["renderer"] = mujoco.Renderer(_worker_ctx["model"], height=shape[1], width=shape[0])
    atexit.register(_close_worker)


def render_frame_job(args):
    """Worker function to render a single frame with MuJoCo's legacy path."""
    state_batch, offsets, transparent, cam_distance, cam_elevation, cam_azimuth = args

    model = _worker_ctx["model"]
    data = _worker_ctx["data"]
    renderer = _worker_ctx["renderer"]

    vopt = mujoco.MjvOption()
    vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
    pert = mujoco.MjvPerturb()
    catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC

    def set_state(d, s, offset=None):
        d.time = s[0]
        d.qpos[:] = s[1 : 1 + model.nq]
        d.qvel[:] = s[1 + model.nq : 1 + model.nq + model.nv]

        apply_root_offset = False

        if offset is not None:
            robot_moved = False
            first_body_jnt = model.body_jntadr[1] if model.nbody > 1 else -1
            if first_body_jnt >= 0 and model.jnt_type[first_body_jnt] == 0:
                d.qpos[0] += offset[0]
                d.qpos[1] += offset[1]
                robot_moved = True

            if not robot_moved:
                apply_root_offset = True

            box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
            if box_id >= 0:
                jnt_adr = model.body_jntadr[box_id]
                if jnt_adr >= 0:
                    qpos_adr = model.jnt_qposadr[jnt_adr]
                    d.qpos[qpos_adr] += offset[0]
                    d.qpos[qpos_adr + 1] += offset[1]

            target_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
            if target_x >= 0:
                d.qpos[model.jnt_qposadr[target_x]] += offset[0]

            target_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")
            if target_y >= 0:
                d.qpos[model.jnt_qposadr[target_y]] += offset[1]

        mujoco.mj_forward(model, d)

        if apply_root_offset and offset is not None:
            box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
            target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mocap_target")

            for i in range(model.ngeom):
                body_id = model.geom_bodyid[i]
                is_box_or_target = (body_id == box_body_id) or (body_id == target_body_id)
                is_plane = model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
                if not is_box_or_target and not is_plane:
                    d.geom_xpos[i, 0] += offset[0]
                    d.geom_xpos[i, 1] += offset[1]

            for i in range(model.nsite):
                body_id = model.site_bodyid[i]
                is_box_or_target = (body_id == box_body_id) or (body_id == target_body_id)
                if not is_box_or_target:
                    d.site_xpos[i, 0] += offset[0]
                    d.site_xpos[i, 1] += offset[1]

    num_envs = state_batch.shape[0]
    set_state(data, state_batch[0], offsets[0] if offsets is not None else None)

    cam = mujoco.MjvCamera()
    if offsets is not None:
        center_x = np.mean(offsets[:, 0])
        center_y = np.mean(offsets[:, 1])
        cam.lookat = [center_x, center_y, 0.0]
        cam.distance = cam_distance
        cam.elevation = cam_elevation
        cam.azimuth = cam_azimuth
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    renderer.update_scene(data, camera=cam, scene_option=vopt)

    for i in range(1, num_envs):
        set_state(data, state_batch[i], offsets[i] if offsets is not None else None)
        mujoco.mjv_addGeoms(model, data, vopt, pert, catmask, renderer.scene)

    return renderer.render()


def _render_states_get_frames_motrix(
    state_list,
    model_path,
    width=1280,
    height=720,
    camera_id=-1,
    cam_distance=2.0,
    cam_elevation=-20,
    cam_azimuth=90,
    camera_mode="fixed",
):
    renderer_cls = _load_motrix_batch_renderer()
    if renderer_cls is None:
        raise ImportError("motrixsim not available")

    num_envs = state_list[0].shape[0]
    with renderer_cls(
        model_file=model_path,
        mujoco_model=mujoco.MjModel.from_xml_path(model_path),
        num_envs=num_envs,
        headless=True,
        width=width,
        height=height,
        camera_id=camera_id,
        cam_distance=cam_distance,
        cam_elevation=cam_elevation,
        cam_azimuth=cam_azimuth,
        camera_mode=camera_mode,
    ) as renderer:
        return [renderer.capture_frame(state) for state in state_list]


def _render_states_get_frames_mujoco(
    state_list,
    model_path,
    width=1280,
    height=720,
    num_processes=8,
    cam_distance=2.0,
    cam_elevation=-20,
    cam_azimuth=90,
):
    num_envs = state_list[0].shape[0]
    offsets = get_grid_offsets(num_envs)
    shape = (width, height)

    print(
        f"Rendering {len(state_list)} frames for {num_envs} envs with {num_processes} processes..."
    )

    tasks = [(s, offsets, False, cam_distance, cam_elevation, cam_azimuth) for s in state_list]
    frames = []

    if num_processes <= 1:
        init_worker(model_path, shape)
        try:
            for task in tasks:
                frames.append(render_frame_job(task))
        finally:
            _close_worker()
    else:
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=num_processes, initializer=init_worker, initargs=(model_path, shape)
        ) as pool:
            frames.extend(pool.map(render_frame_job, tasks))

    return frames


def render_states_get_frames(
    state_list,
    model_path,
    width=1280,
    height=720,
    num_processes=8,
    camera_id=-1,
    cam_distance=2.0,
    cam_elevation=-20,
    cam_azimuth=90,
    camera_mode="fixed",
):
    """Render a list of MuJoCo batch states and return RGB frames."""
    if not state_list:
        print("No states to render.")
        return []

    motrix_renderer_available = _load_motrix_batch_renderer() is not None
    if os.getenv("UNILAB_DISABLE_MOTRIX_BATCH_RENDERER", "0") != "1" and motrix_renderer_available:
        try:
            return _render_states_get_frames_motrix(
                state_list,
                model_path,
                width=width,
                height=height,
                camera_id=camera_id,
                cam_distance=cam_distance,
                cam_elevation=cam_elevation,
                cam_azimuth=cam_azimuth,
                camera_mode=camera_mode,
            )
        except Exception as exc:
            print(f"Falling back to MuJoCo render_many path: {exc}")

    return _render_states_get_frames_mujoco(
        state_list,
        model_path,
        width=width,
        height=height,
        num_processes=num_processes,
        cam_distance=cam_distance,
        cam_elevation=cam_elevation,
        cam_azimuth=cam_azimuth,
    )


def render_states_to_video(
    state_list,
    model_path,
    output_path,
    fps=30,
    width=1280,
    height=720,
    num_processes=8,
    cam_distance=2.0,
    cam_elevation=-20,
    cam_azimuth=90,
    camera_mode="fixed",
):
    """Render a list of physics states to a video file."""
    frames = render_states_get_frames(
        state_list,
        model_path,
        width,
        height,
        num_processes,
        cam_distance=cam_distance,
        cam_elevation=cam_elevation,
        cam_azimuth=cam_azimuth,
        camera_mode=camera_mode,
    )

    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, frames, fps=fps)
    print("Done!")
