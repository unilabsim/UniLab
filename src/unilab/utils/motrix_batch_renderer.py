from __future__ import annotations

import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mujoco
import numpy as np

try:
    import motrixsim as mtx
    from motrixsim.render import RenderApp, RenderSettings

    MOTRIX_BATCH_RENDERER_AVAILABLE = True
except ImportError:
    MOTRIX_BATCH_RENDERER_AVAILABLE = False
    mtx = None
    RenderApp = Any
    RenderSettings = Any

if TYPE_CHECKING:
    import motrixsim


def get_batch_render_offsets(num_envs: int, spacing: float = 1.0) -> np.ndarray:
    cols = int(math.ceil(math.sqrt(num_envs)))
    offsets = np.zeros((num_envs, 3), dtype=np.float64)
    for idx in range(num_envs):
        row = idx // cols
        col = idx % cols
        offsets[idx] = np.array([col * spacing, row * spacing, 0.0], dtype=np.float64)
    return offsets


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return vec / norm


def _format_xml_vec(vec: np.ndarray) -> str:
    return " ".join(f"{float(v):.8g}" for v in vec)


def _read_statistic_center(model_file: str) -> np.ndarray:
    root = ET.parse(model_file).getroot()
    statistic = root.find("statistic")
    if statistic is None:
        return np.zeros(3, dtype=np.float64)

    center_attr = statistic.get("center")
    if not center_attr:
        return np.zeros(3, dtype=np.float64)

    center = np.fromstring(center_attr, sep=" ", dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"Invalid <statistic center> in {model_file}: {center_attr!r}")
    return center


def _compute_camera_axes(position: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = _normalize(target - position)
    camera_z = -forward

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(camera_z, world_up))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    camera_x = _normalize(np.cross(world_up, camera_z))
    camera_y = _normalize(np.cross(camera_z, camera_x))
    return camera_x, camera_y


def compute_batch_render_lookat(model_file: str, offsets: np.ndarray) -> np.ndarray:
    lookat = _read_statistic_center(model_file)
    if offsets.size > 0:
        lookat[:2] += offsets[:, :2].mean(axis=0)
    return lookat


def inject_batch_render_camera(
    model_file: str,
    *,
    lookat: np.ndarray,
    distance: float,
    elevation: float,
    azimuth: float,
    camera_name: str = "__unilab_batch_render_camera__",
) -> str:
    """Inject a fixed camera into an MJCF for Motrix headless capture."""
    elevation_rad = math.radians(elevation)
    azimuth_rad = math.radians(azimuth)
    offset = np.array(
        [
            distance * math.cos(elevation_rad) * math.cos(azimuth_rad),
            distance * math.cos(elevation_rad) * math.sin(azimuth_rad),
            distance * math.sin(elevation_rad),
        ],
        dtype=np.float64,
    )
    position = np.asarray(lookat, dtype=np.float64) + offset
    camera_x, camera_y = _compute_camera_axes(position, np.asarray(lookat, dtype=np.float64))

    tree = ET.parse(model_file)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    ET.SubElement(
        worldbody,
        "camera",
        name=camera_name,
        mode="fixed",
        pos=_format_xml_vec(position),
        xyaxes=_format_xml_vec(np.concatenate([camera_x, camera_y])),
    )

    model_dir = Path(model_file).resolve().parent
    fd, tmp_path = tempfile.mkstemp(
        prefix=".unilab_batch_render_",
        suffix=".xml",
        dir=str(model_dir),
    )
    os.close(fd)
    tree.write(tmp_path)
    return tmp_path


class MujocoStateBridge:
    """Map MuJoCo batch states to MotrixSim qpos layout."""

    def __init__(self, mj_model: mujoco.MjModel):
        self._mj_model = mj_model
        self._idx_qpos = 1
        self._map_qpos_idx_mj_to_mx = np.arange(mj_model.nq, dtype=np.int32)

        for body_id in range(mj_model.nbody):
            body = mj_model.body(body_id)
            has_free_joint = (
                body.dofnum == 6
                and mj_model.body_jntadr[body_id] > -1
                and mj_model.jnt_type[mj_model.body_jntadr[body_id]]
                == int(mujoco.mjtJoint.mjJNT_FREE)
            )
            if not has_free_joint:
                continue

            qpos_adr = mj_model.jnt_qposadr[body.jntadr[0]]
            # MuJoCo free joint quat is wxyz; Motrix expects xyzw.
            self._map_qpos_idx_mj_to_mx[qpos_adr + 3 : qpos_adr + 7] = [
                qpos_adr + 4,
                qpos_adr + 5,
                qpos_adr + 6,
                qpos_adr + 3,
            ]

    @property
    def qpos_index_map(self) -> np.ndarray:
        return self._map_qpos_idx_mj_to_mx

    def physics_state_to_motrix_qpos(self, physics_state: np.ndarray) -> np.ndarray:
        state = np.asarray(physics_state)
        if state.ndim == 1:
            state = state[np.newaxis, :]
        qpos = state[:, self._idx_qpos : self._idx_qpos + self._mj_model.nq]
        return np.ascontiguousarray(qpos[:, self._map_qpos_idx_mj_to_mx])


class MotrixBatchRenderer:
    """Use MotrixSim renderer to visualize MuJoCo batch states."""

    def __init__(
        self,
        *,
        model_file: str,
        mujoco_model: mujoco.MjModel,
        num_envs: int,
        spacing: float = 1.0,
        headless: bool = False,
        width: int = 1280,
        height: int = 720,
        camera_id: int = -1,
        cam_distance: float = 2.0,
        cam_elevation: float = -20.0,
        cam_azimuth: float = 90.0,
    ):
        if not MOTRIX_BATCH_RENDERER_AVAILABLE:
            raise ImportError("motrixsim not available")
        assert mtx is not None

        self._model_file = model_file
        self._headless = headless
        self._bridge = MujocoStateBridge(mujoco_model)
        self._render_offsets = get_batch_render_offsets(num_envs, spacing)
        self._render_app: RenderApp | None = None

        render_model_path = model_file
        temp_model_path: str | None = None
        try:
            if headless and camera_id < 0:
                lookat = compute_batch_render_lookat(model_file, self._render_offsets)
                temp_model_path = inject_batch_render_camera(
                    model_file,
                    lookat=lookat,
                    distance=cam_distance,
                    elevation=cam_elevation,
                    azimuth=cam_azimuth,
                )
                render_model_path = temp_model_path

            self._model = mtx.load_model(render_model_path)
        finally:
            if temp_model_path is not None and os.path.exists(temp_model_path):
                os.remove(temp_model_path)

        self._model.options.timestep = mujoco_model.opt.timestep
        self._data = mtx.SceneData(self._model, batch=[num_envs])
        self._camera_id: int | None = None

        if headless:
            resolved_camera_id = camera_id
            if resolved_camera_id < 0:
                resolved_camera_id = len(self._model.cameras) - 1
            if resolved_camera_id < 0 or resolved_camera_id >= len(self._model.cameras):
                raise ValueError(
                    f"Camera {resolved_camera_id} is unavailable for headless render of {model_file}"
                )
            self._camera_id = resolved_camera_id
            self._model.cameras[self._camera_id].set_render_target("image", width, height)
            self._render_app = RenderApp(log_level="WARN", headless=True)
        else:
            self._render_app = RenderApp()

        settings = RenderSettings.performance()
        if not headless:
            settings.enable_shadow = True

        self._render_app.launch(
            self._model,
            batch=num_envs,
            render_offset=self._render_offsets.tolist(),
            render_settings=settings,
        )

    @property
    def data(self) -> "motrixsim.SceneData":
        return self._data

    def _sync_scene_data(self, physics_state: np.ndarray) -> None:
        qpos_motrix = self._bridge.physics_state_to_motrix_qpos(physics_state)
        self._data.set_dof_pos(qpos_motrix, self._model)
        self._model.forward_kinematic(self._data)

    def render(self, physics_state: np.ndarray) -> None:
        if self._render_app is None:
            raise RuntimeError("Renderer is closed")
        self._sync_scene_data(physics_state)
        self._render_app.sync(self._data)

    def capture_frame(self, physics_state: np.ndarray) -> np.ndarray:
        if not self._headless:
            raise RuntimeError("capture_frame() requires headless=True")
        if self._render_app is None or self._camera_id is None:
            raise RuntimeError("Renderer is closed")

        self._sync_scene_data(physics_state)
        capture_task = self._render_app.get_camera(self._camera_id).capture()
        self._render_app.sync(self._data, wait=True)
        image = capture_task.take_image()
        if image is None:
            raise RuntimeError("Motrix headless capture returned no image")
        return cast(np.ndarray, np.asarray(image.pixels, dtype=np.uint8).copy())

    def close(self) -> None:
        self._render_app = None

    def __enter__(self) -> "MotrixBatchRenderer":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
