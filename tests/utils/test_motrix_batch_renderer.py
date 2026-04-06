import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from unilab.assets import ASSETS_ROOT_PATH
from unilab.utils.motrix_batch_renderer import (
    MujocoStateBridge,
    compute_batch_render_lookat,
    get_batch_render_offsets,
    inject_batch_render_camera,
)


def _xml(robot: str, scene: str = "scene_flat.xml") -> str:
    return str(ASSETS_ROOT_PATH / "robots" / robot / scene)


def test_mujoco_state_bridge_reorders_free_joint_quaternion() -> None:
    model = mujoco.MjModel.from_xml_path(_xml("go1"))
    bridge = MujocoStateBridge(model)

    state = np.zeros((1, 1 + model.nq + model.nv), dtype=np.float64)
    state[0, 1 : 1 + model.nq] = np.arange(1, model.nq + 1, dtype=np.float64)

    qpos_motrix = bridge.physics_state_to_motrix_qpos(state)

    np.testing.assert_array_equal(qpos_motrix[0, :3], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(qpos_motrix[0, 3:7], np.array([5.0, 6.0, 7.0, 4.0]))


def test_compute_batch_render_lookat_uses_statistic_center_and_grid_mean() -> None:
    offsets = get_batch_render_offsets(4, spacing=2.0)

    lookat = compute_batch_render_lookat(_xml("go1"), offsets)

    np.testing.assert_allclose(lookat, np.array([1.0, 1.0, 0.1]), atol=1e-6)


def test_inject_batch_render_camera_adds_fixed_camera() -> None:
    tmp_path = inject_batch_render_camera(
        _xml("go1"),
        lookat=np.array([0.0, 0.0, 0.1], dtype=np.float64),
        distance=2.0,
        elevation=-20.0,
        azimuth=90.0,
    )
    try:
        root = ET.parse(tmp_path).getroot()
        cameras = root.findall("./worldbody/camera[@name='__unilab_batch_render_camera__']")

        assert len(cameras) == 1
        camera = cameras[0]
        assert camera.get("mode") == "fixed"
        assert camera.get("pos")
        assert camera.get("xyaxes")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
