from __future__ import annotations

import inspect

from unilab.tools import render_teaser


def test_set_teaser_system_camera_view_uses_packaged_view():
    class FakeSystemCamera:
        def __init__(self):
            self.view = None

        def set_view(self, lookat, distance, elevation, azimuth):
            self.view = {
                "lookat": lookat,
                "distance": distance,
                "elevation": elevation,
                "azimuth": azimuth,
            }

    class FakeRender:
        def __init__(self):
            self.main_camera = "unset"
            self.system_camera = FakeSystemCamera()

        def set_main_camera(self, camera):
            self.main_camera = camera

    render = FakeRender()
    render_teaser._set_teaser_system_camera_view(render)

    assert render.main_camera is None
    assert render.system_camera.view == {
        "lookat": render_teaser.TEASER_SYSTEM_CAMERA_LOOKAT,
        "distance": render_teaser.TEASER_SYSTEM_CAMERA_DISTANCE,
        "elevation": render_teaser.TEASER_SYSTEM_CAMERA_ELEVATION,
        "azimuth": render_teaser.TEASER_SYSTEM_CAMERA_AZIMUTH,
    }


def test_packaged_teaser_scene_files_exist():
    assert render_teaser.DEFAULT_SCENE.is_file()
    assert (render_teaser.TEASER_DIR / "scene.xml").is_file()


def test_render_teaser_has_no_cli_parameters():
    assert list(inspect.signature(render_teaser.main).parameters) == []
