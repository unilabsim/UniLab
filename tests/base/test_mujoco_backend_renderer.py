import numpy as np

from unilab.assets import ASSETS_ROOT_PATH


def _xml(robot: str, scene: str = "scene_flat.xml") -> str:
    return str(ASSETS_ROOT_PATH / "robots" / robot / scene)


def test_mujoco_backend_uses_motrix_batch_renderer(monkeypatch) -> None:
    from unilab.base.backend import mujoco_backend as mod

    renderer_calls: list[np.ndarray] = []

    class FakeRenderer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def render(self, physics_state):
            renderer_calls.append(np.asarray(physics_state).copy())

    monkeypatch.setattr(mod, "MotrixBatchRenderer", FakeRenderer)

    backend = mod.MuJoCoBackend(_xml("go1"), num_envs=2, sim_dt=0.005, base_name="trunk")
    backend.init_renderer(spacing=1.5)

    assert isinstance(backend._batch_renderer, FakeRenderer)
    assert backend._batch_renderer.kwargs["model_file"] == _xml("go1")
    assert backend._batch_renderer.kwargs["num_envs"] == 2
    assert backend._batch_renderer.kwargs["spacing"] == 1.5

    backend.render()

    assert len(renderer_calls) == 1
    np.testing.assert_array_equal(renderer_calls[0], backend.get_physics_state())
