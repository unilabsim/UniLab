import numpy as np

from unilab.utils import render_many


def test_render_states_get_frames_prefers_motrix(monkeypatch) -> None:
    state_list = [np.zeros((4, 8), dtype=np.float32) for _ in range(3)]
    calls: list[tuple[tuple, dict]] = []

    def fake_motrix(*args, **kwargs):
        calls.append((args, kwargs))
        return [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(len(state_list))]

    def fail_mujoco(*_args, **_kwargs):
        raise AssertionError("legacy MuJoCo path should not be used")

    monkeypatch.setattr(render_many, "_load_motrix_batch_renderer", lambda: object())
    monkeypatch.setattr(render_many, "_render_states_get_frames_motrix", fake_motrix)
    monkeypatch.setattr(render_many, "_render_states_get_frames_mujoco", fail_mujoco)
    monkeypatch.setenv("UNILAB_DISABLE_MOTRIX_BATCH_RENDERER", "0")

    frames = render_many.render_states_get_frames(
        state_list,
        "dummy.xml",
        width=320,
        height=240,
        camera_id=-1,
    )

    assert len(frames) == len(state_list)
    assert len(calls) == 1
    assert calls[0][1]["width"] == 320
    assert calls[0][1]["height"] == 240
    assert calls[0][1]["camera_id"] == -1


def test_render_states_get_frames_falls_back_to_mujoco(monkeypatch) -> None:
    state_list = [np.zeros((2, 8), dtype=np.float32)]
    fallback_calls: list[tuple[tuple, dict]] = []

    def fail_motrix(*_args, **_kwargs):
        raise ImportError("motrixsim not available")

    def fake_mujoco(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return ["legacy-frame"]

    monkeypatch.setattr(render_many, "_load_motrix_batch_renderer", lambda: object())
    monkeypatch.setattr(render_many, "_render_states_get_frames_motrix", fail_motrix)
    monkeypatch.setattr(render_many, "_render_states_get_frames_mujoco", fake_mujoco)
    monkeypatch.setenv("UNILAB_DISABLE_MOTRIX_BATCH_RENDERER", "0")

    frames = render_many.render_states_get_frames(
        state_list,
        "dummy.xml",
        width=160,
        height=120,
        num_processes=2,
    )

    assert frames == ["legacy-frame"]
    assert len(fallback_calls) == 1
    assert fallback_calls[0][1]["width"] == 160
    assert fallback_calls[0][1]["height"] == 120
    assert fallback_calls[0][1]["num_processes"] == 2


def test_render_states_get_frames_uses_legacy_path_when_motrix_missing(monkeypatch) -> None:
    state_list = [np.zeros((2, 8), dtype=np.float32)]
    fallback_calls: list[tuple[tuple, dict]] = []

    def fail_motrix(*_args, **_kwargs):
        raise AssertionError("motrix path should not be touched when dependency is missing")

    def fake_mujoco(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return ["legacy-frame"]

    monkeypatch.setattr(render_many, "_load_motrix_batch_renderer", lambda: None)
    monkeypatch.setattr(render_many, "_render_states_get_frames_motrix", fail_motrix)
    monkeypatch.setattr(render_many, "_render_states_get_frames_mujoco", fake_mujoco)
    monkeypatch.setenv("UNILAB_DISABLE_MOTRIX_BATCH_RENDERER", "0")

    frames = render_many.render_states_get_frames(
        state_list,
        "dummy.xml",
        width=200,
        height=100,
        num_processes=3,
    )

    assert frames == ["legacy-frame"]
    assert len(fallback_calls) == 1
    assert fallback_calls[0][1]["width"] == 200
    assert fallback_calls[0][1]["height"] == 100
    assert fallback_calls[0][1]["num_processes"] == 3
