"""Unit tests for the spring compression monitor (duck-typed fakes)."""

import torch

from mjlab_microduck.tasks.mdp import spring_compression_monitor

_JOINTS = ("passive_left_foot_spring", "passive_right_foot_spring")
_TRAVEL = 0.015


class _Data:
    def __init__(self, q):
        self.joint_pos = torch.tensor(q, dtype=torch.float32)


class _Asset:
    def __init__(self, q):
        self.data = _Data(q)

    def find_joints(self, name):
        # column 0 = left spring, column 1 = right spring
        return [_JOINTS.index(name)], None


class _Scene:
    def __init__(self, q):
        self._a = _Asset(q)

    def __getitem__(self, _k):
        return self._a


class _Env:
    """q: list of [left_compression, right_compression] in metres."""

    def __init__(self, q):
        self.scene = _Scene(q)
        self.num_envs = len(q)
        self.device = "cpu"
        self.extras = {"log": {}}


def test_returns_exactly_zeros():
    env = _Env([[0.005, 0.007]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_reports_mean_and_max_compression():
    env = _Env([[0.004, 0.010]])
    spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    log = env.extras["log"]
    assert abs(float(log["Metrics/spring_compression_mean"]) - 0.007) < 1e-6
    assert abs(float(log["Metrics/spring_compression_max"]) - 0.010) < 1e-6


def test_bottomed_fraction_is_zero_when_well_inside_travel():
    env = _Env([[0.004, 0.005]])
    spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    assert float(env.extras["log"]["Metrics/spring_bottomed_fraction"]) == 0.0


def test_bottomed_fraction_catches_a_bottomed_spring():
    # 0.0149 of 0.015 travel is 99.3% — past the 95% threshold.
    env = _Env([[0.0149, 0.001]])
    spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    # one of two joint-samples is bottomed
    assert abs(float(env.extras["log"]["Metrics/spring_bottomed_fraction"]) - 0.5) < 1e-6


def test_zero_travel_locked_variant_does_not_divide_by_zero():
    env = _Env([[0.0, 0.0]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=0.0)
    assert float(out[0]) == 0.0
    assert torch.isfinite(env.extras["log"]["Metrics/spring_bottomed_fraction"])


def test_nan_safe():
    env = _Env([[float("nan"), 0.006]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    assert torch.isfinite(out).all()
    assert torch.isfinite(env.extras["log"]["Metrics/spring_compression_max"])


def test_missing_joints_return_zeros_without_raising():
    """The locked control arm has NO spring joints, so this path runs in
    production on every step of that arm. mjlab's find_joints RAISES
    ValueError when a name matches nothing rather than returning empty.
    """
    class _RaisingAsset(_Asset):
        def find_joints(self, name):
            raise ValueError("Not all regular expressions are matched!")

    env = _Env([[0.005, 0.005]])
    env.scene._a = _RaisingAsset([[0.005, 0.005]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_empty_joint_lookup_returns_zeros():
    """Defensive variant: also handle the case where find_joints returns
    an empty list (for future mjlab compatibility).
    """
    class _NoJointAsset(_Asset):
        def find_joints(self, name):
            return [], None

    env = _Env([[0.005, 0.005]])
    env.scene._a = _NoJointAsset([[0.005, 0.005]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0
