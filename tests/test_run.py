"""Reward-function unit tests for the Run task (Phase 1 rigid running baseline).

Uses duck-typed fakes rather than a real mjlab env, matching tests/test_wheel_glide.py.
"""

import torch

from mjlab_microduck.tasks.mdp import alternating_flight


class _Data:
    def __init__(self, air):
        self.current_air_time = torch.tensor(air, dtype=torch.float32)


class _Sensor:
    def __init__(self, air):
        self.data = _Data(air)


class _CommandManager:
    def __init__(self, cmd):
        self._cmd = torch.tensor(cmd, dtype=torch.float32)

    def get_command(self, _name):
        return self._cmd


class _Scene:
    def __init__(self, sensors):
        self.sensors = sensors


class _Env:
    """air: list of [left_air_time, right_air_time]; cmd: list of [vx, vy, wz]."""

    def __init__(self, air, cmd=None, sensor_name="feet_ground_contact"):
        if cmd is None:
            cmd = [[0.5, 0.0, 0.0]] * len(air)
        self.scene = _Scene({sensor_name: _Sensor(air)})
        self.command_manager = _CommandManager(cmd)
        self.num_envs = len(air)
        self.device = "cpu"
        self.extras = {"log": {}}


_SENSOR = "feet_ground_contact"
_CMD = "twist"


def test_symmetric_bounce_scores_zero():
    # Both feet airborne with identical air time — the rejected bouncing gait.
    env = _Env([[0.10, 0.10]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) < 1e-4


def test_alternating_flight_scores_high():
    # Trailing foot just left the ground, leading foot about to land.
    env = _Env([[0.02, 0.18]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) > 0.75


def test_both_feet_planted_scores_zero():
    env = _Env([[0.0, 0.0]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_single_support_is_not_flight():
    # One foot in the air is walking, not flight — must not be rewarded.
    env = _Env([[0.10, 0.0]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_inert_at_zero_command():
    env = _Env([[0.02, 0.18]], cmd=[[0.0, 0.0, 0.0]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_nan_safe():
    env = _Env([[float("nan"), 0.18]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert torch.isfinite(out).all()


def test_missing_sensor_returns_zeros():
    env = _Env([[0.02, 0.18]], sensor_name="some_other_sensor")
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_logs_metrics():
    env = _Env([[0.02, 0.18]])
    alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert "Metrics/flight_asymmetry" in env.extras["log"]
    assert "Metrics/flight_fraction" in env.extras["log"]


def test_asymmetry_metric_averages_over_flight_envs_only():
    # env 0 is in flight and symmetric; env 1 is in single support (asymmetry
    # would read 1.0 but must not pollute the metric).
    env = _Env([[0.10, 0.10], [0.10, 0.0]])
    alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(env.extras["log"]["Metrics/flight_asymmetry"]) < 1e-4
    assert abs(float(env.extras["log"]["Metrics/flight_fraction"]) - 0.5) < 1e-6
