"""Reward-function unit tests for the Run task (Phase 1 rigid running baseline).

Uses duck-typed fakes rather than a real mjlab env, matching tests/test_wheel_glide.py.
"""

import torch

from mjlab_microduck.tasks.mdp import alternating_flight, feet_air_time_capped, action_magnitude_monitor


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


def test_capped_both_feet_in_window_scores_one_not_two():
    # THE bug being fixed: stock mjlab feet_air_time returns 2.0 here.
    env = _Env([[0.10, 0.10]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_capped_single_foot_in_window_scores_one():
    env = _Env([[0.10, 0.0]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_capped_below_window_scores_zero():
    env = _Env([[0.01, 0.01]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_capped_above_window_scores_zero():
    env = _Env([[0.40, 0.40]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_capped_inert_at_zero_command():
    env = _Env([[0.10, 0.10]], cmd=[[0.0, 0.0, 0.0]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_capped_nan_safe():
    env = _Env([[float("nan"), 0.10]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert torch.isfinite(out).all()


def test_capped_missing_sensor_returns_zeros():
    env = _Env([[0.10, 0.10]], sensor_name="some_other_sensor")
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


class _ActionManager:
    def __init__(self, actions):
        self.action = torch.tensor(actions, dtype=torch.float32)


class _ActionEnv:
    def __init__(self, actions, with_manager=True):
        self.num_envs = len(actions)
        self.device = "cpu"
        self.extras = {"log": {}}
        if with_manager:
            self.action_manager = _ActionManager(actions)


def test_monitor_contributes_exactly_zero_reward():
    env = _ActionEnv([[0.5, -3.0, 1e9]])
    out = action_magnitude_monitor(env)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_monitor_reports_max_magnitude():
    env = _ActionEnv([[0.5, -3.0, 2.0]])
    action_magnitude_monitor(env)
    assert abs(float(env.extras["log"]["Metrics/action_abs_max"]) - 3.0) < 1e-6


def test_monitor_logs_both_keys():
    env = _ActionEnv([[0.5, -3.0, 2.0]])
    action_magnitude_monitor(env)
    assert "Metrics/action_abs_max" in env.extras["log"]
    assert "Metrics/action_abs_p99" in env.extras["log"]


def test_monitor_survives_blowup_values():
    # The failure mode being watched for: |a| ~ 1e10.
    env = _ActionEnv([[1e10, -1e10]])
    out = action_magnitude_monitor(env)
    assert float(out[0]) == 0.0
    assert torch.isfinite(env.extras["log"]["Metrics/action_abs_max"])


def test_monitor_without_action_manager_returns_zeros():
    env = _ActionEnv([[0.5, 0.5]], with_manager=False)
    out = action_magnitude_monitor(env)
    assert float(out[0]) == 0.0


def test_monitor_survives_non_finite_actions():
    # The nan_to_num(posinf=..., neginf=...) guard exists for exactly this.
    env = _ActionEnv([[float("inf"), float("-inf"), float("nan"), 2.0]])
    out = action_magnitude_monitor(env)
    assert float(out[0]) == 0.0
    assert torch.isfinite(env.extras["log"]["Metrics/action_abs_max"])
    assert torch.isfinite(env.extras["log"]["Metrics/action_abs_p99"])
