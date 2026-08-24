"""Unit tests for the ported hop reward terms (duck-typed fakes)."""

import torch

from mjlab_microduck.tasks.mdp import (
    hop_body_height,
    hop_both_feet_airborne,
    hop_upward_velocity,
)

_SENSOR = "feet_ground_contact"
_CMD = "twist"


class _SensorData:
    def __init__(self, found):
        self.found = torch.tensor(found, dtype=torch.float32)


class _Sensor:
    def __init__(self, found):
        self.data = _SensorData(found)


class _AssetData:
    def __init__(self, vz, z):
        n = len(vz)
        self.root_link_lin_vel_w = torch.zeros((n, 3), dtype=torch.float32)
        self.root_link_lin_vel_w[:, 2] = torch.tensor(vz, dtype=torch.float32)
        self.root_link_pos_w = torch.zeros((n, 3), dtype=torch.float32)
        self.root_link_pos_w[:, 2] = torch.tensor(z, dtype=torch.float32)


class _Asset:
    def __init__(self, vz, z):
        self.data = _AssetData(vz, z)


class _CommandManager:
    def __init__(self, cmd):
        self._cmd = torch.tensor(cmd, dtype=torch.float32)

    def get_command(self, _name):
        return self._cmd


class _Scene:
    def __init__(self, sensors, asset):
        self.sensors = sensors
        self._asset = asset

    def __getitem__(self, _k):
        return self._asset


class _Env:
    """found: per-foot contact flags; cmd: [cos, sin, 0]; vz/z: base state."""

    def __init__(self, found=((0.0, 0.0),), cmd=((0.0, 1.0, 0.0),),
                 vz=(0.0,), z=(0.15,)):
        self.scene = _Scene({_SENSOR: _Sensor(found)}, _Asset(vz, z))
        self.command_manager = _CommandManager(cmd)
        self.num_envs = len(found)
        self.device = "cpu"
        self.extras = {"log": {}}


# --- hop_both_feet_airborne -------------------------------------------------

def test_airborne_rewarded_at_peak_launch_phase():
    # sin(2*pi*phi) = 1 (mid-launch), both feet off the ground
    env = _Env(found=[[0.0, 0.0]], cmd=[[0.0, 1.0, 0.0]])
    out = hop_both_feet_airborne(env, sensor_name=_SENSOR, command_name=_CMD)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_airborne_not_rewarded_when_a_foot_is_down():
    env = _Env(found=[[1.0, 0.0]], cmd=[[0.0, 1.0, 0.0]])
    out = hop_both_feet_airborne(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_airborne_not_rewarded_during_the_recovery_half_cycle():
    """sin < 0 is the recovery half — flight there must not be paid for,
    or the policy is rewarded for simply never landing."""
    env = _Env(found=[[0.0, 0.0]], cmd=[[0.0, -1.0, 0.0]])
    out = hop_both_feet_airborne(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


# --- hop_upward_velocity ----------------------------------------------------

def test_upward_velocity_saturates_at_max_vel():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], vz=[10.0])
    out = hop_upward_velocity(env, command_name=_CMD, max_vel=0.5)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_downward_velocity_is_not_rewarded():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], vz=[-2.0])
    out = hop_upward_velocity(env, command_name=_CMD, max_vel=0.5)
    assert float(out[0]) == 0.0


def test_upward_velocity_scales_below_saturation():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], vz=[0.25])
    out = hop_upward_velocity(env, command_name=_CMD, max_vel=0.5)
    assert abs(float(out[0]) - 0.5) < 1e-6


def test_upward_velocity_is_gated_by_the_launch_phase():
    """sin < 0 is the recovery half — upward velocity there must not be paid
    for, or the policy is rewarded for launching outside the hop cycle."""
    env = _Env(cmd=[[0.0, -1.0, 0.0]], vz=[10.0])
    out = hop_upward_velocity(env, command_name=_CMD, max_vel=0.5)
    assert float(out[0]) == 0.0


# --- hop_body_height --------------------------------------------------------

def test_body_height_peaks_at_the_target():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], z=[0.165])
    out = hop_body_height(env, command_name=_CMD, target_height=0.165, std=0.008)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_body_height_falls_off_away_from_the_target():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], z=[0.145])
    out = hop_body_height(env, command_name=_CMD, target_height=0.165, std=0.008)
    assert float(out[0]) < 0.01


def test_body_height_is_gated_by_the_launch_phase():
    env = _Env(cmd=[[0.0, -1.0, 0.0]], z=[0.165])
    out = hop_body_height(env, command_name=_CMD, target_height=0.165, std=0.008)
    assert float(out[0]) == 0.0


def test_all_three_terms_are_nan_safe():
    env = _Env(found=[[0.0, 0.0]], cmd=[[0.0, 1.0, 0.0]],
               vz=[float("nan")], z=[float("nan")])
    for out in (
        hop_both_feet_airborne(env, sensor_name=_SENSOR, command_name=_CMD),
        hop_upward_velocity(env, command_name=_CMD),
        hop_body_height(env, command_name=_CMD, target_height=0.165),
    ):
        assert torch.isfinite(out).all()
