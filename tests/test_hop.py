"""Unit tests for the ported hop reward terms (duck-typed fakes)."""

import torch

from mjlab_microduck.tasks.mdp import (
    hop_body_height,
    hop_both_feet_airborne,
    hop_energy_monitor,
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


# --- hop_energy_monitor ------------------------------------------------------

_JOINTS = ("passive_left_foot_spring", "passive_right_foot_spring")
_K = 3900.0
_PRELOAD = 0.00074


class _JointData:
    def __init__(self, q):
        self.joint_pos = torch.tensor(q, dtype=torch.float32)


class _JointAsset:
    def __init__(self, q):
        self.data = _JointData(q)

    def find_joints(self, name):
        return [_JOINTS.index(name)], None


class _JointScene:
    def __init__(self, q):
        self._a = _JointAsset(q)

    def __getitem__(self, _k):
        return self._a


class _JointEnv:
    def __init__(self, q):
        self.scene = _JointScene(q)
        self.num_envs = len(q)
        self.device = "cpu"
        self.extras = {"log": {}}


def test_energy_monitor_returns_exactly_zeros():
    env = _JointEnv([[0.005, 0.005]])
    out = hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_energy_matches_the_closed_form():
    """E = 0.5*k*q^2 + k*preload*q per foot, summed over both."""
    q = 0.006
    env = _JointEnv([[q, q]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    expected = 2 * (0.5 * _K * q**2 + _K * _PRELOAD * q)
    got = float(env.extras["log"]["Metrics/hop_spring_energy_mean"])
    # 1e-8, not 1e-9: q is stored as float32 in the fixture (matching real
    # joint_pos), so q=0.006 is already quantized to ~6.0000000522e-3 before
    # any arithmetic runs. That alone puts a ~2.7e-9 floor under this
    # comparison against the float64 closed-form -- confirmed by reproducing
    # the closed-form in pure float64 math off the quantized value.
    assert abs(got - expected) < 1e-8


def test_energy_is_zero_at_rest():
    env = _JointEnv([[0.0, 0.0]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert float(env.extras["log"]["Metrics/hop_spring_energy_mean"]) == 0.0


def test_negative_q_contributes_no_energy():
    """Preload holds the pad past its lower limit when unloaded (measured
    -0.59 mm). That is limit penetration, not stored energy."""
    env = _JointEnv([[-0.00059, -0.00059]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert float(env.extras["log"]["Metrics/hop_spring_energy_mean"]) == 0.0


def test_peak_exceeds_mean_when_feet_differ():
    env = _JointEnv([[0.002, 0.002], [0.010, 0.010]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    log = env.extras["log"]
    assert float(log["Metrics/hop_spring_energy_peak"]) > float(
        log["Metrics/hop_spring_energy_mean"]
    )


def test_energy_monitor_survives_missing_joints():
    """The Locked control arm has NO spring joints; mjlab's find_joints RAISES
    on no match rather than returning empty."""
    class _RaisingAsset(_JointAsset):
        def find_joints(self, name):
            raise ValueError("Not all regular expressions are matched!")

    env = _JointEnv([[0.005, 0.005]])
    env.scene._a = _RaisingAsset([[0.005, 0.005]])
    out = hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert float(out[0]) == 0.0
    assert float(env.extras["log"]["Metrics/hop_spring_energy_mean"]) == 0.0
