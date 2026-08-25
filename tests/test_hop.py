"""Unit tests for the ported hop reward terms (duck-typed fakes)."""

import torch

from mjlab_microduck.tasks.mdp import (
    com_height_target_recovery_only,
    hop_body_height,
    hop_both_feet_airborne,
    hop_energy_monitor,
    hop_load_force,
    hop_upward_velocity,
)

_SENSOR = "feet_ground_contact"
_CMD = "twist"


class _SensorData:
    def __init__(self, found, force=None):
        self.found = torch.tensor(found, dtype=torch.float32)
        # [B, N, 3]. `feet_ground_contact` is reduce="netforce", so this is the
        # summed contact force per foot geom in the GLOBAL frame; MuJoCo reports
        # it pointing DOWN for a loaded foot (probed: -4.905 N under a 4.905 N
        # weight), which is why the fixtures below use negative z.
        self.force = None if force is None else torch.tensor(force, dtype=torch.float32)


class _Sensor:
    def __init__(self, found, force=None):
        self.data = _SensorData(found, force)


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


class _Terrain:
    """Flat ground: `com_height_target` subtracts env_origins[:, 2] from world z."""

    def __init__(self, n):
        self.env_origins = torch.zeros((n, 3), dtype=torch.float32)


class _Scene:
    def __init__(self, sensors, asset, n=1):
        self.sensors = sensors
        self._asset = asset
        self.terrain = _Terrain(n)

    def __getitem__(self, _k):
        return self._asset


class _Env:
    """found: per-foot contact flags; cmd: [cos, sin, 0]; vz/z: base state."""

    def __init__(self, found=((0.0, 0.0),), cmd=((0.0, 1.0, 0.0),),
                 vz=(0.0,), z=(0.15,), force=None):
        self.scene = _Scene({_SENSOR: _Sensor(found, force)}, _Asset(vz, z), len(found))
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


def test_body_height_is_zero_at_the_target_with_both_feet_in_contact():
    """The airborne gate. HOME_FRAME is a parallelogram crouch, so simply
    STRAIGHTENING THE LEGS raises the trunk ~9 mm with both feet still planted.
    Ungated, that ground-level bob — extend while sin > 0, crouch while sin < 0,
    never leave the ground — collects most of the peak reward and is entirely
    spring-irrelevant, which is exactly the confound this experiment exists to
    avoid. Same height as the passing case below; only the contact differs."""
    env = _Env(found=[[1.0, 1.0]], cmd=[[0.0, 1.0, 0.0]], z=[0.165])
    out = hop_body_height(env, command_name=_CMD, target_height=0.165, std=0.008)
    assert float(out[0]) == 0.0


def test_body_height_is_paid_at_the_same_height_when_both_feet_are_airborne():
    env = _Env(found=[[0.0, 0.0]], cmd=[[0.0, 1.0, 0.0]], z=[0.165])
    out = hop_body_height(env, command_name=_CMD, target_height=0.165, std=0.008)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_body_height_is_zero_with_only_one_foot_airborne():
    """A single-foot lift is a step, not a hop."""
    env = _Env(found=[[0.0, 1.0]], cmd=[[0.0, 1.0, 0.0]], z=[0.165])
    out = hop_body_height(env, command_name=_CMD, target_height=0.165, std=0.008)
    assert float(out[0]) == 0.0


def test_body_height_and_airborne_reward_read_the_same_predicate():
    """The gate must be the SAME predicate hop_both_feet_airborne pays for, or
    the policy can be paid for an apex the other term does not call a hop.
    Sweep every contact combination and require the two to agree on zero/non-zero."""
    for found in ([[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]):
        env = _Env(found=found, cmd=[[0.0, 1.0, 0.0]], z=[0.165])
        airborne = float(hop_both_feet_airborne(env, sensor_name=_SENSOR, command_name=_CMD)[0])
        height = float(hop_body_height(env, command_name=_CMD, target_height=0.165, std=0.008)[0])
        assert (airborne > 0.0) == (height > 0.0), found


def test_body_height_treats_a_nan_contact_read_as_in_contact():
    """Never pay for flight we cannot actually see."""
    env = _Env(found=[[float("nan"), float("nan")]], cmd=[[0.0, 1.0, 0.0]], z=[0.165])
    out = hop_body_height(env, command_name=_CMD, target_height=0.165, std=0.008)
    assert float(out[0]) == 0.0


def test_body_height_is_zero_without_the_contact_sensor():
    env = _Env(found=[[0.0, 0.0]], cmd=[[0.0, 1.0, 0.0]], z=[0.165])
    env.scene.sensors = {}
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


# --- hop_load_force ----------------------------------------------------------
#
# The load half was entirely unrewarded: all three terms above gate on
# sin(2*pi*phi) > 0. Without a load-phase signal there is no actuator
# countermovement, and without a countermovement the spring cannot be charged
# (static sag under body weight alone is 0.48 mm at k=3900, ~0.45 mJ, worth
# 0.1 mm of lift) -- so the spring needs a hop to charge and the hop needs a
# charged spring.

_BW = 8.60  # N -- 0.877 kg * 9.81, matching hop.BODY_WEIGHT_N.


def _force(total_n, feet=2):
    """One env, `total_n` newtons of vertical GRF split evenly over `feet` feet.

    Negative z: MuJoCo's reduce="netforce" contact force points DOWN for a foot
    bearing load (probed at -4.905 N under a 4.905 N weight).
    """
    per = -total_n / feet
    return [[[0.0, 0.0, per] for _ in range(feet)]]


def test_load_force_pays_nothing_during_the_launch_half():
    """The gate. sin > 0 is launch -- pressing into the ground there is the
    opposite of what we want, and paying for it would reinstate a reward for
    simply standing on the ground under load for half the cycle."""
    env = _Env(cmd=[[0.0, 1.0, 0.0]], force=_force(10 * _BW))
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert float(out[0]) == 0.0


def test_load_force_pays_during_the_load_half():
    env = _Env(cmd=[[0.0, -1.0, 0.0]], force=_force(1.5 * _BW))
    out = hop_load_force(
        env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW, max_ratio=2.0
    )
    # (1.5 - 1) / (2.0 - 1) = 0.5, times a load gate of 1.0.
    assert abs(float(out[0]) - 0.5) < 1e-5


def test_load_force_is_zero_at_exactly_body_weight():
    """Merely standing must earn nothing -- standing still is the failure mode
    this whole rebalance exists to defeat."""
    env = _Env(cmd=[[0.0, -1.0, 0.0]], force=_force(_BW))
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert abs(float(out[0])) < 1e-6


def test_load_force_is_zero_when_unloaded():
    """Airborne, or hanging: below body weight is still nothing, not negative."""
    env = _Env(cmd=[[0.0, -1.0, 0.0]], force=_force(0.0))
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert float(out[0]) == 0.0


def test_load_force_saturates_at_max_ratio_body_weight():
    env = _Env(cmd=[[0.0, -1.0, 0.0]], force=_force(2.0 * _BW))
    out = hop_load_force(
        env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW, max_ratio=2.0
    )
    assert abs(float(out[0]) - 1.0) < 1e-5


def test_load_force_does_not_exceed_one_above_saturation():
    env = _Env(cmd=[[0.0, -1.0, 0.0]], force=_force(20.0 * _BW))
    out = hop_load_force(
        env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW, max_ratio=2.0
    )
    assert abs(float(out[0]) - 1.0) < 1e-5


def test_load_force_sums_over_both_feet():
    """A one-footed press at body weight is not a two-footed press at body
    weight; the term reads TOTAL vertical GRF, so both feet contribute."""
    both = _Env(cmd=[[0.0, -1.0, 0.0]], force=[[[0.0, 0.0, -_BW], [0.0, 0.0, -_BW]]])
    one = _Env(cmd=[[0.0, -1.0, 0.0]], force=[[[0.0, 0.0, -_BW], [0.0, 0.0, 0.0]]])
    f = dict(sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW, max_ratio=2.0)
    assert abs(float(hop_load_force(both, **f)[0]) - 1.0) < 1e-5
    assert abs(float(hop_load_force(one, **f)[0])) < 1e-6


def test_load_force_ignores_horizontal_shear():
    """Vertical component, not the 3-vector norm. mjlab's `soft_landing` (which
    logs Metrics/landing_force_mean off this same field) takes the norm, which is
    right for an impact magnitude but wrong here: a foot scrubbing sideways would
    otherwise score as loading."""
    env = _Env(cmd=[[0.0, -1.0, 0.0]], force=[[[50.0, 50.0, -_BW / 2], [0.0, 0.0, -_BW / 2]]])
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert abs(float(out[0])) < 1e-6


def test_load_force_is_nan_safe():
    env = _Env(cmd=[[0.0, -1.0, 0.0]], force=_force(float("nan")))
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert torch.isfinite(out).all()
    assert float(out[0]) == 0.0

    nan_cmd = _Env(cmd=[[0.0, float("nan"), 0.0]], force=_force(5 * _BW))
    out = hop_load_force(nan_cmd, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert torch.isfinite(out).all()


def test_load_force_is_zero_without_the_contact_sensor():
    env = _Env(cmd=[[0.0, -1.0, 0.0]], force=_force(5 * _BW))
    env.scene.sensors = {}
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_load_force_is_zero_when_the_sensor_carries_no_force_field():
    """A sensor configured without fields=("force",) reports None. Fail silent
    rather than crash a 8000-iteration run, but never invent a load."""
    env = _Env(cmd=[[0.0, -1.0, 0.0]], force=None)
    out = hop_load_force(env, sensor_name=_SENSOR, command_name=_CMD, body_weight_n=_BW)
    assert float(out[0]) == 0.0


# --- com_height_target_recovery_only -----------------------------------------


def test_com_height_recovery_only_is_zero_during_the_launch_half():
    """The gate. `com_height_target` pays a flat +1 anywhere in band (x1.2), the
    single largest reward for standing perfectly still. During launch we want the
    robot LEAVING the band, so the term must say nothing there."""
    env = _Env(cmd=[[0.0, 1.0, 0.0]], z=[0.15])
    out = com_height_target_recovery_only(
        env, command_name=_CMD, target_height_min=0.14, target_height_max=0.20
    )
    assert float(out[0]) == 0.0


def test_com_height_recovery_only_pays_in_band_during_recovery():
    env = _Env(cmd=[[0.0, -1.0, 0.0]], z=[0.15])
    out = com_height_target_recovery_only(
        env, command_name=_CMD, target_height_min=0.14, target_height_max=0.20
    )
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_com_height_recovery_only_scales_with_the_gate():
    """Partway through the recovery half the gate is |sin| < 1, not 0 or 1."""
    env = _Env(cmd=[[0.0, -0.5, 0.0]], z=[0.15])
    out = com_height_target_recovery_only(
        env, command_name=_CMD, target_height_min=0.14, target_height_max=0.20
    )
    assert abs(float(out[0]) - 0.5) < 1e-6


def test_com_height_recovery_only_matches_the_wrapped_term_at_full_recovery():
    """It must be the EXISTING reward, gated -- not a reimplementation of it.
    Checked out of band too, where the term is a negative quadratic."""
    from mjlab_microduck.tasks.mdp import com_height_target

    for z in (0.10, 0.15, 0.30):
        env = _Env(cmd=[[0.0, -1.0, 0.0]], z=[z])
        gated = float(
            com_height_target_recovery_only(
                env, command_name=_CMD, target_height_min=0.14, target_height_max=0.20
            )[0]
        )
        plain = float(
            com_height_target(env, target_height_min=0.14, target_height_max=0.20)[0]
        )
        assert abs(gated - plain) < 1e-6, z


def test_com_height_recovery_only_is_nan_safe():
    env = _Env(cmd=[[0.0, float("nan"), 0.0]], z=[float("nan")])
    out = com_height_target_recovery_only(
        env, command_name=_CMD, target_height_min=0.14, target_height_max=0.20
    )
    assert torch.isfinite(out).all()
