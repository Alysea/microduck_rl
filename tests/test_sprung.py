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


def test_reports_mean_and_p95_compression():
    env = _Env([[0.004, 0.010]])
    spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    log = env.extras["log"]
    assert abs(float(log["Metrics/spring_compression_mean"]) - 0.007) < 1e-6
    # torch.quantile interpolates linearly: 0.004 + 0.95 * 0.006.
    assert abs(float(log["Metrics/spring_compression_p95"]) - 0.0097) < 1e-6


def test_p95_ignores_a_single_bottomed_sample():
    """The reason _max was replaced.

    One sample pinned at the stop out of many must NOT drag the headline
    compression figure to `travel`; a max over num_envs x 2 samples did exactly
    that, which is why it carried no information.
    """
    q = [[0.003, 0.003] for _ in range(50)]
    q[0][0] = _TRAVEL
    env = _Env(q)
    spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    p95 = float(env.extras["log"]["Metrics/spring_compression_p95"])
    assert p95 < 0.005, p95


def test_loaded_mean_excludes_flight_samples():
    """`loaded_mean` is the series compared against the static-sag table.

    Half these samples are at q=0 (pad in flight). The all-steps mean is
    duty-diluted to 0.002; the loaded mean must report the true 0.004.
    """
    env = _Env([[0.004, 0.0], [0.004, 0.0]])
    spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    log = env.extras["log"]
    assert abs(float(log["Metrics/spring_compression_mean"]) - 0.002) < 1e-6
    assert abs(float(log["Metrics/spring_compression_loaded_mean"]) - 0.004) < 1e-6


def test_loaded_mean_is_zero_when_nothing_is_loaded():
    """Fully airborne (or a spring that never deflects): no loaded samples, so
    the mean is over an empty tensor. Must be 0.0, not NaN — a NaN here would
    poison the wandb series it is meant to diagnose.
    """
    env = _Env([[0.0, 0.0]])
    spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    loaded = env.extras["log"]["Metrics/spring_compression_loaded_mean"]
    assert torch.isfinite(loaded)
    assert float(loaded) == 0.0


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


def test_bottomed_fraction_tracks_the_threshold_magnitude():
    """Just above the threshold counts; just below does not.

    Guards the threshold VALUE (a regression comparing against `travel`
    instead of `bottom_out_frac * travel` would move it by 5%). Deliberately
    straddles rather than testing exact equality: q is float32 and the
    threshold is a Python double, so exact-boundary comparison is unreliable,
    and a compression landing exactly on it has measure zero anyway.
    """
    frac = 0.95
    threshold = frac * _TRAVEL          # 0.01425 m

    env_above = _Env([[threshold * 1.001, 0.0]])
    spring_compression_monitor(env_above, joint_names=_JOINTS, travel=_TRAVEL,
                               bottom_out_frac=frac)
    assert abs(float(env_above.extras["log"]["Metrics/spring_bottomed_fraction"]) - 0.5) < 1e-6

    env_below = _Env([[threshold * 0.999, 0.0]])
    spring_compression_monitor(env_below, joint_names=_JOINTS, travel=_TRAVEL,
                               bottom_out_frac=frac)
    assert float(env_below.extras["log"]["Metrics/spring_bottomed_fraction"]) == 0.0


def test_zero_travel_locked_variant_does_not_divide_by_zero():
    env = _Env([[0.0, 0.0]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=0.0)
    assert float(out[0]) == 0.0
    assert torch.isfinite(env.extras["log"]["Metrics/spring_bottomed_fraction"])


def test_locked_arm_logs_all_four_series_as_zeros():
    """The locked control arm must LOG zeros, not omit the series.

    It has no spring joints at all, so the old code returned before reaching the
    logging block and the arm produced no `spring_*` series whatsoever. That
    made "metric absent" a normal condition and destroyed the diagnostic value
    of absence: with this test, an absent series means a bug.

    Uses an asset whose joint lookup RAISES, i.e. the real locked-arm model, to
    prove the zeros are logged before any lookup is attempted.
    """
    class _RaisingAsset(_Asset):
        def find_joints(self, name):
            raise AssertionError("must short-circuit before the joint lookup")

    env = _Env([[0.0, 0.0]])
    env.scene._a = _RaisingAsset([[0.0, 0.0]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=0.0)
    assert float(out[0]) == 0.0
    log = env.extras["log"]
    for key in (
        "Metrics/spring_compression_mean",
        "Metrics/spring_compression_loaded_mean",
        "Metrics/spring_compression_p95",
        "Metrics/spring_bottomed_fraction",
    ):
        assert key in log, f"{key} missing on the locked arm"
        assert float(log[key]) == 0.0


def test_nan_safe():
    env = _Env([[float("nan"), 0.006]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0
    assert torch.isfinite(env.extras["log"]["Metrics/spring_compression_mean"])
    assert torch.isfinite(env.extras["log"]["Metrics/spring_compression_p95"])
    assert torch.isfinite(env.extras["log"]["Metrics/spring_compression_loaded_mean"])


def test_missing_joints_return_zeros_without_raising():
    """mjlab's find_joints RAISES ValueError when a name matches nothing rather
    than returning empty.

    No longer the locked arm's production path — travel=0.0 short-circuits
    before the lookup (see test_locked_arm_logs_all_four_series_as_zeros). This
    now guards a sprung arm whose joints were renamed: report zeros, do not
    crash a run.
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


def test_negative_q_is_treated_as_zero_compression():
    """Preload pushes the pad past its lower limit when unloaded.

    MuJoCo's soft joint limit lets a preloaded spring settle slightly below
    q=0 (measured -0.59 mm at k=3900). Those samples are limit PENETRATION,
    not compression. Unclamped they cancel positive stance samples and drive
    `spring_compression_mean` to ~0 while p95 and bottomed_fraction stay
    non-zero — the contradictory reading that surfaced in a smoke run.
    """
    # left foot in flight (resting past the limit), right foot loaded in stance
    env = _Env([[-0.00059, 0.008]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    log = env.extras["log"]

    assert float(out[0]) == 0.0
    # mean of (0.0, 0.008) — NOT (-0.00059 + 0.008)/2
    assert abs(float(log["Metrics/spring_compression_mean"]) - 0.004) < 1e-6
    # the clamped sample must not count as loaded
    assert abs(float(log["Metrics/spring_compression_loaded_mean"]) - 0.008) < 1e-6
    assert float(log["Metrics/spring_compression_p95"]) > 0.0


def test_all_negative_q_reads_as_zero_not_negative():
    """A fully airborne robot must report zero compression, never negative."""
    env = _Env([[-0.00059, -0.00061]])
    spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    log = env.extras["log"]
    assert float(log["Metrics/spring_compression_mean"]) == 0.0
    assert float(log["Metrics/spring_compression_loaded_mean"]) == 0.0
    assert float(log["Metrics/spring_bottomed_fraction"]) == 0.0
