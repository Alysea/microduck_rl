"""Config-level assertions for the Run variant transform."""

import pytest

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.run import (
    AIR_TIME_WINDOW,
    RUNNING_THRESHOLD,
    STD_RUNNING,
    VELOCITY_STAGES,
    make_run_variant,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


@pytest.fixture
def run_cfg():
    return make_run_variant(make_microduck_velocity_env_cfg())


def test_running_regime_is_reachable(run_cfg):
    # variable_posture gates on |lin| + |ang| and defaults running_threshold to
    # 1.5, which the stock command ranges can only hit with both maxed. The Run
    # task must set it below what the curriculum actually reaches.
    pose = run_cfg.rewards["pose"]
    assert pose.params["running_threshold"] == RUNNING_THRESHOLD
    max_reachable = VELOCITY_STAGES[-1]["lin_vel_range"]
    assert RUNNING_THRESHOLD < max_reachable


def test_std_running_is_not_aliased_to_std_walking(run_cfg):
    pose = run_cfg.rewards["pose"]
    assert pose.params["std_running"] is not pose.params["std_walking"]
    assert pose.params["std_running"] != pose.params["std_walking"]


def test_hip_roll_tolerance_unchanged_in_running(run_cfg):
    # Loosening hip_roll is what produced leg splay; it must stay tight.
    pose = run_cfg.rewards["pose"]
    assert STD_RUNNING[r".*hip_roll.*"] == pose.params["std_walking"][r".*hip_roll.*"]


def test_air_time_uses_capped_function(run_cfg):
    air = run_cfg.rewards["air_time"]
    assert air.func is microduck_mdp.feet_air_time_capped
    assert air.params["threshold_min"] == AIR_TIME_WINDOW[0]
    assert air.params["threshold_max"] == AIR_TIME_WINDOW[1]


def test_alternating_flight_registered(run_cfg):
    term = run_cfg.rewards["alternating_flight"]
    assert term.func is microduck_mdp.alternating_flight
    assert term.weight > 0.0
    assert term.params["command_name"] == "twist"


def test_action_monitor_weight_is_non_zero(run_cfg):
    # RewardManager.compute skips terms with weight == 0.0 before calling the
    # function, which would silently disable the monitor.
    term = run_cfg.rewards["action_magnitude_monitor"]
    assert term.func is microduck_mdp.action_magnitude_monitor
    assert term.weight != 0.0


def test_velocity_stages_are_monotonic(run_cfg):
    stages = run_cfg.curriculum["velocity_command_ranges"].params["velocity_stages"]
    steps = [s["step"] for s in stages]
    lins = [s["lin_vel_range"] for s in stages]
    assert steps == sorted(steps)
    assert lins == sorted(lins)
    assert len(stages) > 1


def test_angular_range_held_constant(run_cfg):
    # Forward speed must be the only moving variable in the curriculum.
    stages = run_cfg.curriculum["velocity_command_ranges"].params["velocity_stages"]
    angs = {s["ang_vel_range"] for s in stages}
    assert len(angs) == 1


def test_run_rl_cfg_has_its_own_experiment_name():
    # Baseline and sprung runs must not share a wandb grouping.
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
    from mjlab_microduck.tasks.run import MicroduckRunRlCfg

    assert MicroduckRunRlCfg.experiment_name != MicroduckRlCfg.experiment_name
    assert MicroduckRunRlCfg.run_name != MicroduckRlCfg.run_name


def test_run_rl_cfg_keeps_the_plain_gaussian_policy():
    # Phase 1 deliberately does NOT change the distribution; the baseline stays
    # as close to the working velocity config as possible.
    from mjlab_microduck.tasks.run import MicroduckRunRlCfg

    assert (
        MicroduckRunRlCfg.actor.distribution_cfg["class_name"]
        == "GaussianDistribution"
    )
    assert MicroduckRunRlCfg.actor.obs_normalization is True
    assert MicroduckRunRlCfg.critic.obs_normalization is True


def test_run_tasks_are_registered():
    import mjlab_microduck.tasks  # noqa: F401  (import registers)
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    assert "Mjlab-Run-Flat-MicroDuck" in tasks
    assert "Mjlab-Run-Rough-MicroDuck" in tasks


def test_run_task_rl_cfg_round_trips_through_the_registry():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_rl_cfg

    assert load_rl_cfg("Mjlab-Run-Flat-MicroDuck").experiment_name == "run"
