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
    # feet_air_time_capped is deliberately parameter-compatible with mjlab's
    # stock feet_air_time so that swapping `.func` PRESERVES the term's existing
    # params. Pin the two params that only survive because of that: if a future
    # edit rebuilt the dict instead of mutating it, `command_name` would fall
    # back to its None default, the speed gate would vanish, and the term would
    # silently reward air time at zero command. Weight is pinned too — the whole
    # point of capping is that a weight of 5.0 no longer pays double for
    # simultaneous two-foot flight.
    assert air.params["command_name"] == "twist"
    assert air.params.get("command_threshold") is not None
    assert air.weight == 5.0


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


def test_forward_speed_monitor_weight_is_non_zero(run_cfg):
    # Same short-circuit as the action monitor: RewardManager.compute skips
    # terms with weight == 0.0 before calling the function, so the plateau
    # metric would never be logged.
    term = run_cfg.rewards["forward_speed_monitor"]
    assert term.func is microduck_mdp.forward_speed_monitor
    assert term.weight != 0.0


def test_curriculum_ramps_forward_speed_only(run_cfg):
    # velocity_command_ranges_curriculum defaults to forward_only=False and
    # update_lin_vel_y=True, which would ramp BACKWARD and LATERAL speed
    # alongside forward speed (the last stage would set lin_vel_x=(-1.5, 1.5)
    # AND lin_vel_y=(-1.5, 1.5)). Forward speed must be the only moving
    # variable, otherwise the plateau measurement is an isotropic xy error
    # rather than a forward-speed number — and the sprung comparison is made
    # against that number.
    params = run_cfg.curriculum["velocity_command_ranges"].params
    assert params["forward_only"] is True
    assert params["update_lin_vel_y"] is False


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


def test_run_rl_cfg_does_not_share_nested_cfgs_with_velocity():
    # `dataclasses.replace` is SHALLOW: without an explicit deep copy,
    # MicroduckRunRlCfg.actor would be the *same object* as MicroduckRlCfg.actor,
    # and the Phase 3 escape hatch (swapping the policy distribution via
    # actor.distribution_cfg["class_name"]) would silently mutate the Velocity
    # task too — destroying the experimental control this baseline provides.
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
    from mjlab_microduck.tasks.run import MicroduckRunRlCfg

    assert MicroduckRunRlCfg.actor is not MicroduckRlCfg.actor
    assert MicroduckRunRlCfg.critic is not MicroduckRlCfg.critic
    assert MicroduckRunRlCfg.algorithm is not MicroduckRlCfg.algorithm
    # The mutable dict inside the actor must be distinct too, or the deep copy
    # bought nothing for the exact field Phase 3 wants to change.
    assert (
        MicroduckRunRlCfg.actor.distribution_cfg
        is not MicroduckRlCfg.actor.distribution_cfg
    )


def test_run_rl_cfg_hyperparameters_match_velocity():
    # Distinct objects, identical values — Phase 1 changes the task, not the
    # learner.
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
    from mjlab_microduck.tasks.run import MicroduckRunRlCfg

    for name in ("actor", "critic"):
        run_model = getattr(MicroduckRunRlCfg, name)
        vel_model = getattr(MicroduckRlCfg, name)
        assert run_model.hidden_dims == vel_model.hidden_dims
        assert run_model.activation == vel_model.activation
        assert run_model.obs_normalization == vel_model.obs_normalization

    assert (
        MicroduckRunRlCfg.actor.distribution_cfg["class_name"]
        == MicroduckRlCfg.actor.distribution_cfg["class_name"]
        == "GaussianDistribution"
    )
    assert MicroduckRunRlCfg.algorithm == MicroduckRlCfg.algorithm


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


def test_rough_run_task_actually_has_rough_terrain():
    # A copy-paste slip in the registration block (passing the flat cfg to the
    # Rough task id) would silently train the wrong experiment. Assert on the
    # three fields that genuinely differ between the two registered cfgs:
    # make_microduck_velocity_env_cfg switches terrain_type plane<->generator,
    # sets/clears terrain_generator, and deletes the terrain_levels curriculum
    # only in the flat branch.
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    flat = load_env_cfg("Mjlab-Run-Flat-MicroDuck")
    rough = load_env_cfg("Mjlab-Run-Rough-MicroDuck")

    assert flat.scene.terrain.terrain_type == "plane"
    assert rough.scene.terrain.terrain_type == "generator"
    assert flat.scene.terrain.terrain_generator is None
    assert rough.scene.terrain.terrain_generator is not None
    assert "terrain_levels" not in flat.curriculum
    assert "terrain_levels" in rough.curriculum
