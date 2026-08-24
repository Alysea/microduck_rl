"""Config-level assertions for the hop variant transform."""

import pytest

from mjlab_microduck.robot.sprung_foot import H_ADD
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.hop import (
    HOP_HEIGHT_GAIN,
    HOP_PERIOD,
    RIGID_STAND_HEIGHT,
    hop_target_height,
    make_hop_variant,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


@pytest.fixture
def hop_cfg():
    return make_hop_variant(make_microduck_velocity_env_cfg(), h_add=H_ADD)


def test_command_is_the_cyclic_phase_command(hop_cfg):
    term = hop_cfg.commands["twist"]
    assert term.class_type is microduck_mdp.GroundPickPhaseCommand
    assert term.period == pytest.approx(HOP_PERIOD)


def test_height_target_is_shifted_by_h_add(hop_cfg):
    """The ported reward hardcodes 0.135 for the RIGID robot. The sprung robot
    stands H_ADD taller, so an unshifted target asks it to CROUCH."""
    expected = RIGID_STAND_HEIGHT + HOP_HEIGHT_GAIN + H_ADD
    assert hop_cfg.rewards["hop_body_height"].params["target_height"] == pytest.approx(expected)
    assert hop_target_height(H_ADD) == pytest.approx(expected)


def test_rigid_variant_target_is_not_shifted():
    rigid = make_hop_variant(make_microduck_velocity_env_cfg(), h_add=0.0)
    expected = RIGID_STAND_HEIGHT + HOP_HEIGHT_GAIN
    assert rigid.rewards["hop_body_height"].params["target_height"] == pytest.approx(expected)


def test_all_three_hop_rewards_registered_with_positive_weight(hop_cfg):
    for name, func in (
        ("hop_both_feet_airborne", microduck_mdp.hop_both_feet_airborne),
        ("hop_upward_velocity", microduck_mdp.hop_upward_velocity),
        ("hop_body_height", microduck_mdp.hop_body_height),
    ):
        term = hop_cfg.rewards[name]
        assert term.func is func
        assert term.weight > 0.0


def test_energy_monitor_registered_with_non_zero_weight(hop_cfg):
    # RewardManager.compute skips weight==0.0 terms before calling them.
    term = hop_cfg.rewards["hop_energy_monitor"]
    assert term.func is microduck_mdp.hop_energy_monitor
    assert term.weight != 0.0


def test_energy_monitor_stiffness_threads_through_the_variant():
    """make_hop_variant's default (3900.0) is only correct for the k3900 arm.
    Task 4 also registers a k2500 arm; without this parameter threading through,
    hop_energy_monitor would report that arm's stored spring energy 56% high."""
    cfg = make_hop_variant(make_microduck_velocity_env_cfg(), h_add=H_ADD, stiffness=2500.0)
    assert cfg.rewards["hop_energy_monitor"].params["stiffness"] == 2500.0


def test_command_construction_preserves_base_fields():
    """The command-construction step filters vars(command) down to the fields
    GroundPickPhaseCommandCfg declares, then rebuilds it. A regression that
    dropped or mangled behaviour-carrying fields (ranges, rel_standing_envs,
    viz) would not show up as a TypeError -- it would silently ship a policy
    that ignores its velocity-sampling ranges. Assert the carry-over directly."""
    rigid = make_microduck_velocity_env_cfg()
    original = rigid.commands["twist"]
    hop_cfg = make_hop_variant(rigid, h_add=H_ADD)
    rebuilt = hop_cfg.commands["twist"]

    assert rebuilt.ranges == original.ranges
    assert rebuilt.rel_standing_envs == original.rel_standing_envs
    assert rebuilt.rel_heading_envs == original.rel_heading_envs
    assert rebuilt.viz.z_offset == original.viz.z_offset
    assert rebuilt.entity_name == original.entity_name


def test_forward_locomotion_rewards_removed(hop_cfg):
    """This is a hop in place. Leaving velocity tracking in would reward the
    policy for running away instead of hopping, and the phase command has
    overwritten the twist command those terms read."""
    for name in ("track_linear_velocity", "track_angular_velocity"):
        assert name not in hop_cfg.rewards


def test_action_rate_curriculum_untouched(hop_cfg):
    rigid = make_microduck_velocity_env_cfg()
    expected = [dict(s) for s in rigid.curriculum["action_rate_weight"].params["weight_stages"]]
    actual = [dict(s) for s in hop_cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert actual == expected


def test_hop_period_is_above_the_spring_mass_period():
    """At k=3900 and 0.877 kg the spring-mass period is ~94 ms. A hop period at
    or below that would drive the spring faster than it can cycle."""
    assert HOP_PERIOD > 0.094


def test_hop_arms_are_locked_plus_two_stiffnesses():
    from mjlab_microduck.tasks.hop import HOP_ARMS

    labels = [a[0] for a in HOP_ARMS]
    assert "locked" in labels, "the locked arm is the geometric control"
    stiffnesses = {a[1] for a in HOP_ARMS if a[0] != "locked"}
    assert stiffnesses == {2500.0, 3900.0}
    for label, _k, travel, pad in HOP_ARMS:
        assert pad == pytest.approx(0.070), "mass is held; Stage 1 measured it"
        if label == "locked":
            assert travel == 0.0
        else:
            assert travel > 0.0


def test_hop_task_ids_registered():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    for tid in (
        "Mjlab-Hop-Flat-Sprung-Locked-MicroDuck",
        "Mjlab-Hop-Flat-Sprung-K2500-MicroDuck",
        "Mjlab-Hop-Flat-Sprung-K3900-MicroDuck",
    ):
        assert tid in tasks, f"{tid} not registered"


def test_hop_arms_have_distinct_wandb_identities():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_rl_cfg

    names = {
        load_rl_cfg(f"Mjlab-Hop-Flat-Sprung-{s}-MicroDuck").run_name
        for s in ("Locked", "K2500", "K3900")
    }
    assert len(names) == 3


def test_registered_hop_cfgs_carry_the_shifted_height_target():
    """End-to-end: the registered task, not just the transform in isolation."""
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Hop-Flat-Sprung-K3900-MicroDuck")
    assert cfg.rewards["hop_body_height"].params["target_height"] == pytest.approx(
        hop_target_height(H_ADD)
    )


def test_registered_hop_cfgs_carry_their_own_arm_stiffness():
    """End-to-end, through the registered task -- not the transform in
    isolation. The registration loop must thread stiffness=_k into
    make_hop_variant for every arm, not only into make_sprung_variant.
    make_hop_variant's default stiffness (3900.0) is only correct for the
    k3900/locked arms -- if the k2500 arm's registration omitted it,
    hop_energy_monitor would report that arm's stored spring energy 56% high,
    and the spec requires reading the spring instruments
    (hop_spring_energy_*, spring_bottomed_fraction) BEFORE any hop-height
    number, so a wrong energy metric would corrupt the primary result this
    whole phase exists to produce. Loads all three registered tasks so a
    regression in any one arm's call site is caught."""
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks.hop import HOP_ARM_SUFFIX, HOP_ARMS

    for label, k, _travel, _pad in HOP_ARMS:
        tid = f"Mjlab-Hop-Flat-Sprung-{HOP_ARM_SUFFIX[label]}-MicroDuck"
        cfg = load_env_cfg(tid)
        assert cfg.rewards["hop_energy_monitor"].params["stiffness"] == k, tid
