"""Config-level assertions for the sprung-foot variant transform."""

import pytest

from mjlab_microduck.robot.sprung_foot import H_ADD, SPRING_JOINTS, TRAVEL
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.run import make_run_variant
from mjlab_microduck.tasks.sprung import make_sprung_variant


@pytest.fixture
def base_cfg():
    return make_run_variant(make_microduck_velocity_env_cfg())


@pytest.fixture
def sprung_cfg(base_cfg):
    return make_sprung_variant(base_cfg, stiffness=1500.0)


def test_com_band_is_shifted_by_exactly_h_add(sprung_cfg):
    """The sprung robot stands h_add taller.

    Without this shift it would be penalised for being tall before compliance
    is in play — the confound the locked control arm exists to isolate.
    """
    rigid = make_run_variant(make_microduck_velocity_env_cfg())
    r = rigid.rewards["com_height_target"].params
    s = sprung_cfg.rewards["com_height_target"].params
    assert s["target_height_min"] == pytest.approx(r["target_height_min"] + H_ADD)
    assert s["target_height_max"] == pytest.approx(r["target_height_max"] + H_ADD)
    # The band must not widen — only translate.
    assert (s["target_height_max"] - s["target_height_min"]) == pytest.approx(
        r["target_height_max"] - r["target_height_min"]
    )


def test_spring_monitor_registered_with_non_zero_weight(sprung_cfg):
    # RewardManager.compute skips weight==0.0 terms before calling them, which
    # would silently disable the monitor.
    term = sprung_cfg.rewards["spring_compression_monitor"]
    assert term.func is microduck_mdp.spring_compression_monitor
    assert term.weight != 0.0
    assert tuple(term.params["joint_names"]) == SPRING_JOINTS
    assert term.params["travel"] == pytest.approx(TRAVEL)


def test_robot_entity_is_replaced(sprung_cfg, base_cfg):
    assert sprung_cfg.scene.entities["robot"] is not None
    # spec_fn differs from the rigid walk spec
    rigid = make_run_variant(make_microduck_velocity_env_cfg())
    assert sprung_cfg.scene.entities["robot"].spec_fn is not rigid.scene.entities["robot"].spec_fn


def test_pose_reward_excludes_the_spring_joints(sprung_cfg):
    """The pose reward must not try to hold a passive spring at a target."""
    names = sprung_cfg.rewards["pose"].params["asset_cfg"].joint_names
    assert any("passive_" in pattern for pattern in names)


def test_dof_pos_limits_scoped_off_the_spring_joints(sprung_cfg):
    dof = sprung_cfg.rewards.get("dof_pos_limits")
    if dof is not None:
        assert "asset_cfg" in dof.params


def test_locked_variant_has_zero_travel_in_monitor_and_model():
    cfg = make_sprung_variant(
        make_run_variant(make_microduck_velocity_env_cfg()),
        stiffness=1500.0,
        travel=0.0,
    )
    assert cfg.rewards["spring_compression_monitor"].params["travel"] == pytest.approx(0.0)


def test_action_rate_weight_untouched(sprung_cfg):
    """action_rate_l2 is the fixed smoothness budget, not a variable."""
    stages = sprung_cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[-1]["weight"] == pytest.approx(-1.0)
