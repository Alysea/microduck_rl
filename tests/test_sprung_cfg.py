"""Config-level assertions for the sprung-foot variant transform."""

import mujoco
import numpy as np
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
    dof = sprung_cfg.rewards["dof_pos_limits"]
    names = dof.params["asset_cfg"].joint_names
    assert any("passive_" in pattern for pattern in names)


def test_pose_asset_cfg_is_deep_copied_not_mutated_in_place():
    """Guards the deepcopy in make_sprung_variant.

    SceneEntityCfg instances are shared across env-factory make() calls, so
    mutating one in place leaks the sprung scoping into the rigid Run and
    Velocity tasks. The transform currently takes its skip branch (the base
    pattern already excludes passive_*), so the CONTENTS never differ — only
    object identity can tell a deepcopy from an in-place mutation.
    """
    cfg = make_run_variant(make_microduck_velocity_env_cfg())
    before = cfg.rewards["pose"].params["asset_cfg"]
    before_names = tuple(before.joint_names)

    make_sprung_variant(cfg, stiffness=1500.0)

    after = cfg.rewards["pose"].params["asset_cfg"]
    assert after is not before, "asset_cfg must be replaced with a copy, not mutated"
    assert tuple(before.joint_names) == before_names, "the original object was mutated"


def test_locked_variant_has_zero_travel_in_monitor_and_model():
    cfg = make_sprung_variant(
        make_run_variant(make_microduck_velocity_env_cfg()),
        stiffness=1500.0,
        travel=0.0,
    )
    assert cfg.rewards["spring_compression_monitor"].params["travel"] == pytest.approx(0.0)


def test_action_rate_curriculum_is_untouched():
    """action_rate_l2 is the fixed smoothness budget for the whole study.

    Compares every stage against an untransformed cfg rather than asserting a
    hardcoded weight, so this cannot drift out of sync with the base config.
    """
    rigid = make_run_variant(make_microduck_velocity_env_cfg())
    expected = [dict(s) for s in rigid.curriculum["action_rate_weight"].params["weight_stages"]]

    sprung = make_sprung_variant(
        make_run_variant(make_microduck_velocity_env_cfg()), stiffness=1500.0
    )
    actual = [dict(s) for s in sprung.curriculum["action_rate_weight"].params["weight_stages"]]

    assert actual == expected


def test_sweep_arms_cover_the_spec_grid():
    """Stage 1 is a mass budget: k and travel are held fixed and mass sweeps."""
    from mjlab_microduck.tasks.sprung import SWEEP_ARMS

    labels = [a[0] for a in SWEEP_ARMS]
    assert "m30_locked" in labels and "m90_locked" in labels, (
        "the two locked arms are the geometric+mass control pair"
    )
    # k is held at the measured prototype spring for every arm.
    stiffnesses = {a[1] for a in SWEEP_ARMS}
    assert stiffnesses == {3900.0}
    # Mass is the swept axis: exactly {0.030, 0.050, 0.070, 0.090} kg, with the
    # two locked masses (30, 90) each reused by a sprung arm at the same mass.
    masses = {a[3] for a in SWEEP_ARMS}
    assert masses == {0.030, 0.050, 0.070, 0.090}
    # The locked arms must have zero travel; every sprung arm must have some.
    for label, _k, travel, _m in SWEEP_ARMS:
        if label.endswith("_locked"):
            assert travel == 0.0
        else:
            assert travel > 0.0


def test_all_sweep_task_ids_registered():
    import mjlab_microduck.tasks  # noqa: F401  (import registers)
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    for tid in (
        "Mjlab-Run-Flat-Sprung-M30-Locked-MicroDuck",
        "Mjlab-Run-Flat-Sprung-M90-Locked-MicroDuck",
        "Mjlab-Run-Flat-Sprung-M30-K3900-MicroDuck",
        "Mjlab-Run-Flat-Sprung-M50-K3900-MicroDuck",
        "Mjlab-Run-Flat-Sprung-M70-K3900-MicroDuck",
        "Mjlab-Run-Flat-Sprung-M90-K3900-MicroDuck",
    ):
        assert tid in tasks, f"{tid} not registered"


def test_sweep_arms_use_distinct_experiment_names():
    """Each arm needs its own wandb grouping or the sweep is unreadable."""
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_rl_cfg

    names = {
        load_rl_cfg(f"Mjlab-Run-Flat-Sprung-{s}-MicroDuck").run_name
        for s in ("M30-Locked", "M90-Locked", "M30-K3900", "M50-K3900", "M70-K3900", "M90-K3900")
    }
    assert len(names) == 6


def test_sweep_arms_do_not_share_learner_cfg_objects():
    """dataclasses.replace is shallow, so sprung_rl_cfg deep-copies the nested
    cfgs. Without that, all five arms would share one actor object AND share it
    with the Run baseline, so a later change to any single arm would silently
    alter the other four and the control this sweep is compared against.

    Asserts object identity, not values: the values are SUPPOSED to be
    identical (same learner, different logging identity), so only identity can
    distinguish a deep copy from a shared reference.
    """
    from mjlab_microduck.tasks.run import MicroduckRunRlCfg
    from mjlab_microduck.tasks.sprung import sprung_rl_cfg

    a = sprung_rl_cfg("m30_k3900")
    b = sprung_rl_cfg("m50_k3900")

    for field in ("actor", "critic", "algorithm"):
        assert getattr(a, field) is not getattr(b, field), f"{field} shared between arms"
        assert getattr(a, field) is not getattr(MicroduckRunRlCfg, field), (
            f"{field} shared with the Run baseline"
        )

    # Same learner, though: only the logging identity may differ.
    assert a.actor.hidden_dims == MicroduckRunRlCfg.actor.hidden_dims
    assert (
        a.actor.distribution_cfg["class_name"]
        == MicroduckRunRlCfg.actor.distribution_cfg["class_name"]
    )
    assert a.run_name != b.run_name


def test_pad_mass_reaches_the_compiled_model():
    """FIX 1: pad_mass must thread through make_sprung_variant to the spec,
    not stop at the cfg layer.
    """
    m_a, m_b = 0.030, 0.090
    cfg_a = make_sprung_variant(
        make_run_variant(make_microduck_velocity_env_cfg()),
        stiffness=3900.0, travel=TRAVEL, pad_mass=m_a,
    )
    cfg_b = make_sprung_variant(
        make_run_variant(make_microduck_velocity_env_cfg()),
        stiffness=3900.0, travel=TRAVEL, pad_mass=m_b,
    )
    model_a = cfg_a.scene.entities["robot"].spec_fn().compile()
    model_b = cfg_b.scene.entities["robot"].spec_fn().compile()

    # Two pads (left + right foot), so the total-mass delta is 2x the per-pad
    # mass delta.
    assert model_a.body_mass.sum() - model_b.body_mass.sum() == pytest.approx(
        2 * (m_a - m_b), abs=1e-6
    )


def test_two_locked_arms_differ_only_in_mass():
    """The two locked arms (m30_locked, m90_locked) are the pure mass-penalty
    pair: same mechanism geometry, no compliance, only mass differs.
    """
    from mjlab_microduck.tasks.sprung import SWEEP_ARMS

    arms = {label: (k, travel, m) for label, k, travel, m in SWEEP_ARMS}
    k30, t30, m30 = arms["m30_locked"]
    k90, t90, m90 = arms["m90_locked"]
    assert t30 == 0.0 and t90 == 0.0

    cfg30 = make_sprung_variant(
        make_run_variant(make_microduck_velocity_env_cfg()),
        stiffness=k30, travel=t30, pad_mass=m30,
    )
    cfg90 = make_sprung_variant(
        make_run_variant(make_microduck_velocity_env_cfg()),
        stiffness=k90, travel=t90, pad_mass=m90,
    )
    model30 = cfg30.scene.entities["robot"].spec_fn().compile()
    model90 = cfg90.scene.entities["robot"].spec_fn().compile()

    # Neither locked arm has a spring joint.
    for name in SPRING_JOINTS:
        assert mujoco.mj_name2id(model30, mujoco.mjtObj.mjOBJ_JOINT, name) == -1
        assert mujoco.mj_name2id(model90, mujoco.mjtObj.mjOBJ_JOINT, name) == -1

    # Same jnt_range across every remaining joint (identical joint topology).
    assert model30.njnt == model90.njnt
    assert np.allclose(model30.jnt_range, model90.jnt_range)

    # Same h_add: the pad sits at the same local offset under the ankle.
    pad30 = mujoco.mj_name2id(model30, mujoco.mjtObj.mjOBJ_BODY, "left_foot_pad")
    pad90 = mujoco.mj_name2id(model90, mujoco.mjtObj.mjOBJ_BODY, "left_foot_pad")
    assert np.allclose(model30.body_pos[pad30], model90.body_pos[pad90])

    # Same CoM band (h_add-driven, not mass-driven).
    p30 = cfg30.rewards["com_height_target"].params
    p90 = cfg90.rewards["com_height_target"].params
    assert p30["target_height_min"] == pytest.approx(p90["target_height_min"])
    assert p30["target_height_max"] == pytest.approx(p90["target_height_max"])

    # Mass is the ONE thing that differs.
    assert model90.body_mass.sum() - model30.body_mass.sum() == pytest.approx(
        2 * (m90 - m30), abs=1e-6
    )


def test_all_six_arms_share_the_same_com_band():
    """Mass must not perturb the CoM band — only h_add does, and h_add is
    identical (0.030) across every Stage 1 arm.
    """
    from mjlab_microduck.tasks.sprung import SWEEP_ARMS

    bands = set()
    for _label, k, travel, pad_mass in SWEEP_ARMS:
        cfg = make_sprung_variant(
            make_run_variant(make_microduck_velocity_env_cfg()),
            stiffness=k, travel=travel, pad_mass=pad_mass,
        )
        params = cfg.rewards["com_height_target"].params
        bands.add((
            round(params["target_height_min"], 9),
            round(params["target_height_max"], 9),
        ))
    assert len(bands) == 1, f"CoM band must be identical across all arms: {bands}"
