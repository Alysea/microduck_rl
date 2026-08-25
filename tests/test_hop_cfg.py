"""Config-level assertions for the hop variant transform."""

import math

import pytest

from mjlab_microduck.robot.sprung_foot import H_ADD, PAD_MASS
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.hop import (
    AIRBORNE_WEIGHT,
    BODY_HEIGHT_WEIGHT,
    BODY_WEIGHT_N,
    HOP_ARM_SUFFIX,
    HOP_COM_HEIGHT_MAX,
    HOP_HEIGHT_GAIN,
    HOP_HEIGHT_STD,
    HOP_MAX_LAUNCH_VEL,
    HOP_PERIOD,
    LOAD_FORCE_MAX_RATIO,
    LOAD_FORCE_WEIGHT,
    RIGID_STAND_HEIGHT,
    SENSOR_NAME,
    UNLOADED_RIGID_HEIGHT,
    UPWARD_VELOCITY_WEIGHT,
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
    expected = UNLOADED_RIGID_HEIGHT + HOP_HEIGHT_GAIN + H_ADD
    assert hop_cfg.rewards["hop_body_height"].params["target_height"] == pytest.approx(expected)
    assert hop_target_height(H_ADD) == pytest.approx(expected)


def test_rigid_variant_target_is_not_shifted():
    rigid = make_hop_variant(make_microduck_velocity_env_cfg(), h_add=0.0)
    expected = UNLOADED_RIGID_HEIGHT + HOP_HEIGHT_GAIN
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


def test_walking_air_time_reward_removed(hop_cfg):
    """`air_time` (mjlab's feet_air_time, weight 5.0) pays a PER-FOOT indicator
    over 0.10-0.25 s of flight, i.e. it pays continuously for alternating
    single-foot stepping. Integrated over a cycle, marching in place earns
    ~3.0/step against ~1.0/step for a 1.0 s hop, and at most ~1.1/step is
    available from all three hop terms combined — so leaving it in makes a bob in
    place strictly outscore hopping, on all three arms equally, and the campaign
    would conclude "compliance does not help" from three runs that never hopped.

    Its command gate (||cmd[:2]|| + |cmd[2]| > 0.01) is also permanently latched
    on: the phase command's magnitude is identically 1.0."""
    assert "air_time" not in hop_cfg.rewards


def test_air_time_is_present_in_the_walking_env_this_is_removed_from():
    """Guard the guard: if mjlab renamed or dropped the term upstream, the
    assertion above would pass vacuously and stop protecting anything."""
    assert "air_time" in make_microduck_velocity_env_cfg().rewards


def test_hop_body_height_is_gated_on_the_contact_sensor(hop_cfg):
    """The height reward must be gated on both feet airborne, or it pays ~0.57 of
    peak for straightening the legs while planted. The gate reads the same sensor
    as hop_both_feet_airborne, so the two terms cannot disagree about what a hop
    is; make_hop_variant threads it explicitly rather than leaning on the default."""
    height = hop_cfg.rewards["hop_body_height"]
    airborne = hop_cfg.rewards["hop_both_feet_airborne"]
    assert height.params["sensor_name"] == SENSOR_NAME
    assert height.params["sensor_name"] == airborne.params["sensor_name"]


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


def test_rigid_stand_height_is_pinned_to_the_measured_locked_arm_geometry():
    """Pin RIGID_STAND_HEIGHT against a real measurement of the Locked arm.

    MEASURED 2026-08-24 on the registered `Mjlab-Hop-Flat-Sprung-Locked-MicroDuck`
    arm (stiffness 3900, travel 0.0, pad 70 g, h_add 0.030). Method: compile that
    arm's spec_fn, add a floor plane, set HOME_FRAME on every hinge with the
    position actuators pointed at it, PIN THE BASE TO VERTICAL TRAVEL ONLY (it
    topples in ~1 s without a balance policy, so an unpinned settle measures
    tipping, not height) and settle 3000 steps — the horizon this campaign's
    existing settle probes use.

        settled root z (Locked, WEARS the pad)  = 0.13949 m
        minus H_ADD                            = 0.030   m
        => rigid stand height                  = 0.10949 m

    The previous value, 0.120, was the robot_walk.xml SPAWN height (qpos0[2]),
    not a standing height: the sag-free KINEMATIC height below is only 0.1171
    rigid, so the robot could not stand that tall in HOME_FRAME even with
    infinitely stiff actuators. With std = 0.008 in `hop_body_height`, that
    10.5 mm error was most of the reward's dynamic range.

    Two assertions, deliberately independent:
      1. the constant still matches the settle measurement (trips if someone
         edits it without re-measuring);
      2. the compiled Locked arm's sag-free kinematic height still matches what
         was measured (trips if H_ADD, ANKLE_TO_SOLE, the pad box or any leg link
         changes — at which point re-run
         `.superpowers/sdd/2026-08-24-sprung-hop/measure_stand_height.py`).
    """
    import re

    import mujoco
    import numpy as np

    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.robot.microduck_constants import HOME_FRAME

    assert RIGID_STAND_HEIGHT == pytest.approx(0.10949, abs=0.002)

    robot = load_env_cfg("Mjlab-Hop-Flat-Sprung-Locked-MicroDuck").scene.entities["robot"]
    m = robot.spec_fn().compile()
    d = mujoco.MjData(m)
    for i in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm is None or m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        for pat, val in HOME_FRAME.joint_pos.items():
            if re.search(pat.strip("^$").replace(".*", ""), nm):
                d.qpos[m.jnt_qposadr[i]] = val
                break
    d.qpos[:3] = 0.0
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d)

    # Lowest corner of either contact pad, with the base frame at z = 0.
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    lowest = min(
        (d.geom_xpos[g] + (corners * m.geom_size[g]) @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min()
        for g in (
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot_collision")
            for s in ("left", "right")
        )
    )
    kinematic_rigid = -lowest - H_ADD
    assert kinematic_rigid == pytest.approx(0.11710, abs=0.002)
    # UNLOADED_RIGID_HEIGHT *is* this sag-free kinematic height -- it is the datum
    # the airborne-gated hop height reward is built on, so pin it to the geometry
    # rather than to a copied literal.
    assert UNLOADED_RIGID_HEIGHT == pytest.approx(kinematic_rigid, abs=0.002)
    assert RIGID_STAND_HEIGHT < kinematic_rigid, (
        "the robot cannot stand taller in HOME_FRAME than its own kinematics allow"
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


# ---------------------------------------------------------------------------
# The reward ceiling. Four independent mechanisms capped the height reward at
# roughly 15-20 mm of gain while the drop-rig evidence spans 5 mm (Locked) to
# 33 mm (k3900), so all three arms would have sat at the ceiling and the
# arm-to-arm comparison -- which IS the experiment -- would have returned an
# uninformative null. The tests below pin all four, plus the discrimination
# property that motivates them.
# ---------------------------------------------------------------------------


def _hop_task_id(label: str) -> str:
    return f"Mjlab-Hop-Flat-Sprung-{HOP_ARM_SUFFIX[label]}-MicroDuck"


def _registered(label: str):
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    return load_env_cfg(_hop_task_id(label))


def test_height_target_references_the_unloaded_not_the_settled_height():
    """`hop_body_height` is gated on BOTH FEET AIRBORNE, so it is only ever
    evaluated in flight with the legs unloaded. Referencing RIGID_STAND_HEIGHT
    (the SETTLED height, measured with the full 877 g sagging the position
    actuators off their targets) hands the robot the sag for free: it scores
    "gain" for merely unloading its legs, without leaving the ground any higher.
    That is a measurement error in the reward, not a tuning choice.

    Both halves matter: the exact value, and the size of the error the old datum
    introduced.
    """
    assert hop_target_height(0.0) == pytest.approx(UNLOADED_RIGID_HEIGHT + HOP_HEIGHT_GAIN)
    assert hop_target_height(H_ADD) == pytest.approx(
        UNLOADED_RIGID_HEIGHT + HOP_HEIGHT_GAIN + H_ADD
    )

    settled_based = RIGID_STAND_HEIGHT + HOP_HEIGHT_GAIN
    free_gain = hop_target_height(0.0) - settled_based
    assert free_gain == pytest.approx(0.0076, abs=1e-4), (
        "the settled reference under-shoots the flight datum by the actuator sag; "
        "that 7.6 mm was free reward for unloading the legs"
    )


def test_registered_arms_carry_the_widened_height_gaussian():
    """End-to-end through the registered tasks, not the transform in isolation.

    The old (gain 0.015, std 0.008) pair put the entire 5-33 mm evidence band at
    ~0.000 reward. Assert the registered params, and assert the peak still sits
    ABOVE the ~27 mm energetic ceiling estimated for k=3900 so the whole band
    stays on the Gaussian's RISING limb instead of straddling its peak.
    """
    for label in HOP_ARM_SUFFIX:
        cfg = _registered(label)
        params = cfg.rewards["hop_body_height"].params
        tid = _hop_task_id(label)
        assert params["target_height"] == pytest.approx(hop_target_height(H_ADD)), tid
        assert params["std"] == pytest.approx(HOP_HEIGHT_STD), tid
        # Datum guard, independent of HOP_HEIGHT_GAIN's value.
        #
        # The assertion this replaced (`gain > 0.033` where
        # `gain = target_height - (UNLOADED_RIGID_HEIGHT + H_ADD)`) is identically
        # `HOP_HEIGHT_GAIN > 0.033` by construction, so it asserted only "the
        # constant exceeds a hardcoded 0.033" -- and it caught the
        # RIGID_STAND_HEIGHT-vs-UNLOADED_RIGID_HEIGHT datum revert only by
        # coincidence: 0.040 - 0.0076 = 0.0324 happens to land 0.6 mm under that
        # threshold. Someone who raises HOP_HEIGHT_GAIN to 0.045 while reverting
        # the datum back to RIGID_STAND_HEIGHT (double-counting the leg sag
        # again) would pass the whole suite.
        #
        # Assert the datum directly instead, and separately assert it is NOT the
        # settled-height alternative -- the second assertion is the one that
        # survives any future change to the gain.
        assert params["target_height"] == pytest.approx(
            UNLOADED_RIGID_HEIGHT + HOP_HEIGHT_GAIN + H_ADD
        ), tid
        assert params["target_height"] != pytest.approx(
            RIGID_STAND_HEIGHT + HOP_HEIGHT_GAIN + H_ADD
        ), tid


def test_upward_velocity_does_not_saturate_below_the_height_target():
    """`hop_upward_velocity` clamps vel_z/max_vel to [0, 1]. A ballistic launch at
    v rises v**2/(2*g), so max_vel caps the rise the term is willing to pay for.
    At the old 0.5 m/s that cap was 12.7 mm -- below the ENTIRE 5-33 mm band, so
    the velocity term stopped paying long before the height term peaked, and the
    two terms fought each other.

    The second assertion is the real content and is written as a RELATIONSHIP:
    whatever the two constants become, the velocity term must not saturate before
    the height term peaks.
    """
    for label in HOP_ARM_SUFFIX:
        cfg = _registered(label)
        max_vel = cfg.rewards["hop_upward_velocity"].params["max_vel"]
        tid = _hop_task_id(label)
        # The relationship first, so it is the assertion that actually fails when
        # either constant regresses rather than being shadowed by the literal.
        saturating_rise = max_vel**2 / (2 * 9.81)
        assert saturating_rise > HOP_HEIGHT_GAIN, (
            f"{tid}: hop_upward_velocity saturates at {saturating_rise * 1e3:.1f} mm of "
            f"rise, at or below the {HOP_HEIGHT_GAIN * 1e3:.1f} mm the height reward "
            "peaks at -- the velocity term stops paying before the height term does"
        )
        assert max_vel == pytest.approx(HOP_MAX_LAUNCH_VEL), tid
        assert max_vel == pytest.approx(1.0), tid


def test_com_band_ceiling_is_above_the_hop_apex():
    """`com_height_target` pays a flat +1 in band and -(z - max)**2 above it, so
    crossing the top forfeits the whole +1 as a STEP, times its weight of 1.2.
    With the old 0.14 rigid top (0.17 sprung) that step landed at 23 mm of gain --
    inside the range the experiment needs explored, penalising exactly the hops we
    are trying to measure.

    Written as a relationship so it cannot silently regress if HOP_HEIGHT_GAIN,
    UNLOADED_RIGID_HEIGHT or H_ADD moves. Checked on a sprung arm AND on the
    Locked control, which wears the same boot and so gets the same shift.
    """
    apex = hop_target_height(H_ADD)
    for label in ("k3900", "locked"):
        params = _registered(label).rewards["com_height_target"].params
        tid = _hop_task_id(label)
        assert params["target_height_max"] > apex, (
            f"{tid}: CoM band top {params['target_height_max']:.4f} is at or below the "
            f"target apex {apex:.4f} -- reaching the commanded hop height forfeits the "
            "band's +1 as a step penalty"
        )


def test_com_band_floor_is_untouched_by_the_hop_variant():
    """Only the UPPER edge moves. `target_height_min` still pays for not
    collapsing during stance, and the Phase-2 h_add translation in
    make_sprung_variant (explicitly out of scope) must still be the only thing
    acting on it.
    """
    base_min = make_microduck_velocity_env_cfg().rewards["com_height_target"].params[
        "target_height_min"
    ]
    for label in HOP_ARM_SUFFIX:
        params = _registered(label).rewards["com_height_target"].params
        assert params["target_height_min"] == pytest.approx(base_min + H_ADD), _hop_task_id(label)


def test_height_gaussian_discriminates_locked_from_sprung():
    """The property the whole change exists for, and the test that would have
    caught the original ceiling.

    The drop-rig probe rebounded 5 mm on the Locked arm and 33 mm at k=3900. If
    the reward cannot tell those two apart, all three arms score the same and the
    campaign returns an uninformative null after hours of GPU time per arm. Under
    the old (0.015, 0.008) params both read ~1e-8 and the ratio was ~1.0.

    The Gaussian is recomputed here from the REGISTERED params -- exp(-((h -
    target)/std)**2), matching microduck_mdp.hop_body_height -- rather than from
    module constants, so a registration that fails to thread them through fails
    this test too.
    """
    params = _registered("k3900").rewards["hop_body_height"].params
    target, std = params["target_height"], params["std"]
    # The reward is evaluated airborne, so gain is measured from the UNLOADED
    # stand height -- which is exactly `target - HOP_HEIGHT_GAIN`.
    reference = target - HOP_HEIGHT_GAIN

    def reward(gain_m: float) -> float:
        return math.exp(-(((reference + gain_m - target) / std) ** 2))

    locked_like = reward(0.005)
    sprung_like = reward(0.033)
    assert sprung_like > locked_like, "the reward must increase across the band"
    assert sprung_like / locked_like >= 10.0, (
        f"5 mm scores {locked_like:.4f} and 33 mm scores {sprung_like:.4f} -- only "
        f"{sprung_like / locked_like:.1f}x apart. The arms would be indistinguishable."
    )
    # Monotone across the band, so a taller hop is never worth less.
    samples = [reward(g / 1000.0) for g in range(5, 34)]
    assert samples == sorted(samples)


def test_height_gaussian_pays_from_a_standing_start():
    """The >=10x ratio test above is satisfiable by a cliff, not just a slope.

    Reverting HOP_HEIGHT_STD to its old 0.008 while leaving HOP_HEIGHT_GAIN at
    0.040 still passes `sprung_like / locked_like >= 10.0` above (0.465 / 5e-9
    is a huge, monotone ratio) yet pays approximately zero reward below 25 mm of
    gain -- a cliff the policy cannot climb from a standing start (0 mm), which
    is exactly the risk the widening from (0.015, 0.008) to (0.040, 0.020) was
    meant to remove. The ratio test never checks the low end in absolute terms,
    so it cannot see this.

    Two guards:
      1. an absolute floor -- the registered Gaussian evaluated at a 5 mm gain
         (the Locked-arm drop-rig datum) must be worth at least 0.03, so there
         is always a live gradient from a standing start;
      2. the relationship that makes (1) robust to future retuning: std must be
         at least 40% of gain, written as a relationship between the two
         constants rather than as two more hardcoded numbers.
    """
    params = _registered("k3900").rewards["hop_body_height"].params
    target, std = params["target_height"], params["std"]
    reference = target - HOP_HEIGHT_GAIN

    def reward(gain_m: float) -> float:
        return math.exp(-(((reference + gain_m - target) / std) ** 2))

    assert reward(0.005) >= 0.03, (
        f"5 mm of gain scores {reward(0.005):.4f}, below the 0.03 floor -- "
        "there is no usable gradient from a standing start"
    )
    assert HOP_HEIGHT_STD >= 0.4 * HOP_HEIGHT_GAIN, (
        f"HOP_HEIGHT_STD ({HOP_HEIGHT_STD}) is less than 40% of HOP_HEIGHT_GAIN "
        f"({HOP_HEIGHT_GAIN}) -- the Gaussian can be narrowed back into a cliff "
        "without failing the discrimination ratio test above"
    )


# ---------------------------------------------------------------------------
# The reward BUDGET. The first Phase 4 sweep -- three arms, 8000 iterations each
# -- returned a null: all three converged to standing perfectly still, both feet
# airborne 0.07-0.3% of the time. The cause was arithmetic, not tuning, and the
# tests below are the ones that would have caught it before the GPU time was
# spent.
# ---------------------------------------------------------------------------

# Mean of the launch gate clamp(sin(2*pi*phi), 0) over one cycle. This factor --
# not the weight -- is what a hop term is actually worth per step, and it is the
# whole reason the weights are what they are.
_LAUNCH_GATE_MEAN = 1.0 / math.pi


def test_hop_budget_beats_standing_still():
    """The inequality the first Phase 4 sweep violated.

    `launch_weight = clamp(sin(2*pi*phi), 0)` averages 1/pi = 0.318 over a cycle,
    so each hop term's per-step ceiling is `weight * 0.318`. At the original
    3.0/2.0/2.0 that was 0.955 + 0.637 + ~0.3 = ~1.9/step for a PHYSICALLY
    IMPOSSIBLE perfect hopper -- a robot pinned at peak reward on all three terms
    for the entire launch half. Standing perfectly still pays `com_height_target`
    1.2 + `upright` 1.0 = 2.2/step, at lower `action_rate_l2` cost and near-zero
    fall risk.

    A perfect hop lost to standing before any risk was counted. The policy did
    not fail to find hopping; it correctly found that hopping was worse, and all
    three arms learned to stand.

    Both sides are computed from the REGISTERED weights on a REGISTERED task, not
    from hardcoded numbers -- the failure mode this guards is precisely a set of
    weights that look reasonable term by term and lose in aggregate, so the test
    has to do the aggregation itself. Checked on every arm.
    """
    for label in HOP_ARM_SUFFIX:
        cfg = _registered(label)
        tid = _hop_task_id(label)
        rewards = cfg.rewards

        # Hop side: the three launch-gated terms at their per-step ceiling.
        hop_ceiling = _LAUNCH_GATE_MEAN * sum(
            rewards[name].weight
            for name in (
                "hop_both_feet_airborne",
                "hop_upward_velocity",
                "hop_body_height",
            )
        )

        # Standing side: both are flat-+1-shaped terms, evaluated at their
        # maximum, and both are paid EVERY step by a robot that does nothing.
        standing = rewards["com_height_target"].weight * 1.0 + rewards["upright"].weight * 1.0

        assert hop_ceiling >= 2.0 * standing, (
            f"{tid}: a perfect hopper's ceiling is {hop_ceiling:.2f}/step against "
            f"{standing:.2f}/step for standing perfectly still -- only "
            f"{hop_ceiling / standing:.2f}x. The first Phase 4 sweep ran at "
            f"{1.9 / 2.2:.2f}x and all three arms learned to stand."
        )


def test_hop_budget_terms_all_exist_on_both_sides():
    """Guard the guard. The budget test sums named terms; if one were renamed or
    dropped the sum would silently shrink (hop side) or grow (standing side) and
    the inequality could pass for the wrong reason."""
    for label in HOP_ARM_SUFFIX:
        rewards = _registered(label).rewards
        for name in (
            "hop_both_feet_airborne",
            "hop_upward_velocity",
            "hop_body_height",
            "com_height_target",
            "upright",
        ):
            assert name in rewards, f"{_hop_task_id(label)}: {name} missing"
            assert rewards[name].weight > 0.0, f"{_hop_task_id(label)}: {name} weight <= 0"


def test_registered_hop_weights_are_the_rebalanced_ones():
    """Pin the 4x through the registered tasks. The budget test above is written
    as a relationship so it survives retuning; this one catches a silent revert of
    the specific numbers the sweep will be relaunched with."""
    for label in HOP_ARM_SUFFIX:
        rewards = _registered(label).rewards
        tid = _hop_task_id(label)
        assert rewards["hop_both_feet_airborne"].weight == pytest.approx(AIRBORNE_WEIGHT), tid
        assert rewards["hop_upward_velocity"].weight == pytest.approx(UPWARD_VELOCITY_WEIGHT), tid
        assert rewards["hop_body_height"].weight == pytest.approx(BODY_HEIGHT_WEIGHT), tid
    assert (AIRBORNE_WEIGHT, UPWARD_VELOCITY_WEIGHT, BODY_HEIGHT_WEIGHT) == (12.0, 8.0, 8.0)


def test_action_rate_weight_is_untouched_by_the_rebalance():
    """Explicitly out of scope: the hop weights went up 4x, the action-rate cost
    did not move. Raising both would cancel the rebalance."""
    rigid = make_microduck_velocity_env_cfg()
    for label in HOP_ARM_SUFFIX:
        assert _registered(label).rewards["action_rate_l2"].weight == pytest.approx(
            rigid.rewards["action_rate_l2"].weight
        ), _hop_task_id(label)


# --- the load-phase term ----------------------------------------------------


def test_load_force_registered_on_every_arm():
    """Nothing rewarded the load half: all three hop terms gate on sin > 0.
    Without an actuator countermovement the spring cannot be charged (static sag
    under body weight alone is 0.48 mm at k=3900, ~0.45 mJ, worth 0.1 mm of
    lift), so the spring needs a hop to charge and the hop needs a charged
    spring. This term is what breaks that circularity."""
    for label in HOP_ARM_SUFFIX:
        term = _registered(label).rewards["hop_load_force"]
        tid = _hop_task_id(label)
        assert term.func is microduck_mdp.hop_load_force, tid
        assert term.weight == pytest.approx(LOAD_FORCE_WEIGHT), tid
        assert term.params["sensor_name"] == SENSOR_NAME, tid
        assert term.params["command_name"] == "twist", tid
        assert term.params["body_weight_n"] == pytest.approx(BODY_WEIGHT_N), tid
        assert term.params["max_ratio"] == pytest.approx(LOAD_FORCE_MAX_RATIO), tid


def test_load_force_is_identical_on_the_locked_control_arm():
    """It is a FORCE reward, not a compression reward, precisely so the Locked
    control gets the same signal. Both arms can press down; only the sprung arms
    convert that press into stored energy. A compression reward would read
    identically zero on Locked -- it has no spring joint -- and destroy the
    controlled comparison the whole experiment rests on."""
    locked = _registered("locked").rewards["hop_load_force"]
    sprung = _registered("k3900").rewards["hop_load_force"]
    assert locked.func is sprung.func
    assert locked.weight == sprung.weight
    assert locked.params == sprung.params


def test_body_weight_constant_matches_the_measured_mass():
    """0.877 kg (737 g robot + 2 x 70 g boot, worn by all three arms) x 9.81."""
    assert BODY_WEIGHT_N == pytest.approx(0.877 * 9.81, abs=0.005)
    assert PAD_MASS == pytest.approx(0.070)


def test_load_force_stays_below_the_launch_terms():
    """The countermovement is the enabler, not the objective. If pressing down
    paid more than leaving the ground, the policy's best move would be to squat
    hard and never jump -- a new way to reach the same null."""
    for label in HOP_ARM_SUFFIX:
        rewards = _registered(label).rewards
        load_ceiling = _LAUNCH_GATE_MEAN * rewards["hop_load_force"].weight
        launch_ceiling = _LAUNCH_GATE_MEAN * sum(
            rewards[n].weight
            for n in ("hop_both_feet_airborne", "hop_upward_velocity", "hop_body_height")
        )
        assert load_ceiling < launch_ceiling, _hop_task_id(label)


# --- the launch-half silencing of com_height_target --------------------------


def test_com_height_target_is_silenced_during_launch():
    """`com_height_target`'s flat +1-in-band (x1.2) was the single largest reward
    for standing perfectly still. During the launch half we want the robot LEAVING
    the band, so the registered term must be the recovery-gated wrapper."""
    for label in HOP_ARM_SUFFIX:
        term = _registered(label).rewards["com_height_target"]
        tid = _hop_task_id(label)
        assert term.func is microduck_mdp.com_height_target_recovery_only, tid
        assert term.params["command_name"] == "twist", tid


def test_com_height_target_swap_preserves_the_sprung_band_shift():
    """The swap MUST keep the term's key and its two band params, because
    `make_sprung_variant` runs AFTER `make_hop_variant`, looks the term up by the
    key "com_height_target", and shifts `target_height_min`/`target_height_max` by
    h_add in place. Renaming the key or dropping either param would break the band
    on every sprung arm silently -- no error, just a robot penalised for its own
    geometry."""
    base = make_microduck_velocity_env_cfg().rewards["com_height_target"].params
    for label in HOP_ARM_SUFFIX:
        params = _registered(label).rewards["com_height_target"].params
        tid = _hop_task_id(label)
        # Both edges present, and both carry the h_add shift applied afterwards.
        assert params["target_height_min"] == pytest.approx(
            base["target_height_min"] + H_ADD
        ), tid
        assert params["target_height_max"] == pytest.approx(HOP_COM_HEIGHT_MAX + H_ADD), tid


def test_upright_is_deliberately_not_gated():
    """`upright` genuinely pays for not tipping. Suppressing it during launch
    would encourage tipping at exactly the moment of takeoff, so it keeps the
    velocity env's func and weight."""
    rigid = make_microduck_velocity_env_cfg().rewards["upright"]
    for label in HOP_ARM_SUFFIX:
        term = _registered(label).rewards["upright"]
        tid = _hop_task_id(label)
        assert term.func is rigid.func, tid
        assert term.weight == pytest.approx(rigid.weight), tid
