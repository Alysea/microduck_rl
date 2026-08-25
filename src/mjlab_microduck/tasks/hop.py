"""Hop task variant — a phase-commanded periodic hop in place.

``make_hop_variant(cfg, h_add)`` converts a velocity env cfg into a hop task, in
the same shape as ``tasks/run.py`` and ``tasks/backlash.py``. Composed as
``make_sprung_variant(make_hop_variant(cfg), ...)`` so the sprung machinery is
reused unchanged.

Periodic rather than a one-shot jump because the spring's energy comes
overwhelmingly from IMPACT loading: a drop-rig probe measured a 100 mm drop
rebounding 33 mm, while quasi-static actuator loading only just reaches the
49.7 N needed for full travel (52.4 N available, knee-limited) before BAM
back-EMF derates it at launch speed. Each landing charges the next launch.

Six changes:

1. Replace the twist command with the CYCLIC phase command already on develop.
2. Retarget the ported height reward for the robot's actual standing height.
3. Drop the forward-locomotion rewards (this is a hop in place) AND the walking
   `air_time` reward, which outscores hopping with a march in place.
4. Register the three hop rewards, the LOAD-PHASE reward and the energy monitor.
5. Raise the `com_height_target` band's UPPER edge, which otherwise puts a step
   penalty inside the height range the experiment needs explored.
6. Silence `com_height_target` during the launch half — it was the single largest
   reward for standing perfectly still, which is what the first sweep learned.

Steps 4 (the load reward) and 6, together with the 4x on the hop weights, are the
rebalance that followed that null. See the budget comment beside the weights.
"""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg

from mjlab_microduck.robot.sprung_foot import SPRING_JOINTS, SPRING_PRELOAD
from mjlab_microduck.tasks import mdp as microduck_mdp

# Seconds per hop cycle. The spring-mass period at k=3900 and 0.877 kg is
# 2*pi*sqrt(m/k) = 94 ms, so this is deliberately well above it: the cycle must
# accommodate a load segment, a launch, a flight and a landing, not just one
# spring oscillation. Provisional -- sweep only if the first result is ambiguous.
HOP_PERIOD = 1.0

# Rigid standing trunk height, and the height gain the reward asks for on top.
# The ported reward hardcoded 0.135 = 0.12 + 0.015 for the RIGID robot; these
# split that into its two parts so h_add can be added correctly.
#
# MEASURED, 2026-08-24, NOT the ported 0.120. The old value turned out to be the
# robot_walk.xml SPAWN height (model qpos0[2] == 0.120) rather than a settled
# standing height -- the robot cannot actually stand that tall in HOME_FRAME.
#
# Method (see the test with the same number in tests/test_hop_cfg.py): compile
# the registered LOCKED arm, drop it on a floor plane in HOME_FRAME with the base
# pinned to vertical travel only (it topples in ~1 s otherwise, and an unpinned
# settle measures tipping), settle 3000 steps -- the horizon both of this
# campaign's existing settle probes use -- and read the root z. That reads
# 0.1395 m, and the Locked arm WEARS the pad, so the rigid height is
# 0.1395 - H_ADD(0.030) = 0.1095. That is 10.5 mm below the ported 0.120.
#
# Bracketing, because the settle creeps (the hip_roll actuators splay slowly
# against the pin): the sag-free KINEMATIC height -- HOME_FRAME, upright, lowest
# pad corner exactly on the floor -- is 0.1171 rigid, and an 80 s settle reaches
# 0.1035. So the true value is inside [0.1035, 0.1171] on any reading, and the
# old 0.120 sat ABOVE even the sag-free ceiling. Corroboration: the velocity
# env's `com_height_target` band for the rigid robot is 0.11-0.14.
RIGID_STAND_HEIGHT = 0.1095

# The SAG-FREE KINEMATIC height (HOME_FRAME, upright, lowest pad corner exactly
# on the floor), rigid, i.e. with h_add removed. Same measurement as the upper
# bracket quoted above, and pinned by the same test in tests/test_hop_cfg.py.
#
# This -- not RIGID_STAND_HEIGHT -- is the correct reference for the hop height
# reward, because `hop_body_height` is GATED ON BOTH FEET AIRBORNE: it is only
# ever evaluated in flight, with the legs carrying no load and therefore no
# actuator sag. RIGID_STAND_HEIGHT is the SETTLED height, 7.6 mm lower, measured
# with the full 877 g pressing the hip/knee/ankle position actuators off their
# targets. Referencing the settled value from a flight-only reward hands the
# robot 7.6 mm of free "gain" for merely unloading its legs -- a measurement
# error in the reward, not a tuning choice. RIGID_STAND_HEIGHT stays as the
# documented settled value (it is what the CoM band and stance behaviour are
# about); it is simply not the datum for an airborne apex.
#
# This fixes a measurement error, but it does not exhaust the robot's posture
# headroom: a grid search over symmetric hip_pitch/knee/ankle poses (pad kept
# flat) found a maximum sag-free stance root height of 0.16133 -- vs
# HOME_FRAME's 0.14710 -- so about 14.2 mm of headroom survives this fix,
# roughly TWICE the 7.6 mm it removed. That headroom is identical on all three
# arms, so it biases none of them relative to each other, but it means the real
# room between UNLOADED_RIGID_HEIGHT and the hop target's HOP_COM_HEIGHT_MAX
# (0.1871 sprung) is ~26 mm, not the ~40 mm a reader might infer from treating
# this datum fix as pure spring rebound.
UNLOADED_RIGID_HEIGHT = 0.1171

# Height gain above the unloaded reference that the Gaussian peaks at, and its
# width. Both were raised from 0.015 / 0.008: the old pair saturated the reward
# at roughly 15-20 mm of gain, while the drop-rig evidence this campaign is built
# on spans 5 mm (Locked) to 33 mm (k3900), so all three arms sat at the ceiling
# and the arm-to-arm comparison -- which IS the experiment -- measured nothing.
#
# Why 40 mm. Physical ceiling estimate at k=3900: full travel stores
# 0.5*3900*0.012**2 + 2.9*0.012 = 0.3156 J per foot (the second term is work
# against the k*SPRING_PRELOAD ~ 2.9 N preload), so 0.631 J for both feet. At
# zeta = 0.3 a damped oscillator returns exp(-pi*zeta/sqrt(1-zeta**2)) = 0.372 of
# that, i.e. ~0.234 J, which lifts 0.877 kg by 27 mm. Actuator work adds on top.
# So 40 mm sits deliberately just ABOVE the sprung expectation, which keeps the
# whole 5-33 mm discriminating band on the RISING limb of the Gaussian instead of
# straddling its peak (where 27 mm and 53 mm would score alike).
#
# Why std = 20 mm. `hop_body_height` uses exp(-((h - target)/std)**2), so with
# target +40 mm and std 20 mm the term reads: 5 mm gain -> 0.047, 27 mm -> 0.655,
# 33 mm -> 0.885. Monotone increasing across the band, ~19x discrimination
# between a Locked-like 5 mm and a sprung-like 33 mm. Under the old std = 0.008
# all three of those read ~0.000 -- indistinguishable, which was the bug.
#
# The tradeoff, deliberately accepted: at zero gain the term is exp(-4) = 0.018,
# so it gives almost no gradient until the robot is already leaving the ground.
# `hop_both_feet_airborne` (weight AIRBORNE_WEIGHT, binary) is the DISCOVERY term; this one
# only shapes how high once airborne.
HOP_HEIGHT_GAIN = 0.040
HOP_HEIGHT_STD = 0.020

# Upward base velocity at which `hop_upward_velocity` saturates (it clamps
# vel_z/max_vel to [0, 1]). A ballistic launch at v rises v**2/(2*g), so the old
# 0.5 m/s saturated at 0.25/19.62 = 12.7 mm of rise -- BELOW the entire 5-33 mm
# discriminating band, meaning the velocity term stopped paying long before the
# height term peaked. 1.0 m/s saturates at 1.0/19.62 = 51 mm, above the 40 mm
# HOP_HEIGHT_GAIN target, so the two terms now peak in the right order.
#
# This makes the term LESS generous early -- at vz = 0.5 m/s it now reads 0.5
# rather than 1.0 -- which is intended: a 0.5 m/s launch is a 13 mm hop, not a
# finished behaviour.
HOP_MAX_LAUNCH_VEL = 1.0

# Upper edge of `com_height_target`'s band, for the RIGID robot, in the hop task
# only. See the in-place edit in make_hop_variant for the reasoning.
HOP_COM_HEIGHT_MAX = 0.20

SENSOR_NAME = "feet_ground_contact"

# THE REWARD BUDGET. Do not "tidy" these back down -- the 4x is the whole fix
# for the first Phase 4 sweep's null, and the 1/pi below is where it comes from.
#
# All three hop terms gate on `launch = clamp(sin(2*pi*phi), 0)`, whose mean over
# a cycle is exactly 1/pi = 0.318. So a term's per-step ceiling is not its weight,
# it is weight * 0.318 -- and it only reaches that if the shaped factor it
# multiplies is pinned at 1.0 the whole launch half, which no real hop achieves.
#
# At the ORIGINAL weights (3.0 / 2.0 / 2.0):
#
#   term                      weight   ceiling/step
#   hop_both_feet_airborne      3.0       0.955
#   hop_upward_velocity         2.0       0.637
#   hop_body_height             2.0       ~0.3    (Gaussian, never pinned at 1)
#   ------------------------------------------------
#   physically impossible perfect hopper  ~1.9
#
# Standing perfectly still pays `com_height_target` 1.2 + `upright` 1.0 = 2.2 per
# step, at lower `action_rate_l2` cost and with near-zero fall risk. A PERFECT HOP
# LOST TO STANDING STILL before any risk was counted. The first sweep -- three
# arms, 8000 iterations each -- did not fail to find hopping; it correctly found
# that hopping was worse, and all three converged to standing (both feet airborne
# 0.07-0.3% of the time).
#
# 4x on all three restores the inequality: (12 + 8 + 8) * 1/pi = 8.9/step of hop
# ceiling against 2.2/step of standing. `tests/test_hop_cfg.py` pins that ratio at
# >= 2x, computed from the registered weights.
AIRBORNE_WEIGHT = 12.0
UPWARD_VELOCITY_WEIGHT = 8.0
BODY_HEIGHT_WEIGHT = 8.0

# The other half of the fix: the load phase. All three terms above gate on the
# LAUNCH half (sin > 0), so nothing rewarded the load half at all -- see
# `microduck_mdp.hop_load_force` for why that blocks the whole mechanism.
# Held well below the launch terms on purpose: this is the enabling countermovement,
# not the objective. Its ceiling is 4.0 * 1/pi = 1.27/step.
LOAD_FORCE_WEIGHT = 4.0

# Robot weight in newtons, the datum `hop_load_force` normalises against:
# 0.877 kg (737 g robot + 2 x 70 g boot, worn by ALL THREE arms including Locked)
# x 9.81 m/s^2. Named rather than left as a literal because the same 0.877 kg
# appears in the spring-mass period and energy notes above.
BODY_WEIGHT_N = 8.60

# Multiple of body weight at which `hop_load_force` saturates. 2.0 means the term
# reads 0 at a plain stand and 1.0 once the feet push with twice body weight.
# Threaded explicitly rather than left on the function default, for the same
# reason `sensor_name` and `stiffness` are: this campaign has been burned by
# defaults that were only correct for one arm.
LOAD_FORCE_MAX_RATIO = 2.0

ENERGY_MONITOR_WEIGHT = 1.0

# Forward-locomotion rewards read the twist command, which the phase command
# overwrites with [cos, sin, 0]. Left in place they would reward running away.
_LOCOMOTION_REWARDS = ("track_linear_velocity", "track_angular_velocity")

# Removed for a DIFFERENT reason than the locomotion rewards above, hence its own
# list: `air_time` does not reward running away, it rewards WALKING IN PLACE, and
# it pays more for that than the hop rewards pay for hopping.
#
# `air_time` is mjlab's `feet_air_time` at weight 5.0. It sums a PER-FOOT
# indicator over the window 0.10 s < air_time < 0.25 s, so it pays continuously
# for alternating single-foot flight. Integrated over one cycle: in-place
# alternating stepping with ~0.25 s swings earns ~3.0/step, whereas a 1.0 s hop
# with 0.2 s of two-foot flight earns ~1.0/step -- against at most ~1.1/step
# available from all three hop terms combined AT THE ORIGINAL 3.0/2.0/2.0 weights
# (see the budget comment beside them; the 4x raises that ceiling to ~8.9/step,
# which does not make `air_time` safe to keep -- it still pays for the wrong
# behaviour, it just no longer wins outright). Marching in place therefore
# strictly outscored hopping, identically on all three arms, and the campaign
# would conclude "compliance does not help" from three runs that never hopped.
#
# It is also permanently latched ON: its gate is ||cmd[:2]|| + |cmd[2]| > 0.01,
# and the phase command [cos(2*pi*phi), sin(2*pi*phi), 0] has magnitude
# identically 1.0. There is no commanded speed left for it to gate on.
#
# NOT swapped for `feet_air_time_capped`: capping fixes double-payment for
# two-foot flight, but the defeat here comes from the continuous SINGLE-foot
# incentive, which capping leaves intact. And the hop task already has its own
# airborne reward, `hop_both_feet_airborne` at AIRBORNE_WEIGHT -- which pays only
# when BOTH feet are off the ground, i.e. for the behaviour we actually want.
_WALKING_GAIT_REWARDS = ("air_time",)

# NOT removed, deliberately: `foot_swing_height` (weight -0.25, relative-squared
# cost, target_height=0.02) is the same CLASS of walking-gait term as `air_time`
# above -- a retained term whose interaction with the hop rewards needs
# checking, not assuming -- but its shape doesn't create the same conflict.
# It's a bowl centred on 20 mm of foot peak height: harmless across the 5-33 mm
# evidence band (<1% of the per-cycle total there), but quadratic above it
# (-0.36 per landing at 33 mm gain, -0.72 at 40 mm, -2.42 at 60 mm). It is
# IDENTICAL across all three arms, so it biases none of them relative to each
# other -- but it is where the reward's new ceiling actually comes from: the
# hop total now peaks around ~65 mm of gain because this is the term that
# finally overtakes the airborne terms (which grow only as sqrt(h)). Its
# logged `Metrics/peak_height_mean` is the arm-comparison observable to watch.

# LANDING SURVIVAL: the spec asks for a landing-survival term. It is met by the
# terms already present rather than by a new one -- the velocity env's `upright`
# reward (weight 1.0) pays for staying vertical through the landing, and the
# `fell_over` termination (bad_orientation, 70 deg) ends an episode that fails.
# Adding a third redundant survival term would double-count the same behaviour
# and make the hop rewards harder to balance against it.


def hop_target_height(h_add: float) -> float:
    """Target base height for the hop apex, shifted by the boot's added height.

    Built on ``UNLOADED_RIGID_HEIGHT``, not ``RIGID_STAND_HEIGHT``: the reward
    that consumes this is gated on both feet airborne, so it is only evaluated in
    flight with the legs unloaded. See the constant's comment.

    The sprung robot stands ``h_add`` taller, so an unshifted target asks it to
    CROUCH rather than hop -- the same class of bug as the CoM band shift.
    """
    return UNLOADED_RIGID_HEIGHT + HOP_HEIGHT_GAIN + h_add


def make_hop_variant(
    cfg: ManagerBasedRlEnvCfg,
    h_add: float = 0.0,
    stiffness: float = 3900.0,
) -> ManagerBasedRlEnvCfg:
    """Convert a velocity env cfg into the periodic hop task.

    Args:
        h_add: metres of height the foot mechanism adds. 0.0 for the rigid
            robot; pass the sprung model's H_ADD for sprung arms.
        stiffness: N/m spring rate to report through `hop_energy_monitor`. Must
            match the arm's actual spring stiffness -- Task 4 registers a k2500
            arm alongside the k3900 default, and a hardcoded value would report
            that arm's stored spring energy wrong (56% high for k2500 vs k3900).
    """
    # 1. Cyclic phase command, reusing the class already on develop.
    #
    # `make_microduck_velocity_env_cfg` sets cfg.commands["twist"] to
    # VelocityCommandCommandOnlyCfg, which carries a `rel_turn_in_place_envs`
    # field that GroundPickPhaseCommandCfg doesn't declare (it isn't derived
    # from that class -- the ground_pick reference call site starts from
    # mjlab's base UniformVelocityCommandCfg, which has no such field, so it
    # never hits this). Filter vars(command) down to the fields
    # GroundPickPhaseCommandCfg actually accepts instead of forwarding all of
    # them verbatim.
    #
    # `rel_turn_in_place_envs` is the one field we know is safe to drop: it is
    # only ever read by VelocityCommandCommandOnly._resample_command, and
    # GroundPickPhaseCommand's own _resample_command override is a no-op
    # `pass` -- nothing in the built command ever looks at it. Any OTHER
    # dropped field is unproven and must fail loudly rather than vanish
    # silently into a training run (this campaign has been burned by silent
    # drops before), so we raise if drift introduces one.
    command = cfg.commands["twist"]
    valid_fields = {f.name for f in dataclasses.fields(microduck_mdp.GroundPickPhaseCommandCfg)}
    dropped = set(vars(command)) - valid_fields
    _KNOWN_INERT_DROPS = {"rel_turn_in_place_envs"}
    unexpected_drops = dropped - _KNOWN_INERT_DROPS
    if unexpected_drops:
        raise ValueError(
            f"make_hop_variant: cfg.commands['twist'] ({type(command).__name__}) carries "
            f"field(s) {sorted(unexpected_drops)} that GroundPickPhaseCommandCfg does not "
            "declare and that are not in the known-inert allow-list. Either mirror the "
            "field onto GroundPickPhaseCommandCfg or confirm it is unused (like "
            "rel_turn_in_place_envs) and add it to _KNOWN_INERT_DROPS."
        )
    command_kwargs = {k: v for k, v in vars(command).items() if k in valid_fields}
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **command_kwargs,
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": HOP_PERIOD,
        }
    )

    # 2/3. This is a hop in place: forward tracking would reward running away,
    #      and its command has just been overwritten anyway.
    for name in _LOCOMOTION_REWARDS:
        cfg.rewards.pop(name, None)

    # ...and the walking gait reward, which pays more for marching in place than
    # the hop rewards pay for hopping. See _WALKING_GAIT_REWARDS above.
    for name in _WALKING_GAIT_REWARDS:
        cfg.rewards.pop(name, None)

    # 4. Hop rewards. All three gate internally on sin(2*pi*phase) > 0.
    cfg.rewards["hop_both_feet_airborne"] = RewardTermCfg(
        func=microduck_mdp.hop_both_feet_airborne,
        weight=AIRBORNE_WEIGHT,
        params={"sensor_name": SENSOR_NAME, "command_name": "twist"},
    )
    cfg.rewards["hop_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.hop_upward_velocity,
        weight=UPWARD_VELOCITY_WEIGHT,
        params={"command_name": "twist", "max_vel": HOP_MAX_LAUNCH_VEL},
    )
    cfg.rewards["hop_body_height"] = RewardTermCfg(
        func=microduck_mdp.hop_body_height,
        weight=BODY_HEIGHT_WEIGHT,
        params={
            "command_name": "twist",
            "target_height": hop_target_height(h_add),
            "std": HOP_HEIGHT_STD,
            # Threaded explicitly: this term is gated on BOTH FEET AIRBORNE
            # (otherwise it pays for a ground-level bob), and that gate reads the
            # same contact sensor as hop_both_feet_airborne above.
            "sensor_name": SENSOR_NAME,
        },
    )

    # The load half. The three terms above all gate on sin > 0; this one gates on
    # sin < 0, and is what pays for the countermovement that charges the spring.
    cfg.rewards["hop_load_force"] = RewardTermCfg(
        func=microduck_mdp.hop_load_force,
        weight=LOAD_FORCE_WEIGHT,
        params={
            "sensor_name": SENSOR_NAME,
            "command_name": "twist",
            "body_weight_n": BODY_WEIGHT_N,
            "max_ratio": LOAD_FORCE_MAX_RATIO,
        },
    )

    # Energy instrument. Returns zeros, so the weight only has to be non-zero
    # for RewardManager.compute to call it at all.
    cfg.rewards["hop_energy_monitor"] = RewardTermCfg(
        func=microduck_mdp.hop_energy_monitor,
        weight=ENERGY_MONITOR_WEIGHT,
        params={
            "joint_names": SPRING_JOINTS,
            "stiffness": stiffness,
            "preload": SPRING_PRELOAD,
        },
    )

    # 5. Lift the CoM band's ceiling out of the discriminating range.
    #
    #    `com_height_target` returns +1 flat while in band and -(z - max)**2 once
    #    above it, so crossing the top forfeits the whole +1 as a STEP -- times
    #    its weight of 1.2. With the base band at [0.11, 0.14] rigid (shifted to
    #    [0.14, 0.17] by make_sprung_variant), that step landed at
    #    0.17 - 0.1471 = 23 mm of gain, i.e. right inside the 5-33 mm range this
    #    experiment exists to resolve. It penalised exactly the hops we want.
    #
    #    Safe for STANCE, which is what the band is actually for: the rigid
    #    sag-free kinematic maximum is UNLOADED_RIGID_HEIGHT = 0.1171, already
    #    BELOW the old 0.14 top, so the upper edge was unreachable while standing
    #    and only ever fired airborne. Raising it therefore changes nothing about
    #    standing behaviour on any arm. (Sprung, same argument: a 0.1471 stand vs
    #    a 0.17 top.) `target_height_min` is deliberately untouched -- it still
    #    pays for not collapsing during stance.
    #
    #    Only the RIGID upper edge moves, and only in the hop variant. The
    #    Phase-2 `h_add` translation in make_sprung_variant is untouched (that
    #    "CoM band shift" is the out-of-scope item in the spec): running after
    #    this, it shifts both edges by h_add and yields [0.14, 0.23] for the
    #    sprung arms -- comfortably above the 0.1871 target apex.
    cfg.rewards["com_height_target"].params["target_height_max"] = HOP_COM_HEIGHT_MAX

    # 6. ...and stop paying it at all during the LAUNCH half.
    #
    #    Even with the ceiling lifted, the term's flat +1-in-band (x1.2) was the
    #    single largest reward for standing perfectly still, which is what all
    #    three arms of the first sweep learned. During launch we want the robot
    #    LEAVING the band, so swap the func for the recovery-gated wrapper.
    #
    #    MUTATE IN PLACE, do not rebuild the term. `make_sprung_variant` runs
    #    AFTER this transform and looks the term up by the key
    #    "com_height_target", then shifts `target_height_min`/`target_height_max`
    #    by h_add. Renaming the key or dropping either param silently breaks the
    #    band shift on EVERY sprung arm -- which is why the wrapper takes those
    #    two params through unchanged and only adds `command_name`.
    com = cfg.rewards["com_height_target"]
    com.func = microduck_mdp.com_height_target_recovery_only
    com.params["command_name"] = "twist"

    return cfg


from copy import deepcopy
from dataclasses import replace

from mjlab_microduck.robot.sprung_foot import H_ADD, PAD_MASS, TRAVEL
from mjlab_microduck.tasks.run import MicroduckRunRlCfg

# (label, stiffness N/m, travel m, pad mass kg).
#
# Three arms, not six. The drop-rig probe already narrowed stiffness: at a
# 100 mm drop k2500 rebounded 35.3 mm, k3900 32.8 mm, k5500 28.3 mm, and k1500
# only 20.6 mm because it bottoms out and slams. So the optimum is the softest
# spring that does NOT bottom, and 2500-3900 brackets it.
#
# Mass is held at the measured 70 g because Stage 1's locked arms already
# measured the mass penalty separately (-17.7% at 30 g, -61.9% at 90 g vs the
# rigid running baseline).
HOP_ARMS = (
    ("locked", 3900.0, 0.0, PAD_MASS),
    ("k2500", 2500.0, TRAVEL, PAD_MASS),
    ("k3900", 3900.0, TRAVEL, PAD_MASS),
)

HOP_ARM_SUFFIX = {"locked": "Locked", "k2500": "K2500", "k3900": "K3900"}


def hop_rl_cfg(label: str):
    """Per-arm RL cfg: identical learner, distinct logging identity.

    ``replace`` is shallow, so the nested cfgs are deep-copied -- otherwise all
    three arms would share one actor object AND share it with the Run baseline,
    so a later change to any arm would silently alter the others.
    """
    return replace(
        MicroduckRunRlCfg,
        actor=deepcopy(MicroduckRunRlCfg.actor),
        critic=deepcopy(MicroduckRunRlCfg.critic),
        algorithm=deepcopy(MicroduckRunRlCfg.algorithm),
        experiment_name=f"hop_{label}",
        run_name=f"hop_{label}",
    )
