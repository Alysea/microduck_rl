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

Four changes:

1. Replace the twist command with the CYCLIC phase command already on develop.
2. Retarget the ported height reward for the robot's actual standing height.
3. Drop the forward-locomotion rewards (this is a hop in place) AND the walking
   `air_time` reward, which outscores hopping with a march in place.
4. Register the three hop rewards and the energy monitor.
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
HOP_HEIGHT_GAIN = 0.015

HOP_HEIGHT_STD = 0.008
SENSOR_NAME = "feet_ground_contact"

AIRBORNE_WEIGHT = 3.0
UPWARD_VELOCITY_WEIGHT = 2.0
BODY_HEIGHT_WEIGHT = 2.0
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
# available from all three hop terms combined. Marching in place therefore
# strictly outscores hopping, identically on all three arms, and the campaign
# would conclude "compliance does not help" from three runs that never hopped.
#
# It is also permanently latched ON: its gate is ||cmd[:2]|| + |cmd[2]| > 0.01,
# and the phase command [cos(2*pi*phi), sin(2*pi*phi), 0] has magnitude
# identically 1.0. There is no commanded speed left for it to gate on.
#
# NOT swapped for `feet_air_time_capped`: capping fixes double-payment for
# two-foot flight, but the defeat here comes from the continuous SINGLE-foot
# incentive, which capping leaves intact. And the hop task already has its own
# airborne reward, `hop_both_feet_airborne` at weight 3.0 -- which pays only when
# BOTH feet are off the ground, i.e. for the behaviour we actually want.
_WALKING_GAIT_REWARDS = ("air_time",)

# LANDING SURVIVAL: the spec asks for a landing-survival term. It is met by the
# terms already present rather than by a new one -- the velocity env's `upright`
# reward (weight 1.0) pays for staying vertical through the landing, and the
# `fell_over` termination (bad_orientation, 70 deg) ends an episode that fails.
# Adding a third redundant survival term would double-count the same behaviour
# and make the hop rewards harder to balance against it.


def hop_target_height(h_add: float) -> float:
    """Target base height for the hop apex, shifted by the boot's added height.

    The sprung robot stands ``h_add`` taller, so an unshifted target asks it to
    CROUCH rather than hop -- the same class of bug as the CoM band shift.
    """
    return RIGID_STAND_HEIGHT + HOP_HEIGHT_GAIN + h_add


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
        params={"command_name": "twist", "max_vel": 0.5},
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
