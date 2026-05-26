"""MicroDuck (sprung shank) — velocity tracking on the sprung-leg
prototype, using the same proven reward stack as the rigid microduck.

Key difference from `Mjlab-HopForward-MicroDuck-Sprung` (the previous
sprung task): this one is built on top of mjlab's stock
`make_velocity_env_cfg()` template and uses the `feet_air_time` reward
machinery + foot contact sensor that the rigid microduck task relies on
to learn alternating gait.  Our home-grown `airborne` / `alternation` /
`co_compression` rewards are gone — replaced by the cadence-based
`air_time` reward that explicitly drives single-leg-stance behaviour.

Sprung-specific adjustments vs rigid microduck:
  * Entity = MICRODUCK_WALK_SPRUNG_ROBOT_CFG (with HOME_FRAME_SPRUNG)
  * `reset_base` z-range = (0.131, 0.155) — sprung HOME standing height
    is 11 mm higher than rigid + a small drop bootstrap.
  * `air_time` thresholds 0.10-0.20 s (vs rigid's 0.10-0.25) — slightly
    tighter upper bound because the springs naturally produce a faster
    cadence than the rigid robot's underpowered XL330 walk.
  * Adds `head_vel_l2` penalty to keep the head still for the
    body-mounted IMU.
  * Skipped (for first prototype): domain randomization, neck-offset
    randomization, observation noise / delays, curriculum, push events.
    These should be added incrementally once the baseline trains.
"""

from __future__ import annotations

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.manager_term_config import (
    CurriculumTermCfg, RewardTermCfg, TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_SPRUNG_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp


def make_microduck_velocity_sprung_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # ── Pose-reward stds ── (same per-joint stds as the rigid microduck —
    # how tight to hold each joint near its default during standing vs walking)
    std_standing = {
        r".*hip_yaw.*":   0.1,
        r".*hip_roll.*":  0.1,
        r".*hip_pitch.*": 0.15,
        r".*knee.*":      0.15,
        r".*ankle.*":     0.1,
        r".*neck.*":      0.1,
        r".*head.*":      0.1,
    }
    std_walking = {
        r".*hip_yaw.*":   0.3,
        r".*hip_roll.*":  0.1,
        r".*hip_pitch.*": 0.4,
        r".*knee.*":      0.4,
        r".*ankle.*":     0.25,
        r".*neck.*":      0.1,
        r".*head.*":      0.1,
    }

    site_names = ["left_foot", "right_foot"]

    # Foot contact sensor.  `track_air_time=True` is what enables the
    # cadence-based `feet_air_time` reward to actually score anything.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # ── Pull in the stock velocity env (rewards + actions + events) ──
    cfg = make_velocity_env_cfg()

    # Robot + scene
    cfg.scene.entities = {"robot": MICRODUCK_WALK_SPRUNG_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg,)
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.viewer.body_name = "trunk_base"

    # The critic obs has a `foot_height` term that needs its site refs
    cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

    # Action — full hip range, standard JointPositionAction (no neck offset)
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards ──
    # Pose reward — tight stds at standing, looser at walking.
    # Restrict to the 14 actuated joints (exclude the 2 passive spring
    # joints) so the std-dict matches what `variable_posture` sees.
    # WEIGHT 1.0 (was 2.0): the previous training converged to "stand
    # nicely" because pose + upright together dominated tracking.  Lower
    # weight makes pose a regularizer rather than a primary objective.
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!.*spring).*",),
    )
    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_walking
    cfg.rewards["pose"].params["std_running"] = std_walking
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 1.0

    # Upright torso
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 1.0

    # Foot rewards — bind to our foot site IDs
    for reward_name in ("foot_clearance", "foot_swing_height", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["foot_clearance"].params["target_height"] = 0.02
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    # Body angular velocity penalty
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05

    # ── AIR TIME — the cadence-based alternating-gait reward ──
    # Rigid microduck uses 0.10-0.25 s; we tighten the upper bound a bit
    # because our springs naturally bounce at ~6 Hz and we don't want to
    # encourage extremely slow swing phases.
    # Weight bumped from 5 → 7 to push the policy toward proactive
    # stepping rather than the "lean and stumble + catch with foot"
    # reactive-recovery local optimum the previous run found.
    cfg.rewards["air_time"].weight = 7.0
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    # Loosened from 0.10-0.20 to 0.12-0.30 — the tight 0.20 upper bound
    # caps swing time at walking-cadence values, which is why iter-17000
    # "walks faster" instead of running.  At 0.30 the policy can be
    # rewarded for longer swing phases (a running-gait characteristic)
    # without losing the lower-bound enforcement against tiny twitchy
    # steps.
    cfg.rewards["air_time"].params["threshold_min"] = 0.12
    cfg.rewards["air_time"].params["threshold_max"] = 0.30

    # ── NEW: flight-phase bonus ── ENCOURAGES actual running.
    # Returns 1 per step when BOTH feet are simultaneously in the air,
    # 0 otherwise.  Command-gated so it only activates when moving (we
    # don't want the policy to hop in place when commanded to stand).
    # Modest weight: this is a permission signal, not the primary
    # objective — the air_time reward (weight 7.0) still dominates and
    # enforces alternating gait most of the time.  At low command
    # magnitude → walks (alternating, brief or no flight).  At high
    # command magnitude → policy *can* trade some of the alternating
    # reward for the flight-phase bonus and start producing a brief
    # both-feet-airborne phase.
    cfg.rewards["flight_phase"] = RewardTermCfg(
        func=microduck_mdp.flight_phase_reward,
        weight=1.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.1,
        },
    )

    # Velocity tracking — BUMPED weights to make tracking the dominant
    # reward signal.  Previous training at weight=3.0 lost out to the
    # combined pose+upright "stand still" attractor.  At 6.0 / 4.0 the
    # tracking reward is now larger than standing rewards combined when
    # tracking is good, so the policy is incentivised to actually move.
    cfg.rewards["track_linear_velocity"].weight = 6.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.15)
    cfg.rewards["track_angular_velocity"].weight = 4.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.40)

    cfg.rewards["soft_landing"].weight = -1e-5

    # Action rate — moderate.  Start at -0.4 like the rigid microduck's
    # baseline (it later curricula up to -1.0; we skip curriculum for now).
    cfg.rewards["action_rate_l2"].weight = -0.4

    # ── NEW: head stability penalty (sprung-specific) ──
    # Keep the head and neck still so the body-mounted IMU sees clean data,
    # and so the visual head doesn't bob like a counterweight.
    cfg.rewards["head_vel_l2"] = RewardTermCfg(
        func=base_mdp.joint_vel_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=(r".*neck.*", r".*head.*"),
            ),
        },
    )

    # ── Commands ──
    # Use deepcopy so we don't share mutable state with other envs that
    # might construct from the same make_velocity_env_cfg() — defensive
    # pattern lifted from microduck_velocity_env_cfg.py.
    command = deepcopy(cfg.commands["twist"])
    cfg.commands["twist"] = command
    # rel_standing_envs=0: the policy NEVER sees a "stand still" command
    # during training, so it can't learn standing as a default behaviour
    # to fall back on.  Has to do something with every step it sees.
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    # Placeholder values — overwritten by the velocity-curriculum below
    # starting at step 0.  Symmetric since the curriculum function uses
    # symmetric ranges (lin_vel_x = ±r).
    command.ranges.lin_vel_x = (-0.05, 0.05)
    command.ranges.lin_vel_y = (-0.05, 0.05)
    command.ranges.ang_vel_z = (-0.1,  0.1)

    # ── Base reset — sprung HOME is ~131 mm; small drop bootstrap on top ──
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.131, 0.155)

    # Drop default curriculum terms we don't need
    for term in ("terrain_levels", "command_vel"):
        if term in cfg.curriculum:
            del cfg.curriculum[term]

    # ── NaN-state safety termination ──
    # The rigid microduck env has this; ours did not.  Sprung-leg
    # dynamics can produce NaN MuJoCo states under extreme contact
    # impulses (foot landing hard, springs at limit, etc.), which then
    # propagate into the policy obs → into the action → into NaN losses.
    # Terminating immediately on NaN resets the env *before* the
    # corruption reaches the rollout buffer.  This is almost certainly
    # missing-safeguard cause of the resume-from-checkpoint NaN losses
    # we've been chasing.
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Velocity command curriculum ──
    # Start with very narrow ranges and grow to the target over the first
    # ~4000 iterations (out of 30k).  The previous training (no curriculum,
    # full range from iter 0) converged to "lean and stumble" because the
    # policy never had a phase to learn slow controlled motion first.
    # Starting at ±0.05 m/s gives the policy time to find proper gait
    # patterns before being asked to track higher speeds.
    cfg.curriculum["velocity_command_ranges"] = CurriculumTermCfg(
        # SMOOTH version — linearly interpolates between adjacent stages
        # over `ramp_steps` instead of step-function jumps.  Step-function
        # transitions repeatedly caused NaN-loss collapse (~iter 12000)
        # because a converged policy can't absorb instantaneous
        # distribution shifts; value estimates go stale, TD errors spike,
        # advantages explode, gradients explode, weights → NaN.
        # Spreading the same total shift over 2000 iters means each
        # gradient step sees ~0.05% command-range expansion — well within
        # the value function's tolerance.
        func=microduck_mdp.velocity_command_ranges_curriculum_smooth,
        params={
            "command_name": "twist",
            "ramp_steps": 2000 * 24,
            # Phase A — narrow ramp from very-slow to walking speed.
            # Used by from-scratch training to find proper gait patterns
            # before being asked to track higher speeds.
            # Phase B — push toward running speeds.  On from-scratch
            # training the policy passes through these stages naturally;
            # the gradual expansion doesn't trigger the converged-policy
            # advantage explosion that resume-from-checkpoint did.
            # Phase A — narrow ramp from very-slow to walking speed.
            # Phase B — push toward running, with FINER stages than the
            # previous attempt.  The iter-17000 jump from 0.25 → 0.35
            # (+40%) collapsed training around iter 18000 — too big a
            # distribution shift for the converged policy to handle in
            # one step.  Stages here cap any single jump at ≤ +20% and
            # space them ≥ 5000 iters apart, giving the critic time to
            # re-fit value estimates before the next expansion.
            "velocity_stages": [
                {"step": 0,         "lin_vel_range": 0.05, "ang_vel_range": 0.10},
                {"step": 1000 * 24, "lin_vel_range": 0.10, "ang_vel_range": 0.20},
                {"step": 2500 * 24, "lin_vel_range": 0.15, "ang_vel_range": 0.30},
                {"step": 4000 * 24, "lin_vel_range": 0.20, "ang_vel_range": 0.40},
                {"step": 12000 * 24, "lin_vel_range": 0.25, "ang_vel_range": 0.45},
                {"step": 17000 * 24, "lin_vel_range": 0.30, "ang_vel_range": 0.50},
                {"step": 22000 * 24, "lin_vel_range": 0.35, "ang_vel_range": 0.60},
                {"step": 27000 * 24, "lin_vel_range": 0.42, "ang_vel_range": 0.70},
                {"step": 32000 * 24, "lin_vel_range": 0.50, "ang_vel_range": 0.80},
            ],
        },
    )

    return cfg


# === PPO config ===
MicroduckVelocitySprungRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # Same network as the rigid microduck velocity task
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        # Tightened from 1.0 to 0.5 as a safety net against the
        # advantage-explosion mechanism that has caused NaN losses at
        # curriculum transitions.  The smooth curriculum should already
        # eliminate the explosions, but a tighter grad-norm clip means
        # even an unexpected distribution shift can't blow up the update.
        max_grad_norm=0.5,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_velocity_sprung",
    run_name="velocity_sprung",
    save_interval=200,
    num_steps_per_env=24,
    # Bumped to 40k to accommodate the finer Phase B curriculum (final
    # stage at iter 32000 needs ~8000 iters to consolidate the
    # higher-speed gait).  Rigid microduck uses 50k for the same task.
    max_iterations=40000,
)
