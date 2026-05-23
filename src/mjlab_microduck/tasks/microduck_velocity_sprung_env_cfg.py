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
from mjlab.managers.manager_term_config import CurriculumTermCfg, RewardTermCfg
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
    # Widened downward to admit shorter swing phases.  Walking cadence
    # produces ~100-200 ms swings (microduck's setting); running cadence
    # produces ~50-100 ms swings.  Original 0.10-0.20 made running gaits
    # ineligible for the reward.  0.05-0.20 covers both walking and running.
    cfg.rewards["air_time"].params["threshold_min"] = 0.05
    cfg.rewards["air_time"].params["threshold_max"] = 0.20

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

    # ── Velocity command curriculum ──
    # Start with very narrow ranges and grow to the target over the first
    # ~4000 iterations (out of 30k).  The previous training (no curriculum,
    # full range from iter 0) converged to "lean and stumble" because the
    # policy never had a phase to learn slow controlled motion first.
    # Starting at ±0.05 m/s gives the policy time to find proper gait
    # patterns before being asked to track higher speeds.
    cfg.curriculum["velocity_command_ranges"] = CurriculumTermCfg(
        func=microduck_mdp.velocity_command_ranges_curriculum,
        params={
            "command_name": "twist",
            "velocity_stages": [
                # Phase A: from-scratch easy ramp (4k iters).  Used by the
                # initial training run.  On a resume from iter 12k these
                # all already fired, so no effect.
                {"step": 0,         "lin_vel_range": 0.05, "ang_vel_range": 0.10},
                {"step": 1000 * 24, "lin_vel_range": 0.10, "ang_vel_range": 0.20},
                {"step": 2500 * 24, "lin_vel_range": 0.15, "ang_vel_range": 0.30},
                {"step": 4000 * 24, "lin_vel_range": 0.20, "ang_vel_range": 0.40},
                # Phase B: extension toward running speeds.  Activated
                # gradually after the walking policy is solid.  At lin_vel=
                # 0.5 the natural gait should require flight phases.
                {"step": 12000 * 24, "lin_vel_range": 0.25, "ang_vel_range": 0.45},
                {"step": 17000 * 24, "lin_vel_range": 0.35, "ang_vel_range": 0.60},
                {"step": 25000 * 24, "lin_vel_range": 0.50, "ang_vel_range": 0.80},
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
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="microduck_velocity_sprung",
    run_name="velocity_sprung",
    save_interval=200,
    num_steps_per_env=24,
    # Bumped from 10k → 30k — the previous run was undertrained for the
    # complexity of "track velocity while walking on sprung legs".  The
    # rigid microduck uses 50k iters for the same task with the same
    # reward stack, so 30k is still on the conservative side.
    max_iterations=30000,
)
