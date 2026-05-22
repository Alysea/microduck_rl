"""MicroDuck (sprung shank) — hop forward at a commanded velocity.

Step up from `Mjlab-Hop-MicroDuck-Sprung`:
  * Adds a forward velocity command (vx ∈ [0, 0.3] m/s, no lateral, no yaw).
  * Replaces `stay_in_place` with `track_linear_velocity` (same Gaussian
    form as the SLIP run task).
  * Adds an explicit head-stability penalty — pen-alising neck/head joint
    velocities so the policy uses the legs (not the head as a
    counterweight) for balance.  This is the user requirement: the real
    robot's IMU sits on the trunk, so a stable head + body gives cleaner
    sensor data and better sim-to-real.

Bakes in the lessons from the SLIP work:
  * Initial drop bootstrap (z range 0.131-0.171) — every episode starts
    airborne, so PPO sees the airborne reward from step 0.
  * Explicit reset of springs + actuated joints (defaults) before adding
    leg-noise — otherwise post-crash resets leave the robot mid-fall.
  * entropy_coef = 0.02 to keep exploration alive once the policy starts
    converging on standing.
  * Moderate `action_rate_l2` = -0.05 (cyclic hip motion is good).

NOT included (deliberately):
  * Yaw penalty — we learned on the SLIP run task that adding an L2 yaw
    penalty crushes early exploration.  The microduck's foot pads have
    some extent and CAN anchor yaw, but the policy has to find that on
    its own.  Accept some yaw drift in the first cut.
"""

from __future__ import annotations

import math
import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs import mdp as base_mdp
from mjlab.managers.manager_term_config import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.viewer import ViewerConfig

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_SPRUNG_ROBOT_CFG


# ---------------------------------------------------------------------------
# Custom MDP terms
# ---------------------------------------------------------------------------


def upright_bonus(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    grav_xy = asset.data.projected_gravity_b[:, :2]
    return torch.exp(-torch.sum(grav_xy ** 2, dim=1) / (std ** 2))


def both_feet_airborne(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.0005,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    spring_ids, _ = asset.find_joints([r".*spring"])
    spring_q = asset.data.joint_pos[:, spring_ids]
    return (spring_q < threshold).all(dim=1).float()


def track_velocity_gaussian(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """exp(-|v_xy_body − v_cmd_xy|² / std²) — Gaussian on body-frame xy
    velocity error.

    Switched from world-frame to body-frame so the command means "move
    forward / sideways relative to current heading" — the right semantics
    once a yaw command is active.  With ωz=0 this behaves identically to
    the previous world-frame tracker (body and world frames differ only
    by yaw, which we track separately).
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    vxy = asset.data.root_link_lin_vel_b[:, :2]
    err_sq = torch.sum((vxy - cmd[:, :2]) ** 2, dim=1)
    return torch.exp(-err_sq / (std ** 2))


def alternating_legs(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward out-of-phase spring loading between left and right legs.

    `|q_left − q_right|` normalised to [0, 1] by the spring's 10 mm travel
    range — peaks at 1.0 when one leg is at maximum compression and the
    other is fully unloaded.  Combined with `co_compression` (penalty)
    this triple drives the policy toward proper alternating gait.
    """
    asset: Entity = env.scene[asset_cfg.name]
    spring_ids, _ = asset.find_joints([r".*spring"])
    q = asset.data.joint_pos[:, spring_ids]              # (num_envs, 2), in metres
    qn = torch.clamp(q / 0.01, 0.0, 1.0)                  # normalise to [0, 1]
    return torch.abs(qn[:, 0] - qn[:, 1])


def co_compression(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Indicator that BOTH springs are compressed simultaneously.

    Returns the product q_L × q_R (normalised), so it's only large when
    both legs are loaded at the same time — bilateral stance.  Used with
    a NEGATIVE weight to specifically penalise that configuration, which
    `alternating_legs` alone doesn't distinguish from genuine alternation
    (both have low |q_L - q_R| if amplitudes match).

    Three-term combination drives the gait:
      airborne:        peaks when both q ≈ 0 (flight phase)
      alternation:     peaks when |q_L - q_R| is large (one loaded, one not)
      co_compression:  peaks (and is penalised) when BOTH loaded

    Bilateral hop: hits co_compression hard, no alternation → punished.
    Walking:       no co_compression, full alternation, no flight → OK.
    Running:       no co_compression, alternation in stance, airborne in
                   flight → best.
    """
    asset: Entity = env.scene[asset_cfg.name]
    spring_ids, _ = asset.find_joints([r".*spring"])
    q = asset.data.joint_pos[:, spring_ids]
    qn = torch.clamp(q / 0.01, 0.0, 1.0)
    return qn[:, 0] * qn[:, 1]                            # ∈ [0, 1]


def track_yaw_rate_gaussian(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """exp(-(ωz − ωz_cmd)² / std²) — Gaussian on yaw-rate error.

    Same shape as the xy-velocity tracker.  Wider std than the xy tracker
    because yaw control is intrinsically harder on a sprung biped — even
    with the microduck's hip_yaw + mesh foot pads, getting precise yaw
    rate is a separate skill the policy has to find.
    """
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    wz = asset.data.root_link_ang_vel_b[:, 2]
    err_sq = (wz - cmd[:, 2]) ** 2
    return torch.exp(-err_sq / (std ** 2))


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

ROBOT_CFG    = SceneEntityCfg("robot")
SPRING_CFG   = SceneEntityCfg("robot", joint_names=(r".*spring",))
LEG_CFG      = SceneEntityCfg(
    "robot",
    joint_names=(r".*hip_(pitch|roll|yaw)", r".*knee", r".*ankle"),
)
HEAD_CFG     = SceneEntityCfg(
    "robot",
    joint_names=(r".*neck.*", r".*head.*"),
)
ALL_ACTUATED_CFG = SceneEntityCfg(
    "robot",
    joint_names=(r"^(?!.*spring).*",),
)


def make_microduck_hop_forward_sprung_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # Forward-only velocity command.  Always positive (no zero-cmd envs)
    # so the policy can't satisfy the task by standing.
    commands = {
        "twist": UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                # Expanded after first training got forward+turn working —
                # now adding backward (negative vx) and proper lateral
                # (sidestep) so the policy learns omnidirectional commands.
                # Range is asymmetric on vx since the robot's geometry has
                # a clear forward, and humans/bipeds naturally walk forward
                # faster than backward.
                lin_vel_x=(-0.15, 0.25),   # forward + backward
                lin_vel_y=(-0.15, 0.15),   # proper lateral
                ang_vel_z=(-0.5,  0.5),    # ~ ±30 deg/s yaw rate (unchanged)
            ),
            debug_vis=False,
        ),
    }

    policy_terms = {
        "base_lin_vel":      ObservationTermCfg(func=base_mdp.base_lin_vel),
        "base_ang_vel":      ObservationTermCfg(func=base_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity),
        "joint_pos":         ObservationTermCfg(func=base_mdp.joint_pos_rel),
        "joint_vel":         ObservationTermCfg(func=base_mdp.joint_vel_rel),
        "actions":           ObservationTermCfg(func=base_mdp.last_action),
        "velocity_command":  ObservationTermCfg(
            func=base_mdp.generated_commands,
            params={"command_name": "twist"},
        ),
    }
    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms=dict(policy_terms),
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    actions = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(r".*",),
            scale=0.5,
            use_default_offset=True,
        ),
    }

    # Reset events — same defensive pattern as the SLIP run task.
    events = {
        "reset_base": EventTermCfg(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"z": (0.131, 0.171)},   # 0-40 mm drop bootstrap
                "velocity_range": {},
                "asset_cfg": ROBOT_CFG,
            },
        ),
        "reset_springs": EventTermCfg(
            func=base_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SPRING_CFG,
            },
        ),
        "reset_joints_default": EventTermCfg(
            func=base_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": ALL_ACTUATED_CFG,
            },
        ),
        "reset_leg_noise": EventTermCfg(
            func=base_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.05, 0.05),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": LEG_CFG,
            },
        ),
    }

    # Reward design philosophy for this iteration: stay as close as possible
    # to the hop-in-place reward structure that converged cleanly.  The
    # ONLY change is replacing `stay_in_place` (Gaussian on v_xy = 0) with
    # `track_velocity_gaussian` (same Gaussian shape, centred on the
    # commanded velocity instead of zero).  Everything else — weights,
    # std, action_rate_l2 — matches the working hop-in-place config.
    #
    # Previous run failed because adding `track_linear_velocity` with
    # weight 3.0 + std 0.39 changed too much at once relative to the
    # baseline that worked.  This iteration changes one thing.
    rewards = {
        "alive": RewardTermCfg(func=base_mdp.is_alive, weight=0.5),
        "upright": RewardTermCfg(
            func=upright_bonus,
            weight=1.0,
            params={"std": 0.2, "asset_cfg": ROBOT_CFG},
        ),
        # Main signal: dominate the "stand still" optimum, same weight that
        # made the hop-in-place task converge.
        "airborne": RewardTermCfg(
            func=both_feet_airborne,
            weight=3.0,
            params={"asset_cfg": ROBOT_CFG, "threshold": 0.0005},
        ),
        # Reward out-of-phase leg loading (one loaded, the other not).
        "alternation": RewardTermCfg(
            func=alternating_legs,
            weight=3.0,
            params={"asset_cfg": ROBOT_CFG},
        ),
        # NEW: penalise bilateral stance.  `alternation` alone can't tell
        # apart genuine alternation from amplitude-mismatched-but-phase-
        # aligned bilateral hopping (both have low |q_L - q_R|).  This
        # term fires only when BOTH springs are simultaneously compressed
        # — bilateral stance specifically.  Negative weight makes it a
        # cost that proper running / walking gait pays zero of.
        "co_compression": RewardTermCfg(
            func=co_compression,
            weight=-2.0,
            params={"asset_cfg": ROBOT_CFG},
        ),
        # Replaces `stay_in_place` from hop-in-place — same Gaussian form,
        # same weight, same std — centred on the commanded body-frame xy
        # velocity instead of zero.
        "track_velocity": RewardTermCfg(
            func=track_velocity_gaussian,
            weight=1.0,
            params={
                "std": 0.1,
                "command_name": "twist",
                "asset_cfg": ROBOT_CFG,
            },
        ),
        # NEW: yaw-rate tracking.  Smaller weight (0.5) and wider std
        # (0.3 rad/s) than xy tracking — yaw is harder, give the policy
        # more slack while it learns to use hip_yaw + foot-placement
        # asymmetry for turning.
        "track_yaw_rate": RewardTermCfg(
            func=track_yaw_rate_gaussian,
            weight=0.5,
            params={
                "std": 0.3,
                "command_name": "twist",
                "asset_cfg": ROBOT_CFG,
            },
        ),
        # Stable head for clean body-mounted IMU.
        "head_vel_l2": RewardTermCfg(
            func=base_mdp.joint_vel_l2,
            weight=-0.1,
            params={"asset_cfg": HEAD_CFG},
        ),
        # Back to hop-in-place's -0.05 weight — the -0.02 last iteration
        # was an unnecessary deviation from the baseline.
        "action_rate_l2": RewardTermCfg(
            func=base_mdp.action_rate_l2,
            weight=-0.05,
        ),
        "joint_vel_l2": RewardTermCfg(
            func=base_mdp.joint_vel_l2,
            weight=-1e-3,
            params={"asset_cfg": LEG_CFG},
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=base_mdp.bad_orientation,
            params={"limit_angle": 1.0, "asset_cfg": ROBOT_CFG},
        ),
        "fallen_height": TerminationTermCfg(
            func=base_mdp.root_height_below_minimum,
            params={"minimum_height": 0.06, "asset_cfg": ROBOT_CFG},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(terrain_type="plane"),
            entities={"robot": MICRODUCK_WALK_SPRUNG_ROBOT_CFG},
            num_envs=4096 if not play else 1,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum={},
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            asset_name="robot",
            body_name="trunk_base",
            distance=0.7,
            elevation=-15.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=40,
            njmax=200,
            mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
        ),
        decimation=4,
        episode_length_s=8.0,
    )


# ---------------------------------------------------------------------------
# PPO config
# ---------------------------------------------------------------------------

MicroduckHopForwardSprungRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(256, 128, 64),
        critic_hidden_dims=(256, 128, 64),
        activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,     # back to default — previous 0.02 kept the
                               # action noise std at 0.76 forever, never
                               # let the policy converge to a clean gait.
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
    experiment_name="microduck_hop_forward_sprung",
    run_name="hop_forward_sprung",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=5000,
)
