"""SLIP biped 3D — forward velocity tracking.

First locomotion task on the SLIP biped.  The robot gets a commanded
forward speed (`vx ∈ [0, 0.3] m/s`, no lateral, no yaw) and must move
forward at that speed without falling.

Builds on what hop-3D validated:
  * Springs can be exploited for vertical thrust via hip-torque pumping.
  * The policy can find a stable cyclic gait.
  * 3D balance is solved.

This task adds the horizontal piece:
  * Velocity command sampled uniformly per episode (resampled mid-episode
    via UniformVelocityCommand's standard resampling timer).
  * Tracking reward = exp(-(v_actual - v_cmd)² / std²).
  * Command goes into the policy observation so the agent knows what to
    track.

The policy can in principle solve this by either stepping (alternating
legs) or hopping forward (both legs in phase) — we don't force a gait.
If it ends up sliding or dragging feet, we add foot air-time / clearance
rewards in a follow-up.
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

from mjlab_microduck.robot.slip_biped_constants import SLIP_BIPED_3D_ROBOT_CFG


# ---------------------------------------------------------------------------
# Custom MDP term — upright bonus (Gaussian on tilt, same form as balance/hop)
# ---------------------------------------------------------------------------


def upright_bonus(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    grav_xy = asset.data.projected_gravity_b[:, :2]
    err_sq = torch.sum(grav_xy ** 2, dim=1)
    return torch.exp(-err_sq / (std ** 2))


def ang_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Squared yaw rate.  Applied with negative weight as a spinning penalty.

    The stock `track_angular_velocity` is an exp(-err) bonus — once the
    yaw rate is large the term saturates near 0 and gives no signal at
    the margin (so "spin faster" costs nothing).  This L2 version grows
    monotonically, so spinning is always worse than not spinning."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_ang_vel_b[:, 2])


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

ROBOT_CFG = SceneEntityCfg("robot")
HIP_CFG = SceneEntityCfg("robot", joint_names=(r".*hip_(roll|pitch)",))
SPRING_CFG = SceneEntityCfg("robot", joint_names=(r".*spring",))


def make_slip_biped_velocity_3d_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # Commands: forward-only twist.  Other components clamped to zero so the
    # task is one-dimensional for now.
    commands = {
        "twist": UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),    # resample at 4 s mid-episode
            rel_standing_envs=0.1,               # 10% of envs get cmd = 0 (anchor "stand still")
            rel_heading_envs=0.0,
            heading_command=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.3),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
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
        # The command goes into the observation so the agent knows the target.
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
            actuator_names=(r".*hip_(roll|pitch)",),
            scale=0.5,
            use_default_offset=True,
        ),
    }

    events = {
        "reset_base": EventTermCfg(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
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
        "reset_hips": EventTermCfg(
            func=base_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.05, 0.05),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": HIP_CFG,
            },
        ),
    }

    rewards = {
        # Main signal: track the commanded forward velocity.
        # std=sqrt(0.15) ≈ 0.39 — same scale as MicroDuck velocity env,
        # gives a smooth gradient for vx errors up to ~0.5 m/s.
        "track_linear_velocity": RewardTermCfg(
            func=vel_mdp.track_linear_velocity,
            weight=3.0,
            params={
                "std": math.sqrt(0.15),
                "command_name": "twist",
                "asset_cfg": ROBOT_CFG,
            },
        ),
        # Yaw drift penalty.  Replaces the stock exp-form track_angular_velocity
        # (which saturates at 0 for large errors so it can't bite once the
        # robot is already spinning).  L2 grows without bound so "spin faster"
        # is always worse than "spin less".
        "ang_vel_z_l2": RewardTermCfg(
            func=ang_vel_z_l2,
            weight=-0.1,    # tuned: at observed steady-state yaw² ≈ 8, gives
                            # penalty −0.8/step vs +0.35/step linear tracking
                            # — strong gradient to stop spinning, doesn't kill
                            # exploration at training start
            params={"asset_cfg": ROBOT_CFG},
        ),
        "alive": RewardTermCfg(func=base_mdp.is_alive, weight=0.5),
        "upright": RewardTermCfg(
            func=upright_bonus,
            weight=1.0,
            params={"std": 0.2, "asset_cfg": ROBOT_CFG},
        ),
        # Stronger action_rate than hop — for locomotion we particularly want
        # to discourage high-frequency jitter on top of any rhythmic gait.
        "action_rate_l2": RewardTermCfg(
            func=base_mdp.action_rate_l2,
            weight=-0.05,
        ),
        "joint_vel_l2": RewardTermCfg(
            func=base_mdp.joint_vel_l2,
            weight=-1e-3,
            params={"asset_cfg": HIP_CFG},
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
            params={"minimum_height": 0.05, "asset_cfg": ROBOT_CFG},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(terrain_type="plane"),
            entities={"robot": SLIP_BIPED_3D_ROBOT_CFG},
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
            body_name="torso",
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
# PPO config — longer training budget than hop (locomotion is harder to
# find than a fixed-point or a vertical cycle).
# ---------------------------------------------------------------------------

SlipBipedVelocity3DRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="slip_biped_velocity_3d",
    run_name="velocity_3d",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=5000,
)
