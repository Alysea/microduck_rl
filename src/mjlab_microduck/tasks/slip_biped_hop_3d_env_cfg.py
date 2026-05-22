"""SLIP biped 3D — "hop in place" RL task.

Same robot as balance-3D, different reward.  Goal: bounce vertically using
the leg springs without drifting horizontally.

Detecting "airborne":
    Each leg's spring is a slide joint with range [0, 0.03] — q=0 is fully
    extended (rest), q>0 is compression.  When the foot is on the ground
    bearing load, q>0; when the foot lifts off, gravity tries to pull q
    below 0 but the joint limit pins it at 0.  So q≈0 on a leg is a clean
    proxy for "that foot is unloaded".  Both springs unloaded ⇒ both feet
    off the ground ⇒ robot is airborne.

Reward weights tuned so the airborne bonus dominates (the *new* thing the
policy must discover) while the upright + stay-in-place terms remain
strong enough to prevent obvious gaming (tipping over forwards to get
"both feet off" doesn't satisfy the upright term).
"""

from __future__ import annotations

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
from mjlab.terrains import TerrainImporterCfg
from mjlab.viewer import ViewerConfig

from mjlab_microduck.robot.slip_biped_constants import SLIP_BIPED_3D_ROBOT_CFG


# ---------------------------------------------------------------------------
# Custom MDP terms
# ---------------------------------------------------------------------------


def upright_bonus(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """exp(-|grav_b_xy|² / std²) — peaks at 1 when torso is upright."""
    asset: Entity = env.scene[asset_cfg.name]
    grav_xy = asset.data.projected_gravity_b[:, :2]
    err_sq = torch.sum(grav_xy ** 2, dim=1)
    return torch.exp(-err_sq / (std ** 2))


def both_feet_airborne(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.0005,
) -> torch.Tensor:
    """1.0 when both spring compressions are below `threshold` (legs unloaded).

    The slide-joint range[0]=0 pins q at zero when the leg isn't compressed
    by ground contact, so a near-zero qpos on a spring joint reliably means
    that foot is off the ground.
    """
    asset: Entity = env.scene[asset_cfg.name]
    spring_ids, _ = asset.find_joints([r".*spring"])
    spring_q = asset.data.joint_pos[:, spring_ids]   # (num_envs, 2)
    return (spring_q < threshold).all(dim=1).float()


def stay_in_place(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """exp(-|v_xy|² / std²) — peaks at zero horizontal velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    vxy = asset.data.root_link_lin_vel_w[:, :2]
    err_sq = torch.sum(vxy ** 2, dim=1)
    return torch.exp(-err_sq / (std ** 2))


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

ROBOT_CFG = SceneEntityCfg("robot")
HIP_CFG = SceneEntityCfg("robot", joint_names=(r".*hip_(roll|pitch)",))
SPRING_CFG = SceneEntityCfg("robot", joint_names=(r".*spring",))


def make_slip_biped_hop_3d_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    policy_terms = {
        "base_lin_vel":      ObservationTermCfg(func=base_mdp.base_lin_vel),
        "base_ang_vel":      ObservationTermCfg(func=base_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity),
        "joint_pos":         ObservationTermCfg(func=base_mdp.joint_pos_rel),
        "joint_vel":         ObservationTermCfg(func=base_mdp.joint_vel_rel),
        "actions":           ObservationTermCfg(func=base_mdp.last_action),
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
        # Survival floor — small so it doesn't dwarf the hop signal.
        "alive": RewardTermCfg(func=base_mdp.is_alive, weight=0.5),
        # Posture regularizer — halved vs balance task; we still want the
        # robot upright while it hops, but not so strongly that minimising
        # tilt dominates over the hop reward.
        "upright": RewardTermCfg(
            func=upright_bonus,
            weight=1.0,
            params={"std": 0.2, "asset_cfg": ROBOT_CFG},
        ),
        # Main signal: both legs off the ground.
        "airborne": RewardTermCfg(
            func=both_feet_airborne,
            weight=3.0,
            params={"asset_cfg": ROBOT_CFG, "threshold": 0.0005},
        ),
        # Anti-drift: penalises any strategy that "hops" by running off.
        "stay_in_place": RewardTermCfg(
            func=stay_in_place,
            weight=1.0,
            params={"std": 0.1, "asset_cfg": ROBOT_CFG},
        ),
        # Smoothness — moderate, lower than I'd want for pure balance
        # because hopping needs cyclic hip motion (rhythmic ≠ vibration).
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
        # Looser than balance — during a hop landing the torso transiently
        # dips below the balance threshold of 0.07 m.
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
        commands={},
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
# PPO config — same network as balance-3D
# ---------------------------------------------------------------------------

SlipBipedHop3DRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(128, 128, 64),
        critic_hidden_dims=(128, 128, 64),
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
    experiment_name="slip_biped_hop_3d",
    run_name="hop_3d",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=3000,
)
