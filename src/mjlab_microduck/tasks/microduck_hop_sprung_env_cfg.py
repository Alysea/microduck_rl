"""MicroDuck (sprung shank) — hop in place RL task.

First training task on the sprung-microduck prototype.  Goal: bounce
vertically using the passive shank springs, without horizontal drift.
Reward structure ports directly from the SLIP-biped hop task which
trained cleanly:

    alive            +0.5  (survival floor)
    upright          +1.0  exp(-|grav_b_xy|²/0.2²)
    airborne         +3.0  1 when both shank springs are uncompressed
    stay_in_place    +1.0  exp(-|v_xy|²/0.1²)
    action_rate_l2  -0.05  smoothness (lighter than balance because
                            hopping NEEDS cyclic hip motion)
    joint_vel_l2    -1e-3  on actuated joints

Reset:
  * Base drop from 0-4 cm above the sprung HOME standing height (0.131 m).
    Bootstraps airborne behaviour — every episode starts with at least
    the first few control steps unloaded, giving PPO non-zero airborne
    reward from step 0 instead of having to discover it cold.
  * All non-actuated joints (the two shank springs) explicitly reset to
    default — without this, after the first crashed episode the next
    one starts mid-fall and ep length collapses to 1.
  * Small ±0.05 rad noise on the sagittal leg joints (hip_pitch, knee,
    ankle) to break left-right symmetry.

Termination: time_out (8 s), bad_orientation (1.0 rad tilt), fallen_height
(trunk z < 0.06 m → dropped ~70 mm below standing).
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

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_SPRUNG_ROBOT_CFG


# ---------------------------------------------------------------------------
# Custom MDP terms (same forms as the SLIP-biped hop task)
# ---------------------------------------------------------------------------


def upright_bonus(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Bell on torso tilt: exp(-|grav_b_xy|² / std²)."""
    asset: Entity = env.scene[asset_cfg.name]
    grav_xy = asset.data.projected_gravity_b[:, :2]
    return torch.exp(-torch.sum(grav_xy ** 2, dim=1) / (std ** 2))


def both_feet_airborne(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.0005,
) -> torch.Tensor:
    """1 when both shank springs are uncompressed (feet unloaded).

    Spring range[0]=0 pins q at the joint limit when the leg isn't bearing
    load, so spring qpos < ~0.5 mm on a leg is a clean proxy for that
    foot being off the ground.  Both → robot is airborne.
    """
    asset: Entity = env.scene[asset_cfg.name]
    spring_ids, _ = asset.find_joints([r".*spring"])
    spring_q = asset.data.joint_pos[:, spring_ids]
    return (spring_q < threshold).all(dim=1).float()


def stay_in_place(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Bell on horizontal velocity: exp(-|v_xy|² / std²)."""
    asset: Entity = env.scene[asset_cfg.name]
    vxy = asset.data.root_link_lin_vel_w[:, :2]
    return torch.exp(-torch.sum(vxy ** 2, dim=1) / (std ** 2))


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

ROBOT_CFG       = SceneEntityCfg("robot")
SPRING_CFG      = SceneEntityCfg("robot", joint_names=(r".*spring",))
LEG_CFG         = SceneEntityCfg(
    "robot",
    joint_names=(r".*hip_(pitch|roll|yaw)", r".*knee", r".*ankle"),
)
ALL_ACTUATED_CFG = SceneEntityCfg(
    "robot",
    joint_names=(r"^(?!.*spring).*",),   # everything except springs
)


def make_microduck_hop_sprung_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # 14 actuated joints (hip yaw/roll/pitch × 2 + knee × 2 + ankle × 2
    # + neck/head × 4).  Position offsets from default HOME pose.
    actions = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(r".*",),     # match all joints with an actuator
            scale=0.5,
            use_default_offset=True,
        ),
    }

    policy_terms = {
        "base_lin_vel":      ObservationTermCfg(func=base_mdp.base_lin_vel),
        "base_ang_vel":      ObservationTermCfg(func=base_mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=base_mdp.projected_gravity),
        "joint_pos":         ObservationTermCfg(func=base_mdp.joint_pos_rel),  # 16 (incl. springs)
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

    # Reset order matters — events fire in declaration order.
    # 1) Root pose: drop from 0-40 mm above the half-bent HOME standing
    #    height (0.131 m) → every episode bootstraps airborne behaviour.
    # 2) Springs restored to default (q=0) — the SLIP-2D lesson: never
    #    skip this or post-crash resets leave springs in their final
    #    fallen state.
    # 3) Actuated joints restored to HOME with small noise on the
    #    sagittal-plane leg joints to break left-right symmetry.
    events = {
        "reset_base": EventTermCfg(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"z": (0.131, 0.171)},   # standing height + 0-40 mm drop
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

    rewards = {
        "alive": RewardTermCfg(func=base_mdp.is_alive, weight=0.5),
        "upright": RewardTermCfg(
            func=upright_bonus,
            weight=1.0,
            params={"std": 0.2, "asset_cfg": ROBOT_CFG},
        ),
        # MAIN SIGNAL — keep it the dominant reward term, same shape that
        # worked for the SLIP hop task.
        "airborne": RewardTermCfg(
            func=both_feet_airborne,
            weight=3.0,
            params={"asset_cfg": ROBOT_CFG, "threshold": 0.0005},
        ),
        "stay_in_place": RewardTermCfg(
            func=stay_in_place,
            weight=1.0,
            params={"std": 0.1, "asset_cfg": ROBOT_CFG},
        ),
        # Moderate action smoothness — hopping needs rhythmic hip motion
        # so we don't want the harsh -0.1 we'd pick for pure balance.
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
        commands={},
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
        decimation=4,           # 200 Hz physics, 50 Hz control
        episode_length_s=8.0,
    )


# ---------------------------------------------------------------------------
# PPO config — entropy_coef bumped per the SLIP-run lesson (more
# exploration helps escape the "stand still" local optimum).
# ---------------------------------------------------------------------------

MicroduckHopSprungRlCfg = RslRlOnPolicyRunnerCfg(
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
        entropy_coef=0.02,         # SLIP-run lesson — bumped from default 0.01
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
    experiment_name="microduck_hop_sprung",
    run_name="hop_sprung",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=3000,
)
