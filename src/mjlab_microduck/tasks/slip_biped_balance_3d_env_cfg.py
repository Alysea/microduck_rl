"""SLIP biped 3D — "don't fall" RL task with hip_roll + freejoint.

Step up from the 2D version:
  * Floating base (freejoint) instead of three planar joints — stock mjlab
    helpers (projected_gravity_b, base_lin_vel, reset_root_state_…) now
    work directly, so we use them and drop most of the custom MDP shims
    we needed in 2D.
  * Each hip gains a roll joint → 4-D action.
  * Lateral stability becomes a real failure mode, so the task is harder
    than 2D even with the same reward shape.

Observation (25-D):
    base_lin_vel (3) + base_ang_vel (3) + projected_gravity (3)
    + joint_pos_rel (6) + joint_vel_rel (6) + last_action (4)

Reward:
    alive            +1.0
    upright bonus    +2.0   exp(-|grav_b_xy|² / 0.2²)  (custom — bell on upright)
    action_rate_l2   -0.01
    joint_vel_l2     -1e-3  (hip joints only)

Termination:
    bad_orientation  (tilt > 1.0 rad ≈ 57° from upright)
    root_height < 0.07 m  (torso dropped >5 cm below nominal)
    timeout at 8 s
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
# Custom MDP term — the only one we still need (a Gaussian upright bonus
# instead of mjlab's L2 penalty, to match the 2D reward shape and keep the
# alive bonus dominant at training start)
# ---------------------------------------------------------------------------


def upright_bonus(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """exp(-|grav_b_xy|² / std²) — peaks at 1 when torso is upright.

    `projected_gravity_b` is gravity expressed in the torso body frame: for
    a perfectly upright torso it is (0, 0, -1); any tilt rotates that into
    a non-zero xy component, with |xy| ≈ sin(tilt_angle).  So the squared
    xy norm is a tilt² measure that's well-behaved through any axis of tilt.
    """
    asset: Entity = env.scene[asset_cfg.name]
    grav_xy = asset.data.projected_gravity_b[:, :2]
    err_sq = torch.sum(grav_xy ** 2, dim=1)
    return torch.exp(-err_sq / (std ** 2))


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

ROBOT_CFG = SceneEntityCfg("robot")
HIP_CFG = SceneEntityCfg("robot", joint_names=(r".*hip_(roll|pitch)",))
SPRING_CFG = SceneEntityCfg("robot", joint_names=(r".*spring",))


def make_slip_biped_balance_3d_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # 25-D policy observation, all from stock mjlab terms.
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

    # Action: position offsets on all 4 hip joints (roll + pitch on each leg).
    # `actuator_names` is matched against actuated joint names — `.*hip_(roll|
    # pitch)` hits exactly the four hip hinges.
    actions = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(r".*hip_(roll|pitch)",),
            scale=0.5,
            use_default_offset=True,
        ),
    }

    # Reset order matters — events fire in declaration order.
    # 1) Root freejoint → default pose (no noise, just restore from fallen state).
    # 2) Springs       → defaults (zero compression, no velocity).
    # 3) Hip joints    → defaults + small noise to break left-right symmetry.
    events = {
        "reset_base": EventTermCfg(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},        # all six pose axes default to (0, 0)
                "velocity_range": {},    # zero velocity
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
        "alive": RewardTermCfg(func=base_mdp.is_alive, weight=1.0),
        "upright": RewardTermCfg(
            func=upright_bonus,
            weight=2.0,
            params={"std": 0.2, "asset_cfg": ROBOT_CFG},
        ),
        "action_rate_l2": RewardTermCfg(
            func=base_mdp.action_rate_l2,
            weight=-0.01,
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
            params={"minimum_height": 0.07, "asset_cfg": ROBOT_CFG},
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
        decimation=4,           # 200 Hz physics, 50 Hz control
        episode_length_s=8.0,
    )


# ---------------------------------------------------------------------------
# PPO config — slightly larger network than 2D for the bigger obs / action.
# ---------------------------------------------------------------------------

SlipBipedBalance3DRlCfg = RslRlOnPolicyRunnerCfg(
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
    experiment_name="slip_biped_balance_3d",
    run_name="balance_3d",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=2000,
)
