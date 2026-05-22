"""SLIP biped 2D — minimal "don't fall" RL task.

First RL experiment on the 2D toy.  Single objective: keep the torso
upright for as long as possible.  No velocity command, no curriculum,
no domain randomization, no terrain.

Why a custom env (not built on `make_velocity_env_cfg()`):
- mjlab's velocity-env template wires in twist commands, foot-clearance
  rewards, terrain, etc. that this toy doesn't need.
- The toy's "base" is three scalar joints (slider_x + slider_z + hinge_pitch)
  rather than a free joint.  Most mjlab stock functions read from `xpos` /
  `xquat` and work transparently for both layouts, but `reset_base` is a
  no-op on fixed-base entities, so resets go through `reset_joints_by_offset`.

Observation (13-D, all from stock mjlab terms with one custom):
    slider_z + hinge_pitch + their velocities + slider_x velocity   (5)
    hip positions × 2  + hip velocities × 2                         (4)
    spring compressions × 2                                         (2)
    last_action × 2                                                 (2)

Action (2-D):  position offsets from default (0 rad) for each hip pitch.

Reward (weights tuned so "stay alive" dominates early training):
    alive            +1.0
    upright bonus    +2.0    exp(-pitch²/0.1²) — bell on pitch=0
    action_rate_l2   -0.01
    joint_vel_l2     -1e-3
    action_l2        -0.001

Termination:
    |hinge_pitch| > 1.0 rad (~57°)   → fallen
    slider_z < -0.05 m               → torso collapsed > 5 cm
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

from mjlab_microduck.robot.slip_biped_constants import SLIP_BIPED_2D_ROBOT_CFG


# ---------------------------------------------------------------------------
# Custom MDP terms
# ---------------------------------------------------------------------------


def torso_upright_bonus(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """exp(-pitch² / std²) — bell-shaped reward peaking at upright torso."""
    asset: Entity = env.scene[asset_cfg.name]
    pitch_ids, _ = asset.find_joints(["hinge_pitch"])
    pitch = asset.data.joint_pos[:, pitch_ids[0]]
    return torch.exp(-(pitch ** 2) / (std ** 2))


def torso_pitch_out_of_range(
    env: ManagerBasedRlEnv,
    max_pitch_rad: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when the torso pitch exceeds ±max_pitch_rad."""
    asset: Entity = env.scene[asset_cfg.name]
    pitch_ids, _ = asset.find_joints(["hinge_pitch"])
    pitch = asset.data.joint_pos[:, pitch_ids[0]]
    return torch.abs(pitch) > max_pitch_rad


def torso_too_low(
    env: ManagerBasedRlEnv,
    min_slider_z: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when the torso has dropped below `min_slider_z` (slider_z
    joint position, in metres relative to the body's nominal location)."""
    asset: Entity = env.scene[asset_cfg.name]
    sz_ids, _ = asset.find_joints(["slider_z"])
    z = asset.data.joint_pos[:, sz_ids[0]]
    return z < min_slider_z


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

ROBOT_CFG = SceneEntityCfg("robot")
HIP_CFG = SceneEntityCfg("robot", joint_names=(r".*hip_pitch",))
# Everything except the actuated hip pitches: planar mount + springs.
# We reset these to defaults explicitly because we have no freejoint — without
# this, after the first episode terminates with the robot tipped over, the
# next "reset" leaves slider_z / hinge_pitch / springs at the fallen state
# and the new episode terminates again on step 1.
NON_HIP_CFG = SceneEntityCfg(
    "robot",
    joint_names=("slider_x", "slider_z", "hinge_pitch",
                 "left_spring", "right_spring"),
)


def make_slip_biped_balance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # Observations: stock joint_pos_rel / joint_vel_rel give all 7 joints
    # (slider_x, slider_z, hinge_pitch, hip_L, spring_L, hip_R, spring_R).
    # We send all of them — the policy can learn to weigh them.  slider_x
    # is included as well; for "don't fall" it's irrelevant (task is shift-
    # invariant), but reaching it into a custom slice isn't worth the
    # complexity for a first pass.
    policy_terms = {
        "joint_pos": ObservationTermCfg(
            func=base_mdp.joint_pos_rel,
        ),
        "joint_vel": ObservationTermCfg(
            func=base_mdp.joint_vel_rel,
        ),
        "actions": ObservationTermCfg(func=base_mdp.last_action),
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

    # Actions: 0.5 rad full-scale on each hip.
    # NOTE: despite the kwarg name, `actuator_names` is matched against
    # joint names (filtered to those with an actuator).  Using the joint
    # regex r".*hip_pitch" hits exactly the two actuated hips.
    actions = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(r".*hip_pitch",),
            scale=0.5,
            use_default_offset=True,
        ),
    }

    # Events execute in declaration order on each reset.
    # 1) Restore all non-hip joints to their keyframe defaults (zero offset).
    #    This is the freejoint-equivalent reset for our planar-mount robot.
    # 2) Then add small ±0.05 rad noise to the hips, breaking left-right
    #    symmetry so the policy sees a non-degenerate task.
    events = {
        "reset_state": EventTermCfg(
            func=base_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": NON_HIP_CFG,
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
            func=torso_upright_bonus,
            weight=2.0,
            params={"std": 0.1, "asset_cfg": ROBOT_CFG},
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
        "fallen_pitch": TerminationTermCfg(
            func=torso_pitch_out_of_range,
            params={"max_pitch_rad": 1.0, "asset_cfg": ROBOT_CFG},
        ),
        "fallen_height": TerminationTermCfg(
            func=torso_too_low,
            params={"min_slider_z": -0.05, "asset_cfg": ROBOT_CFG},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(terrain_type="plane"),
            entities={"robot": SLIP_BIPED_2D_ROBOT_CFG},
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
            distance=0.6,
            elevation=-10.0,
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
# PPO config — small network for a tiny observation/action space
# ---------------------------------------------------------------------------

SlipBipedBalanceRlCfg = RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=(128, 64),
        critic_hidden_dims=(128, 64),
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
    experiment_name="slip_biped_balance_2d",
    run_name="balance",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=2000,
)
