"""SLIP biped 3D — running (fast forward velocity tracking with flight phases).

Step up from velocity-3D in three ways:
  * Commanded forward velocity is always non-zero (vx ∈ [0.2, 0.6] m/s) —
    the policy can't satisfy the task by standing still.
  * Small `both_feet_airborne` bonus (+1.0) to push the gait toward real
    flight phases instead of sliding/shuffling.
  * Yaw drift is no longer penalised.  The sphere-footed SLIP biped has no
    yaw-anchoring contact geometry, so the policy can't actually control
    body heading.  Asking it to do so just confuses the learning.  We use
    body-frame velocity tracking, accept whatever heading the policy settles
    on, and call it running.

If the trained policy slides forward without flight phases, bump the airborne
weight or tighten action_rate_l2 to discourage low-amplitude shuffling.
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
# Custom MDP terms — reused from hop env (upright + airborne detection)
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


def both_feet_airborne(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.0005,
) -> torch.Tensor:
    """1.0 when both spring compressions are below `threshold` (legs unloaded).

    Spring qpos sits at the 0 limit when the leg isn't bearing load — so
    near-zero on both is a reliable airborne signal.  Same proxy used in
    the hop-in-place task.
    """
    asset: Entity = env.scene[asset_cfg.name]
    spring_ids, _ = asset.find_joints([r".*spring"])
    spring_q = asset.data.joint_pos[:, spring_ids]
    return (spring_q < threshold).all(dim=1).float()


def vertical_motion(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """vz² of the torso — rewards vertical bounciness.

    Walking has near-zero vz; running has large vz oscillation as the body
    pumps up and down.  Unlike the binary `both_feet_airborne` signal, this
    is continuous and gives a smooth gradient that the walking policy can
    climb toward running incrementally (bigger bounces always score more).
    """
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w[:, 2] ** 2


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

ROBOT_CFG = SceneEntityCfg("robot")
HIP_CFG = SceneEntityCfg("robot", joint_names=(r".*hip_(roll|pitch)",))
SPRING_CFG = SceneEntityCfg("robot", joint_names=(r".*spring",))


def make_slip_biped_run_3d_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    # Commands: forward-only twist, ALWAYS positive vx so the policy can't
    # satisfy the task by standing.  No lateral, no yaw command (the latter
    # is moot for this robot — sphere feet can't generate yaw torque).
    commands = {
        "twist": UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(4.0, 4.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(0.2, 0.6),    # always running speed
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
        # Drop from 2-5 cm above nominal each episode.  Impact velocity ~1 m/s
        # compresses springs ~25 mm (close to the full 30 mm range), pre-loading
        # them with elastic energy.  This bootstraps airborne behaviour: every
        # episode starts with both legs unloaded (free "airborne" reward at
        # step 0) and a fully sprung landing — the policy only needs to learn
        # to *maintain* the bounce, not discover how to start one from rest.
        "reset_base": EventTermCfg(
            func=base_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"z": (0.02, 0.05)},
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
        # Main signal — track the commanded forward speed.
        "track_linear_velocity": RewardTermCfg(
            func=vel_mdp.track_linear_velocity,
            weight=3.0,
            params={
                "std": math.sqrt(0.15),
                "command_name": "twist",
                "asset_cfg": ROBOT_CFG,
            },
        ),
        # Continuous "vertical bounciness" signal: walking has vz ≈ 0,
        # running has vz oscillation of ~0.5-1 m/s peak.  Unlike the binary
        # airborne bonus this gives a smooth gradient — a slightly bouncier
        # gait scores slightly higher, so PPO can climb from walking toward
        # running incrementally without needing to discover full airborne
        # state in one exploration step.
        "vertical_motion": RewardTermCfg(
            func=vertical_motion,
            weight=2.0,
            params={"asset_cfg": ROBOT_CFG},
        ),
        # Binary airborne bonus — kicks in once the policy is bouncy enough
        # to actually leave the ground.  vertical_motion provides the
        # gradient TO this regime; this term provides the payoff for being
        # IN it.
        "airborne": RewardTermCfg(
            func=both_feet_airborne,
            weight=3.0,
            params={"asset_cfg": ROBOT_CFG, "threshold": 0.0005},
        ),
        "alive": RewardTermCfg(func=base_mdp.is_alive, weight=0.5),
        "upright": RewardTermCfg(
            func=upright_bonus,
            weight=1.0,
            params={"std": 0.2, "asset_cfg": ROBOT_CFG},
        ),
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
# PPO config — same shape as velocity-3D.
# ---------------------------------------------------------------------------

SlipBipedRun3DRlCfg = RslRlOnPolicyRunnerCfg(
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
        entropy_coef=0.02,    # bumped from 0.01 — more exploration to escape
                              # the walking local optimum on the way to running
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
    experiment_name="slip_biped_run_3d",
    run_name="run_3d",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=5000,
)
