"""Sprung-foot task variant — an idealised 1-DoF compliant foot.

``make_sprung_variant(cfg, stiffness)`` converts a Run-task env cfg into its
sprung counterpart, in the same shape as ``tasks/backlash.py``. Four changes:

1. Swap the robot for a sprung-foot model at the requested stiffness.
2. Shift the ``com_height_target`` band by ``h_add``. The sprung robot stands
   taller, so without this it is penalised for its geometry before compliance
   is in play — and the whole point of the locked control arm is to isolate
   geometry from compliance.
3. Scope the ``pose`` and ``dof_pos_limits`` rewards off the spring joints. A
   passive spring has no pose target and legitimately rides its limits.
4. Register the compression monitor, whose reading decides whether any speed
   number from this variant means anything.

``travel=0.0`` produces the LOCKED control variant: identical geometry and mass,
no compliance.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.robot.sprung_foot import (
    H_ADD,
    SPRING_JOINTS,
    TRAVEL,
    make_sprung_foot_robot_cfg,
)
from mjlab_microduck.tasks import mdp as microduck_mdp

SPRING_MONITOR_WEIGHT = 1.0

# Excludes the two foot springs while keeping every other joint, including the
# neck/head exclusions the velocity env already applies.
_NO_SPRING = r"^(?!passive_).*"


def make_sprung_variant(
    cfg: ManagerBasedRlEnvCfg,
    stiffness: float,
    travel: float = TRAVEL,
    h_add: float = H_ADD,
) -> ManagerBasedRlEnvCfg:
    """Convert a Run-task env cfg into its sprung-foot counterpart."""
    # 1. Robot.
    cfg.scene.entities = {
        **cfg.scene.entities,
        "robot": make_sprung_foot_robot_cfg(
            stiffness=stiffness, travel=travel, h_add=h_add
        ),
    }

    # 2. The sprung robot stands h_add taller — translate the CoM band, do not
    #    widen it.
    com = cfg.rewards["com_height_target"]
    com.params["target_height_min"] = com.params["target_height_min"] + h_add
    com.params["target_height_max"] = com.params["target_height_max"] + h_add

    # 3. A passive spring has no pose target, and rides its own limits by
    #    design. Deepcopy first: base templates share SceneEntityCfg objects
    #    across make() calls, so mutating in place would leak into other tasks.
    pose = cfg.rewards.get("pose")
    if pose is not None and "asset_cfg" in pose.params:
        ac = deepcopy(pose.params["asset_cfg"])
        if not any("passive_" in p for p in ac.joint_names):
            ac.joint_names = tuple(ac.joint_names) + (_NO_SPRING,)
        pose.params["asset_cfg"] = ac

    dof_limits = cfg.rewards.get("dof_pos_limits")
    if dof_limits is not None and "asset_cfg" not in dof_limits.params:
        dof_limits.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=(_NO_SPRING,)
        )

    # 4. Compression monitor. Returns zeros, so the weight only has to be
    #    non-zero for RewardManager.compute to call it at all.
    cfg.rewards["spring_compression_monitor"] = RewardTermCfg(
        func=microduck_mdp.spring_compression_monitor,
        weight=SPRING_MONITOR_WEIGHT,
        params={"joint_names": SPRING_JOINTS, "travel": travel},
    )

    return cfg
