import os
from pathlib import Path

import mujoco
from mjlab.actuator import DelayedActuatorCfg, XmlPositionActuatorCfg
from mjlab_microduck.actuator.bam_params import make_bam_m6_actuator_cfg, make_bam_m4_actuator_cfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg


_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "microduck"

MICRODUCK_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"
MICRODUCK_STANDUP_XML: Path = _ROBOT_DIR / "robot_standup.xml"
MICRODUCK_GROUND_PICK_XML: Path = _ROBOT_DIR / "robot_ground_pick.xml"
MICRODUCK_WALK_ROLLERS_XML: Path = _ROBOT_DIR / "robot_walk_rollers.xml"
MICRODUCK_WALK_SPRUNG_XML: Path = _ROBOT_DIR / "robot_walk_sprung.xml"

assert MICRODUCK_WALK_XML.exists(), f"XML not found: {MICRODUCK_WALK_XML}"
assert MICRODUCK_STANDUP_XML.exists(), f"XML not found: {MICRODUCK_STANDUP_XML}"
assert MICRODUCK_GROUND_PICK_XML.exists(), f"XML not found: {MICRODUCK_GROUND_PICK_XML}"
assert MICRODUCK_WALK_ROLLERS_XML.exists(), f"XML not found: {MICRODUCK_WALK_ROLLERS_XML}"
assert MICRODUCK_WALK_SPRUNG_XML.exists(), f"XML not found: {MICRODUCK_WALK_SPRUNG_XML}"


def get_walk_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_XML))


def get_walk_sprung_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_SPRUNG_XML))


def get_standup_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_STANDUP_XML))


def get_ground_pick_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_GROUND_PICK_XML))


def get_walk_rollers_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MICRODUCK_WALK_ROLLERS_XML))


HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # Lower body
        r".*hip_yaw.*": 0.0,
        r".*hip_roll.*": 0.0,
        r".*left_hip_pitch.*": 0.6,
        r".*right_hip_pitch.*": -0.6,
        r".*left_knee.*": -1.2,
        r".*right_knee.*": 1.2,
        r".*left_ankle.*": 0.6,
        r".*right_ankle.*": -0.6,
        # Head
        r".*neck_pitch.*": -0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (1.0,)},
)

# -- Old actuator (XML position, MuJoCo built-in PD + friction) --
# actuators = DelayedActuatorCfg(
    # delay_min_lag=0,
    # delay_max_lag=3,
    # base_cfg=XmlPositionActuatorCfg(joint_names_expr=(r".*",)),
# )

# -- BAM M6 actuator (full voltage control + load-dependent friction) --
actuators = DelayedActuatorCfg(
    delay_min_lag=0,
    delay_max_lag=3,
    base_cfg=make_bam_m6_actuator_cfg(),
)

# -- BAM M4 actuator
# actuators = DelayedActuatorCfg(
    # delay_min_lag=0,
    # delay_max_lag=3,
    # base_cfg=make_bam_m4_actuator_cfg(),
# )

MICRODUCK_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)


# ── Sprung-shank prototype ─────────────────────────────────────────────────
# Same as the walk model but with a passive prismatic spring inserted in
# each shank between knee and ankle (see robot_walk_sprung.xml).  Two extra
# passive DoFs (left_shank_spring, right_shank_spring); the 14 actuated
# joints are identical to the walk model in name, order, and dynamics.
#
# The BAM actuator regex needs to skip the spring joints — otherwise mjlab
# would try to actuate them.  Negative-lookahead pattern `^(?!.*spring).*`
# matches every joint EXCEPT those containing "spring".
sprung_actuators = DelayedActuatorCfg(
    delay_min_lag=0,
    delay_max_lag=3,
    base_cfg=make_bam_m6_actuator_cfg(joint_names_expr=(r"^(?!.*spring).*",)),
)

# Sprung-specific HOME pose ("half-bent"): each knee bent half as much as
# in the rigid HOME (-0.6 instead of -1.2), with hip and ankle scaled to
# keep the body upright.  The shank world angle drops from ~34° to ~17°,
# so spring compression has roughly half the horizontal foot-drift
# component → much less destabilising tipping moment under body weight.
# Longer effective leg means trunk starts ~131 mm above ground (vs 120 for
# rigid).  See conversation on viewer behaviour for full reasoning.
HOME_FRAME_SPRUNG = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),       # reset event adds the trunk z offset
    rot=(1.0, 0.0, 0.0, 0.0),
    joint_pos={
        r".*hip_yaw.*": 0.0,
        r".*hip_roll.*": 0.0,
        r".*left_hip_pitch.*": 0.3,
        r".*right_hip_pitch.*": -0.3,
        r".*left_knee.*": -0.6,
        r".*right_knee.*": 0.6,
        r".*left_ankle.*": 0.3,
        r".*right_ankle.*": -0.3,
        r".*neck_pitch.*": -0.3491,
        r".*head_pitch.*": 0.3491,
        r".*head_yaw.*": 0.0,
        r".*head_roll.*": 0.0,
        r".*spring.*": 0.0,
    },
    joint_vel={".*": 0.0},
)

MICRODUCK_WALK_SPRUNG_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_sprung_spec,
    init_state=HOME_FRAME_SPRUNG,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(sprung_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_STANDUP_ROBOT_CFG = EntityCfg(
    spec_fn=get_standup_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

MICRODUCK_GROUND_PICK_ROBOT_CFG = EntityCfg(
    spec_fn=get_ground_pick_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

# Roller skate robot: passive wheel joints have no actuators in the XML.
# Use a separate actuator config that explicitly excludes passive joints so
# the action space stays 14-dimensional (same as the walk robot).
roller_actuators = DelayedActuatorCfg(
    delay_min_lag=0,
    delay_max_lag=3,
    base_cfg=XmlPositionActuatorCfg(joint_names_expr=(r"^(?!passive_).*",)),
)

MICRODUCK_WALK_ROLLERS_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_rollers_spec,
    init_state=HOME_FRAME,
    collisions=(),  # roller wheel collision geoms have no explicit names; XML defaults apply
    articulation=EntityArticulationInfoCfg(
        actuators=(roller_actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainImporterCfg

    SCENE_CFG = SceneCfg(
        terrain=TerrainImporterCfg(terrain_type="plane"),
        entities={"robot": MICRODUCK_WALK_ROBOT_CFG},
    )

    scene = Scene(SCENE_CFG, device="cuda:0")
    viewer.launch(scene.compile())
