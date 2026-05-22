"""SLIP biped (2D toy) entity configuration.

Small bipedal robot with two actuated hip-pitch joints and two passive
linear springs in the legs.  The torso is mounted to the world via three
scalar joints (slider_x, slider_z, hinge_pitch) instead of a free joint,
constraining motion to the sagittal plane.

mjlab treats this entity as "fixed base" (no free joint at the root) but
articulated, so root-pose writes are no-ops — episode resets must set the
planar joints explicitly via reset_joints_by_offset.

The 2D constraint is intentional for the first RL pass; the model will be
extended to a 3D version (free joint + hip_roll) once we have a baseline.
"""

import os
from pathlib import Path

import mujoco
from mjlab.actuator import DelayedActuatorCfg, XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


_SLIP_BIPED_DIR: Path = Path(os.path.dirname(__file__)) / "slip_biped"
SLIP_BIPED_2D_XML: Path = _SLIP_BIPED_DIR / "slip_biped_2d.xml"
SLIP_BIPED_3D_XML: Path = _SLIP_BIPED_DIR / "slip_biped_3d.xml"

assert SLIP_BIPED_2D_XML.exists(), f"XML not found: {SLIP_BIPED_2D_XML}"
assert SLIP_BIPED_3D_XML.exists(), f"XML not found: {SLIP_BIPED_3D_XML}"


def get_slip_biped_2d_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(SLIP_BIPED_2D_XML))


def get_slip_biped_3d_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(SLIP_BIPED_3D_XML))


# Home pose: planar joints at zero (torso at nominal pos="0 0 0.132", foot
# just touching floor), hips straight down, springs uncompressed.
HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        # Match every joint with a regex so mjlab's keyframe construction
        # gets a value for each.  All zero is the nominal standing pose.
        r".*": 0.0,
    },
    joint_vel={r".*": 0.0},
)


# Position-controlled hips, matching the <position kp=5 kv=0.1> defaults
# already declared in the XML.  Delay 0–3 ctrl steps (matches MicroDuck) —
# light realism for the toy, also breaks degenerate "instant response"
# assumptions that can lead to brittle policies.
slip_biped_actuators = DelayedActuatorCfg(
    delay_min_lag=0,
    delay_max_lag=3,
    base_cfg=XmlPositionActuatorCfg(joint_names_expr=(r".*hip_pitch",)),
)


SLIP_BIPED_2D_ROBOT_CFG = EntityCfg(
    spec_fn=get_slip_biped_2d_spec,
    init_state=HOME_FRAME,
    collisions=(),  # foot friction already set in XML defaults
    articulation=EntityArticulationInfoCfg(
        actuators=(slip_biped_actuators,),
        soft_joint_pos_limit_factor=0.95,
    ),
)


# ── 3D version ──────────────────────────────────────────────────────────────
# Floating base via <freejoint>, plus a hip_roll hinge added on each leg.
# Actuator regex picks up roll + pitch on both legs → 4 actuators.

HOME_FRAME_3D = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.132),       # torso centre = nominal standing height
    rot=(1.0, 0.0, 0.0, 0.0),    # identity quaternion (w, x, y, z)
    joint_pos={r".*": 0.0},
    joint_vel={r".*": 0.0},
)

slip_biped_3d_actuators = DelayedActuatorCfg(
    delay_min_lag=0,
    delay_max_lag=3,
    base_cfg=XmlPositionActuatorCfg(
        joint_names_expr=(r".*hip_(roll|pitch)",),
    ),
)

SLIP_BIPED_3D_ROBOT_CFG = EntityCfg(
    spec_fn=get_slip_biped_3d_spec,
    init_state=HOME_FRAME_3D,
    collisions=(),
    articulation=EntityArticulationInfoCfg(
        actuators=(slip_biped_3d_actuators,),
        soft_joint_pos_limit_factor=0.95,
    ),
)
