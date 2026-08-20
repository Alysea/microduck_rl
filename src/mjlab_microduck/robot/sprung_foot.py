"""Sprung-foot robot model — an idealised 1-DoF compliant foot accessory.

Built PROGRAMMATICALLY from the canonical ``robot_walk.xml`` rather than as a
forked XML. The abandoned ``test_spring`` branch forked the XML, and its 50-line
delta became unusable once ``robot_walk.xml`` moved by 310 insertions. Adding two
bodies to the live spec tracks every upstream change to the base model for free.

The mechanism modelled here is deliberately idealised: one prismatic spring per
foot. That is not a shortcut — it is the design target. A rigid 1-DoF
translating mechanism (a prismatic slide, or a Sarrus linkage) maps exactly onto
a MuJoCo ``slide`` joint, so the kinematics carry no sim-to-real gap. A
Kangoo-style leaf flexure would need a discretised multi-body chain or
deformables, and was rejected on that basis. See the Phase 2 spec.
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

from mjlab_microduck.robot.microduck_constants import (
    FULL_COLLISION,
    HOME_FRAME,
    actuators,
    get_walk_spec,
)

# Local +y of the ankle bodies maps to world [0, 0.087, 0.996] — almost straight
# up. So a slide along +y means positive q = compression (pad moves toward the
# body). Local +z is nearly HORIZONTAL; using it would slide the foot sideways.
SPRING_AXIS = (0.0, 1.0, 0.0)

# Distance from the ankle body origin down to the existing sole's contact plane,
# measured at the home pose. The pad is placed h_add BELOW that, which is what
# makes the sprung robot taller than the rigid one.
ANKLE_TO_SOLE = 0.025

H_ADD = 0.025      # height the mechanism adds under the foot (m)
PAD_MASS = 0.020   # mechanism mass per foot (kg) — distal, so it is modelled
TRAVEL = 0.015     # spring stroke (m)
DAMPING = 0.5      # N.s/m — represents a good steel spring, low hysteresis

SPRING_JOINTS = ("passive_left_foot_spring", "passive_right_foot_spring")

# Contact pad half-extents (m). Local y is world-up here, so the middle number
# is half the pad thickness.
_PAD_HALF_EXTENTS = (0.020, 0.004, 0.014)


def make_sprung_foot_spec_fn(
    stiffness: float,
    travel: float = TRAVEL,
    damping: float = DAMPING,
    h_add: float = H_ADD,
    pad_mass: float = PAD_MASS,
) -> Callable[[], mujoco.MjSpec]:
    """Build a zero-argument ``spec_fn`` for a sprung-foot MicroDuck.

    ``EntityCfg.spec_fn`` must take no arguments, so the spring parameters are
    captured in a closure. ``travel=0.0`` yields the LOCKED control variant:
    identical geometry and mass, no compliance.

    Args:
        stiffness: spring rate in N/m, applied to both feet.
        travel: stroke in m. 0.0 locks the spring.
        damping: N.s/m on the spring DoF.
        h_add: metres of height the mechanism adds below the existing sole.
        pad_mass: mass per pad in kg.
    """

    def _spec_fn() -> mujoco.MjSpec:
        spec = get_walk_spec()
        for side in ("left", "right"):
            ankle = spec.body(f"ankle_{side}")

            # Retire the rigid sole: rename it and switch off its contact, so
            # the name `{side}_foot_collision` is free for the pad below. Left
            # in place it would keep answering the feet_ground_contact sensor
            # while floating h_add above the ground.
            old_geom = spec.geom(f"{side}_foot_collision")
            old_geom.name = f"{side}_sole_disabled"
            old_geom.contype = 0
            old_geom.conaffinity = 0
            spec.site(f"{side}_foot").name = f"{side}_foot_old"
            # -y is downward in world at the home pose, so a negative y offset
            # puts the pad below the ankle.
            pad = ankle.add_body(
                name=f"{side}_foot_pad", pos=[0.0, -(ANKLE_TO_SOLE + h_add), 0.0]
            )
            joint = pad.add_joint(
                name=f"passive_{side}_foot_spring",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
            )
            joint.axis = list(SPRING_AXIS)
            joint.range = [0.0, travel]
            # Leave `limited` at its default (mjLIMITED_AUTO) rather than
            # forcing 1: MuJoCo's compile-time check requires range[0] <
            # range[1] whenever limited is explicitly true, which breaks the
            # travel=0.0 locked variant (range [0, 0]). AUTO enables the limit
            # only when range differs from the [0, 0] default, which is
            # exactly what we want in both cases.
            # These MUST be 3-arrays; MjsJoint rejects a scalar. Only element 0
            # is used by the compiler.
            joint.stiffness = np.array([stiffness, 0.0, 0.0])
            joint.damping = np.array([damping, 0.0, 0.0])
            # Re-use the ORIGINAL names so the contact sensor, the terrain
            # height-scan frames, foot_clearance and foot_slip all keep working
            # with no config change.
            pad.add_geom(
                name=f"{side}_foot_collision",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(_PAD_HALF_EXTENTS),
                pos=[0.0, 0.0, 0.0],
                mass=pad_mass,
            )
            pad.add_site(name=f"{side}_foot", pos=[0.0, 0.0, 0.0])
        return spec

    return _spec_fn


def make_sprung_foot_robot_cfg(
    stiffness: float,
    travel: float = TRAVEL,
    damping: float = DAMPING,
    h_add: float = H_ADD,
    pad_mass: float = PAD_MASS,
) -> EntityCfg:
    """EntityCfg for a sprung-foot MicroDuck, spawned h_add higher.

    The spawn must rise by exactly ``h_add`` or the taller foot starts inside
    the floor.
    """
    init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, h_add),
        joint_pos=dict(HOME_FRAME.joint_pos),
        joint_vel={".*": 0.0},
    )
    return EntityCfg(
        spec_fn=make_sprung_foot_spec_fn(stiffness, travel, damping, h_add, pad_mass),
        init_state=init_state,
        collisions=(FULL_COLLISION,),
        articulation=EntityArticulationInfoCfg(
            actuators=(actuators,),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
