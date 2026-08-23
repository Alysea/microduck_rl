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
#
# Tuned (not the naive mesh measurement of 0.025) so that the settled
# rigid-vs-sprung trunk-height delta lands on H_ADD: the rigid sole is a mesh
# and the pad is a box, and the two settle to slightly different contact
# penetration depths under gravity, so the naive value overshot the delta by
# ~3-4 mm. See task-1-report.md fix-round-1 notes.
#
# Retuned for the measured H_ADD=0.030 (was 0.0215, tuned for the old
# H_ADD=0.025): the mesh-vs-box penetration mismatch this constant corrects
# for is unaffected by H_ADD, so it re-manifested as the same ~4 mm overshoot
# on the LOCKED (zero-compliance) arm's settled delta and was re-tuned down
# by that amount. See FIX 5's settling measurement in the prototype-update
# report.
ANKLE_TO_SOLE = 0.01744

H_ADD = 0.030      # measured on the Sarrus prototype (was an assumed 0.025)
PAD_MASS = 0.070   # measured (was an assumed 0.020) — 70 g per boot
TRAVEL = 0.012     # measured (was an assumed 0.015)
DAMPING = 0.5      # N.s/m — represents a good steel spring, low hysteresis

# Intentional spring preload in the Sarrus mechanism, as a DISPLACEMENT.
# Measured: 2.9 N offset at k = 3920 N/m -> 2.9/3920 = 0.74 mm of precompression.
# Parameterised as displacement rather than force because the linkage geometry
# fixes the precompression at assembly: a stiffer spring in the same boot keeps
# the 0.74 mm and produces proportionally MORE preload force. Consequence, which
# is physically faithful rather than a modelling artifact: preload force varies
# across the sweep (1.1 N at k=1500, 4.1 N at k=5500).
# Preload holds the pad firmly at full extension during flight instead of
# letting it float within its travel and chatter against the hard stop.
SPRING_PRELOAD = 0.00074   # m of precompression at rest

# These exist to OVERRIDE the `microduck` childclass joint defaults
# (frictionloss=0.1, armature=0.005 in robot_walk.xml), which the spring joint
# would otherwise inherit silently — the joint is added inside that childclass
# scope. Zero is not a physical claim about a real mechanism: it makes the model
# match the spec's *idealised* spring, whose only dissipation is DAMPING.
# Mechanism stiction and mechanism inertia are hardware-phase concerns the spec
# explicitly defers.
SPRING_FRICTIONLOSS = 0.0
SPRING_ARMATURE = 0.0

SPRING_JOINTS = ("passive_left_foot_spring", "passive_right_foot_spring")

# The `<default class="collision">` block in robot_walk.xml (group=3). The pad's
# contact geom MUST inherit it: `foot_height_scan` rays are restricted to
# `include_geom_groups=(0,)` (terrain only), so a group-0 pad geom would be hit
# by the opposite foot's height ray and reported as ground, corrupting
# `foot_clearance` and `foot_swing_height`.
_COLLISION_CLASS = "collision"

# Contact pad half-extents (m). Local y is world-up here, so the middle number
# is half the pad thickness.
_PAD_HALF_EXTENTS = (0.020, 0.004, 0.014)


def make_sprung_foot_spec_fn(
    stiffness: float,
    travel: float = TRAVEL,
    damping: float = DAMPING,
    h_add: float = H_ADD,
    pad_mass: float = PAD_MASS,
    preload: float = SPRING_PRELOAD,
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
        preload: metres of precompression built into the mechanism at
            assembly. Applied as ``springref = -preload`` (see below).
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
            # travel == 0.0 is the LOCKED control arm: no joint at all, so the
            # pad is a rigid child of the ankle (identical mass and height,
            # zero DoF). A slide joint with range [0, 0] compiles fine but is
            # NOT locked -- MuJoCo leaves `limited` at AUTO in that case
            # (range == the joint-type default), so the joint is actually
            # unconstrained and held only by the spring. That silently turns
            # the control arm into an infinite-travel spring, which defeats
            # its purpose of isolating "extra height/mass" from "compliance".
            if travel > 0.0:
                joint = pad.add_joint(
                    name=f"passive_{side}_foot_spring",
                    type=mujoco.mjtJoint.mjJNT_SLIDE,
                )
                joint.axis = list(SPRING_AXIS)
                joint.range = [0.0, travel]
                joint.limited = 1
                # These MUST be 3-arrays; MjsJoint rejects a scalar. Only
                # element 0 is used by the compiler.
                joint.stiffness = np.array([stiffness, 0.0, 0.0])
                joint.damping = np.array([damping, 0.0, 0.0])
                # MuJoCo's spring force is -stiffness * (qpos - springref).
                # Our convention is q=0 extended, q>0 compressed, so a NEGATIVE
                # springref puts a compression-resisting force at q=0: the
                # spring is pressed against its own extension stop, i.e. the
                # preload. Unlike stiffness/damping, springref is a SCALAR on
                # MjsJoint (a 3-array raises TypeError).
                joint.springref = -preload
                # Set explicitly to override the `microduck` childclass joint
                # defaults (0.1 / 0.005) this joint would otherwise inherit.
                # See the constants above. Unlike stiffness/damping these two
                # are SCALARS on MjsJoint (a 3-array raises TypeError).
                joint.frictionloss = SPRING_FRICTIONLOSS
                joint.armature = SPRING_ARMATURE
            # Re-use the ORIGINAL names so the contact sensor, the terrain
            # height-scan frames, foot_clearance and foot_slip all keep working
            # with no config change.
            pad.add_geom(
                spec.find_default(_COLLISION_CLASS),
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
    preload: float = SPRING_PRELOAD,
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
        spec_fn=make_sprung_foot_spec_fn(
            stiffness, travel, damping, h_add, pad_mass, preload
        ),
        init_state=init_state,
        collisions=(FULL_COLLISION,),
        articulation=EntityArticulationInfoCfg(
            actuators=(actuators,),
            soft_joint_pos_limit_factor=0.9,
        ),
    )
