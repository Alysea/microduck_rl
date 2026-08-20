"""Sprung-foot robot model: geometry, spring parameters, and compression sign.

These tests legitimately compile a real MjSpec — the thing under test IS the
model, so a duck-typed fake would test nothing.
"""

import mujoco
import numpy as np
import pytest

from mjlab_microduck.robot.sprung_foot import (
    H_ADD,
    PAD_MASS,
    SPRING_JOINTS,
    TRAVEL,
    make_sprung_foot_spec_fn,
)


@pytest.fixture(scope="module")
def model():
    return make_sprung_foot_spec_fn(stiffness=1500.0)().compile()


def test_both_spring_joints_exist_and_are_passive(model):
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    for j in SPRING_JOINTS:
        assert j in names
        # The passive_ prefix is load-bearing: every actuator/reward/obs regex
        # of the form ^(?!passive_).* relies on it to exclude these joints.
        assert j.startswith("passive_")


def test_spring_joints_are_slide_with_the_requested_range(model):
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
        assert model.jnt_range[jid][0] == pytest.approx(0.0)
        assert model.jnt_range[jid][1] == pytest.approx(TRAVEL)


def test_stiffness_and_damping_reach_the_compiled_model(model):
    # MjsJoint.stiffness/.damping need a 3-array; a scalar raises. This test
    # is what catches a regression to scalar assignment.
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert model.jnt_stiffness[jid] == pytest.approx(1500.0)
        assert model.dof_damping[model.jnt_dofadr[jid]] == pytest.approx(0.5)


def test_stiffness_is_actually_parameterised():
    m800 = make_sprung_foot_spec_fn(stiffness=800.0)().compile()
    m3000 = make_sprung_foot_spec_fn(stiffness=3000.0)().compile()
    jid800 = mujoco.mj_name2id(m800, mujoco.mjtObj.mjOBJ_JOINT, SPRING_JOINTS[0])
    jid3000 = mujoco.mj_name2id(m3000, mujoco.mjtObj.mjOBJ_JOINT, SPRING_JOINTS[0])
    assert m800.jnt_stiffness[jid800] == pytest.approx(800.0)
    assert m3000.jnt_stiffness[jid3000] == pytest.approx(3000.0)


def test_positive_q_is_compression(model):
    """The sign convention the whole study rests on.

    q=0 must be the extended (unloaded) pose and q>0 must move the pad UP,
    toward the body. If this inverts, the spring pushes the robot into the
    ground and every sweep result is meaningless.
    """
    data = mujoco.MjData(model)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, SPRING_JOINTS[0])
    pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_pad")
    adr = model.jnt_qposadr[jid]

    data.qpos[adr] = 0.0
    mujoco.mj_forward(model, data)
    z_extended = data.xpos[pad][2]

    data.qpos[adr] = TRAVEL
    mujoco.mj_forward(model, data)
    z_compressed = data.xpos[pad][2]

    assert z_compressed > z_extended
    # Nearly all of the travel should show up as vertical motion; the small
    # shortfall is the ~5 deg ankle tilt at the home pose.
    assert (z_compressed - z_extended) == pytest.approx(TRAVEL, rel=0.05)


def test_pad_mass_is_added_not_idealised_away(model):
    from mjlab_microduck.robot.microduck_constants import get_walk_spec
    rigid_mass = get_walk_spec().compile().body_mass.sum()
    assert model.body_mass.sum() == pytest.approx(rigid_mass + 2 * PAD_MASS, abs=1e-6)


def test_locked_variant_has_zero_travel():
    """The locked (travel=0) variant must be a true rigid control, not a
    spring with an unenforced [0, 0] range (that would leave it unconstrained
    -- see fix-round-1 notes in the report).
    """
    m = make_sprung_foot_spec_fn(stiffness=1500.0, travel=0.0)().compile()
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert jid == -1, f"{j} should not exist in the locked (travel=0) variant"


def test_locked_variant_has_two_fewer_dofs(model):
    locked = make_sprung_foot_spec_fn(stiffness=1500.0, travel=0.0)().compile()
    assert locked.nv == model.nv - 2


def test_contact_geom_and_site_live_on_the_pad(model):
    """The most load-bearing assertion in this file.

    feet_ground_contact matches ^(left_foot_collision|right_foot_collision)$ and
    foot_height_scan frames off the left_foot/right_foot sites. If either still
    resolves to the ankle body, contact is read from a geom floating above the
    ground and every gait metric silently reads garbage.
    """
    for side in ("left", "right"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
        assert gid >= 0 and sid >= 0
        pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot_pad")
        assert model.geom_bodyid[gid] == pad
        assert model.site_bodyid[sid] == pad


def test_old_sole_no_longer_collides(model):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_sole_disabled")
    assert gid >= 0, "the rigid sole should be renamed, not deleted"
    assert model.geom_contype[gid] == 0
    assert model.geom_conaffinity[gid] == 0


def test_h_add_lowers_the_pad(model):
    """Larger h_add must put the contact pad further below the ankle."""
    shallow = make_sprung_foot_spec_fn(stiffness=1500.0, h_add=0.010)().compile()
    d_deep, d_shallow = mujoco.MjData(model), mujoco.MjData(shallow)
    mujoco.mj_forward(model, d_deep)
    mujoco.mj_forward(shallow, d_shallow)
    pad_deep = d_deep.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_pad")]
    pad_shallow = d_shallow.xpos[mujoco.mj_name2id(shallow, mujoco.mjtObj.mjOBJ_BODY, "left_foot_pad")]
    assert pad_deep[2] < pad_shallow[2]
