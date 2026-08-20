# Sprung-Foot Stiffness Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an idealised 1-DoF sprung foot and five sweep task ids (a locked geometric control plus k = 800/1500/2200/3000 N/m) so the useful (stiffness, travel) window can be measured and handed to the hardware phase.

**Architecture:** The sprung robot is built **programmatically** from the canonical `robot_walk.xml` via `MjSpec` — a pad body with one `slide` joint added under each ankle — rather than as a forked XML. A `make_sprung_variant()` transform in the shape of `tasks/backlash.py` then swaps the robot cfg and shifts the CoM reward band by the mechanism's added height.

**Tech Stack:** Python 3.12, MuJoCo `MjSpec`, PyTorch, mjlab 1.3.0, rsl-rl-lib 5.0.1, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-20-sprung-foot-design.md`

## Deviation from the spec, with justification

The spec says "derive `robot_walk_sprung_foot.xml` from the current
`robot_walk.xml`". **This plan builds the sprung model programmatically instead**,
with no new XML file.

Reason: a forked XML is exactly what killed the abandoned `test_spring` branch —
its sprung XML was a 50-line delta on a `robot_walk.xml` that then changed by
310 insertions, leaving the delta un-portable. A `spec_fn` that loads the
canonical XML and adds two bodies tracks every upstream change to the base model
for free. The repo already uses `spec_fn` factories and spec editing (the BAM
actuator rewrites actuators this way), so this follows an established pattern
rather than inventing one.

Everything else in the spec is implemented as written.

## Global Constraints

- **Branch:** all work on `spring_v2`. Standing commit permission on this branch — commit directly, do not ask. **Never `git push`**, on any branch.
- **Never read `.envrc`** or `..envrc.~undo-tree~` — secrets.
- **Training is remote.** Do not launch a campaign locally; unit tests and at most a very short smoke run.
- **Naming:** the new joints MUST be `passive_left_foot_spring` / `passive_right_foot_spring`. Every existing `^(?!passive_).*` regex depends on the `passive_` prefix to exclude them.
- **Reward functions** take `env` first, return a `torch.Tensor` of shape `(num_envs,)`, guard missing state, and are NaN-safe. Match `wheel_speed_reward` / `action_magnitude_monitor` in `tasks/mdp.py`.
- **Tests** use lightweight duck-typed fakes (`tests/test_wheel_glide.py`, `tests/test_run.py`), never a real mjlab env — except the robot-model tests in Task 1, which legitimately compile a real `MjSpec`.
- **Monitors are registered with a NON-ZERO weight.** `RewardManager.compute` (`mjlab/managers/reward_manager.py:122`) short-circuits before calling a term whose weight is `0.0`; a zero-weight monitor is silently dead.
- **`action_rate_l2` stays at -1.0.** Not a variable in this study.
- Test command: `uv run pytest tests/<file> -v`

## Verified API facts — use these, do not re-derive

All four checked against the installed MuJoCo before this plan was written:

1. **`MjsJoint.stiffness` and `.damping` require a 3-element array, NOT a scalar.**
   `j.stiffness = 1500.0` raises `TypeError`. `j.stiffness = np.array([1500.0, 0.0, 0.0])`
   works, and only element 0 is used — the compiled model gets
   `jnt_stiffness = 1500.0` and `dof_damping = 0.5`.
2. **The slide axis is `[0.0, 1.0, 0.0]`.** The ankle bodies are NOT world-aligned:
   local `+y` maps to world `[0, 0.087, 0.996]`, i.e. almost straight up. With this
   axis, **positive q = compression** (pad moves toward the body), verified as
   +14.94 mm of pad rise over 15 mm of travel. Do not use `[0,0,1]`; local `+z` is
   nearly horizontal.
3. **Spec editing API:** `spec.body("ankle_left")`, `body.add_body(name=..., pos=...)`,
   `pad.add_joint(name=..., type=mujoco.mjtJoint.mjJNT_SLIDE)`, `pad.add_geom(...)`,
   `spec.compile()`. All confirmed working on this model.
4. **The sole collision geom and foot site MUST be relocated to the pad, and
   this is the single most important detail in Task 1.** The
   `feet_ground_contact` sensor matches geoms by the pattern
   `^(left_foot_collision|right_foot_collision)$`, and `foot_height_scan` takes
   its frames from the `left_foot` / `right_foot` sites. If the pad is added
   while those stay on the ankle body, the contact sensor watches a geom now
   floating `h_add` above the ground — never in contact — and `air_time`,
   `alternating_flight`, `flight_fraction`, `foot_clearance` and `foot_slip` all
   silently read garbage. Verified working approach: rename the old geom and
   site, disable the old geom's contact, then add a new geom and site on the pad
   **under the original names**, so every downstream regex and sensor keeps
   working untouched:
   ```python
   g = spec.geom("left_foot_collision"); g.name = "left_sole_disabled"
   g.contype = 0; g.conaffinity = 0
   spec.site("left_foot").name = "left_foot_old"
   # ... then on the pad body:
   pad.add_geom(name="left_foot_collision", ...)
   pad.add_site(name="left_foot", pos=[0, 0, 0])
   ```
   Confirmed after compile: both `left_foot_collision` and the `left_foot` site
   resolve to body `left_foot_pad`.
5. **`EntityCfg` takes `spec_fn`** (a zero-argument callable returning `MjSpec`),
   plus `init_state`, `collisions`, `articulation`. See `MICRODUCK_WALK_ROBOT_CFG`
   at `microduck_constants.py:159`.

---

### Task 1: Sprung-foot robot model

**Files:**
- Create: `src/mjlab_microduck/robot/sprung_foot.py`
- Test: `tests/test_sprung_foot_model.py`

**Interfaces:**
- Consumes: `get_walk_spec`, `HOME_FRAME`, `FULL_COLLISION`, `actuators` from `mjlab_microduck.robot.microduck_constants`.
- Produces:
  - `SPRING_AXIS = (0.0, 1.0, 0.0)`, `ANKLE_TO_SOLE = 0.025`, `H_ADD = 0.025`, `PAD_MASS = 0.020`, `TRAVEL = 0.015`, `DAMPING = 0.5`
  - `make_sprung_foot_spec_fn(stiffness: float, travel: float = TRAVEL, damping: float = DAMPING, h_add: float = H_ADD, pad_mass: float = PAD_MASS) -> Callable[[], mujoco.MjSpec]`
  - `make_sprung_foot_robot_cfg(stiffness: float, **kw) -> EntityCfg`
  - `SPRING_JOINTS = ("passive_left_foot_spring", "passive_right_foot_spring")`
  Used by Tasks 3 and 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sprung_foot_model.py`:

```python
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
    base = make_sprung_foot_spec_fn(stiffness=1500.0, pad_mass=0.0)().compile()
    assert model.body_mass.sum() == pytest.approx(base.body_mass.sum() + 2 * PAD_MASS, abs=1e-6)


def test_locked_variant_has_zero_travel():
    m = make_sprung_foot_spec_fn(stiffness=1500.0, travel=0.0)().compile()
    for j in SPRING_JOINTS:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert m.jnt_range[jid][1] == pytest.approx(0.0)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sprung_foot_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mjlab_microduck.robot.sprung_foot'`

- [ ] **Step 3: Write the implementation**

Create `src/mjlab_microduck/robot/sprung_foot.py`:

```python
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
            joint.limited = 1
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sprung_foot_model.py -v`
Expected: 10 passed.

If `HOME_FRAME.joint_vel` does not exist as an attribute, drop that kwarg — the
base `HOME_FRAME` at `microduck_constants.py:73` is the reference for which
fields `InitialStateCfg` accepts.

- [ ] **Step 5: Measure the standing geometry — do not skip**

The `H_ADD`/`ANKLE_TO_SOLE` numbers place the pad by arithmetic; only a settling
test shows whether the robot actually stands correctly. Write
`/tmp/sprung_settle.py`:

```python
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, numpy as np, re
from mjlab_microduck.robot.microduck_constants import HOME_FRAME
from mjlab_microduck.robot.sprung_foot import H_ADD, TRAVEL, make_sprung_foot_spec_fn

for k in (800.0, 1500.0, 3000.0):
    m = make_sprung_foot_spec_fn(stiffness=k)().compile()
    d = mujoco.MjData(m)
    for i in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if nm is None or m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        for pat, val in HOME_FRAME.joint_pos.items():
            if re.search(pat.strip("^$").replace(".*", ""), nm):
                d.qpos[m.jnt_qposadr[i]] = val
                break
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    # free joint z, raised so the taller foot starts clear of the floor
    d.qpos[2] += H_ADD
    for _ in range(3000):
        mujoco.mj_step(m, d)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "passive_left_foot_spring")
    q = d.qpos[m.jnt_qposadr[jid]]
    print(f"k={k:6.0f}  trunk z={d.xpos[trunk][2]*1000:6.1f} mm  "
          f"spring q={q*1000:5.2f} mm ({100*q/TRAVEL:4.1f}% of travel)")
```

Run: `uv run python /tmp/sprung_settle.py`

Check three things and record them in your report:
1. **Trunk height** should be ~125 mm + `H_ADD` = ~150 mm. Far off means
   `ANKLE_TO_SOLE` is wrong; adjust it so the rigid and sprung robots differ by
   exactly `H_ADD`.
2. **Spring compression** should be small at rest (~2-5 mm at 1500 N/m) and must
   NOT sit at 100% of travel. Pinned at 100% means it is bottomed out standing
   still — the exact failure the abandoned branch had.
3. **The robot must not sink through the floor** or jitter. Either means the pad
   geom or the spawn height needs work.

- [ ] **Step 6: Commit**

```bash
git add src/mjlab_microduck/robot/sprung_foot.py tests/test_sprung_foot_model.py
git commit -m "feat(sprung): programmatic sprung-foot robot model

One prismatic spring per foot, added to the canonical robot_walk.xml spec
rather than a forked XML. Slide axis 0 1 0 (local +y is world-up on the ankle
bodies); positive q is compression."
```

---

### Task 2: Spring compression monitor

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (append at end of file, after `forward_speed_monitor`)
- Test: `tests/test_sprung.py` (create)

**Interfaces:**
- Consumes: `SPRING_JOINTS` from `mjlab_microduck.robot.sprung_foot` is NOT imported here — the monitor resolves joints by name argument, so `tasks/mdp.py` gains no dependency on the robot module.
- Produces: `spring_compression_monitor(env, joint_names: tuple[str, ...], travel: float, bottom_out_frac: float = 0.95, asset_cfg=...) -> torch.Tensor` returning zeros. Logs `Metrics/spring_compression_mean`, `Metrics/spring_compression_max`, `Metrics/spring_bottomed_fraction`. Used by Task 3.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sprung.py`:

```python
"""Unit tests for the spring compression monitor (duck-typed fakes)."""

import torch

from mjlab_microduck.tasks.mdp import spring_compression_monitor

_JOINTS = ("passive_left_foot_spring", "passive_right_foot_spring")
_TRAVEL = 0.015


class _Data:
    def __init__(self, q):
        self.joint_pos = torch.tensor(q, dtype=torch.float32)


class _Asset:
    def __init__(self, q):
        self.data = _Data(q)

    def find_joints(self, name):
        # column 0 = left spring, column 1 = right spring
        return [_JOINTS.index(name)], None


class _Scene:
    def __init__(self, q):
        self._a = _Asset(q)

    def __getitem__(self, _k):
        return self._a


class _Env:
    """q: list of [left_compression, right_compression] in metres."""

    def __init__(self, q):
        self.scene = _Scene(q)
        self.num_envs = len(q)
        self.device = "cpu"
        self.extras = {"log": {}}


def test_returns_exactly_zeros():
    env = _Env([[0.005, 0.007]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_reports_mean_and_max_compression():
    env = _Env([[0.004, 0.010]])
    spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    log = env.extras["log"]
    assert abs(float(log["Metrics/spring_compression_mean"]) - 0.007) < 1e-6
    assert abs(float(log["Metrics/spring_compression_max"]) - 0.010) < 1e-6


def test_bottomed_fraction_is_zero_when_well_inside_travel():
    env = _Env([[0.004, 0.005]])
    spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    assert float(env.extras["log"]["Metrics/spring_bottomed_fraction"]) == 0.0


def test_bottomed_fraction_catches_a_bottomed_spring():
    # 0.0149 of 0.015 travel is 99.3% — past the 95% threshold.
    env = _Env([[0.0149, 0.001]])
    spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    # one of two joint-samples is bottomed
    assert abs(float(env.extras["log"]["Metrics/spring_bottomed_fraction"]) - 0.5) < 1e-6


def test_zero_travel_locked_variant_does_not_divide_by_zero():
    env = _Env([[0.0, 0.0]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=0.0)
    assert float(out[0]) == 0.0
    assert torch.isfinite(env.extras["log"]["Metrics/spring_bottomed_fraction"])


def test_nan_safe():
    env = _Env([[float("nan"), 0.006]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    assert torch.isfinite(out).all()
    assert torch.isfinite(env.extras["log"]["Metrics/spring_compression_max"])


def test_missing_joint_returns_zeros_without_raising():
    class _NoJointAsset(_Asset):
        def find_joints(self, name):
            return [], None

    env = _Env([[0.005, 0.005]])
    env.scene._a = _NoJointAsset([[0.005, 0.005]])
    out = spring_compression_monitor(env, joint_names=_JOINTS, travel=_TRAVEL)
    assert float(out[0]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sprung.py -v`
Expected: FAIL — `ImportError: cannot import name 'spring_compression_monitor'`

- [ ] **Step 3: Write the implementation**

Append to `src/mjlab_microduck/tasks/mdp.py`:

```python
def spring_compression_monitor(
    env: ManagerBasedRlEnv,
    joint_names: tuple,
    travel: float,
    bottom_out_frac: float = 0.95,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Log sprung-foot compression. Contributes exactly zero reward.

    This is the first thing to read in the stiffness sweep: a spring that never
    deflects, or one pinned against its hard stop, is measuring nothing. The
    abandoned sprung branch ran at 500 N/m over 10 mm of travel, which is 5 N
    before the stop — it was riding a rigid limit, not a spring, and nothing
    logged that fact.

    Returns zeros, so the reward total is unaffected at any weight. **Register
    it with a non-zero weight anyway**: ``RewardManager.compute``
    (mjlab/managers/reward_manager.py:122) short-circuits before calling the
    term function when ``weight == 0.0``.

    Args:
        joint_names: the spring joint names, resolved by name (so this function
            has no dependency on the robot module).
        travel: the spring's stroke in metres. 0.0 (the locked control variant)
            reports zero compression rather than dividing by zero.
        bottom_out_frac: fraction of travel counted as bottomed out.

    Returns:
        A zeros tensor (num_envs,).
    """
    zeros = torch.zeros(env.num_envs, device=env.device)
    asset: Entity = env.scene[asset_cfg.name]

    ids = []
    for name in joint_names:
        found, _ = asset.find_joints(name)
        if not found:
            return zeros
        ids.append(found[0])
    if not ids:
        return zeros

    q = torch.nan_to_num(
        asset.data.joint_pos[:, ids].float(), nan=0.0, posinf=0.0, neginf=0.0
    )

    log = env.extras.get("log") if hasattr(env, "extras") else None
    if log is not None:
        log["Metrics/spring_compression_mean"] = q.mean()
        log["Metrics/spring_compression_max"] = q.max()
        if travel > 0.0:
            log["Metrics/spring_bottomed_fraction"] = (
                q >= bottom_out_frac * travel
            ).float().mean()
        else:
            # Locked control variant: no travel, so nothing can bottom out.
            log["Metrics/spring_bottomed_fraction"] = torch.zeros((), device=env.device)

    return zeros
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sprung.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_sprung.py
git commit -m "feat(sprung): spring_compression_monitor

Logs mean/max compression and a bottomed-out fraction. The abandoned branch's
spring rode its hard stop and nothing recorded it; this makes that visible."
```

---

### Task 3: `make_sprung_variant` transform

**Files:**
- Create: `src/mjlab_microduck/tasks/sprung.py`
- Test: `tests/test_sprung_cfg.py` (create)

**Interfaces:**
- Consumes: `make_sprung_foot_robot_cfg`, `SPRING_JOINTS`, `H_ADD`, `TRAVEL` (Task 1); `spring_compression_monitor` (Task 2); `make_run_variant` from `mjlab_microduck.tasks.run`.
- Produces: `make_sprung_variant(cfg, stiffness: float, travel: float = TRAVEL, h_add: float = H_ADD) -> ManagerBasedRlEnvCfg`, plus `SPRING_MONITOR_WEIGHT = 1.0`. Used by Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sprung_cfg.py`:

```python
"""Config-level assertions for the sprung-foot variant transform."""

import pytest

from mjlab_microduck.robot.sprung_foot import H_ADD, SPRING_JOINTS, TRAVEL
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.run import make_run_variant
from mjlab_microduck.tasks.sprung import make_sprung_variant


@pytest.fixture
def base_cfg():
    return make_run_variant(make_microduck_velocity_env_cfg())


@pytest.fixture
def sprung_cfg(base_cfg):
    return make_sprung_variant(base_cfg, stiffness=1500.0)


def test_com_band_is_shifted_by_exactly_h_add(sprung_cfg):
    """The sprung robot stands h_add taller.

    Without this shift it would be penalised for being tall before compliance
    is in play — the confound the locked control arm exists to isolate.
    """
    rigid = make_run_variant(make_microduck_velocity_env_cfg())
    r = rigid.rewards["com_height_target"].params
    s = sprung_cfg.rewards["com_height_target"].params
    assert s["target_height_min"] == pytest.approx(r["target_height_min"] + H_ADD)
    assert s["target_height_max"] == pytest.approx(r["target_height_max"] + H_ADD)
    # The band must not widen — only translate.
    assert (s["target_height_max"] - s["target_height_min"]) == pytest.approx(
        r["target_height_max"] - r["target_height_min"]
    )


def test_spring_monitor_registered_with_non_zero_weight(sprung_cfg):
    # RewardManager.compute skips weight==0.0 terms before calling them, which
    # would silently disable the monitor.
    term = sprung_cfg.rewards["spring_compression_monitor"]
    assert term.func is microduck_mdp.spring_compression_monitor
    assert term.weight != 0.0
    assert tuple(term.params["joint_names"]) == SPRING_JOINTS
    assert term.params["travel"] == pytest.approx(TRAVEL)


def test_robot_entity_is_replaced(sprung_cfg, base_cfg):
    assert sprung_cfg.scene.entities["robot"] is not None
    # spec_fn differs from the rigid walk spec
    rigid = make_run_variant(make_microduck_velocity_env_cfg())
    assert sprung_cfg.scene.entities["robot"].spec_fn is not rigid.scene.entities["robot"].spec_fn


def test_pose_reward_excludes_the_spring_joints(sprung_cfg):
    """The pose reward must not try to hold a passive spring at a target."""
    names = sprung_cfg.rewards["pose"].params["asset_cfg"].joint_names
    assert any("passive_" in pattern for pattern in names)


def test_dof_pos_limits_scoped_off_the_spring_joints(sprung_cfg):
    dof = sprung_cfg.rewards.get("dof_pos_limits")
    if dof is not None:
        assert "asset_cfg" in dof.params


def test_locked_variant_has_zero_travel_in_monitor_and_model():
    cfg = make_sprung_variant(
        make_run_variant(make_microduck_velocity_env_cfg()),
        stiffness=1500.0,
        travel=0.0,
    )
    assert cfg.rewards["spring_compression_monitor"].params["travel"] == pytest.approx(0.0)


def test_action_rate_weight_untouched(sprung_cfg):
    """action_rate_l2 is the fixed smoothness budget, not a variable."""
    stages = sprung_cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[-1]["weight"] == pytest.approx(-1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sprung_cfg.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mjlab_microduck.tasks.sprung'`

- [ ] **Step 3: Write the implementation**

Create `src/mjlab_microduck/tasks/sprung.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sprung_cfg.py -v`
Expected: 7 passed.

If `test_pose_reward_excludes_the_spring_joints` fails because the velocity
env's pose `asset_cfg` already carries a `passive_` exclusion (it uses
`^(?!passive_|.*neck.*|.*head.*).*`), that is fine — the guard in the
implementation skips redundant additions, and the test asserts only that some
pattern mentions `passive_`.

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/sprung.py tests/test_sprung_cfg.py
git commit -m "feat(sprung): make_sprung_variant transform

Swaps the sprung robot, shifts the CoM band by h_add so the taller robot is not
penalised for its geometry, scopes pose/dof_pos_limits off the spring joints,
and registers the compression monitor."
```

---

### Task 4: Register the five sweep arms

**Files:**
- Modify: `src/mjlab_microduck/tasks/sprung.py` (append the arm table)
- Modify: `src/mjlab_microduck/tasks/__init__.py` (import + register, after the Run registrations)
- Test: `tests/test_sprung_cfg.py` (extend)

**Interfaces:**
- Consumes: `make_sprung_variant` (Task 3); `make_run_variant`, `MicroduckRunRlCfg` from `mjlab_microduck.tasks.run`; `make_microduck_velocity_env_cfg`; `MicroduckOnPolicyRunner`.
- Produces: `SWEEP_ARMS: tuple[tuple[str, float, float], ...]` of `(label, stiffness, travel)`, and five registered task ids.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sprung_cfg.py`:

```python
def test_sweep_arms_cover_the_spec_grid():
    from mjlab_microduck.tasks.sprung import SWEEP_ARMS

    labels = [a[0] for a in SWEEP_ARMS]
    assert "locked" in labels, "the locked arm is the geometric control"
    stiffnesses = {a[1] for a in SWEEP_ARMS if a[0] != "locked"}
    assert stiffnesses == {800.0, 1500.0, 2200.0, 3000.0}
    # The locked arm must have zero travel; every other arm must have some.
    for label, _k, travel in SWEEP_ARMS:
        if label == "locked":
            assert travel == 0.0
        else:
            assert travel > 0.0


def test_all_sweep_task_ids_registered():
    import mjlab_microduck.tasks  # noqa: F401  (import registers)
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    for tid in (
        "Mjlab-Run-Flat-Sprung-Locked-MicroDuck",
        "Mjlab-Run-Flat-Sprung-K800-MicroDuck",
        "Mjlab-Run-Flat-Sprung-K1500-MicroDuck",
        "Mjlab-Run-Flat-Sprung-K2200-MicroDuck",
        "Mjlab-Run-Flat-Sprung-K3000-MicroDuck",
    ):
        assert tid in tasks, f"{tid} not registered"


def test_sweep_arms_use_distinct_experiment_names():
    """Each arm needs its own wandb grouping or the sweep is unreadable."""
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_rl_cfg

    names = {
        load_rl_cfg(f"Mjlab-Run-Flat-Sprung-{s}-MicroDuck").run_name
        for s in ("Locked", "K800", "K1500", "K2200", "K3000")
    }
    assert len(names) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sprung_cfg.py -v`
Expected: FAIL — `ImportError: cannot import name 'SWEEP_ARMS'`

- [ ] **Step 3: Write the implementation**

Append to `src/mjlab_microduck/tasks/sprung.py`:

```python
from dataclasses import replace

from mjlab_microduck.tasks.run import MicroduckRunRlCfg

# (label, stiffness N/m, travel m). The locked arm is the geometric control:
# identical height and mass, zero compliance. It — not the 0.468 m/s rigid
# baseline — is what the sprung arms are compared against, because the rigid
# baseline differs in geometry as well as compliance.
#
# k=800 is expected to bottom out (22.5 mm of deflection at an 18 N peak against
# 15 mm of travel). It is included deliberately: it should reproduce the
# abandoned branch's failure and establish the travel floor empirically.
SWEEP_ARMS = (
    ("locked", 1500.0, 0.0),
    ("k800", 800.0, TRAVEL),
    ("k1500", 1500.0, TRAVEL),
    ("k2200", 2200.0, TRAVEL),
    ("k3000", 3000.0, TRAVEL),
)

ARM_TASK_SUFFIX = {
    "locked": "Locked",
    "k800": "K800",
    "k1500": "K1500",
    "k2200": "K2200",
    "k3000": "K3000",
}


def sprung_rl_cfg(label: str):
    """Per-arm RL cfg: identical learner, distinct logging identity.

    ``replace`` is shallow, so deepcopy the nested cfgs — otherwise every arm
    would share one actor object and a later change to any of them would alter
    all five plus the Run baseline.
    """
    return replace(
        MicroduckRunRlCfg,
        actor=deepcopy(MicroduckRunRlCfg.actor),
        critic=deepcopy(MicroduckRunRlCfg.critic),
        algorithm=deepcopy(MicroduckRunRlCfg.algorithm),
        experiment_name=f"sprung_{label}",
        run_name=f"sprung_{label}",
    )
```

Then in `src/mjlab_microduck/tasks/__init__.py`, add the import beside the
others (it must come **after** the `.run` import, since `sprung.py` imports
from `run.py`):

```python
from .sprung import SWEEP_ARMS, make_sprung_variant, sprung_rl_cfg, ARM_TASK_SUFFIX
```

and register after the Run block:

```python
# Sprung-foot stiffness sweep — Phase 2. See
# docs/superpowers/specs/2026-08-20-sprung-foot-design.md
for _label, _k, _travel in SWEEP_ARMS:
    _tid = f"Mjlab-Run-Flat-Sprung-{ARM_TASK_SUFFIX[_label]}-MicroDuck"
    register_mjlab_task(
        task_id=_tid,
        env_cfg=make_sprung_variant(
            make_run_variant(make_microduck_velocity_env_cfg()),
            stiffness=_k,
            travel=_travel,
        ),
        play_env_cfg=make_sprung_variant(
            make_run_variant(make_microduck_velocity_env_cfg(play=True)),
            stiffness=_k,
            travel=_travel,
        ),
        rl_cfg=sprung_rl_cfg(_label),
        runner_cls=MicroduckOnPolicyRunner,
    )
    print(f"✓ Sprung task registered: {_tid}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sprung_cfg.py -v`
Expected: 10 passed.

Then confirm registration end to end:

Run: `uv run python -c "import mjlab_microduck.tasks"`
Expected: five `✓ Sprung task registered:` lines.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest tests/ -v`

Expected: every pre-existing test still passes, plus **exactly 4 failures, all in
`tests/test_wheel_glide.py`** (`KeyError: 'passive_LF_?wheel'`). Those are
pre-existing repo debt, confirmed to reproduce before this branch's first code
commit, and are NOT yours to fix.

Any OTHER failure is yours. In particular, if a `run`, `velocity` or `backlash`
test breaks, `make_sprung_variant` is mutating shared config state — fix it by
deep-copying inside the transform, **never** by editing the failing test.

- [ ] **Step 6: Commit**

```bash
git add src/mjlab_microduck/tasks/sprung.py src/mjlab_microduck/tasks/__init__.py tests/test_sprung_cfg.py
git commit -m "feat(sprung): register the five stiffness sweep arms

Locked geometric control plus k=800/1500/2200/3000 N/m, each with its own wandb
identity and a deep-copied learner cfg so no arm shares mutable state."
```

---

## Handoff to the remote box

After Task 4 the sweep is code-complete but unmeasured. Training runs remotely —
do **not** start a campaign locally. A short smoke run first is worth it, since
the reward terms have only ever seen synthetic tensors:

```bash
uv run train Mjlab-Run-Flat-Sprung-K1500-MicroDuck \
    --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard
```

Then the five arms, 8000 iterations each, compared over the 7000-8000 window:

```bash
for ARM in Locked K800 K1500 K2200 K3000; do
  uv run train Mjlab-Run-Flat-Sprung-$ARM-MicroDuck \
      --env.scene.num-envs 4096 --agent.max-iterations 8000 \
      --agent.run-name sweep_$ARM
done
```

**Read the compression metrics before any speed number.** In order:

1. `Metrics/spring_compression_mean` — non-zero on the sprung arms. If it is ~0,
   the spring is not deflecting and no speed result means anything.
2. `Metrics/spring_bottomed_fraction` — near zero on the arms that matter. k800
   is expected to bottom out; if k1500 also does, travel is the binding
   constraint and 15 mm is too little.
3. `Metrics/forward_speed_mean` — **sprung vs the locked arm**, not vs 0.468 m/s.
4. `Metrics/flight_asymmetry` — must hold near 0.70. Faster via a degenerate
   bounce is not a win.
5. `Episode_Reward/com_height_target` — confirms the shifted band is satisfied,
   i.e. the height compensation worked.

## Deliberately not in this plan

- Choosing the physical mechanism (its own phase, opened with the `k` target in hand).
- Path-coupled mechanisms — the sim models pure translation only.
- Spring hysteresis and stiction.
- Damping as a sweep axis (fixed at 0.5 N.s/m).
- Any change to `action_rate_l2`.
