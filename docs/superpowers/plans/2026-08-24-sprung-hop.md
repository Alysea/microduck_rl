# Sprung-Foot Periodic Hop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a phase-commanded periodic hop task with three arms (a Locked geometric control plus k=2500 and k=3900) so the spring hypothesis can be tested in the regime most favourable to it.

**Architecture:** A `make_hop_variant(cfg)` transform in the shape of `tasks/run.py` and `tasks/sprung.py`, composed as `make_sprung_variant(make_hop_variant(velocity_cfg))`. The cyclic phase command already exists on `develop` (`GroundPickPhaseCommand`) and is reused with a shorter period. Three reward functions are ported from `origin/jump`; a new energy-return monitor is added.

**Tech Stack:** Python 3.12, MuJoCo `MjSpec`, PyTorch, mjlab 1.3.0, rsl-rl-lib 5.0.1, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-24-sprung-hop-design.md`

## Two discoveries that shrink this plan

Both found while reading the source, and both reduce scope versus the spec:

1. **No command class needs porting.** The spec says to reuse the `ground_pick`
   phase-command pattern. It is stronger than that: `GroundPickPhaseCommand`
   (`tasks/mdp.py:4951`) is already on `develop`, is already **cyclic**
   (`self._gp_phase = (self._gp_phase + dt / self._period) % 1.0`), and already
   takes a `period` cfg field. Configure it, do not reimplement it.
2. **Only THREE of the four jump rewards are needed.** `jump_phase_complete` is a
   termination for the *one-shot* jump variant — it fires when a clamped phase
   reaches 1.0. A cyclic phase never reaches 1.0, so that function is
   meaningless here. Porting it would add a termination that never fires.

## Global Constraints

- **Branch:** all work on `spring_v2`. Standing commit permission — commit directly, do not ask. **Never `git push`**, on any branch.
- **Never read `.envrc`** or `..envrc.~undo-tree~` — secrets.
- **Training is remote.** Do not launch a campaign locally; unit tests and at most a very short smoke run.
- **Naming:** passive joints stay `passive_*`; every existing `^(?!passive_).*` regex depends on the prefix.
- **Reward functions** take `env` first, return a `torch.Tensor` of shape `(num_envs,)`, and are NaN-safe. Match `alternating_flight` / `spring_compression_monitor` in `tasks/mdp.py`.
- **Monitors are registered with a NON-ZERO weight.** `RewardManager.compute` (`mjlab/managers/reward_manager.py:122`) short-circuits before calling a term whose weight is `0.0`; a zero-weight monitor is silently dead.
- **`action_rate_l2` stays at −1.0.** The fixed smoothness budget for the campaign, not a variable.
- **Tests** use lightweight duck-typed fakes (`tests/test_sprung.py` is the model), never a real mjlab env — except model-level tests that legitimately compile an `MjSpec`.
- Test command: `uv run pytest tests/<file> -v`

## Verified facts — use these, do not re-derive

Checked against the working tree before this plan was written:

1. **`GroundPickPhaseCommand`** at `tasks/mdp.py:4951` subclasses `UniformVelocityCommand`, emits `[cos(2πφ), sin(2πφ), 0]` into `vel_command_b`, reads `period` and `randomize_phase` from its cfg, and exposes `_gp_phase`. Configured via `microduck_mdp.GroundPickPhaseCommandCfg(**{**vars(command), "class_type": ..., "period": ...})` — see `microduck_ground_pick_env_cfg.py:465`.
2. **All three ported rewards gate on `cmd[:, 1]`** — i.e. `sin(2πφ)`, positive over the first half-cycle — via `launch_weight = torch.clamp(cmd[:, 1], min=0.0)`. They read the command by name, defaulting to `"twist"`, which is the command term the velocity env already registers.
3. **`jump_body_height` has a hardcoded `target_height=0.135`** = the rigid robot's ~0.12 m standing height plus 0.015 m of gain. **The sprung robot stands `H_ADD` = 0.030 m taller.** This target must be shifted or the reward asks the sprung robot to *crouch*. Same class of bug as the CoM band shift in Phase 2.
4. **`fell_over`** is `mdp.bad_orientation` with a 70° limit — orientation-based, not height-based — so the boot's added height does not perturb it. No change needed.
5. **`make_sprung_variant` works on any velocity-family cfg**, despite its
   docstring saying "Run-task env cfg". It touches only `com_height_target`,
   `pose`, `dof_pos_limits` and the robot entity — all of which come from the
   base velocity env, not from `make_run_variant`. So composing it over a hop
   cfg is safe. Its signature is
   `(cfg, stiffness, travel=TRAVEL, h_add=H_ADD, pad_mass=PAD_MASS)`.
6. **The velocity env already carries an `upright` reward** at weight 1.0
   (`microduck_velocity_env_cfg.py:267`), retained by the hop transform.
7. Spring-mass period at k=3900, m=0.877 kg is `2π√(m/k)` = **94 ms**.

---

### Task 1: Port the three jump reward functions

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (append at end of file)
- Test: `tests/test_hop.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, all returning shape `(num_envs,)`:
  - `hop_both_feet_airborne(env, sensor_name="feet_ground_contact", command_name="twist")`
  - `hop_upward_velocity(env, asset_cfg=..., command_name="twist", max_vel=0.5)`
  - `hop_body_height(env, asset_cfg=..., command_name="twist", target_height=..., std=0.008)`
  Used by Task 3.

Renamed `jump_*` → `hop_*` because the task is a cyclic hop, not a one-shot jump, and the repo already has enough near-duplicate names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hop.py`:

```python
"""Unit tests for the ported hop reward terms (duck-typed fakes)."""

import torch

from mjlab_microduck.tasks.mdp import (
    hop_body_height,
    hop_both_feet_airborne,
    hop_upward_velocity,
)

_SENSOR = "feet_ground_contact"
_CMD = "twist"


class _SensorData:
    def __init__(self, found):
        self.found = torch.tensor(found, dtype=torch.float32)


class _Sensor:
    def __init__(self, found):
        self.data = _SensorData(found)


class _AssetData:
    def __init__(self, vz, z):
        n = len(vz)
        self.root_link_lin_vel_w = torch.zeros((n, 3), dtype=torch.float32)
        self.root_link_lin_vel_w[:, 2] = torch.tensor(vz, dtype=torch.float32)
        self.root_link_pos_w = torch.zeros((n, 3), dtype=torch.float32)
        self.root_link_pos_w[:, 2] = torch.tensor(z, dtype=torch.float32)


class _Asset:
    def __init__(self, vz, z):
        self.data = _AssetData(vz, z)


class _CommandManager:
    def __init__(self, cmd):
        self._cmd = torch.tensor(cmd, dtype=torch.float32)

    def get_command(self, _name):
        return self._cmd


class _Scene:
    def __init__(self, sensors, asset):
        self.sensors = sensors
        self._asset = asset

    def __getitem__(self, _k):
        return self._asset


class _Env:
    """found: per-foot contact flags; cmd: [cos, sin, 0]; vz/z: base state."""

    def __init__(self, found=((0.0, 0.0),), cmd=((0.0, 1.0, 0.0),),
                 vz=(0.0,), z=(0.15,)):
        self.scene = _Scene({_SENSOR: _Sensor(found)}, _Asset(vz, z))
        self.command_manager = _CommandManager(cmd)
        self.num_envs = len(found)
        self.device = "cpu"
        self.extras = {"log": {}}


# --- hop_both_feet_airborne -------------------------------------------------

def test_airborne_rewarded_at_peak_launch_phase():
    # sin(2*pi*phi) = 1 (mid-launch), both feet off the ground
    env = _Env(found=[[0.0, 0.0]], cmd=[[0.0, 1.0, 0.0]])
    out = hop_both_feet_airborne(env, sensor_name=_SENSOR, command_name=_CMD)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_airborne_not_rewarded_when_a_foot_is_down():
    env = _Env(found=[[1.0, 0.0]], cmd=[[0.0, 1.0, 0.0]])
    out = hop_both_feet_airborne(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_airborne_not_rewarded_during_the_recovery_half_cycle():
    """sin < 0 is the recovery half — flight there must not be paid for,
    or the policy is rewarded for simply never landing."""
    env = _Env(found=[[0.0, 0.0]], cmd=[[0.0, -1.0, 0.0]])
    out = hop_both_feet_airborne(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


# --- hop_upward_velocity ----------------------------------------------------

def test_upward_velocity_saturates_at_max_vel():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], vz=[10.0])
    out = hop_upward_velocity(env, command_name=_CMD, max_vel=0.5)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_downward_velocity_is_not_rewarded():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], vz=[-2.0])
    out = hop_upward_velocity(env, command_name=_CMD, max_vel=0.5)
    assert float(out[0]) == 0.0


def test_upward_velocity_scales_below_saturation():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], vz=[0.25])
    out = hop_upward_velocity(env, command_name=_CMD, max_vel=0.5)
    assert abs(float(out[0]) - 0.5) < 1e-6


# --- hop_body_height --------------------------------------------------------

def test_body_height_peaks_at_the_target():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], z=[0.165])
    out = hop_body_height(env, command_name=_CMD, target_height=0.165, std=0.008)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_body_height_falls_off_away_from_the_target():
    env = _Env(cmd=[[0.0, 1.0, 0.0]], z=[0.145])
    out = hop_body_height(env, command_name=_CMD, target_height=0.165, std=0.008)
    assert float(out[0]) < 0.01


def test_body_height_is_gated_by_the_launch_phase():
    env = _Env(cmd=[[0.0, -1.0, 0.0]], z=[0.165])
    out = hop_body_height(env, command_name=_CMD, target_height=0.165, std=0.008)
    assert float(out[0]) == 0.0


def test_all_three_terms_are_nan_safe():
    env = _Env(found=[[0.0, 0.0]], cmd=[[0.0, 1.0, 0.0]],
               vz=[float("nan")], z=[float("nan")])
    for out in (
        hop_both_feet_airborne(env, sensor_name=_SENSOR, command_name=_CMD),
        hop_upward_velocity(env, command_name=_CMD),
        hop_body_height(env, command_name=_CMD, target_height=0.165),
    ):
        assert torch.isfinite(out).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hop.py -v`
Expected: FAIL — `ImportError: cannot import name 'hop_body_height'`

- [ ] **Step 3: Write the implementation**

Append to `src/mjlab_microduck/tasks/mdp.py`. These are ported from the
`origin/jump` branch (April, four months behind `develop`), renamed `jump_*` →
`hop_*`, with NaN guards added — the originals had none.

```python
def hop_both_feet_airborne(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
) -> torch.Tensor:
    """Reward both feet simultaneously airborne during the LAUNCH half-cycle.

    Ported from the abandoned `jump` branch. Gated on sin(2*pi*phase) > 0 so
    flight during the recovery half earns nothing — without that gate the policy
    is rewarded for simply never landing.
    """
    zeros = torch.zeros(env.num_envs, device=env.device)
    if sensor_name not in env.scene.sensors:
        return zeros
    found = env.scene.sensors[sensor_name].data.found
    if found is None or found.shape[1] < 2:
        return zeros
    found = torch.nan_to_num(found[:, :2].float(), nan=1.0)   # NaN -> "in contact"
    both_airborne = ((found[:, 0] <= 0) & (found[:, 1] <= 0)).float()

    cmd = env.command_manager.get_command(command_name)
    launch = torch.clamp(torch.nan_to_num(cmd[:, 1], nan=0.0), min=0.0)
    return launch * both_airborne


def hop_upward_velocity(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    max_vel: float = 0.5,
) -> torch.Tensor:
    """Reward upward base velocity during the launch half-cycle.

    Gives a gradient toward liftoff BEFORE the feet actually leave the ground,
    which `hop_both_feet_airborne` alone cannot provide (it is all-or-nothing).
    Clamped to [0, 1] so it saturates and never rewards falling.
    """
    asset: Entity = env.scene[asset_cfg.name]
    vel_z = torch.nan_to_num(
        asset.data.root_link_lin_vel_w[:, 2].float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    upward = torch.clamp(vel_z / max_vel, min=0.0, max=1.0)

    cmd = env.command_manager.get_command(command_name)
    launch = torch.clamp(torch.nan_to_num(cmd[:, 1], nan=0.0), min=0.0)
    return launch * upward


def hop_body_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_name: str = "twist",
    target_height: float = 0.135,
    std: float = 0.008,
) -> torch.Tensor:
    """Gaussian reward for base height reaching the hop target during launch.

    NOTE: `target_height` is NOT a safe default here. The ported value of 0.135
    is the RIGID robot's ~0.12 m standing height plus 0.015 m of gain. The sprung
    robot stands H_ADD (0.030 m) taller, so a caller that leaves this at the
    default is asking the sprung robot to CROUCH. `make_hop_variant` computes it
    from the robot's actual standing height; see tasks/hop.py.
    """
    asset: Entity = env.scene[asset_cfg.name]
    height = torch.nan_to_num(
        asset.data.root_link_pos_w[:, 2].float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    height_reward = torch.exp(-(((height - target_height) / std) ** 2))

    cmd = env.command_manager.get_command(command_name)
    launch = torch.clamp(torch.nan_to_num(cmd[:, 1], nan=0.0), min=0.0)
    return launch * height_reward
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hop.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_hop.py
git commit -m "feat(hop): port three jump reward terms from the abandoned jump branch

Renamed jump_* -> hop_* (cyclic hop, not a one-shot jump) and added the NaN
guards the originals lacked. jump_phase_complete is deliberately NOT ported:
it terminates when a CLAMPED phase reaches 1.0, which a cyclic phase never
does, so it would add a termination that never fires."
```

---

### Task 2: Energy-return monitor

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (append after the Task 1 terms)
- Test: `tests/test_hop.py` (extend)

**Interfaces:**
- Consumes: nothing.
- Produces: `hop_energy_monitor(env, joint_names, stiffness, preload, asset_cfg=...) -> torch.Tensor` returning zeros. Logs `Metrics/hop_spring_energy_mean` and `Metrics/hop_spring_energy_peak` (joules, summed over both feet). Used by Task 3.

The spec makes restitution the crux: `zeta = 0.3` discards ~63% of stored energy,
so knowing how much energy the spring is *storing* per cycle is what makes a hop
height interpretable.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hop.py`:

```python
from mjlab_microduck.tasks.mdp import hop_energy_monitor

_JOINTS = ("passive_left_foot_spring", "passive_right_foot_spring")
_K = 3900.0
_PRELOAD = 0.00074


class _JointData:
    def __init__(self, q):
        self.joint_pos = torch.tensor(q, dtype=torch.float32)


class _JointAsset:
    def __init__(self, q):
        self.data = _JointData(q)

    def find_joints(self, name):
        return [_JOINTS.index(name)], None


class _JointScene:
    def __init__(self, q):
        self._a = _JointAsset(q)

    def __getitem__(self, _k):
        return self._a


class _JointEnv:
    def __init__(self, q):
        self.scene = _JointScene(q)
        self.num_envs = len(q)
        self.device = "cpu"
        self.extras = {"log": {}}


def test_energy_monitor_returns_exactly_zeros():
    env = _JointEnv([[0.005, 0.005]])
    out = hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_energy_matches_the_closed_form():
    """E = 0.5*k*q^2 + k*preload*q per foot, summed over both."""
    q = 0.006
    env = _JointEnv([[q, q]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    expected = 2 * (0.5 * _K * q**2 + _K * _PRELOAD * q)
    got = float(env.extras["log"]["Metrics/hop_spring_energy_mean"])
    assert abs(got - expected) < 1e-9


def test_energy_is_zero_at_rest():
    env = _JointEnv([[0.0, 0.0]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert float(env.extras["log"]["Metrics/hop_spring_energy_mean"]) == 0.0


def test_negative_q_contributes_no_energy():
    """Preload holds the pad past its lower limit when unloaded (measured
    -0.59 mm). That is limit penetration, not stored energy."""
    env = _JointEnv([[-0.00059, -0.00059]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert float(env.extras["log"]["Metrics/hop_spring_energy_mean"]) == 0.0


def test_peak_exceeds_mean_when_feet_differ():
    env = _JointEnv([[0.002, 0.002], [0.010, 0.010]])
    hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    log = env.extras["log"]
    assert float(log["Metrics/hop_spring_energy_peak"]) > float(
        log["Metrics/hop_spring_energy_mean"]
    )


def test_energy_monitor_survives_missing_joints():
    """The Locked control arm has NO spring joints; mjlab's find_joints RAISES
    on no match rather than returning empty."""
    class _RaisingAsset(_JointAsset):
        def find_joints(self, name):
            raise ValueError("Not all regular expressions are matched!")

    env = _JointEnv([[0.005, 0.005]])
    env.scene._a = _RaisingAsset([[0.005, 0.005]])
    out = hop_energy_monitor(env, joint_names=_JOINTS, stiffness=_K, preload=_PRELOAD)
    assert float(out[0]) == 0.0
    assert float(env.extras["log"]["Metrics/hop_spring_energy_mean"]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hop.py -v`
Expected: FAIL — `ImportError: cannot import name 'hop_energy_monitor'`

- [ ] **Step 3: Write the implementation**

Append to `src/mjlab_microduck/tasks/mdp.py`:

```python
def hop_energy_monitor(
    env: ManagerBasedRlEnv,
    joint_names: tuple,
    stiffness: float,
    preload: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Log elastic energy stored in the foot springs. Contributes zero reward.

    A hop height is uninterpretable without knowing how much energy the spring
    actually stored: the drop-rig probe measured restitution 0.57-0.70 at
    zeta = 0.3, i.e. ~63% of stored energy is dissipated, so a poor hop could
    mean "stored little" or "stored plenty and lost it". This separates them.

    E = 0.5*k*q^2 + k*preload*q per foot (the second term is the work done
    against the preload), summed over both feet. Negative q is limit
    penetration, not compression, and is clamped away.

    Returns zeros, so the reward total is unaffected. **Register with a non-zero
    weight anyway**: RewardManager.compute short-circuits before calling a term
    whose weight is 0.0. The Locked arm has no spring joints, so the lookup
    raises there; that is normal and reports zero energy.
    """
    zeros = torch.zeros(env.num_envs, device=env.device)
    log = env.extras.get("log") if hasattr(env, "extras") else None

    def _log_zero():
        if log is not None:
            z = torch.zeros((), device=env.device)
            log["Metrics/hop_spring_energy_mean"] = z
            log["Metrics/hop_spring_energy_peak"] = z

    asset: Entity = env.scene[asset_cfg.name]
    ids = []
    try:
        for name in joint_names:
            found, _ = asset.find_joints(name)
            if not found:
                _log_zero()
                return zeros
            ids.append(found[0])
    except ValueError:
        # Locked control arm: no spring joints exist.
        _log_zero()
        return zeros

    q = torch.nan_to_num(
        asset.data.joint_pos[:, ids].float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp(min=0.0)

    energy = (0.5 * stiffness * q.pow(2) + stiffness * preload * q).sum(dim=1)
    if log is not None:
        log["Metrics/hop_spring_energy_mean"] = energy.mean()
        log["Metrics/hop_spring_energy_peak"] = energy.max()
    return zeros
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hop.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_hop.py
git commit -m "feat(hop): energy-return monitor

Logs elastic energy stored per cycle so a hop height can be read against how
much the spring actually stored -- at zeta=0.3 the drop probe measured ~63% of
stored energy dissipated, so a poor hop is ambiguous without this."
```

---

### Task 3: The hop variant transform

**Files:**
- Create: `src/mjlab_microduck/tasks/hop.py`
- Test: `tests/test_hop_cfg.py` (create)

**Interfaces:**
- Consumes: `hop_both_feet_airborne`, `hop_upward_velocity`, `hop_body_height` (Task 1); `hop_energy_monitor` (Task 2); `GroundPickPhaseCommand`/`GroundPickPhaseCommandCfg` from `tasks/mdp.py`; `H_ADD` from `robot/sprung_foot.py`.
- Produces: `make_hop_variant(cfg, h_add=0.0) -> ManagerBasedRlEnvCfg`, plus `HOP_PERIOD`, `HOP_HEIGHT_GAIN`, `RIGID_STAND_HEIGHT`, `hop_target_height(h_add)`. Used by Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hop_cfg.py`:

```python
"""Config-level assertions for the hop variant transform."""

import pytest

from mjlab_microduck.robot.sprung_foot import H_ADD
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.hop import (
    HOP_HEIGHT_GAIN,
    HOP_PERIOD,
    RIGID_STAND_HEIGHT,
    hop_target_height,
    make_hop_variant,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


@pytest.fixture
def hop_cfg():
    return make_hop_variant(make_microduck_velocity_env_cfg(), h_add=H_ADD)


def test_command_is_the_cyclic_phase_command(hop_cfg):
    term = hop_cfg.commands["twist"]
    assert term.class_type is microduck_mdp.GroundPickPhaseCommand
    assert term.period == pytest.approx(HOP_PERIOD)


def test_height_target_is_shifted_by_h_add(hop_cfg):
    """The ported reward hardcodes 0.135 for the RIGID robot. The sprung robot
    stands H_ADD taller, so an unshifted target asks it to CROUCH."""
    expected = RIGID_STAND_HEIGHT + HOP_HEIGHT_GAIN + H_ADD
    assert hop_cfg.rewards["hop_body_height"].params["target_height"] == pytest.approx(expected)
    assert hop_target_height(H_ADD) == pytest.approx(expected)


def test_rigid_variant_target_is_not_shifted():
    rigid = make_hop_variant(make_microduck_velocity_env_cfg(), h_add=0.0)
    expected = RIGID_STAND_HEIGHT + HOP_HEIGHT_GAIN
    assert rigid.rewards["hop_body_height"].params["target_height"] == pytest.approx(expected)


def test_all_three_hop_rewards_registered_with_positive_weight(hop_cfg):
    for name, func in (
        ("hop_both_feet_airborne", microduck_mdp.hop_both_feet_airborne),
        ("hop_upward_velocity", microduck_mdp.hop_upward_velocity),
        ("hop_body_height", microduck_mdp.hop_body_height),
    ):
        term = hop_cfg.rewards[name]
        assert term.func is func
        assert term.weight > 0.0


def test_energy_monitor_registered_with_non_zero_weight(hop_cfg):
    # RewardManager.compute skips weight==0.0 terms before calling them.
    term = hop_cfg.rewards["hop_energy_monitor"]
    assert term.func is microduck_mdp.hop_energy_monitor
    assert term.weight != 0.0


def test_forward_locomotion_rewards_removed(hop_cfg):
    """This is a hop in place. Leaving velocity tracking in would reward the
    policy for running away instead of hopping, and the phase command has
    overwritten the twist command those terms read."""
    for name in ("track_linear_velocity", "track_angular_velocity"):
        assert name not in hop_cfg.rewards


def test_action_rate_curriculum_untouched(hop_cfg):
    rigid = make_microduck_velocity_env_cfg()
    expected = [dict(s) for s in rigid.curriculum["action_rate_weight"].params["weight_stages"]]
    actual = [dict(s) for s in hop_cfg.curriculum["action_rate_weight"].params["weight_stages"]]
    assert actual == expected


def test_hop_period_is_above_the_spring_mass_period():
    """At k=3900 and 0.877 kg the spring-mass period is ~94 ms. A hop period at
    or below that would drive the spring faster than it can cycle."""
    assert HOP_PERIOD > 0.094
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hop_cfg.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mjlab_microduck.tasks.hop'`

- [ ] **Step 3: Write the implementation**

Create `src/mjlab_microduck/tasks/hop.py`:

```python
"""Hop task variant — a phase-commanded periodic hop in place.

``make_hop_variant(cfg, h_add)`` converts a velocity env cfg into a hop task, in
the same shape as ``tasks/run.py`` and ``tasks/backlash.py``. Composed as
``make_sprung_variant(make_hop_variant(cfg), ...)`` so the sprung machinery is
reused unchanged.

Periodic rather than a one-shot jump because the spring's energy comes
overwhelmingly from IMPACT loading: a drop-rig probe measured a 100 mm drop
rebounding 33 mm, while quasi-static actuator loading only just reaches the
49.7 N needed for full travel (52.4 N available, knee-limited) before BAM
back-EMF derates it at launch speed. Each landing charges the next launch.

Four changes:

1. Replace the twist command with the CYCLIC phase command already on develop.
2. Retarget the ported height reward for the robot's actual standing height.
3. Drop forward-locomotion rewards — this is a hop in place.
4. Register the three hop rewards and the energy monitor.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg

from mjlab_microduck.robot.sprung_foot import SPRING_JOINTS, SPRING_PRELOAD
from mjlab_microduck.tasks import mdp as microduck_mdp

# Seconds per hop cycle. The spring-mass period at k=3900 and 0.877 kg is
# 2*pi*sqrt(m/k) = 94 ms, so this is deliberately well above it: the cycle must
# accommodate a load segment, a launch, a flight and a landing, not just one
# spring oscillation. Provisional -- sweep only if the first result is ambiguous.
HOP_PERIOD = 1.0

# Rigid standing trunk height, and the height gain the reward asks for on top.
# The ported reward hardcoded 0.135 = 0.12 + 0.015 for the RIGID robot; these
# split that into its two parts so h_add can be added correctly.
RIGID_STAND_HEIGHT = 0.120
HOP_HEIGHT_GAIN = 0.015

HOP_HEIGHT_STD = 0.008
SENSOR_NAME = "feet_ground_contact"

AIRBORNE_WEIGHT = 3.0
UPWARD_VELOCITY_WEIGHT = 2.0
BODY_HEIGHT_WEIGHT = 2.0
ENERGY_MONITOR_WEIGHT = 1.0

# Forward-locomotion rewards read the twist command, which the phase command
# overwrites with [cos, sin, 0]. Left in place they would reward running away.
_LOCOMOTION_REWARDS = ("track_linear_velocity", "track_angular_velocity")

# LANDING SURVIVAL: the spec asks for a landing-survival term. It is met by the
# terms already present rather than by a new one -- the velocity env's `upright`
# reward (weight 1.0) pays for staying vertical through the landing, and the
# `fell_over` termination (bad_orientation, 70 deg) ends an episode that fails.
# Adding a third redundant survival term would double-count the same behaviour
# and make the hop rewards harder to balance against it.


def hop_target_height(h_add: float) -> float:
    """Target base height for the hop apex, shifted by the boot's added height.

    The sprung robot stands ``h_add`` taller, so an unshifted target asks it to
    CROUCH rather than hop -- the same class of bug as the CoM band shift.
    """
    return RIGID_STAND_HEIGHT + HOP_HEIGHT_GAIN + h_add


def make_hop_variant(
    cfg: ManagerBasedRlEnvCfg,
    h_add: float = 0.0,
) -> ManagerBasedRlEnvCfg:
    """Convert a velocity env cfg into the periodic hop task.

    Args:
        h_add: metres of height the foot mechanism adds. 0.0 for the rigid
            robot; pass the sprung model's H_ADD for sprung arms.
    """
    # 1. Cyclic phase command, reusing the class already on develop.
    command = cfg.commands["twist"]
    cfg.commands["twist"] = microduck_mdp.GroundPickPhaseCommandCfg(
        **{
            **vars(command),
            "class_type": microduck_mdp.GroundPickPhaseCommand,
            "period": HOP_PERIOD,
        }
    )

    # 2/3. This is a hop in place: forward tracking would reward running away,
    #      and its command has just been overwritten anyway.
    for name in _LOCOMOTION_REWARDS:
        cfg.rewards.pop(name, None)

    # 4. Hop rewards. All three gate internally on sin(2*pi*phase) > 0.
    cfg.rewards["hop_both_feet_airborne"] = RewardTermCfg(
        func=microduck_mdp.hop_both_feet_airborne,
        weight=AIRBORNE_WEIGHT,
        params={"sensor_name": SENSOR_NAME, "command_name": "twist"},
    )
    cfg.rewards["hop_upward_velocity"] = RewardTermCfg(
        func=microduck_mdp.hop_upward_velocity,
        weight=UPWARD_VELOCITY_WEIGHT,
        params={"command_name": "twist", "max_vel": 0.5},
    )
    cfg.rewards["hop_body_height"] = RewardTermCfg(
        func=microduck_mdp.hop_body_height,
        weight=BODY_HEIGHT_WEIGHT,
        params={
            "command_name": "twist",
            "target_height": hop_target_height(h_add),
            "std": HOP_HEIGHT_STD,
        },
    )

    # Energy instrument. Returns zeros, so the weight only has to be non-zero
    # for RewardManager.compute to call it at all.
    cfg.rewards["hop_energy_monitor"] = RewardTermCfg(
        func=microduck_mdp.hop_energy_monitor,
        weight=ENERGY_MONITOR_WEIGHT,
        params={
            "joint_names": SPRING_JOINTS,
            "stiffness": 3900.0,
            "preload": SPRING_PRELOAD,
        },
    )

    return cfg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hop_cfg.py -v`
Expected: 8 passed.

If `GroundPickPhaseCommandCfg` rejects any key carried over by `vars(command)`,
mirror exactly what `microduck_ground_pick_env_cfg.py:465` does — that call site
is the working reference. Do not invent a different construction.

- [ ] **Step 5: Commit**

```bash
git add src/mjlab_microduck/tasks/hop.py tests/test_hop_cfg.py
git commit -m "feat(hop): make_hop_variant transform

Reuses the cyclic GroundPickPhaseCommand already on develop rather than porting
a command class. Retargets the ported height reward by h_add -- the ported 0.135
is the RIGID robot's height, so unshifted it would ask the sprung robot to crouch."
```

---

### Task 4: Register the three hop arms

**Files:**
- Modify: `src/mjlab_microduck/tasks/hop.py` (append the arm table and RL cfg)
- Modify: `src/mjlab_microduck/tasks/__init__.py` (import + register, after the sprung registrations)
- Test: `tests/test_hop_cfg.py` (extend)

**Interfaces:**
- Consumes: `make_hop_variant` (Task 3); `make_sprung_variant`, `sprung_rl_cfg` from `tasks/sprung.py`; `H_ADD`, `PAD_MASS`, `TRAVEL` from `robot/sprung_foot.py`; `MicroduckRunRlCfg` from `tasks/run.py`; `MicroduckOnPolicyRunner`.
- Produces: `HOP_ARMS`, `HOP_ARM_SUFFIX`, `hop_rl_cfg(label)`, and three registered task ids.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hop_cfg.py`:

```python
def test_hop_arms_are_locked_plus_two_stiffnesses():
    from mjlab_microduck.tasks.hop import HOP_ARMS

    labels = [a[0] for a in HOP_ARMS]
    assert "locked" in labels, "the locked arm is the geometric control"
    stiffnesses = {a[1] for a in HOP_ARMS if a[0] != "locked"}
    assert stiffnesses == {2500.0, 3900.0}
    for label, _k, travel, pad in HOP_ARMS:
        assert pad == pytest.approx(0.070), "mass is held; Stage 1 measured it"
        if label == "locked":
            assert travel == 0.0
        else:
            assert travel > 0.0


def test_hop_task_ids_registered():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    for tid in (
        "Mjlab-Hop-Flat-Sprung-Locked-MicroDuck",
        "Mjlab-Hop-Flat-Sprung-K2500-MicroDuck",
        "Mjlab-Hop-Flat-Sprung-K3900-MicroDuck",
    ):
        assert tid in tasks, f"{tid} not registered"


def test_hop_arms_have_distinct_wandb_identities():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_rl_cfg

    names = {
        load_rl_cfg(f"Mjlab-Hop-Flat-Sprung-{s}-MicroDuck").run_name
        for s in ("Locked", "K2500", "K3900")
    }
    assert len(names) == 3


def test_registered_hop_cfgs_carry_the_shifted_height_target():
    """End-to-end: the registered task, not just the transform in isolation."""
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Hop-Flat-Sprung-K3900-MicroDuck")
    assert cfg.rewards["hop_body_height"].params["target_height"] == pytest.approx(
        hop_target_height(H_ADD)
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_hop_cfg.py -v`
Expected: FAIL — `ImportError: cannot import name 'HOP_ARMS'`

- [ ] **Step 3: Write the implementation**

Append to `src/mjlab_microduck/tasks/hop.py`:

```python
from copy import deepcopy
from dataclasses import replace

from mjlab_microduck.robot.sprung_foot import H_ADD, PAD_MASS, TRAVEL
from mjlab_microduck.tasks.run import MicroduckRunRlCfg

# (label, stiffness N/m, travel m, pad mass kg).
#
# Three arms, not six. The drop-rig probe already narrowed stiffness: at a
# 100 mm drop k2500 rebounded 35.3 mm, k3900 32.8 mm, k5500 28.3 mm, and k1500
# only 20.6 mm because it bottoms out and slams. So the optimum is the softest
# spring that does NOT bottom, and 2500-3900 brackets it.
#
# Mass is held at the measured 70 g because Stage 1's locked arms already
# measured the mass penalty separately (-17.7% at 30 g, -61.9% at 90 g vs the
# rigid running baseline).
HOP_ARMS = (
    ("locked", 3900.0, 0.0, PAD_MASS),
    ("k2500", 2500.0, TRAVEL, PAD_MASS),
    ("k3900", 3900.0, TRAVEL, PAD_MASS),
)

HOP_ARM_SUFFIX = {"locked": "Locked", "k2500": "K2500", "k3900": "K3900"}


def hop_rl_cfg(label: str):
    """Per-arm RL cfg: identical learner, distinct logging identity.

    ``replace`` is shallow, so the nested cfgs are deep-copied -- otherwise all
    three arms would share one actor object AND share it with the Run baseline,
    so a later change to any arm would silently alter the others.
    """
    return replace(
        MicroduckRunRlCfg,
        actor=deepcopy(MicroduckRunRlCfg.actor),
        critic=deepcopy(MicroduckRunRlCfg.critic),
        algorithm=deepcopy(MicroduckRunRlCfg.algorithm),
        experiment_name=f"hop_{label}",
        run_name=f"hop_{label}",
    )
```

Then in `src/mjlab_microduck/tasks/__init__.py`, add the import beside the
others — it must come **after** the `.run` and `.sprung` imports, since `hop.py`
imports from both:

```python
from .hop import HOP_ARMS, HOP_ARM_SUFFIX, hop_rl_cfg, make_hop_variant
```

and register after the sprung block:

```python
# Periodic hop on the sprung foot — Phase 4. See
# docs/superpowers/specs/2026-08-24-sprung-hop-design.md
for _label, _k, _travel, _pad in HOP_ARMS:
    _tid = f"Mjlab-Hop-Flat-Sprung-{HOP_ARM_SUFFIX[_label]}-MicroDuck"
    register_mjlab_task(
        task_id=_tid,
        env_cfg=make_sprung_variant(
            make_hop_variant(make_microduck_velocity_env_cfg(), h_add=H_ADD),
            stiffness=_k, travel=_travel, pad_mass=_pad,
        ),
        play_env_cfg=make_sprung_variant(
            make_hop_variant(make_microduck_velocity_env_cfg(play=True), h_add=H_ADD),
            stiffness=_k, travel=_travel, pad_mass=_pad,
        ),
        rl_cfg=hop_rl_cfg(_label),
        runner_cls=MicroduckOnPolicyRunner,
    )
    print(f"✓ Hop task registered: {_tid}")
```

Note the composition order: `make_hop_variant` **first**, then
`make_sprung_variant`. The sprung transform swaps the robot and shifts the CoM
band; the hop transform swaps the command and rewards. They touch disjoint parts
of the cfg, but this order matches the spec's stated composition.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_hop_cfg.py -v`
Expected: 12 passed.

Then confirm registration end to end:

Run: `uv run python -c "import mjlab_microduck.tasks"`
Expected: three `✓ Hop task registered:` lines.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest tests/ -v`

Expected: every pre-existing test still passes, plus **exactly 4 failures, all in
`tests/test_wheel_glide.py`** (`KeyError: 'passive_LF_?wheel'`) — pre-existing
repo debt, confirmed to reproduce before this branch's first code commit, NOT
yours to fix. Report the passed total.

Any OTHER failure is yours. If a `run`, `sprung`, `velocity` or `backlash` test
breaks, a transform is mutating shared config state — fix it by deep-copying
inside the transform, **never** by editing the failing test.

- [ ] **Step 6: Commit**

```bash
git add src/mjlab_microduck/tasks/hop.py src/mjlab_microduck/tasks/__init__.py tests/test_hop_cfg.py
git commit -m "feat(hop): register the three periodic-hop arms

Locked geometric control plus k=2500 and k=3900, all at the measured 70 g pad.
Stiffness narrowed by the drop-rig probe; mass held because Stage 1 already
measured the mass penalty."
```

---

## Handoff to the remote box

Code-complete but unmeasured after Task 4. Training runs remotely — do **not**
start a campaign locally. Smoke one arm first; these rewards have only seen
synthetic tensors:

```bash
uv run train Mjlab-Hop-Flat-Sprung-K3900-MicroDuck \
    --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard
```

Then the three arms, 8000 iterations each:

```bash
for ARM in Locked K2500 K3900; do
  uv run train Mjlab-Hop-Flat-Sprung-$ARM-MicroDuck \
      --env.scene.num-envs 4096 --agent.max-iterations 8000 \
      --agent.run-name hop_$ARM
done
```

**Read the instruments before any height number**, in this order:

1. `Metrics/hop_spring_energy_mean` / `_peak` — is the spring storing anything?
   Near zero means no result is interpretable.
2. `Metrics/spring_bottomed_fraction` — near zero on the arms that matter. The
   end-stop retains ~9% overshoot under the hardest impacts, so a high value
   means "this arm bottoms out", not a precise number.
3. `Metrics/spring_compression_loaded_mean` — not `_mean`, which is diluted by
   flight.
4. Base height / hop apex — **sprung against the Locked arm**, never against the
   rigid running baseline.
5. `Episode_Termination/fell_over` — landings must not be materially worse than
   Locked.

## Deliberately not in this plan

- Skipping and bounding gaits (spec: out of scope until a hop result exists).
- Sweeping `HOP_PERIOD` — one value first; sweep only if the result is ambiguous.
- Sweeping pad mass — Stage 1's locked arms already measured it.
- Changing `zeta` without a hysteresis measurement on the prototype.
- Any change to `action_rate_l2`, the pad geometry, or the CoM band shift.
- `jump_phase_complete` from the `origin/jump` branch — it terminates on a
  clamped phase reaching 1.0, which a cyclic phase never does.
