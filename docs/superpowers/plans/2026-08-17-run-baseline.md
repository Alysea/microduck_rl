# Rigid Running Baseline (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the rigid MicroDuck a `Run` task whose rewards target *alternating* flight rather than symmetric bouncing, so its speed plateau can serve as the control for the later sprung-leg comparison.

**Architecture:** A variant transform (`tasks/run.py::make_run_variant`) in the shape of the existing `tasks/backlash.py`, applied to the velocity env cfg. Three new reward functions land in `tasks/mdp.py`. No new robot model, no new env cfg file — this composes so Phase 3 can wrap it as `make_sprung_variant(make_run_variant(cfg))`.

**Tech Stack:** Python 3.12, PyTorch, mjlab 1.3.0 (MuJoCo-Warp), rsl-rl-lib 5.0.1, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-17-sprung-running-design.md`

## Global Constraints

Every task's requirements implicitly include these.

- **Branch:** all work happens on `spring_v2`. **Never** edit files while on `develop` or `main`.
- **Commits:** Steve granted standing commit permission **on the `spring_v2` branch** (2026-08-17), so commit freely here. That permission is branch-scoped and does not carry elsewhere. Never `git push` under any circumstances, on any branch — ask Steve to push.
- **Never read `.envrc`** or `..envrc.~undo-tree~`. Secrets; loaded by `direnv`.
- **Training is remote.** Do not launch a full training run locally. Local scope is config, unit tests, and at most a very short smoke run.
- **Versions:** `mjlab==1.3.0`, `rsl-rl-lib==5.0.1`. Run `uv sync` if the venv drifts.
- **Naming:** any passive joint must be named `passive_*` — every existing `^(?!passive_).*` regex depends on it.
- **Reward functions** take `env` first, return a `torch.Tensor` of shape `(num_envs,)`, guard missing sensors with `torch.zeros(env.num_envs, device=env.device)`, and are NaN-safe. Match the style of `wheel_speed_reward` / `feet_grounded_reward` in `tasks/mdp.py`.
- **Tests** use lightweight duck-typed fakes (see `tests/test_wheel_glide.py`), never a real mjlab env.
- **Test command:** `uv run pytest tests/<file> -v`
- Symmetry stays OFF (`SYMMETRY_CFG` is hardcoded for the retired 51-D obs layout; obs is 61-D).

---

### Task 1: `alternating_flight` reward

Separates real running from the symmetric bouncing gait the previous campaign produced, using only air-time asymmetry — no phase state machine, no history buffer.

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (append near the other contact-sensor rewards, after `feet_grounded_reward` ~line 1627)
- Test: `tests/test_run.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `alternating_flight(env, sensor_name: str, command_name: str, command_threshold: float = 0.01, eps: float = 1e-6) -> torch.Tensor`, shape `(num_envs,)`. Logs `Metrics/flight_asymmetry` and `Metrics/flight_fraction` into `env.extras["log"]`. Used by Task 4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run.py`:

```python
"""Reward-function unit tests for the Run task (Phase 1 rigid running baseline).

Uses duck-typed fakes rather than a real mjlab env, matching tests/test_wheel_glide.py.
"""

import torch

from mjlab_microduck.tasks.mdp import alternating_flight


class _Data:
    def __init__(self, air):
        self.current_air_time = torch.tensor(air, dtype=torch.float32)


class _Sensor:
    def __init__(self, air):
        self.data = _Data(air)


class _CommandManager:
    def __init__(self, cmd):
        self._cmd = torch.tensor(cmd, dtype=torch.float32)

    def get_command(self, _name):
        return self._cmd


class _Scene:
    def __init__(self, sensors):
        self.sensors = sensors


class _Env:
    """air: list of [left_air_time, right_air_time]; cmd: list of [vx, vy, wz]."""

    def __init__(self, air, cmd=None, sensor_name="feet_ground_contact"):
        if cmd is None:
            cmd = [[0.5, 0.0, 0.0]] * len(air)
        self.scene = _Scene({sensor_name: _Sensor(air)})
        self.command_manager = _CommandManager(cmd)
        self.num_envs = len(air)
        self.device = "cpu"
        self.extras = {"log": {}}


_SENSOR = "feet_ground_contact"
_CMD = "twist"


def test_symmetric_bounce_scores_zero():
    # Both feet airborne with identical air time — the rejected bouncing gait.
    env = _Env([[0.10, 0.10]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) < 1e-4


def test_alternating_flight_scores_high():
    # Trailing foot just left the ground, leading foot about to land.
    env = _Env([[0.02, 0.18]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) > 0.75


def test_both_feet_planted_scores_zero():
    env = _Env([[0.0, 0.0]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_single_support_is_not_flight():
    # One foot in the air is walking, not flight — must not be rewarded.
    env = _Env([[0.10, 0.0]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_inert_at_zero_command():
    env = _Env([[0.02, 0.18]], cmd=[[0.0, 0.0, 0.0]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_nan_safe():
    env = _Env([[float("nan"), 0.18]])
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert torch.isfinite(out).all()


def test_missing_sensor_returns_zeros():
    env = _Env([[0.02, 0.18]], sensor_name="some_other_sensor")
    out = alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_logs_metrics():
    env = _Env([[0.02, 0.18]])
    alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert "Metrics/flight_asymmetry" in env.extras["log"]
    assert "Metrics/flight_fraction" in env.extras["log"]


def test_asymmetry_metric_averages_over_flight_envs_only():
    # env 0 is in flight and symmetric; env 1 is in single support (asymmetry
    # would read 1.0 but must not pollute the metric).
    env = _Env([[0.10, 0.10], [0.10, 0.0]])
    alternating_flight(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(env.extras["log"]["Metrics/flight_asymmetry"]) < 1e-4
    assert abs(float(env.extras["log"]["Metrics/flight_fraction"]) - 0.5) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run.py -v`
Expected: FAIL — `ImportError: cannot import name 'alternating_flight' from 'mjlab_microduck.tasks.mdp'`

- [ ] **Step 3: Write the implementation**

Append to `src/mjlab_microduck/tasks/mdp.py`:

```python
def alternating_flight(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reward flight phases whose feet ALTERNATE, not flight per se.

    Both feet airborne is a flight phase, but that alone does not distinguish
    running from the symmetric two-foot bouncing gait the previous sprung
    campaign converged to. Air-time asymmetry does:

      - Running: at any flight instant the trailing foot has just left the
        ground (small air time) while the leading foot is about to land (large
        air time). |dt| / sum -> 1.
      - Bouncing: both feet leave and land together, air times equal.
        |dt| / sum -> 0.

    Note this must be paired with `feet_air_time_capped` — the stock mjlab
    `feet_air_time` sums its per-foot indicator, paying 2.0 for simultaneous
    flight versus 1.0 for alternating, which pulls the other way.

    Args:
        sensor_name: ContactSensorCfg name with ``track_air_time=True`` and two
            primary foot geoms, LEFT in column 0, RIGHT in column 1.
        command_name: velocity command; the term is inert below
            ``command_threshold``.

    Returns:
        Reward tensor (num_envs,) in [0, 1].
    """
    zeros = torch.zeros(env.num_envs, device=env.device)
    if sensor_name not in env.scene.sensors:
        return zeros

    air = env.scene.sensors[sensor_name].data.current_air_time
    if air is None or air.dim() < 2 or air.shape[1] < 2:
        return zeros
    air = torch.nan_to_num(air, nan=0.0, posinf=0.0, neginf=0.0)

    air_l, air_r = air[:, 0], air[:, 1]
    flight = ((air_l > 0.0) & (air_r > 0.0)).float()
    asymmetry = torch.abs(air_l - air_r) / (air_l + air_r + eps)

    command = env.command_manager.get_command(command_name)
    speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    reward = flight * asymmetry * (speed > command_threshold).float()

    log = env.extras.get("log") if hasattr(env, "extras") else None
    if log is not None:
        # Average asymmetry over FLIGHT envs only — in single support one air
        # time is 0, so asymmetry reads 1.0 and would swamp the metric.
        n_flight = flight.sum().clamp(min=1.0)
        log["Metrics/flight_asymmetry"] = (asymmetry * flight).sum() / n_flight
        log["Metrics/flight_fraction"] = flight.mean()

    return reward
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

Commit directly — `spring_v2` carries standing commit permission.

```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_run.py
git commit -m "feat(run): alternating_flight reward separating running from bouncing"
```

---

### Task 2: `feet_air_time_capped` reward

Removes the double payment for simultaneous two-foot flight that the stock mjlab reward makes.

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (immediately after `alternating_flight`)
- Test: `tests/test_run.py` (extend)

**Interfaces:**
- Consumes: the `_Env` / `_Sensor` / `_CommandManager` fakes from Task 1.
- Produces: `feet_air_time_capped(env, sensor_name: str, threshold_min: float = 0.05, threshold_max: float = 0.15, command_name: str | None = None, command_threshold: float = 0.01) -> torch.Tensor`. Signature is deliberately parameter-compatible with mjlab's `feet_air_time`, so Task 4 can swap `.func` on the existing term and keep its params. Used by Task 4.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run.py`:

```python
from mjlab_microduck.tasks.mdp import feet_air_time_capped


def test_capped_both_feet_in_window_scores_one_not_two():
    # THE bug being fixed: stock mjlab feet_air_time returns 2.0 here.
    env = _Env([[0.10, 0.10]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_capped_single_foot_in_window_scores_one():
    env = _Env([[0.10, 0.0]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_capped_below_window_scores_zero():
    env = _Env([[0.01, 0.01]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_capped_above_window_scores_zero():
    env = _Env([[0.40, 0.40]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_capped_inert_at_zero_command():
    env = _Env([[0.10, 0.10]], cmd=[[0.0, 0.0, 0.0]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0


def test_capped_nan_safe():
    env = _Env([[float("nan"), 0.10]])
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert torch.isfinite(out).all()


def test_capped_missing_sensor_returns_zeros():
    env = _Env([[0.10, 0.10]], sensor_name="some_other_sensor")
    out = feet_air_time_capped(env, sensor_name=_SENSOR, command_name=_CMD)
    assert float(out[0]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run.py -v`
Expected: FAIL — `ImportError: cannot import name 'feet_air_time_capped'`

- [ ] **Step 3: Write the implementation**

Append to `src/mjlab_microduck/tasks/mdp.py`:

```python
def feet_air_time_capped(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.15,
    command_name: str | None = None,
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """mjlab's ``feet_air_time`` with the per-foot indicator sum capped at 1.0.

    Stock ``feet_air_time`` (mjlab/tasks/velocity/mdp/rewards.py:209) does
    ``torch.sum(in_range.float(), dim=1)``, so both feet airborne scores 2.0
    against 1.0 for alternating — at weight 5.0 that is a standing incentive
    toward symmetric bouncing, and it grows with commanded speed. Capping at
    1.0 removes the double payment without forbidding flight; genuine flight is
    rewarded separately by `alternating_flight`.

    Parameter-compatible with the stock term so a config can swap ``.func``
    and keep the existing params dict.

    Returns:
        Reward tensor (num_envs,) in [0, 1].
    """
    zeros = torch.zeros(env.num_envs, device=env.device)
    if sensor_name not in env.scene.sensors:
        return zeros

    air = env.scene.sensors[sensor_name].data.current_air_time
    if air is None or air.dim() < 2:
        return zeros
    air = torch.nan_to_num(air, nan=0.0, posinf=0.0, neginf=0.0)

    in_range = (air > threshold_min) & (air < threshold_max)
    reward = torch.clamp(in_range.float().sum(dim=1), max=1.0)

    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward = reward * (speed > command_threshold).float()

    log = env.extras.get("log") if hasattr(env, "extras") else None
    if log is not None:
        # Preserve the dashboard metric the stock term emitted.
        in_air = air > 0
        n_in_air = in_air.float().sum().clamp(min=1.0)
        log["Metrics/air_time_mean"] = (air * in_air.float()).sum() / n_in_air

    return reward
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run.py -v`
Expected: 16 passed.

- [ ] **Step 5: Commit**



```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_run.py
git commit -m "feat(run): feet_air_time_capped, removing the two-foot flight double payment"
```

---

### Task 3: `action_magnitude_monitor`

Watches for the action blow-up that destroyed the previous campaign, without altering the reward.

**Files:**
- Modify: `src/mjlab_microduck/tasks/mdp.py` (after `feet_air_time_capped`)
- Test: `tests/test_run.py` (extend)

**Interfaces:**
- Consumes: nothing from earlier tasks (it needs its own fake, defined below, because it reads `env.action_manager` rather than a sensor).
- Produces: `action_magnitude_monitor(env) -> torch.Tensor` returning exactly zeros, shape `(num_envs,)`. Logs `Metrics/action_abs_max` and `Metrics/action_abs_p99`. Used by Task 4, which **must** register it with a non-zero weight.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run.py`:

```python
from mjlab_microduck.tasks.mdp import action_magnitude_monitor


class _ActionManager:
    def __init__(self, actions):
        self.action = torch.tensor(actions, dtype=torch.float32)


class _ActionEnv:
    def __init__(self, actions, with_manager=True):
        self.num_envs = len(actions)
        self.device = "cpu"
        self.extras = {"log": {}}
        if with_manager:
            self.action_manager = _ActionManager(actions)


def test_monitor_contributes_exactly_zero_reward():
    env = _ActionEnv([[0.5, -3.0, 1e9]])
    out = action_magnitude_monitor(env)
    assert out.shape == (1,)
    assert float(out[0]) == 0.0


def test_monitor_reports_max_magnitude():
    env = _ActionEnv([[0.5, -3.0, 2.0]])
    action_magnitude_monitor(env)
    assert abs(float(env.extras["log"]["Metrics/action_abs_max"]) - 3.0) < 1e-6


def test_monitor_logs_both_keys():
    env = _ActionEnv([[0.5, -3.0, 2.0]])
    action_magnitude_monitor(env)
    assert "Metrics/action_abs_max" in env.extras["log"]
    assert "Metrics/action_abs_p99" in env.extras["log"]


def test_monitor_survives_blowup_values():
    # The failure mode being watched for: |a| ~ 1e10.
    env = _ActionEnv([[1e10, -1e10]])
    out = action_magnitude_monitor(env)
    assert float(out[0]) == 0.0
    assert torch.isfinite(env.extras["log"]["Metrics/action_abs_max"])


def test_monitor_without_action_manager_returns_zeros():
    env = _ActionEnv([[0.5, 0.5]], with_manager=False)
    out = action_magnitude_monitor(env)
    assert float(out[0]) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run.py -v`
Expected: FAIL — `ImportError: cannot import name 'action_magnitude_monitor'`

- [ ] **Step 3: Write the implementation**

Append to `src/mjlab_microduck/tasks/mdp.py`:

```python
def action_magnitude_monitor(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Log action magnitude. Contributes exactly zero reward.

    Watches for the failure that killed the previous sprung campaign: a
    converged policy emitting |a| ~ 1e8-1e10, which drove action_rate_l2 to
    ~-1e25 and corrupted the value function. If these traces climb off their
    baseline, the fix is to swap in a tanh-squashed distribution via
    ``RslRlModelCfg.distribution_cfg["class_name"]``.

    Returns zeros, so the reward total is unaffected at any weight. **Register
    it with a non-zero weight anyway**: ``RewardManager.compute``
    (mjlab/managers/reward_manager.py:122) short-circuits before calling the
    term function when ``weight == 0.0``, which would silently disable this.

    Returns:
        A zeros tensor (num_envs,).
    """
    zeros = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "action_manager"):
        return zeros

    actions = env.action_manager.action
    if actions is None or actions.numel() == 0:
        return zeros

    abs_actions = torch.nan_to_num(
        actions.abs().float(), nan=0.0, posinf=0.0, neginf=0.0
    )

    log = env.extras.get("log") if hasattr(env, "extras") else None
    if log is not None:
        log["Metrics/action_abs_max"] = abs_actions.max()
        log["Metrics/action_abs_p99"] = torch.quantile(abs_actions.flatten(), 0.99)

    return zeros
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run.py -v`
Expected: 21 passed.

- [ ] **Step 5: Commit**



```bash
git add src/mjlab_microduck/tasks/mdp.py tests/test_run.py
git commit -m "feat(run): action_magnitude_monitor for the action blow-up failure mode"
```

---

### Task 4: `make_run_variant` transform

**Files:**
- Create: `src/mjlab_microduck/tasks/run.py`
- Test: `tests/test_run_cfg.py` (create)

**Interfaces:**
- Consumes: `alternating_flight` (Task 1), `feet_air_time_capped` (Task 2), `action_magnitude_monitor` (Task 3).
- Produces: `make_run_variant(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg`, plus module constants `STD_RUNNING`, `RUNNING_THRESHOLD`, `VELOCITY_STAGES`, `AIR_TIME_WINDOW`, `SENSOR_NAME`. Used by Task 5.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_cfg.py`:

```python
"""Config-level assertions for the Run variant transform."""

import pytest

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.run import (
    AIR_TIME_WINDOW,
    RUNNING_THRESHOLD,
    STD_RUNNING,
    VELOCITY_STAGES,
    make_run_variant,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


@pytest.fixture
def run_cfg():
    return make_run_variant(make_microduck_velocity_env_cfg())


def test_running_regime_is_reachable(run_cfg):
    # variable_posture gates on |lin| + |ang| and defaults running_threshold to
    # 1.5, which the stock command ranges can only hit with both maxed. The Run
    # task must set it below what the curriculum actually reaches.
    pose = run_cfg.rewards["pose"]
    assert pose.params["running_threshold"] == RUNNING_THRESHOLD
    max_reachable = VELOCITY_STAGES[-1]["lin_vel_range"]
    assert RUNNING_THRESHOLD < max_reachable


def test_std_running_is_not_aliased_to_std_walking(run_cfg):
    pose = run_cfg.rewards["pose"]
    assert pose.params["std_running"] is not pose.params["std_walking"]
    assert pose.params["std_running"] != pose.params["std_walking"]


def test_hip_roll_tolerance_unchanged_in_running(run_cfg):
    # Loosening hip_roll is what produced leg splay; it must stay tight.
    pose = run_cfg.rewards["pose"]
    assert STD_RUNNING[r".*hip_roll.*"] == pose.params["std_walking"][r".*hip_roll.*"]


def test_air_time_uses_capped_function(run_cfg):
    air = run_cfg.rewards["air_time"]
    assert air.func is microduck_mdp.feet_air_time_capped
    assert air.params["threshold_min"] == AIR_TIME_WINDOW[0]
    assert air.params["threshold_max"] == AIR_TIME_WINDOW[1]


def test_alternating_flight_registered(run_cfg):
    term = run_cfg.rewards["alternating_flight"]
    assert term.func is microduck_mdp.alternating_flight
    assert term.weight > 0.0
    assert term.params["command_name"] == "twist"


def test_action_monitor_weight_is_non_zero(run_cfg):
    # RewardManager.compute skips terms with weight == 0.0 before calling the
    # function, which would silently disable the monitor.
    term = run_cfg.rewards["action_magnitude_monitor"]
    assert term.func is microduck_mdp.action_magnitude_monitor
    assert term.weight != 0.0


def test_velocity_stages_are_monotonic(run_cfg):
    stages = run_cfg.curriculum["velocity_command_ranges"].params["velocity_stages"]
    steps = [s["step"] for s in stages]
    lins = [s["lin_vel_range"] for s in stages]
    assert steps == sorted(steps)
    assert lins == sorted(lins)
    assert len(stages) > 1


def test_angular_range_held_constant(run_cfg):
    # Forward speed must be the only moving variable in the curriculum.
    stages = run_cfg.curriculum["velocity_command_ranges"].params["velocity_stages"]
    angs = {s["ang_vel_range"] for s in stages}
    assert len(angs) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_cfg.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mjlab_microduck.tasks.run'`

- [ ] **Step 3: Write the implementation**

Create `src/mjlab_microduck/tasks/run.py`:

```python
"""Run task variant — push the velocity task toward an ALTERNATING running gait.

``make_run_variant(cfg)`` turns any microduck velocity-family env cfg into its
running counterpart, in the same shape as ``tasks/backlash.py``. Kept as a
transform rather than a new env cfg so it composes: the sprung phase becomes
``make_sprung_variant(make_run_variant(cfg))`` instead of a fourth copy of the
velocity env — the duplication that stranded the previous campaign.

Four changes:

1. Activate the posture running regime. ``variable_posture`` gates on
   ``|lin| + |ang|`` with ``running_threshold`` defaulting to 1.5, which the
   stock command ranges only reach with both maxed — so the regime is dead code
   today, and ``std_running`` is aliased to ``std_walking`` anyway.
2. Swap ``air_time`` to ``feet_air_time_capped`` and shorten its window. The
   stock reward pays double for simultaneous two-foot flight, which rewards the
   bouncing gait; the stock window (0.10-0.25 s) was tuned to slow the gait.
3. Add ``alternating_flight``, which rewards flight only when the feet are
   genuinely alternating.
4. Add ``action_magnitude_monitor`` (zero contribution, non-zero weight).

The speed curriculum ramps ``lin_vel_range`` only; ``ang_vel_range`` is held so
forward speed is the single moving variable.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp

SENSOR_NAME = "feet_ground_contact"

# Posture tolerances for the running regime. Looser than walking on the joints
# that must swing, but hip_roll is deliberately UNCHANGED — loosening roll is
# what produced leg splay (see the tuning notes in microduck_velocity_env_cfg.py
# lines 168 and 177).
STD_RUNNING = {
    r".*hip_yaw.*": 0.5,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.8,
    r".*knee.*": 0.8,
    r".*ankle.*": 0.5,
}

# Total commanded speed (|lin| + |ang|) above which the running posture regime
# engages. Provisional — revisit once the plateau is measured.
RUNNING_THRESHOLD = 0.6

# Swing-time window. Stock is (0.10, 0.25), explicitly raised to slow the gait
# down; running needs faster strides.
AIR_TIME_WINDOW = (0.05, 0.15)

# Steps are env steps (iteration * num_steps_per_env=24).
VELOCITY_STAGES = [
    {"step": 0,         "lin_vel_range": 0.5, "ang_vel_range": 1.0},
    {"step": 1000 * 24, "lin_vel_range": 0.7, "ang_vel_range": 1.0},
    {"step": 2000 * 24, "lin_vel_range": 0.9, "ang_vel_range": 1.0},
    {"step": 3000 * 24, "lin_vel_range": 1.2, "ang_vel_range": 1.0},
    {"step": 4000 * 24, "lin_vel_range": 1.5, "ang_vel_range": 1.0},
]

ALTERNATING_FLIGHT_WEIGHT = 3.0


def make_run_variant(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Convert a microduck velocity-family env cfg into the Run task."""
    # 1. Posture: activate the running regime with its own tolerances.
    pose = cfg.rewards["pose"]
    pose.params["std_running"] = dict(STD_RUNNING)
    pose.params["running_threshold"] = RUNNING_THRESHOLD

    # 2. Air time: stop paying double for simultaneous two-foot flight, and
    #    shorten the swing window. Params are unchanged — the capped function is
    #    deliberately signature-compatible with the stock one.
    air = cfg.rewards["air_time"]
    air.func = microduck_mdp.feet_air_time_capped
    air.params["sensor_name"] = SENSOR_NAME
    air.params["threshold_min"] = AIR_TIME_WINDOW[0]
    air.params["threshold_max"] = AIR_TIME_WINDOW[1]

    # 3. Reward genuinely alternating flight.
    cfg.rewards["alternating_flight"] = RewardTermCfg(
        func=microduck_mdp.alternating_flight,
        weight=ALTERNATING_FLIGHT_WEIGHT,
        params={
            "sensor_name": SENSOR_NAME,
            "command_name": "twist",
            "command_threshold": 0.01,
        },
    )

    # 4. Action-magnitude watchdog. Returns zeros, so the weight only has to be
    #    non-zero for RewardManager.compute to call it at all.
    cfg.rewards["action_magnitude_monitor"] = RewardTermCfg(
        func=microduck_mdp.action_magnitude_monitor,
        weight=1.0,
        params={},
    )

    # 5. Speed curriculum.
    cfg.curriculum["velocity_command_ranges"].params["velocity_stages"] = [
        dict(stage) for stage in VELOCITY_STAGES
    ]

    return cfg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run_cfg.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**



```bash
git add src/mjlab_microduck/tasks/run.py tests/test_run_cfg.py
git commit -m "feat(run): make_run_variant transform for the rigid running baseline"
```

---

### Task 5: Register the Run tasks

**Files:**
- Modify: `src/mjlab_microduck/tasks/run.py` (add `MicroduckRunRlCfg`)
- Modify: `src/mjlab_microduck/tasks/__init__.py` (import + register, after the Velocity registrations ~line 105)
- Test: `tests/test_run_cfg.py` (extend)

**Interfaces:**
- Consumes: `make_run_variant`, `VELOCITY_STAGES` (Task 4); `MicroduckRlCfg` and `make_microduck_velocity_env_cfg` from `microduck_velocity_env_cfg`; `MicroduckOnPolicyRunner` from `tasks/__init__.py`.
- Produces: task IDs `Mjlab-Run-Flat-MicroDuck` and `Mjlab-Run-Rough-MicroDuck`; `MicroduckRunRlCfg`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_run_cfg.py`:

```python
def test_run_rl_cfg_has_its_own_experiment_name():
    # Baseline and sprung runs must not share a wandb grouping.
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
    from mjlab_microduck.tasks.run import MicroduckRunRlCfg

    assert MicroduckRunRlCfg.experiment_name != MicroduckRlCfg.experiment_name
    assert MicroduckRunRlCfg.run_name != MicroduckRlCfg.run_name


def test_run_rl_cfg_keeps_the_plain_gaussian_policy():
    # Phase 1 deliberately does NOT change the distribution; the baseline stays
    # as close to the working velocity config as possible.
    from mjlab_microduck.tasks.run import MicroduckRunRlCfg

    assert (
        MicroduckRunRlCfg.actor.distribution_cfg["class_name"]
        == "GaussianDistribution"
    )
    assert MicroduckRunRlCfg.actor.obs_normalization is True
    assert MicroduckRunRlCfg.critic.obs_normalization is True


def test_run_tasks_are_registered():
    import mjlab_microduck.tasks  # noqa: F401  (import registers)
    from mjlab.tasks.registry import list_tasks

    tasks = list_tasks()
    assert "Mjlab-Run-Flat-MicroDuck" in tasks
    assert "Mjlab-Run-Rough-MicroDuck" in tasks


def test_run_task_rl_cfg_round_trips_through_the_registry():
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_rl_cfg

    assert load_rl_cfg("Mjlab-Run-Flat-MicroDuck").experiment_name == "run"
```

`mjlab.tasks.registry` exposes `list_tasks()` (returns a `list[str]` of task
IDs), `load_env_cfg`, `load_rl_cfg` and `load_runner_cls` — verified against the
installed 1.3.0. There is no `get_task_spec`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_run_cfg.py -v`
Expected: FAIL — `ImportError: cannot import name 'MicroduckRunRlCfg'`

- [ ] **Step 3: Write the implementation**

Append to `src/mjlab_microduck/tasks/run.py`:

```python
from dataclasses import replace

from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg

# Same hyperparameters as the velocity task — Phase 1 changes the task, not the
# learner. Only the logging identity differs, so the baseline and the later
# sprung runs land in separate wandb groups.
MicroduckRunRlCfg = replace(
    MicroduckRlCfg,
    experiment_name="run",
    run_name="run",
)
```

Verified against the installed 1.3.0: `RslRlOnPolicyRunnerCfg` is a dataclass,
`replace` succeeds, and `MicroduckRlCfg` keeps its own `experiment_name`.

**Caveat:** `replace` is shallow — `MicroduckRunRlCfg.actor` is the *same
object* as `MicroduckRlCfg.actor`. That is fine as long as nothing mutates it.
If a later phase needs to change the Run task's actor (for example swapping in a
squashed distribution), deep-copy the nested cfg first rather than assigning
into the shared one, or the velocity task silently changes too.

Then in `src/mjlab_microduck/tasks/__init__.py`, add the import alongside the other task imports:

```python
from .run import make_run_variant, MicroduckRunRlCfg
```

and register after the Velocity block:

```python
# Run task — rigid running baseline (Phase 1 of the sprung-leg campaign).
# Control for the later sprung comparison; see
# docs/superpowers/specs/2026-08-17-sprung-running-design.md
register_mjlab_task(
    task_id="Mjlab-Run-Flat-MicroDuck",
    env_cfg=make_run_variant(make_microduck_velocity_env_cfg()),
    play_env_cfg=make_run_variant(make_microduck_velocity_env_cfg(play=True)),
    rl_cfg=MicroduckRunRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Run task registered: Mjlab-Run-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-Run-Rough-MicroDuck",
    env_cfg=make_run_variant(make_microduck_velocity_env_cfg(rough=True)),
    play_env_cfg=make_run_variant(
        make_microduck_velocity_env_cfg(play=True, rough=True)
    ),
    rl_cfg=MicroduckRunRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Run task registered: Mjlab-Run-Rough-MicroDuck")
```

> **Import-cycle warning:** `tasks/run.py` imports from
> `tasks/microduck_velocity_env_cfg`, and `tasks/__init__.py` imports both. If
> importing `mjlab_microduck.tasks` raises a circular-import error, move the
> `MicroduckRlCfg` import inside `run.py` to module scope *below* the
> `make_run_variant` definition, or import it lazily inside a factory function.
> `backlash.py` has the same shape and works, so this should be fine.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_run_cfg.py -v`
Expected: 12 passed.

Then confirm registration end to end:

Run: `uv run python -c "import mjlab_microduck.tasks"`
Expected: the output includes `✓ Run task registered: Mjlab-Run-Flat-MicroDuck`.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest tests/ -v`
Expected: all pre-existing tests still pass. If a `backlash` or `velocity` test breaks, `make_run_variant` has mutated shared config state — the velocity factory returns shared mutable refs (see the `deepcopy` note in CLAUDE.md's Conventions). Fix by deep-copying in `make_run_variant` rather than by editing the failing test.

- [ ] **Step 6: Commit**



```bash
git add src/mjlab_microduck/tasks/run.py src/mjlab_microduck/tasks/__init__.py tests/test_run_cfg.py
git commit -m "feat(run): register Mjlab-Run-Flat/Rough-MicroDuck"
```

---

## Handoff to the remote box

After Task 5, Phase 1 is code-complete but unmeasured. Training runs remotely — do **not** start a campaign locally.

Command for Steve to run on the GPU machine:

```bash
uv run train Mjlab-Run-Flat-MicroDuck --env.scene.num-envs 4096
```

Watch, per the spec's success criteria:

- `Metrics/flight_asymmetry` — must rise and stay high. Near 0 means bouncing.
- `Metrics/flight_fraction` — flight is happening at all.
- `Metrics/action_abs_max` — must stay flat. Climbing means the blow-up is recurring; the response is a tanh-squashed distribution via `distribution_cfg["class_name"]`, not a reward tweak.
- Tracking error against `lin_vel_range` across curriculum stages — where it stops improving is the plateau, and the deliverable.

A short smoke run (a few iterations, 64 envs) is worth doing first to confirm the terms evaluate without shape errors.

## Deliberately not in this plan

- The spring mechanism (Phase 2 — its own brainstorm, once the plateau is known).
- Any tanh-squashed / Beta distribution (escape hatch only, triggered by the monitor).
- Tuning `RUNNING_THRESHOLD`, `ALTERNATING_FLIGHT_WEIGHT`, or the curriculum step spacing. These are provisional; tuning them before the first run would be guessing.
