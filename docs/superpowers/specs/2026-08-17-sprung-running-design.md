# Spec — Sprung-leg running campaign (restart)

Branch: `spring_v2` (off `develop`).
Supersedes the abandoned `test_spring` / `tanh-squashed-gauss` branches.

## Goal

Test the hypothesis that **passive leg compliance is what makes a running gait
possible** on the MicroDuck.

That is a two-variable hypothesis (new mechanism *and* new behaviour), so the
campaign is sequenced to keep attribution possible: establish what the rigid
robot can do first, then change one thing.

## Framed decisions

Settled during design, not to be relitigated without a reason:

1. **Rigid baseline first.** The rigid robot's speed plateau is the number the
   sprung robot must beat. Without it, a successful sprung run tells us nothing
   about the springs.
2. **Running means alternating flight.** Not raw speed, not stride length, and
   explicitly not the symmetric two-foot bouncing gait the previous campaign
   produced and rejected. Flight phase must be bracketed by opposite-foot
   stances.
3. **The spring mechanism is a fresh design.** The old `robot_walk_sprung.xml`
   is not cherry-picked — see "Why not reuse the old branches".
4. **Plain Gaussian policy, instrumented.** The shipped `GaussianDistribution`
   stays. The known action-blowup failure is watched for, not pre-empted, so the
   baseline stays as close to the current working velocity config as possible.
5. **Step by step.** Each phase completes and is measured before the next is
   designed.

## Why not reuse the old branches

`test_spring` carried a working sprung model, but it is not portable:

- The sprung XML was a 50-line delta on the **then-current** `robot_walk.xml`.
  That file has since changed by 310 insertions / 236 deletions. The delta
  applies to a model that no longer exists.
- The old approach duplicated the whole velocity env into a 400-line
  `microduck_velocity_sprung_env_cfg.py`. Three such files existed
  (`velocity_sprung`, `hop_sprung`, `hop_forward_sprung`), plus five SLIP-biped
  envs. That duplication is why the work could not follow `develop`.
- The 800-line `policy/` package (`squashed_actor_critic` + forked
  `squashed_ppo` + forked `squashed_rollout_storage`) targets the rsl_rl 3.x
  API. rsl_rl is now 5.0.1 and the fork is unnecessary — see "Policy layer".

What *is* reused is the knowledge: spring scale (~k=500 N/m, c=2.0, 10 mm
travel, one passive prismatic DoF per leg) and the two documented failure modes,
recorded in `docs/action_obs_normalization_report.md` and
`docs/phase_b_touchpoint_map.md`.

## Stack facts verified for this design

Checked against the pinned versions, not against `CLAUDE.md` (which is stale on
several of these):

- `mjlab==1.3.0` from PyPI (no longer a git rev); `rsl-rl-lib==5.0.1`.
  **The local `.venv` is stale** (mjlab 0.1.0 / rsl_rl 3.3.0) — `uv sync` is
  Phase 0.
- Policy obs is **61-D**, not 51-D. `SYMMETRY_CFG` is hardcoded for the old
  51-D layout, so symmetry stays OFF (as in every v1.5+ env).
- `obs_normalization=True` on both actor and critic already
  (`microduck_velocity_env_cfg.py:948,958`). The old report's "Step 1" is done.
- `MicroduckRollersOnPolicyRunner` no longer exists; 1.3.0 fixed passive-joint
  metadata export upstream. Only `MicroduckOnPolicyRunner` remains.

## Phases

| Phase | Content | Deliverable |
|---|---|---|
| 0 | `uv sync`; confirm `Mjlab-Velocity-Flat-MicroDuck` still trains | working env |
| 1 | Rigid running baseline (this spec) | plateau speed + gait signature |
| 2 | Design the spring mechanism (own brainstorm) | mechanism decision |
| 3 | Sprung variant + comparison against Phase 1 | hypothesis answered |

Phases 2 and 3 are deliberately not specified here. Phase 2 is a real design
question (prismatic shank vs. series-elastic at the servo vs. compliant ankle
vs. parallel knee spring — different sim2real stories each) and it should be
answered with Phase 1's evidence in hand.

---

# Phase 1 — rigid running baseline

## Architecture

`tasks/run.py` exposing `make_run_variant(cfg)`, modelled directly on
`tasks/backlash.py`.

A **variant transform**, not a new env cfg file. This is the repo's newest idiom
and it composes: Phase 3 becomes

```python
make_sprung_variant(make_run_variant(make_microduck_velocity_env_cfg()))
```

rather than a fourth copy of the velocity env. It also applies to `velocity2` or
to the backlash models at no extra cost.

New registrations in `tasks/__init__.py`, existing style:

- `Mjlab-Run-Flat-MicroDuck`
- `Mjlab-Run-Rough-MicroDuck`

with `MicroduckRunRlCfg` — a copy of `MicroduckRlCfg` carrying its own
`experiment_name` / `run_name`, so baseline and sprung runs do not share a wandb
grouping.

## Rewards

### Traps verified in mjlab (must be handled explicitly)

**`feet_air_time` pays double for two-foot flight.** In
`mjlab/tasks/velocity/mdp/rewards.py:209`:

```python
in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
reward = torch.sum(in_range.float(), dim=1)   # summed over feet
```

Both feet airborne scores 2.0; alternating scores 1.0. The velocity env runs
this at `weight = 5.0` (`microduck_velocity_env_cfg.py:295`). Carried into a
speed curriculum unchanged, this **actively rewards the bouncing gait** — and it
pays more the faster the command, because higher speed means more air time.

This is a plausible co-cause of the previous campaign's 46 %-flight bouncing
attractor, which was attributed at the time to `clip_actions=π` alone. A
squashed Gaussian would not have fixed it.

**The running posture regime is currently dead code.** `variable_posture`
(`rewards.py:385`) gates on `total_speed = |lin| + |ang|` with
`running_threshold` defaulting to **1.5**. The velocity env never sets it, and
its command ranges max at `lin 0.5 + ang 1.0 = 1.5` — so the running branch only
engages when both are simultaneously maxed. With `std_running = std_walking`
(`microduck_velocity_env_cfg.py:253`) it makes no difference anyway.

### New reward — `alternating_flight`

Written in `tasks/mdp.py`. No new sensor plumbing: `feet_ground_contact` already
has `track_air_time=True` and `current_air_time` of shape `[n_envs, 2]`, left in
column 0 and right in column 1 (pattern at `microduck_velocity_env_cfg.py:190`).

```
flight    = (air_L > 0) & (air_R > 0)
asymmetry = |air_L − air_R| / (air_L + air_R + eps)
reward    = flight · asymmetry · speed_gate
```

Asymmetry alone separates the two gaits, with no phase state machine and no
history buffer:

- **Real running** — at any flight instant the trailing foot has just left the
  ground (small air time) and the leading foot is about to land (large air
  time). Asymmetry → 1.
- **Symmetric bouncing** — both feet leave and land together. Asymmetry → 0,
  reward → 0.

`speed_gate` follows the existing `command_threshold` convention (a binary gate
on commanded speed, as in `feet_air_time`) so the term is inert at zero command.

The term also logs `Metrics/flight_asymmetry` and `Metrics/flight_fraction` into
`env.extras["log"]` — success criterion 1 is stated in terms of the asymmetry
metric, so it has to be observable during training, not only inferable from the
reward total.

### Modified reward — `feet_air_time_capped`

Local copy of mjlab's `feet_air_time` with the per-foot sum clamped to 1.0.
Removes the double payment without forbidding flight. Replaces the stock
`air_time` term.

Its window also changes: **0.10–0.25 s → 0.05–0.15 s**. The current values carry
the comment "increased to slow down gait" (`microduck_velocity_env_cfg.py:297`)
— a walking tuning that caps stride frequency.

### Posture regime

- `running_threshold` set explicitly to **0.6**.
- `std_running` given real values, no longer aliased to `std_walking`:
  hip_pitch ≈ 0.8, knee ≈ 0.8, ankle ≈ 0.5, hip_yaw ≈ 0.5.
- **`hip_roll` stays at 0.05.** Loosening roll is what produced leg splay; see
  the tuning history at lines 168 and 177.

## Curriculum

Re-enable the stages commented out at `microduck_velocity_env_cfg.py:851`, as a
ramp on `lin_vel_range` only. `ang_vel_range` is held at 1.0 so forward speed is
the single moving variable.

| step | `lin_vel_range` | `ang_vel_range` |
|---|---|---|
| 0 | 0.5 | 1.0 |
| 1000 × 24 | 0.7 | 1.0 |
| 2000 × 24 | 0.9 | 1.0 |
| 3000 × 24 | 1.2 | 1.0 |
| 4000 × 24 | 1.5 | 1.0 |

Where this stops improving *is* the deliverable.

## Policy layer

No change: the shipped `GaussianDistribution` stays, `joint_pos_action.scale`
stays at 1.0, no `clip_actions`.

**Instrumentation instead.** A reward term `action_magnitude_monitor` that
writes `max|action|` and `p99|action|` into `env.extras["log"]` and **returns a
zeros tensor**, so its contribution to the reward is exactly 0 at any weight.
Registered with `weight = 1.0`.

The weight must be non-zero: `RewardManager.compute`
(`mjlab/managers/reward_manager.py:122`) short-circuits before calling the term
function when `term_cfg.weight == 0.0`, so a zero-weight monitor would never
execute. Returning zeros rather than leaning on a tiny weight keeps the
contribution exactly nil instead of merely small — which matters here, since the
thing being monitored is precisely the case where action values explode.

Logging via `env.extras["log"]` is the idiomatic hook; it is how
`feet_air_time` already reports `Metrics/air_time_mean`.

If that trace climbs off its baseline, the escape hatch is now cheap. rsl_rl
5.0.1 replaced the monolithic `ActorCritic` with a pluggable
`rsl_rl.modules.distribution.Distribution`, selected via
`RslRlModelCfg.distribution_cfg["class_name"]`, which accepts a fully-qualified
`"module:Class"` string. PPO calls `log_prob`, `entropy`, `params` and
`kl_divergence` **through** the distribution, so a tanh-squashed variant is one
~70-line file and a config string — no PPO fork, no rollout-storage fork, and
the two approximations the old design conceded (Normal KL, Normal entropy)
become genuinely overridable.

One gotcha if that hatch is ever used: `as_deterministic_output_module()` feeds
ONNX export, so a squashed distribution must return a module that *contains* the
tanh, or exported policies emit unsquashed means.

## Tests

Following the repo convention of `test_<name>.py` (reward logic on synthetic
tensors) plus `test_<name>_cfg.py` (config assertions). Written before the
implementation.

`tests/test_run.py`:
- symmetric bounce (equal air times, both airborne) scores ≈ 0
- alternating flight (one long, one short) scores high
- both feet planted scores 0
- `feet_air_time_capped` never exceeds 1.0, including both-feet-airborne
- both terms are inert at zero command
- `action_magnitude_monitor` returns exactly zeros and populates its two log keys

`tests/test_run_cfg.py`:
- `running_threshold` is set, and is below the curriculum's reachable maximum
- `std_running` differs from `std_walking`, and `hip_roll` is unchanged
- curriculum stages are monotonic in `lin_vel_range`
- `action_magnitude_monitor` has a **non-zero** weight (guards the
  `RewardManager` short-circuit — a well-meaning cleanup to `weight = 0.0`
  would silently disable the monitor)
- both task IDs are registered and construct

## Success criteria

1. A gait with sustained flight phase whose `asymmetry` metric stays high — i.e.
   alternating, not bouncing.
2. A measured speed plateau: the `lin_vel_range` beyond which tracking error
   stops improving.
3. `max|action|` stable across the whole run (no repeat of the 10⁸–10¹⁰ blowup).

Criterion 2 is the one Phase 3 is measured against.

## Plan B if the baseline plateaus early

If the rigid robot cannot reach flight at all, that is a *result*, not a
failure — it strengthens the spring hypothesis and Phase 2 proceeds with a
clearer target. In that case, record the plateau and do not tune further; tuning
the baseline to look good would destroy its value as a control.

## Known uncertainties

Stated rather than papered over:

- **`running_threshold` is a guess.** Set to `1.2` (was `0.6`). mjlab's
  `variable_posture` gates on the MIXED total `|lin| + |ang|`, and
  `ang_vel_range` is held at 1.0 through every stage, so any threshold at or
  below 1.0 was reachable by yaw alone — spin-in-place at zero linear velocity
  was being granted the loose hip_pitch/knee running tolerance. `1.2` sits above
  the max `|ang|`, so yaw can no longer trigger it by itself; on forward speed
  alone only the last two curriculum stages (1.2, 1.5) reach it. Revisit once
  the plateau is measured.
- **`alternating_flight` could be gamed by a limp** — one leg taking long
  flights, the other short hops, which also scores high asymmetry. If that
  appears, the fix is a symmetry term on stride length rather than air time.
  Not pre-built; YAGNI until observed.
- **Bouncing is discouraged but not excluded.** Capping `air_time` makes it
  gait-*neutral* rather than anti-bouncing: symmetric bouncing still collects the
  full `air_time` weight of 5.0 and forfeits only `alternating_flight`'s ≤3.0, so
  it retains roughly 62% of the flight-related reward and remains a plausible
  local optimum. Two further terms mildly favour it — `body_ang_vel` (−0.05) and
  `angular_momentum` (−0.02) penalise the trunk rotation intrinsic to an
  alternating gait while symmetric bouncing generates almost none. Magnitudes are
  small (~−0.004/step against `alternating_flight`'s ~+0.06/step) so they should
  not flip the balance. Deliberately not retuned before the first run; revisit
  with data.
- **Three untouched shaping terms are a Phase 3 confound.**
  `com_height_target` (+1.2, band 0.11–0.14 m, i.e. only ~3 cm of CoM travel),
  `foot_clearance` (−2.0, target 0.02 m) and `foot_swing_height` (−0.25,
  quadratic around 0.02 m) are all tuned for a 2 cm-lift walking gait and oppose
  the ballistic CoM arc and higher swing that flight requires. The plateau may be
  set by these rather than by rigid-leg dynamics. That is survivable for the
  baseline, but springs change exactly CoM oscillation and foot height, so these
  penalties are **not neutral between the rigid and sprung conditions** —
  "springs helped" could be "springs relieved a shaping penalty". Phase 3 must
  hold all three fixed and report their episode sums alongside the plateau.
  Widening them for both conditions is an open decision for Steve.

## Out of scope

- The spring mechanism itself (Phase 2).
- The SLIP-biped investigation from `test_spring` (5 envs + Raibert controller).
  Dropped unless a specific question needs it.
- Tanh-squashed / Beta policy distributions (escape hatch only).
- Symmetry augmentation — `SYMMETRY_CFG` needs a 61-D rewrite first, unrelated
  to this campaign.
- Sim2real transfer of the running gait. Flight phase on XL330 servos is its own
  question.
