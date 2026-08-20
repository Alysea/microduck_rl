# Phase 1 results — rigid running baseline

Run: `pollen-robotics/mjlab_microduck/jhxknvhw` (`2026-08-17_16-26-04_run`)
Task: `Mjlab-Run-Flat-MicroDuck`, 4096 envs, 50 000 iterations, completed.
Spec: `2026-08-17-sprung-running-design.md`

**Verdict: all three success criteria met. The baseline is established.**

## The number Phase 3 is measured against

**`Metrics/forward_speed_mean` = 0.468 m/s**, plateaued.

It is flat to four decimal places over the last third of the run — 0.4674,
0.4677, 0.4676, 0.4678 across the final four bands (iters 33 439 → 49 986).
Growth had already decayed to noise by ~iteration 33 000; the remaining
17 000 iterations bought nothing.

**Compare Phase 3 against this raw metric, not a derived one.** It is a mean
over *all* envs including the ~25 % commanded to stand
(`Curriculum/standing_envs` = 0.25), so it understates the speed of a moving
robot — but Phase 3 runs the same config with the same standing fraction, so
the raw numbers are directly comparable. Deriving a "true" moving-env speed
(≈ 0.62 m/s, dividing by 0.75) is an *inference* from the aggregates, not a
measurement, and should not become the comparator.

Supporting: `forward_speed_max` 1.525 m/s (top command is reached by some
envs), `error_vel_xy` 0.457 — tracking error rose monotonically with each
curriculum stage (0.335 → 0.354 → 0.368 → 0.443 → 0.486), which is the
signature of a robot running out of capability rather than out of training.

## Criterion 1 — alternating flight, not bouncing: MET

| Metric | Final |
|---|---|
| `Metrics/flight_asymmetry` | **0.698** |
| `Metrics/flight_fraction` | 0.221 |

Asymmetry sat in a 0.694–0.700 band in *every* time slice of the run.
Symmetric bouncing drives this toward 0; 0.70 with 22 % of steps in flight is
a genuinely alternating gait. The gait signature to reproduce in Phase 3 is
this pair — 22 % flight at ~0.70 asymmetry.

## Criterion 2 — measured plateau: MET

See above. The curriculum completed (`velocity_command_ranges` = 1.5) and the
speed asymptoted well inside the run.

## Criterion 3 — action stability: MET, with a caveat worth keeping

Steady state `action_abs_max` ≈ 3.8–4.6, `action_abs_p99` ≈ 1.83, `mean_std`
0.215 — flat across all 50 000 iterations. No drift.

**But the watchdog did fire.** 27 of 2000 sampled points exceed 10, eleven
exceed 20, three exceed 50, and one reached **119.0** at iteration 22 535 —
mostly a single burst spanning iters 21 215–22 857.

These were benign and did not cascade: through the 119.0 spike, reward held
215–223, `action_rate_l2` held ≈ −2.3, and `mean_std` held 0.222. Nothing
propagated. This is 6–7 orders of magnitude below the 1e8–1e10 excursions
that destroyed the previous campaign, and the value function never noticed.

Keep watching it in Phase 3 — springs add a passive DoF and could plausibly
excite this — but no action is warranted now, and **no switch to a
tanh-squashed distribution is justified by this run.**

## Two deferred decisions, now resolved by data

**1. Reward weights: leave them alone.** The realized
`air_time : alternating_flight` ratio is **6.19 : 1**, far off the 1.67 : 1
the weights (5.0 / 3.0) imply — because `air_time` pays on most steps while
`alternating_flight` pays only during flight (22 %) and is then scaled by
asymmetry. The final review worried this left bouncing too attractive.
**It did not matter**: asymmetry reached 0.70 regardless. The concern was
real arithmetic but the wrong conclusion; raising
`ALTERNATING_FLIGHT_WEIGHT` would now be a change with no evidence behind it.

**2. The Phase 3 confound is RETIRED — see the diagnostic sweep below.**
The concern as originally written follows, but a four-arm sweep (2026-08-19)
showed the foot-height and CoM terms are not limiting the gait, and the foot
terms are if anything *generous*. Superseded; kept for the record.

**2 (original, superseded). The Phase 3 confound is real and must be controlled.**
`Metrics/peak_height_mean` = **0.0205 m** — pinned to the `foot_swing_height`
/ `foot_clearance` target of 0.02 m, with both penalties driven to near zero
(−0.011, −0.014). The policy is not choosing a 2 cm foot lift; the shaping
terms are holding it there. `com_height_target` is meanwhile the third-largest
positive term at **+1.174**, constraining the CoM band to 0.11–0.14 m.

Springs change exactly these quantities. **Phase 3 must hold all three terms
fixed and report their episode sums beside the plateau**, or "springs helped"
cannot be separated from "springs relieved a shaping penalty". Whether to
widen them for *both* conditions before Phase 3 remains open — doing so
invalidates this baseline and requires a rerun.

## Reference — final values

```
flight_asymmetry      0.699      forward_speed_mean    0.469
flight_fraction       0.221      forward_speed_max     1.527
action_abs_max        4.36       error_vel_xy          0.457
action_abs_p99        1.83       fell_over             0.121
peak_height_mean      0.0205     air_time_mean         0.0806
mean_reward         220.07       track_linear_velocity 3.099
com_height_target     1.174      action_rate_l2       -2.360
air_time              2.810      alternating_flight    0.454
```


---

# Diagnostic sweep (2026-08-19/20)

Four 8 000-iteration arms, each a single-variable CLI override on the Run task,
compared against the control's matched 7 000–8 000 iteration window. The control
needed no rerun — the 50 k baseline supplies that window directly.

| Arm | Change | `forward_speed_mean` | Verdict |
|---|---|---|---|
| control | — | 0.4301 | — |
| A | foot target 0.02 → 0.04 | 0.4272 (−0.7%) | negative; flawed instrument |
| **B** | **`action_rate_l2` −1.0 → −0.5** | **0.4772 (+11.0%)** | **the sole limiter** |
| C | CoM band 0.11–0.14 → 0.09–0.17 | 0.4322 (+0.5%) | negative; band was slack |
| A′ | foot weights → ~0 | 0.4284 (−0.4%) | negative; terms *raise* lift |

## The plateau is actuation-bandwidth-limited

Arm B is the whole story. At 8 000 iterations it reached **0.4772 m/s — already
above the control's fully converged 50 k value of 0.4691** — at one sixth the
compute, and it had *not* plateaued: speed and flight fraction were still rising
monotonically at the cutoff (0.4473 → 0.4776, flight 0.2005 → 0.2426).

Cost: action acceleration +29.8%, `action_abs_max` +21.7%, policy std +18.4%.
`action_rate_l2` is sim2real protection, so this is a real trade, not free speed.

**This sharpens Phase 3 rather than invalidating the baseline.** If the rigid
robot is limited by how fast it can actively change actions, that is exactly the
limitation passive compliance removes — a spring stores and returns energy with
no action-rate cost at all. So keep the penalty at −1.0 and treat it as a fixed
smoothness budget. Phase 3's prediction becomes concrete and falsifiable:

> **The sprung robot beats 0.468 m/s at `action_rate_l2` = −1.0.**

## Why the other three arms came back negative

**Foot height is not a constraint — it is a subsidy.** Arm A′ relaxed both foot
penalties to ~1/200 of their weight (verified: costs fell to −0.0001 and
−0.0000, i.e. the constraint was genuinely gone) and foot lift *dropped*, from
2.02 cm to 1.87 cm, with speed flat. Those terms hold the foot **higher** than
the policy would choose. ~2 cm is this robot's natural swing height, not a
ceiling imposed by shaping. Corroborated by arm B, where the policy had 30% more
action acceleration available and still lifted only 1.92 cm: extra speed comes
from cadence and flight, never from bigger steps.

**The CoM band was slack.** `com_height_target` returns ~1.0 in-range at weight
1.2; the control's 1.1655 means the CoM was already inside 0.11–0.14 m about 97%
of the time. Roughly doubling the band moved the reward only +0.9% (to ~98%
in-range). Nothing was being suppressed.

**Arm A's instrument was wrong, and this is worth remembering.**
`feet_clearance` is `Σ |foot_height − target| × ‖foot_vel_xy‖` — the height error
is *multiplied by foot speed*. Raising the target while the robot cannot comply
makes the cheapest remedy **slowing the feet down**, not lifting them higher.
That is exactly what happened: −14% flight fraction, −4% peak height, and the
`foot_clearance` cost rose 170% as it ate the unavoidable error. To test whether
a constraint binds, **relax its weight — never move its setpoint.**

One unexplained detail, flagged rather than papered over: in arm A,
`foot_swing_height`'s episode cost *improved* (−0.0103 → −0.0087) where its
`(peak/target − 1)²` form predicts a ~2700× larger error. The class does reset
its peak tracker on landing, so that is not the cause. It does not affect any
verdict here (speed and peak height are measured independently) but it is worth
a look if foot shaping is ever revisited.

## What still needs watching in Phase 3

The three terms are slack or generous **for the rigid robot**. That is not the
same as neutral for the sprung one:

- `com_height_target` — a spring's function is to oscillate the CoM. The moment
  that oscillation leaves the 0.11–0.14 m band it starts paying, and only the
  sprung condition pays. Report its episode sum.
- foot terms — they currently subsidise ~2 cm of lift. If springs change the
  natural foot trajectory, that subsidy lands differently. Report both sums.

Monitor them; do not pre-emptively widen them. Widening changes the control for
no measured benefit, and the sweep found no evidence any of them binds today.
