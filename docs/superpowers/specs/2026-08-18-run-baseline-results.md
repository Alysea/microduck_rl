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

**2. The Phase 3 confound is real and must be controlled.**
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
