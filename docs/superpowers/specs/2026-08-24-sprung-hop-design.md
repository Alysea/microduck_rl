# Spec — Phase 4: commanded periodic hop on the sprung foot

Branch: `spring_v2`.
Follows `2026-08-20-sprung-foot-design.md` (Phase 2/3, the sprung-foot stiffness
and mass study) and `2026-08-18-run-baseline-results.md` (Phase 1, the rigid
running baseline).

## Goal

Test the spring hypothesis in the regime where a spring is theoretically
strongest, and where the running phases suggested it is weakest.

Phase 1 established that this robot is **actuation-bandwidth limited**: a
diagnostic sweep found `action_rate_l2` to be the sole limiter on forward speed,
worth +11% when relaxed. That is exactly the constraint a spring relieves — it
**decouples the timescale of energy input from output**, storing slowly where
motor torque is plentiful and releasing fast where back-EMF has taken the torque
away. That is the catapult mechanism (fleas, locusts, Salto).

Forward running is a poor test of it: the spring must absorb and return inside a
single ~51 ms stance, and the boot's distal mass is paid on **every swing cycle**.
A hop pays that mass once per cycle against a much larger energy return, so the
cost/benefit ratio is far more favourable.

## What the drop-rig probe already established

A throwaway drop-rig probe (base constrained to vertical travel; the robot
topples in ~1 s without a balance policy, so a free drop measures tipping rather
than rebound) compared identical locked and sprung robots:

| drop | locked rebound | k3900 rebound |
|---|---|---|
| 20 mm | 5.8 mm | 9.0 mm (+56%) |
| 50 mm | 5.3 mm | 17.7 mm (+234%) |
| 100 mm | 4.9 mm | 32.8 mm (+568%) |

The locked robot's rebound is **flat at ~5 mm regardless of drop height** — it
absorbs everything. The sprung robot's rebound **scales with drop height**, the
signature of an energy-storing element. So the spring returns usable energy in
this model, and the hop phase is worth running.

Three findings from that probe shape this spec:

1. **There is an optimum stiffness, and it is the softest spring that does not
   bottom out.** At a 100 mm drop, k2500 rebounds 35.3 mm, k3900 32.8 mm,
   k5500 28.3 mm, and k1500 only 20.6 mm because it bottoms and slams. That
   narrows the stiffness axis to roughly 2500-3900 N/m.
2. **Damping is what limits energy return, and it is a guess.** Measured
   restitution for k3900 is 0.57-0.70. Damped-oscillator theory predicts
   `exp(-pi*zeta/sqrt(1-zeta^2))` = 0.372 of stored energy retained per
   half-cycle at `zeta = 0.3`, i.e. restitution 0.61 — the measured 0.594 at the
   50 mm drop matches almost exactly. **`zeta = 0.3` discards ~63% of the energy
   the spring stores.** That value is not measured; it was chosen to suppress a
   pad resonance (see "The damping tension").
3. **The end-stop is not perfectly rigid.** Stiffened to the solver's stability
   floor (`solref_limit` timeconst 0.004 = 2*timestep, `solimp` dmax 0.99),
   worst-case penetration fell from 149% to 109% of travel. MuJoCo cannot make it
   rigid at dt = 0.002, so a high `spring_bottomed_fraction` should be read as
   "this arm bottoms out" rather than trusted to the millimetre.

## Framed decisions

1. **Periodic hop, not a single commanded jump.** The spring's energy comes
   overwhelmingly from **impact loading**. Quasi-statically the actuators can
   reach ~52.4 N per foot (knee-limited, moment arm 18.3 mm at the home pose,
   0.96 N.m joint limit) against the 49.7 N needed for full travel — just
   barely, and BAM back-EMF derates that further at launch speed. A 100 mm drop,
   by contrast, stores enough to rebound 33 mm. Periodic hopping lets each
   landing load the spring for the next launch; a single jump from standstill
   must do all the loading with a countermovement, which is the harder control
   problem. A single jump is the degenerate long-period case of this design, so
   nothing is excluded.
2. **Reuse the `ground_pick` phase-command pattern**: `[cos(2*pi*phi),
   sin(2*pi*phi), 0]` with a segmented phase profile (load / launch / flight /
   land). Giving the policy explicit timing is easier to learn than discovering a
   rhythm, and the pattern already exists in this repo.
3. **Three arms, not six.** `Locked`, `k2500`, `k3900`, all at the measured 70 g
   pad. The drop probe already narrowed stiffness; mass is held because Stage 1's
   locked arms measured the mass penalty separately.
4. **`action_rate_l2` stays at -1.0.** The fixed smoothness budget for the whole
   campaign, not a variable.
5. **Port the reward functions, not the env cfg.** `origin/jump` (April, four
   months behind `develop`) has `jump_phase_complete`, `jump_both_feet_airborne`,
   `jump_upward_velocity`, `jump_body_height` — self-contained tensor logic that
   should carry. Its 380-line env cfg will have rotted the same way the abandoned
   sprung env did; rebuild against current `develop`.

## The damping tension

This is the central unresolved parameter, and it deserves stating plainly because
it points in two directions at once:

- **Low damping** gives good energy return, but the pad-on-spring subsystem
  resonates. At an absolute 0.5 N.s/m, `zeta` was 0.013-0.023, the pad rang at
  33-57 Hz against a 50 Hz controller, and it retained 65-87% of its amplitude
  across a stance — so it never settled between steps. That artifact invalidated
  the first Stage 1 sweep, where sprung speed *improved* monotonically with pad
  mass (lighter pad = faster ringing = worse), the opposite of the locked arms.
- **High damping** stabilises the pad but turns the boot into a shock absorber.
  At `zeta = 0.3` the drop probe measured only 0.57-0.70 restitution.

Hopping is the task where this matters most, because **hopping IS the
energy-return test**. The real figure is measurable on the prototype as
loading-versus-unloading hysteresis: push the boot down through several
deflections recording force, then release through the same points; the area
between the curves is the dissipation. `make_sprung_foot_spec_fn` accepts an
absolute `damping=` override precisely so a measured value can replace the
estimate.

Until then `zeta = 0.3` stands, and **every hop result should be read as a lower
bound on what the mechanism could return.**

## Task design

**Command.** A phase signal `[cos(2*pi*phi), sin(2*pi*phi), 0]` over a segmented
profile, driven externally at inference (as `scripts/infer_policy.py` already
does for ground-pick). Segments: **load** (crouch, compress), **launch**
(extend, release), **flight**, **land** (absorb, return to a stable pose).

**Hop period** is a config parameter. It determines whether the spring is driven
near its natural frequency: at k = 3900 and 0.877 kg the spring-mass period is
~94 ms, so a hop period near that resonates and a much longer one does not. Start
at a single value; sweep only if the first result is ambiguous.

**Rewards.** The four ported jump terms, plus a landing-survival term. The
existing `fell_over` tilt termination (`mdp.bad_orientation`, 70 deg) handles the
failure case unchanged — it is orientation-based, not height-based, so the boot's
30 mm of added height does not perturb it.

**Metrics.** Peak CoM height per cycle is the headline. Alongside it, the spring
instruments already built — `spring_compression_loaded_mean`,
`spring_compression_p95`, `spring_bottomed_fraction` — plus a **new
energy-return metric**, since finding 2 makes restitution the crux rather than a
detail.

## Success criteria

1. **Sprung peak hop height exceeds Locked at matched mass.** This is the
   hypothesis. If it fails here — the regime most favourable to a spring — then
   passive compliance does not help this robot, and that is a real answer.
2. **The spring is demonstrably working**: non-zero
   `spring_compression_loaded_mean`, and `spring_bottomed_fraction` near zero on
   the arms that matter. Read this BEFORE any height number; a spring that never
   deflects or rides its stop is measuring nothing.
3. **Landings survive**: `fell_over` not materially worse than the Locked arm.

## Out of scope

- Skipping and bounding gaits. They re-introduce forward speed as a confound and
  pay the swing-mass penalty every cycle; worth doing only after a hop result.
- Sweeping the hop period, unless criterion 1 comes back ambiguous.
- Sweeping pad mass. Stage 1's locked arms already measured the mass penalty
  (-17.7% at 30 g, -61.9% at 90 g vs the rigid baseline).
- Changing `zeta` without a measurement. See "The damping tension".
- Any change to `action_rate_l2`, the pad geometry, or the CoM band shift.

## Known uncertainties

- **`zeta = 0.3` is unmeasured** and costs ~63% of stored energy. The single most
  decision-relevant measurement outstanding on the hardware.
- **The 52.4 N foot-force figure is computed at one pose.** The knee's moment arm
  changes through a crouch, so quasi-static loading capacity may differ
  materially in the poses a hop actually uses.
- **The end-stop retains ~9% overshoot** under the hardest impacts (finding 3).
  Arms that bottom out will have their rebound slightly overstated.
- **The hop period is a guess.** ~94 ms is the spring-mass period at k=3900, but
  the right hop period also depends on flight time and the load segment's
  duration, which the policy partly chooses.

## Amendment, 2026-08-24: the reward ceiling

Recorded after implementation, because it changes one item this spec listed as
out of scope. The "Out of scope" list above stands except where noted here.

A whole-plan review found the reward as first built **saturated at roughly
15-20 mm of height gain**, against the 5-33 mm span of the drop-rig evidence
this spec argues from. Worse, the per-cycle total was **non-monotone** — a local
peak at 15 mm and decreasing across 12-32 mm — so all three arms could have
converged to the same height and criterion 1 would have been untestable. Four
mechanisms caused it, and all four were changed:

1. The height reward is gated on both feet airborne, so it is evaluated only in
   flight with the legs unloaded, yet its datum was the **settled** standing
   height. Moved to the sag-free unloaded height (0.1171 rigid). This one was a
   measurement error, not a tuning choice.
2. `HOP_HEIGHT_GAIN` 0.015 -> 0.040 and `HOP_HEIGHT_STD` 0.008 -> 0.020, putting
   the whole 5-33 mm band on the Gaussian's rising limb. At k=3900 full travel
   stores 0.631 J across both feet; at `zeta = 0.3` a damped oscillator returns
   37%, lifting 0.877 kg by ~27 mm — so a 40 mm peak sits deliberately just
   above the sprung expectation.
3. `max_vel` 0.5 -> 1.0. At 0.5 m/s the upward-velocity term saturated at
   `v^2/(2g)` = 12.7 mm, below the entire discriminating band.
4. **This is the scope change:** `com_height_target`'s *upper* edge was raised
   from 0.14 to 0.20 in the hop variant only. Crossing the old top forfeited the
   flat `+1` (x1.2 weight) as a step, at 23 mm of gain — inside the band. The
   Phase-2 `h_add` **translation** that the out-of-scope list protects is
   untouched; only the rigid upper edge moves, and only for the hop task. It is
   safe because the maximum sag-free stance root height is 0.16133, already below
   the old sprung band top of 0.17, so that edge was unreachable while standing
   on every arm and only ever fired airborne.

Three findings from the same review were **declined** and remain out of scope by
decision, not oversight: a horizontal-velocity penalty (so a bounding gait stays
available, and is a differential confound the sprung arms benefit more from); a
segmented load/launch/flight/land phase profile (the policy gets a raw sine and
one half-cycle gate, and must discover the catapult unaided); and the 0.5 mm
`com_height_target` **floor** mismatch, which is long-standing repo behaviour
that every prior campaign result including the Phase 1 baseline was obtained
under — fixing it would break that comparability.

Two things to carry into reading the results:

- **The Gaussian is no longer the dominant term.** At 33 mm of gain it
  contributes 8.3 of a 44.8 per-cycle total, against 24.0 from the airborne term
  and 9.6 from the CoM band. The reward is effectively air-time-shaped with
  height as a secondary shaper. Defensible, since flight time is monotone in
  apex, but it means the 40 mm choice buys less than it appears to.
- **"Gain" is not pure spring rebound.** About 14.2 mm of sag-free posture
  headroom survives the datum fix (max stance root 0.16133 vs HOME_FRAME
  0.14710), so real headroom to the target is ~26 mm, not 40 mm. Identical on all
  three arms, so it biases nothing — but do not read the logged gain as rebound.
- The total now peaks near **65 mm**, where the retained `foot_swing_height`
  penalty (a bowl centred on 20 mm of foot peak, so quadratic in height above
  that) finally overtakes the air-time terms, which grow as `sqrt(h)`. That term
  is retained deliberately: arm-identical and under 1% of the per-cycle total
  inside the band.
