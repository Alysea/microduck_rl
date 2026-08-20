# Spec — Phase 2: sprung-foot stiffness study

Branch: `spring_v2`.
Follows `2026-08-17-sprung-running-design.md` and its results in
`2026-08-18-run-baseline-results.md`.

## Goal

Find the **useful (stiffness, travel) window** for a compliant foot on the
MicroDuck, and hand the hardware phase a target spec instead of a guess.

Phase 2 was originally "design the spring mechanism". It is now a simulation
study, because the mechanism cannot be chosen well before the stiffness is
known — a stiff, short-travel result and a soft, long-travel result imply
different mechanisms.

## Framed decisions

1. **Sim first.** The sweep's output is a `(k, travel)` target that constrains
   mechanism design.
2. **Simulatability is a first-class design criterion, not a convenience.**
   The whole programme is sim-driven RL, so a mechanism that MuJoCo cannot model
   exactly is a mechanism whose experiment cannot be trusted. This is why the
   design target is *rigid links plus a single prismatic spring* and NOT a
   Kangoo-style flexure: a rigid 1-DoF translating mechanism maps one-to-one
   onto a MuJoCo `slide` joint with `stiffness`/`damping`, while a leaf spring
   requires either a discretised multi-body chain (expensive, and stiff enough
   to force a smaller timestep) or deformables (worse). The kinematics then
   carry **zero sim-to-real gap**.
3. **The spring goes in the foot, as a swappable accessory**, sharing the
   interface the roller accessory already uses (a child body of `ankle_left` /
   `ankle_right`, replacing the `sole_*` mesh).
4. **The mechanism choice is deferred**, and deferring costs nothing — see
   "Both candidates are sim-equivalent" below.
5. **`action_rate_l2` stays at -1.0.** The diagnostic sweep showed the rigid
   plateau is actuation-bandwidth-limited; that is precisely the limitation
   passive compliance is meant to relieve, so it is the budget the experiment is
   run inside, not a variable.

## Kinematic analysis (recorded input to the mechanism phase)

### Why the first prototype had several stable positions

The prototype is a single closed loop of 6 links joined by 6 revolute joints.
Grübler for a planar mechanism:

```
DoF = 3(L - 1) - 2J = 3(6-1) - 2(6) = 3
```

Three DoF, not one. Beyond the intended vertical compression the pad may also
shear fore-aft and rotate; one spring cannot constrain a 3-D configuration
space, so the assembly settles into whichever local energy minimum it finds.
Near the flattened pose the links approach collinearity, which is a toggle
singularity. **This is kinematic — no spring rate or preload fixes it.**

A 1-DoF planar mechanism requires `J = (3L-4)/2`: `L=4 -> J=4` (four-bar),
`L=6 -> J=7`, `L=8 -> J=10`. A 6-link loop with only 6 joints is always 3-DoF.

### Why a single four-bar is NOT the fix

A parallelogram four-bar is 1-DoF and its pad does not tilt, but the pad
travels on an **arc**, so vertical travel is bought with fore-aft slide. At
L = 40 mm links and 15 mm of travel:

| rest angle | end angle | fore-aft shift | shift per mm vertical |
|---|---|---|---|
| 70 deg | 34.4 deg | 19.3 mm | 1.29 |
| 55 deg | 26.4 deg | 12.9 mm | 0.86 |
| 40 deg | 15.5 deg | 7.9 mm | 0.53 |
| 30 deg | 7.2 deg | 5.0 mm | 0.34 |

The ratio only becomes acceptable as the stroke ends within ~7 deg of
collinearity — back inside the bistable region that broke the prototype.
**There is no safe operating point.** A single four-bar is rejected.

(An earlier draft of this analysis quoted ~2.9 mm of shift. That was computed
with the links starting horizontal, i.e. *at* the singularity, and was wrong.)

### The two live candidates

- **Prismatic slide + catalogue coil spring.** 1 DoF by construction, exactly
  vertical, no path coupling, and the spring rate is certified so sim `k` equals
  hardware `k`. Real cost is **stiction** — a sliding interface dissipates the
  energy a running spring must return. Mitigable with bushings, not removable.
  Needs precision, which CNC aluminium provides.
- **Sarrus linkage.** Six links, six revolutes in two perpendicular planes:
  exact pure translation, 1 DoF, no sliding friction, compact. It is
  overconstrained (Grübler says 0 DoF; the special geometry yields 1), hence
  tolerance-sensitive — it jams if the planes are not square. Not viable in FDM;
  viable in CNC aluminium.

Rejected, with reasons: single four-bar (no safe operating point);
synchronised double-parallelogram (1-DoF but 10 pin joints of mass and
backlash in a foot); leaf/oval flexure (not exactly simulatable — decision 2 —
plus its stiffness must be measured rather than specified and cannot be
retuned); accepting a diagonal path (trades coupling for foot scrub on the
return stroke, since the pad moves forward while the body also moves forward;
testable as elevated `foot_slip`).

### Both candidates are sim-equivalent

A prismatic slide and a Sarrus linkage both deliver **pure 1-DoF translation**,
so both reduce to the *same* MuJoCo model: one `slide` joint with `stiffness`
and `damping`. The sweep is therefore valid for either, and the mechanism can
be chosen on manufacturing grounds later without invalidating any result. That
is the concrete payoff of decision 2, and the reason deferring costs nothing.

## Load and stiffness analysis

Robot mass is **0.737 kg** (summed body masses in `robot_walk.xml`); standing
trunk height 0.125 m. Static load is ~3.6 N per foot on two feet, ~7.2 N in
single support, and perhaps 18 N at a 2.5x body-weight running peak.

Three independent arguments all point at **stiff**:

1. **Asymmetric failure.** Too stiff degrades gracefully toward the rigid
   baseline — nothing is lost. Too soft sinks, drags stance, compromises
   push-off and bottoms out. Prefer the harmless failure.
2. **Stance-time matching.** A SLIP spring's stance is `t ~ pi*sqrt(m/k)`. The
   measured baseline gait (`flight_fraction` 0.221 -> per-leg duty ~0.39, with
   `air_time_mean` 0.0795 s -> step period ~0.130 s) has a stance of **~51 ms**,
   which corresponds to k ~ 2800 N/m. Softer springs lengthen stance — a
   legitimate gait change that also demands less actuation bandwidth — but are a
   larger departure from what already works.
3. **The CoM reward sets a floor.** `com_height_target` holds the trunk in
   0.11-0.14 m and is currently satisfied ~97% of the time. Static sag eats into
   that: below roughly **240 N/m** the robot sags out of band while merely
   standing. This retroactively explains the abandoned branch, whose 500 N/m was
   close enough to that floor to be fighting the CoM reward *and* bottoming out
   at the same time.

**Travel is 15 mm, not 25.** 25 mm is 20% of standing height inside a foot
region only ~26 mm tall — the mechanism would nearly fully collapse. 15 mm is
self-consistent with the stiffness answer, since less travel forces a stiffer
spring: keeping peak deflection under 12 mm at an 18 N peak needs
**k >= 1500 N/m**, which is the same figure argument 2 reached independently.

## Height offset is a first-class parameter

**Any mechanism adds height under the foot**, and that is not cosmetic:

- **It shifts the CoM out of the reward band.** At a nominal +25 mm the trunk
  stands at 0.150 m against a 0.11-0.14 m band, so the sprung robot would be
  penalised for being tall before compliance is in play. The band must shift by
  *exactly* the geometric offset, preserving width — an equivalent
  normalisation, not a reward change.
- **It changes leg length**, so `k~ = k*L0/mg` shifts (at L0 = 0.150 m,
  k = 1500 N/m gives k~ = 31) and Froude numbers for a given speed change.
- **It raises the CoM**, lengthening moment arms and reducing stability. This is
  a genuine physical consequence of the design, so it belongs in the model.
- **It adds distal mass** — the worst place to add it, since it is paid through
  the whole swing. The mechanism's own mass must be modelled, not idealised
  away.

Nominal for the study: **`H_add` = 25 mm, mechanism mass 20 g per foot**, both
stated as assumptions to be revised once a mechanism exists. Also requires the
foot contact site and the per-foot terrain-height ray sensor to move down to the
new pad, or `foot_clearance` / `foot_swing_height` will measure the wrong body.

## The sweep

**Five arms, 8000 iterations each**, matching the diagnostic-sweep protocol and
compared over the 7000-8000 iteration window. Travel fixed at 15 mm, damping
fixed low (0.5 N.s/m, representing a good steel spring — hardware hysteresis
will be worse and is a hardware-phase concern, not a sweep axis).

| Arm | k (N/m) | static sag (2-foot) | peak @ 18 N | purpose |
|---|---|---|---|---|
| **locked** | rigid (travel 0) | 0 | 0 | **geometric control** |
| soft | 800 | 4.5 mm | 22.5 mm — bottoms out | deliberate failure case |
| mid | 1500 | 2.4 mm | 12.0 mm | plausible |
| stiff | 2200 | 1.6 mm | 8.2 mm | plausible |
| very stiff | 3000 | 1.2 mm | 6.0 mm | approaches rigid |

### The locked arm is the real control

The 0.468 m/s baseline was measured on a robot **without** the added height and
mass. Comparing a sprung foot against it changes two things at once —
compliance *and* geometry — which is exactly the attribution failure this
campaign was restructured to avoid.

The locked arm carries identical geometry, identical mass and a spring locked to
zero travel. **The hypothesis is tested as sprung vs locked.** The 0.468 m/s
figure remains a useful reference for how much the geometry alone costs, but it
is not the control.

The soft arm is included on purpose: it should reproduce the abandoned branch's
bottoming-out failure, validating the compression monitor and establishing the
travel floor empirically rather than from the arithmetic above.

## Implementation

Follows paths already proven in this repo:

1. **`robot_walk_sprung_foot.xml`** — derived from the *current* `robot_walk.xml`
   (never cherry-picked from the abandoned branch, whose parent model no longer
   exists). Adds one `slide` joint per foot between the `ankle_*` body and a new
   pad body, named **`passive_left_foot_spring` / `passive_right_foot_spring`**
   so every existing `^(?!passive_).*` regex excludes them for free. Moves the
   foot site and the terrain-height ray frame to the pad.
2. **`tasks/sprung.py`** — `make_sprung_variant(cfg, robot_cfg, stiffness,
   travel)`, shaped like `tasks/backlash.py`: swap the robot cfg, scope
   `dof_pos_limits` and the `pose` reward's `asset_cfg` off the new joints, and
   shift the `com_height_target` band by `H_add`.
3. **`spring_compression_monitor`** in `tasks/mdp.py` — returns a zeros tensor
   and logs `Metrics/spring_compression_mean`, `_max`, and
   `Metrics/spring_bottomed_fraction`. Registered with a **non-zero** weight:
   `RewardManager.compute` short-circuits before calling terms whose weight is
   0.0, which would silently disable it.
4. **Registration** — one task id per sweep arm
   (`Mjlab-Run-Flat-Sprung-K800-MicroDuck`, ...). MuJoCo joint stiffness lives
   in the model rather than the env cfg, so it is not CLI-overridable the way
   the reward params were; a spec-edit parameter on the factory plus one id per
   arm keeps it declarative and needs no new plumbing. Explicitly throwaway
   scaffolding for the sweep.
5. **Tests** — `tests/test_sprung.py` (monitor logic on synthetic tensors) and
   `tests/test_sprung_cfg.py` (joints named `passive_*`; stiffness and travel
   reach the model; the CoM band is shifted by exactly `H_add`; the monitor's
   weight is non-zero; all arm ids registered and constructing).

## Success criteria

1. **The compression monitor shows the spring is actually working** — non-zero
   mean compression, and `bottomed_fraction` near zero for the arms that matter.
   A sprung model that never deflects, or that rides its hard stop, is measuring
   nothing. This is the first thing to check, before any speed number.
2. **A stiffness that beats the locked arm on `forward_speed_mean`**, with
   `flight_asymmetry` holding near 0.70 — faster *and* still alternating, not
   faster by degenerating into a bounce.
3. **A `(k, travel)` target for the hardware**: the best k, and the travel it
   actually consumed.

If no arm beats the locked control, that is a real answer: passive foot
compliance does not help this robot within this smoothness budget, and the
mechanism phase should not be entered.

## Out of scope

- Choosing the mechanism (its own phase, opened with the `k` target in hand and
  the kinematic analysis above as recorded input).
- Path-coupled mechanisms. If a coupled mechanism is later chosen, the sim needs
  that coupling added before any comparison is trustworthy — which is itself an
  argument for preferring an uncoupled candidate.
- Spring hysteresis and stiction. Fixed low here; a hardware-phase concern.
- Damping as a sweep axis.
- Any change to `action_rate_l2` (decision 5).
