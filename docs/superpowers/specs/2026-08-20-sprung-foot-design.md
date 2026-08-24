# Spec — Phase 2: sprung-foot design-space search

Branch: `spring_v2`.
Follows `2026-08-17-sprung-running-design.md` and its results in
`2026-08-18-run-baseline-results.md`.

## Goal

Find the **constraints on (mass, stiffness, travel)** that produce good
compliant-foot behaviour on the MicroDuck, and hand the hardware phase a
target spec instead of a guess.

Phase 2 was originally "design the spring mechanism". It is now a simulation
study, because the mechanism cannot be chosen well before the design point is
known — a stiff, short-travel result and a soft, long-travel result imply
different mechanisms, and likewise a boot that must stay under some mass
budget rules out mechanisms a heavier boot could use.

This is a **design-space search**, not a validation of the built prototype.
The Sarrus boot that has been measured (70 g, 12 mm travel, k≈3920 N/m) is an
early proof-of-principle and one datapoint on that space — a nominal point to
calibrate the sim against, not a fixed property to defend. The question this
phase answers is what the *next* boot should be built to, and analysis below
(see "Design-space structure") shows the three axes are not equally free:
travel is effectively fixed by geometry that already exists, stiffness tracks
mass almost for free once travel is fixed, and mass is the axis actually worth
spending a sweep on. That is why the study is staged (mass, then stiffness,
then — likely unnecessary — travel) rather than run as a single sweep or a
full factorial.

## Framed decisions

1. **Sim first.** The sweep's output is a `(mass, k, travel)` target that
   constrains mechanism design.
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

## Design-space structure: why mass is the binding axis

Before spending sweeps, it is worth asking algebraically which of the three
axes (mass, stiffness, travel) actually constrains the design and which are
along for the ride.

**Stance-matching pins stiffness to mass and stance time.** Treating the
stance phase as a quarter period of a mass-spring oscillator,
`t ~= pi*sqrt(m/k)`, so a spring tuned to match a given stance time `t` must
satisfy

```
k ~= pi^2 * m / t^2
```

**Avoiding bottom-out sets a minimum travel.** The peak running load is taken
as `2.5 * m * g` (a 2.5x body-weight landing peak), so the deflection at that
peak must not exceed the available travel:

```
travel >= 2.5 * m * g / k
```

Substituting the stance-matched `k` from above, **mass cancels**:

```
travel_min ~= 2.5 * m * g / (pi^2 * m / t^2) = (2.5 * g / pi^2) * t^2 ~= 2.485 * t^2
```

`travel_min` depends only on the stance time being matched, not on mass. At
the measured 12 mm of travel, solving for `t` gives `t ~= sqrt(0.012 / 2.485)
~= 0.0695 s`, i.e. **12 mm of travel supports stance-matched compliance up to
~69 ms of stance time.** The rigid gait's measured stance is ~51 ms, comfortably
under that ceiling. **Travel is therefore not the binding axis for this
robot's actual gait speed** — it would only bind for a much slower, longer-stance
gait than this robot runs (see the SLIP-band correction below).

**Stiffness and mass are nearly decoupled over the mass range this study
sweeps.** Moving the pad mass from 30 g to 90 g moves total robot mass from
0.797 kg to 0.917 kg — only ~15%. Since stance-matched `k` is linear in `m` at
fixed `t`, the stance-matched stiffness across that whole range shifts only
**3024 -> 3479 N/m** (`k = pi^2*m/t^2` at the rigid gait's measured t = 51 ms;
the prototype's own 0.877 kg sits at 3327 N/m, and its built k = 3900
corresponds to a slightly shorter 47 ms stance) — a small correction relative to
the built spring,
and small enough that a single stiffness (the measured prototype's) is a
reasonable common value to hold across a mass sweep.

**Conclusion: mass is the axis actually worth sweeping.** Travel is
oversized relative to what this robot's gait needs, and stiffness barely moves
across the mass range of interest. What is NOT yet known is how heavy a boot
can get before its swing-inertia penalty (see "It adds distal mass" below)
eats the compliance benefit — that is a question about mass, not about
stiffness or travel, hence Stage 1's mass-budget sweep.

## Load and stiffness analysis

**Updated for the measured Sarrus prototype.** The boot has now been built and
measured: **70 g per boot** (was assumed 20 g), **30 mm** of added height (was
assumed 25 mm), **12 mm** of travel (was assumed 15 mm). The spring itself was
measured at two points (3 mm -> 1500 g, 8 mm -> 3500 g) and fits to
**k ~= 3920 N/m with a 2.9 N force offset at zero deflection** — confirmed to be
*intentional preload* built into the Sarrus linkage's assembly geometry, not
stiction. Numbers below are recomputed from these measurements, not the
original assumptions.

Robot mass is **0.737 kg** rigid (summed body masses in `robot_walk.xml`), so
the sprung robot is **0.737 + 2 x 0.070 = 0.877 kg**. Weight is 8.60 N; static
load is 4.30 N per foot on two feet, 8.60 N in single support, and ~21.5 N at a
2.5x body-weight running peak.

Three independent arguments still point at **stiff**, now re-run against the
measured numbers:

1. **Asymmetric failure.** Unchanged: too stiff degrades gracefully toward the
   rigid baseline; too soft sinks, drags stance, and bottoms out. Prefer the
   harmless failure.
2. **Stance-time matching.** `t ~ pi*sqrt(m/k)` at m = 0.877 kg: **76 ms at
   k=1500, 59 ms at k=2500, 47 ms at k=3900, 40 ms at k=5500**. The rigid gait's
   measured stance is ~51 ms, so **k=3900 — the spring Steve actually built —
   sits closest**, which is a useful cross-check that the built spring landed
   near the value this analysis would have picked anyway.
3. **The CoM reward sets a floor.** Unchanged in kind: `com_height_target`
   holds the trunk in a band (now shifted by `H_ADD`, see below) and static sag
   below some stiffness will violate it while merely standing. Not recomputed
   here — the earlier ~240 N/m floor was for a 0.737 kg robot on two feet; at
   0.877 kg the floor is proportionally higher but still well under every arm in
   the new grid.

**Peak deflection at the 21.5 N landing peak** (ignoring the small preload
offset, i.e. `F/k`): **14.3 mm at k=1500 — exceeds the 12 mm of travel and
bottoms out**, 8.6 mm at k=2500, 5.5 mm at k=3900, 3.9 mm at k=5500. Including
the preload (the mechanism doesn't start compressing until the applied force
exceeds the preload's own holding force, `k * 0.00074 m`, so the usable
deflection is `F/k - 0.00074`) shifts these down slightly: 13.6 mm at k=1500
(still bottoms out), 7.9 mm at k=2500, **4.8 mm at k=3900**, 3.2 mm at k=5500.
k1500 is retained in the sweep specifically *because* it bottoms out — a
deliberate marker, replacing the role k=800 played in the old grid.

**A finding this measurement forces onto the table, corrected below: the
biological SLIP band is out of reach for a long-stance gait, but that is not
the same as saying it is out of reach for this robot.** The classic
dimensionless SLIP stiffness for running gaits is `k~ = k*L0/(m*g)` in the
range 10-30. At this mass (0.877 kg) and a leg length `L0` of 0.155 m, that
band is **k = 555-1665 N/m**, which corresponds to stance times of roughly
**72-125 ms** (`t ~= pi*sqrt(m/k)` over that k range). The softest end of that
band (k=555) needs `w/k = 8.60/555 ~= 15.5 mm` of deflection from static
single support *alone* — before any running load is added, more than the
12 mm of stroke available.

**Correction to an earlier draft of this spec:** that draft concluded from
this "the mechanism cannot operate in the biological SLIP band at all," full
stop. That is only true against the 72-125 ms stance the classic SLIP band
implies — a much slower gait than this robot actually runs (~51 ms measured
stance). The honest statement, per the design-space analysis above, is
narrower: **12 mm of travel rules out *long-stance* biological compliance, not
compliance at this robot's actual gait speed** — at ~51 ms stance, 12 mm of
travel is comfortably sufficient (the ceiling is ~69 ms). This boot is
committed to stiffer-than-biological-SLIP-band compliance only in the sense
that it cannot slow down to a bouncier, longer-stance gait and still fit its
travel; it is not committed to a harder stance than the gait it actually runs
calls for. If a bouncier, longer-stance gait were ever wanted, more travel —
not a softer spring within the current stroke — is the design lever.

The 2.9 N preload measured at k=3920 N/m is **0.74 mm** of precompression
(`2.9 / 3920`). It is modelled as the spring joint's `springref`, not as extra
stiffness or an added force term — see `robot/sprung_foot.py::SPRING_PRELOAD`.
Because it is parameterised as a displacement (fixed by the linkage geometry at
assembly) rather than a force, preload force scales with the swept stiffness:
**1.1 N at k=1500, 4.1 N at k=5500.** That is a physically faithful consequence
of the design, not a modelling artifact.

The prototype is explicitly a rough, unoptimised first build, so **70 g should
be read as a pessimistic mass figure** — a refined mechanism is expected to
weigh less, not more.

## Height offset is a first-class parameter

**Any mechanism adds height under the foot**, and that is not cosmetic. Note up
front that height is only one of **four** ways a sprung arm differs from the
rigid 0.468 m/s baseline — height, mass, compliance, and **foot contact
geometry** (see "The foot contact geometry also changes" below):

- **It shifts the CoM out of the reward band.** At the measured +30 mm the
  trunk stands 30 mm taller than rigid, so the sprung robot would be penalised
  for being tall before compliance is in play. The band must shift by
  *exactly* the geometric offset, preserving width — an equivalent
  normalisation, not a reward change.
- **It changes leg length**, so `k~ = k*L0/mg` shifts (at L0 = 0.155 m and the
  built spring, k = 3900 N/m gives k~ ~= 70 — well above the 10-30 SLIP band,
  consistent with the new finding above that this boot cannot run
  biologically) and Froude numbers for a given speed change.
- **It raises the CoM**, lengthening moment arms and reducing stability. This is
  a genuine physical consequence of the design, so it belongs in the model.
- **It adds distal mass** — the worst place to add it, since it is paid through
  the whole swing. The mechanism's own mass must be modelled, not idealised
  away. Measured, this is worse than assumed: **70 g per boot is a 19% increase
  in total robot mass (0.737 -> 0.877 kg), concentrated entirely in the foot**
  — the existing `ankle_left` body is ~30 g, so foot mass rises **~3.3x**
  (30 g -> 100 g). This creates real tension with this campaign's central
  finding: Phase 1 established the rigid robot's plateau is
  **actuation-bandwidth limited**, and passive compliance was attractive
  precisely because it adds flight energy without costing actuation budget.
  But 3.3x distal mass *increases* swing-torque demand against servos that are
  already the binding constraint — the mechanism could be fighting the same
  limitation it is meant to relieve. **The locked arm is what decomposes the
  two effects**: locked-vs-rigid-baseline measures the boot's cost (added
  height, mass, and contact geometry, with zero compliance), and
  sprung-vs-locked measures compliance's benefit in isolation. Only the second
  comparison speaks to whether the mechanism is worth its own weight.

Measured for the study: **`H_ADD` = 30 mm, mechanism mass 70 g per foot,
12 mm of travel** — no longer assumptions; see `robot/sprung_foot.py`. Also
requires the foot contact site and the per-foot terrain-height ray sensor to
move down to the new pad, or `foot_clearance` / `foot_swing_height` will
measure the wrong body.

## The foot contact geometry also changes

The pad that carries contact is a **40 x 28 x 8 mm box** (`rbound` 0.0247). The
sole it replaces was a **mesh**, oriented-bbox **54.0 x 41.1 x 12.9 mm**
(`rbound` 0.0355). So the sprung foot has:

- **~half the contact footprint** — 1120 mm^2 against 2219 mm^2, and
- **26% less fore-aft length** (40.0 mm against 54.0 mm), which is the axis that
  matters most for push-off and for pitch stability at speed, plus
- **box-vs-mesh contact**, which settles to a different penetration depth under
  the same load (this is why `ANKLE_TO_SOLE` is tuned to 0.0215 rather than the
  naive 0.025 mesh measurement).

**This is deliberate and is NOT to be "fixed" by widening the pad.** The pad is
a *hardware design parameter* handed forward to the mechanism phase: a 1-DoF
prismatic slide or Sarrus linkage has to fit inside the foot, and its contact
plate is the size it is. Changing it here to flatter the comparison would make
the sim model a mechanism nobody intends to build.

What it does mean is that the sprung arms differ from the rigid baseline in
**four** ways, not three: height, mass, compliance, and foot contact geometry.
Hence the next section.

## The sweep is staged, not a single grid

A single sweep or a full factorial over (mass, stiffness, travel) would be
wasteful: the design-space analysis above shows travel is not binding at this
robot's gait speed and stiffness barely moves across the mass range of
interest, so most cells of even a modest 4x4x3 grid (mass x stiffness x
travel) would be answering a question the algebra already answered for free.
Instead the study runs in **stages**, each one informed by the previous:

- **Stage 1 — mass budget (this change).** How heavy can the boot get before
  the swing-inertia penalty eats the compliance benefit? k held at the
  measured prototype spring (3900 N/m), travel held at the measured 12 mm,
  because both are near-decoupled from mass over this range (see above). Mass
  is the only variable.
- **Stage 2 — stiffness confirmation (deferred).** Once Stage 1 has picked a
  viable mass, confirm the stiffness choice around that mass by reusing the
  1500/2500/3900/5500 N/m grid this campaign already built (see below) —
  cheaper to re-run at the chosen mass than to have swept stiffness x mass
  jointly from the start.
- **Stage 3 — travel (likely unnecessary).** The `travel_min ~= 2.485*t^2`
  result above says 12 mm already covers this robot's gait with margin, so
  this stage is expected to confirm the measured travel rather than change it.
  Only worth running if Stage 1 or 2 turn up a surprise the algebra didn't
  predict.

**Common protocol across stages**: 8000 iterations per arm, matching the
diagnostic-sweep protocol, compared over the 7000-8000 iteration window.
Damping specified as a **ratio**, `c = 2*zeta*sqrt(k*pad_mass)` with
`zeta = 0.3`, rather than as an absolute rate.

**This replaces an earlier absolute 0.5 N.s/m ("a good steel spring, low
hysteresis"), which produced a pathological model.** That figure is right for
the *spring* but leaves the *pad-on-spring* subsystem essentially undamped:
`zeta` came out at 0.013-0.023, the pad resonated at 33-57 Hz against a 50 Hz
controller, and it retained 65-87% of its amplitude across a 51 ms stance, so
it rang on through 2-7 subsequent steps and never settled. The 30 g pad rang
*above* the control rate entirely, where the policy cannot observe it.

The first Stage 1 sweep is invalidated by this. Its sprung arms produced
0.054-0.085 m/s against locked arms at 0.179-0.386, and — the tell — sprung
speed *improved* monotonically with pad mass while the locked arms fell with
it. A mass penalty cannot produce that; pad resonance can, because a lighter
pad on a stiff spring rings faster. The sprung ranking was measuring resonance,
not compliance.

A ratio also holds resonance **constant across the mass sweep** instead of
letting it confound the axis, and is physically defensible: a larger mechanism
carries proportionally more joint friction. At `zeta = 0.3` the derived rate is
6.5-11.2 N.s/m across the 30-90 g range, and decay `tau` falls from 120-360 ms
to 9-16 ms — inside a single stance.

`zeta = 0.3` is provisional. The real figure is measurable on the prototype as
loading-versus-unloading hysteresis, and that measurement should replace it;
`make_sprung_foot_spec_fn` accepts an absolute `damping=` override for exactly
that purpose. The Stage 1 mass-penalty result from the LOCKED arms is unaffected
by any of this — those carry no spring DoF.

For that "idealised spring" claim to be true of what is actually built, the
spring joint's `frictionloss` and `armature` are **explicitly set to 0.0**
(`SPRING_FRICTIONLOSS` / `SPRING_ARMATURE` in `robot/sprung_foot.py`). They have
to be stated: the joint is created inside the `microduck` childclass, whose
`<joint frictionloss="0.1" armature="0.005"/>` default it would otherwise
inherit silently — a dry-friction term worth roughly a third of total
dissipation, plus proportionally-scaled effective inertia on the pad (invisible
in the total-mass check). Uniform across arms, so the *ranking* would survive,
but the absolute energy-return figure and the design point handed to the
hardware phase would both be wrong. Zero is not a claim about a real
mechanism: stiction and mechanism inertia are hardware-phase concerns this
spec defers. The one dissipative/reactive term that IS modelled now is the
measured **preload** (`springref`), because unlike stiction it is intentional,
not incidental.

### Stage 1 — mass budget

**Six arms**: two locked (no spring joint) at the mass extremes, and four
sprung arms at k=3900 N/m spanning the same mass range. `h_add` (30 mm) is
identical across every arm — same mechanism geometry, lighter materials — so
mass is isolated as the one variable.

| Arm | pad mass (per boot) | total robot mass | travel | purpose |
|---|---|---|---|---|
| m30_locked | 30 g | 0.797 kg | 0 (locked) | mass-penalty floor |
| m90_locked | 90 g | 0.917 kg | 0 (locked) | mass-penalty ceiling |
| m30_k3900 | 30 g | 0.797 kg | 12 mm | compliance at light mass |
| m50_k3900 | 50 g | 0.837 kg | 12 mm | interpolation point |
| m70_k3900 | 70 g | 0.877 kg | 12 mm | **the built prototype's mass** |
| m90_k3900 | 90 g | 0.917 kg | 12 mm | compliance at heavy mass |

The two locked arms are what turn this into a **budget** rather than a
ranking — see "The locked arm is the real control" below for how the pairing
decomposes mass penalty from compliance benefit.

### Stage 2 — stiffness confirmation (deferred, reuses this grid)

The grid below was built for the single-mass (877 g, the built prototype)
stiffness sweep this campaign ran before the mass axis was opened up. It is
kept here as the input Stage 2 will reuse once Stage 1 picks a viable mass,
rather than re-derived: the old grid (800/1500/2200/3000) is mostly invalid at
877 g total with only 12 mm of travel — 800 and 1500 both bottom out before
doing useful work. Static sag and peak deflection below both include the
preload (a spring does not start compressing until the applied force exceeds
its own preload-holding force, `k * 0.00074 m`):

| Arm | k (N/m) | static sag (2-foot) | peak @ 21.5 N | purpose |
|---|---|---|---|---|
| **locked** | inert (3900, travel 0) | 0 | 0 | **geometric control** |
| k1500 | 1500 | 2.1 mm | 13.6 mm — bottoms out | deliberate bottom-out marker |
| k2500 | 2500 | 1.0 mm | 7.9 mm | plausible |
| k3900 | 3900 | 0.4 mm | 4.8 mm | **the spring Steve actually built** |
| k5500 | 5500 | ~0.0 mm | 3.2 mm | approaches rigid |

Note how the preload changes the qualitative picture at the stiff end: at
k=5500 the preload-holding force (4.1 N) is nearly equal to the static
two-foot load per foot (4.3 N), so the mechanism is predicted to sit
essentially fully extended (q ~= 0) under ordinary standing load, not
partially compressed — see FIX 5's settling measurement for confirmation.

The k1500 arm was included on purpose (the role k800 played in the grid
before it): it should reproduce the abandoned branch's bottoming-out failure,
validating the compression monitor and establishing the travel floor
empirically rather than from the arithmetic above. When Stage 2 re-runs this
grid at the mass Stage 1 picks, the same bottom-out marker logic applies —
swap in whichever k, at that mass, is predicted to exceed the 12 mm stroke.

The locked arm's stiffness (3900) in this grid is set purely for tidiness,
matching the built spring — it has **zero effect**, since `travel=0.0` omits
the spring joint from the model entirely (see `make_sprung_foot_spec_fn` in
`robot/sprung_foot.py`). The same is true of both Stage 1 locked arms.

### The locked arm is the real control

The 0.468 m/s baseline was measured on a robot **without** the added height,
without the added mass, and on the **original mesh sole**. Comparing a sprung
foot against it changes **four** things at once:

1. **height** (+30 mm under the foot),
2. **mass** (+70 g per foot, distal — a 19% total-mass increase, ~3.3x on the
   foot alone),
3. **compliance** (the spring itself), and
4. **foot contact geometry** — a 40 x 28 x 8 mm box (`rbound` 0.0247) in place
   of the sole mesh (oriented-bbox 54.0 x 41.1 x 12.9 mm, `rbound` 0.0355):
   ~half the contact footprint (1120 vs 2219 mm^2), 26% shorter fore-aft, and
   box-vs-mesh contact.

That is exactly the attribution failure this campaign was restructured to avoid,
and (4) is the one an earlier draft of this spec missed. With the measured 70 g
pad, (2) is also a materially bigger confound than assumed — see the
actuation-bandwidth tension noted above.

**The locked arm controls for all four.** It carries the same +30 mm, the same
mass, and — crucially — **the same pad**, with no compliance. So sprung-vs-
locked isolates variable (3) alone, which is the hypothesis. **The hypothesis is
tested as sprung vs locked.** The 0.468 m/s figure remains a useful reference for
how much the geometry (1, 2 and 4 together) costs, but it is not the control —
and this four-way difference is precisely why.

### The two-locked-arm decomposition (Stage 1)

Stage 1 sweeps mass, which the single-locked-arm design above did not need to
handle: mass was fixed at the built prototype's 70 g, so one locked arm was
enough to control for it. Once mass is a swept axis, "the locked arm" has to
become **two** locked arms — one at each mass extreme (`m30_locked`,
`m90_locked`) — because a single locked arm can only control for *its own*
mass, not for mass in general.

With two locked arms, Stage 1's six-arm grid decomposes into two orthogonal
comparisons, each holding one variable fixed:

- **sprung vs locked at matched mass** (e.g. `m30_k3900` vs `m30_locked`, or
  `m90_k3900` vs `m90_locked`) — mass is identical on both sides, so the gap
  is **compliance's benefit at that mass**, exactly as sprung-vs-locked
  isolated compliance in the single-mass design.
- **locked vs locked across mass** (`m30_locked` vs `m90_locked`) — compliance
  is absent from both sides (travel=0 on both), so the gap is the **pure mass
  penalty**: added swing inertia and static load, with no compliance to offset
  it.

Plotting both comparisons against mass gives two curves — a "compliance
benefit" curve and a "mass penalty" curve — and **where they cross is the mass
constraint**: the heaviest boot for which compliance's benefit still exceeds
the mass penalty it drags in. That crossing point is the number this stage
exists to produce.

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
   and logs `Metrics/spring_compression_mean`, `_loaded_mean`, `_p95`, and
   `Metrics/spring_bottomed_fraction`, on **every** arm including `locked`
   (as explicit zeros), so an absent series means a bug rather than a normal
   arm. `_loaded_mean` (mean over samples with q > 1e-4 m) is the figure to
   compare against the static-sag table: the all-steps `_mean` is duty-diluted
   by flight, and reads ~1.9 mm at k=1500 against a 2.4 mm static sag. `_p95`
   replaces an earlier `_max`, which read ~`travel` as soon as any single
   sample of 4096 x 2 touched the stop and so carried no information.
   Registered with a **non-zero** weight:
   `RewardManager.compute` short-circuits before calling terms whose weight is
   0.0, which would silently disable it.
4. **Registration** — one task id per sweep arm
   (`Mjlab-Run-Flat-Sprung-M30-K3900-MicroDuck`, ...). MuJoCo joint stiffness
   and the pad's mass both live in the model rather than the env cfg, so
   neither is CLI-overridable the way the reward params were; a spec-edit
   parameter on the factory plus one id per arm keeps it declarative and needs
   no new plumbing. Explicitly throwaway scaffolding for the sweep — expect
   `SWEEP_ARMS` to be replaced wholesale when Stage 2 (and, if needed, Stage 3)
   run.
5. **Tests** — `tests/test_sprung.py` (monitor logic on synthetic tensors) and
   `tests/test_sprung_cfg.py` (joints named `passive_*`; stiffness and travel
   reach the model; the CoM band is shifted by exactly `H_add`; the monitor's
   weight is non-zero; all arm ids registered and constructing).

## Success criteria

1. **The compression monitor shows the spring is actually working** — non-zero
   `spring_compression_loaded_mean` in the neighbourhood of the static-sag
   table, and `bottomed_fraction` near zero for the arms that matter. Read
   `_loaded_mean`, not `_mean`: the latter is diluted by flight and will look
   low even on a healthy spring. A sprung model that never deflects, or that
   rides its hard stop, is measuring nothing. This is the first thing to check,
   before any speed number.
2. **A mass at which the sprung arm beats its matched-mass locked arm on
   `forward_speed_mean`**, with `flight_asymmetry` holding near 0.70 — faster
   *and* still alternating, not faster by degenerating into a bounce. Stage 1
   answers this per the two-locked-arm decomposition above: the mass budget is
   wherever the compliance-benefit curve stops beating the mass-penalty curve.
3. **A `(mass, k, travel)` target for the hardware**: the heaviest mass at
   which compliance still wins, the k that produces that win (Stage 2), and
   the travel it actually consumed (Stage 3, if run).

If no sprung arm beats its matched-mass locked arm at any mass in the grid,
that is a real answer: passive foot compliance does not help this robot within
this smoothness budget, and the mechanism phase should not be entered.

## Out of scope

- Choosing the mechanism (its own phase, opened with the `k` target in hand and
  the kinematic analysis above as recorded input).
- Path-coupled mechanisms. If a coupled mechanism is later chosen, the sim needs
  that coupling added before any comparison is trustworthy — which is itself an
  argument for preferring an uncoupled candidate.
- Spring hysteresis and stiction. Fixed low here; a hardware-phase concern.
- Damping as a sweep axis.
- Any change to `action_rate_l2` (decision 5).
