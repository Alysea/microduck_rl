# Sim-to-Real Transfer for MicroDuck: A Complete Analysis

*MicroDuck is a miniature bipedal robot (346 g, ~12 cm tall) trained entirely in MuJoCo simulation using deep reinforcement learning. This document is an exhaustive account of every sim2real technique explored, every failure mode found, and the current understanding of the remaining gap.*

---

## 1. The Robot and the Challenge

MicroDuck has 14 actuated joints: 5 per leg (hip yaw, hip roll, hip pitch, knee, ankle) and 4 in the neck/head. All joints are driven by Dynamixel XL330 servo motors — tiny, cheap brushless motors with firmware PD position control running at 50 Hz. The robot weighs 347 g and stands roughly 12 cm tall.

The core sim2real challenge for small bipeds is harsher than for large ones:

- **Actuator nonlinearity dominates.** At small scales, motor friction and armature inertia are a larger fraction of the available torque budget. A 10% error in the torque constant has more effect on gait stability than it would on a 10 kg humanoid.
- **Body asymmetry matters more.** A lateral CoM offset of 2 mm on a 100 g leg segment represents a meaningful bias torque. The same offset on a Boston Dynamics leg is negligible.
- **Control latency is a larger fraction of the gait cycle.** At 50 Hz, 3 steps of delay = 60 ms, which is a significant fraction of a ~300 ms step period.

The goal: train a walking policy in MuJoCo that transfers to the real robot with `action_scale = 1.0` (no gain reduction required).

---

## 2. System Identification with BAM

### 2.1 What is BAM?

BAM (Better Actuator Model) is a motor identification framework that fits nonlinear friction and electrical models to testbench data. A pendulum arm is attached to the motor; position, velocity, and PWM voltage are logged while the motor tracks various reference trajectories. The model is then fitted by minimising the residual between simulated and measured position traces.

### 2.2 The M1 and M6 Motor Models

BAM provides several model families of increasing complexity:

**M1 (simple):** Voltage-controlled motor with Coulomb + viscous friction.

```
τ_out = (vin × duty × kt / R) - (kt² / R) × dq - friction_base × sign(dq) - friction_viscous × dq
```

Parameters: `kt`, `R`, `armature`, `friction_base`, `friction_viscous`

**M6 (load-dependent friction):** Adds Stribeck friction and friction terms proportional to motor torque and external (gravitational) load:

```
τ_friction = friction_base × sign(dq)
           + friction_stribeck × sign(dq) × f_stribeck(dq)
           + load_friction_motor × |τ_motor| × sign(dq)
           + load_friction_external × |τ_gravity| × sign(dq)
           + friction_viscous × dq
```

where `f_stribeck(dq) = exp(-(dq/dtheta_stribeck)²)` approximates the Stribeck effect (higher friction near zero velocity).

### 2.3 Data Collection and Contamination

The testbench records sessions with different arm masses (`m`) and spring constants (`k`). The original dataset had 53 recordings spanning three trajectory types:

| Trajectory | Description |
|---|---|
| `sin_time_square` | Sinusoidal position target, amplitude varies as t² |
| `sin_sin` | Double sinusoid — richer frequency content |
| `up_and_down` | Slow lift-and-lower, emphasises static friction |

**Contamination discovery:** Recordings with heavy arm mass (`m = 0.223 kg`) and stiff springs (`k ≥ 800`) hit motor torque saturation. The motor physically could not track the target — it was stuck at the PWM limit with near-zero velocity. These are not "hard points" (mechanical friction spikes) but **torque saturation** events: the motor runs out of voltage.

**Stall detection criterion:**

```python
stall = (|ω| < 0.05 rad/s)  &  (|goal - pos| > 0.2 rad)  &  (|V| > 3 V)
# A frame is "stalled" when: almost stopped, far from target, motor straining
```

**Effect on identification:** Saturated frames artificially inflate the estimated resistance `R` (the model explains "why didn't it move? must be high R") and the torque constant `kt` compensates downward to preserve the duty-cycle–torque ratio. Result: old M1 identified a motor that was **29% weaker** than reality.

**Filter applied:** Files with `stall_fraction ≥ 10%` were removed. This reduced the dataset from 53 → 22 clean recordings.

```
Before filter: 53 files  (17 sin_time_square + 22 sin_sin + 14 up_and_down)
After filter:  22 files  (17 sin_time_square +  3 sin_sin +  2 up_and_down)
Removed:       31 files  (all heavy-mass, high-spring recordings)
```

### 2.4 Identification Results: M1

| Parameter | Old M1 | New M1 | Change |
|---|---|---|---|
| `kt` (Nm/A) | 0.2007 | 0.1819 | -9% |
| `R` (Ω) | 2.867 | 2.009 | **-30%** |
| `armature` (kg·m²) | 0.00153 | 0.00207 | +35% |
| `friction_base` (Nm) | 0.0161 | **0.0317** | **+97%** |
| `friction_viscous` (Nm·s/rad) | 0.0182 | 0.0243 | +34% |
| **Stall torque** (7.4V) | 0.518 Nm | **0.670 Nm** | **+29%** |

The lower `R` is the key fix. With clean data, the model correctly attributes the missing torque to friction rather than to high winding resistance. The resulting motor is 29% stronger and has twice the Coulomb friction.

### 2.5 Identification Results: M6

M6 is fitted on the same clean data but with the richer load-dependent friction model.

| Parameter | Old M6 | New M6 | Change |
|---|---|---|---|
| `kt` | 0.325 | 0.247 | -24% |
| `R` | 2.649 | 2.437 | -8% |
| `armature` | 0.00196 | 0.00223 | +14% |
| `friction_base` | 0.00599 | 0.00781 | +30% |
| `friction_stribeck` | 0.00158 | **0.01299** | +721% |
| `load_friction_motor` | **0.4288** | 0.1768 | **-59%** |
| `load_friction_external` | 0.00214 | **0.33285** | **+15483%** |
| `dtheta_stribeck` (rad/s) | 2.579 | **0.108** | **-96%** |
| `friction_viscous` | 0.00814 | 0.01675 | +106% |
| **Stall torque** (7.4V) | 0.908 Nm | **0.750 Nm** | -17% |

**Three major changes explained:**

1. **`dtheta_stribeck`: 2.579 → 0.108 rad/s.** The old fit put the Stribeck transition at 2.58 rad/s, which means every normal motion was in the "Stribeck regime" — physically implausible. With clean data, the Stribeck effect correctly concentrates at near-zero velocities (< 0.1 rad/s).

2. **`load_friction_external`: 0.002 → 0.333.** This term captures friction proportional to the gravitational load on the arm. The old contaminated data gave no clean signal for this term (the motor was stalled, not moving under gravity), so it was effectively zero. Clean data at multiple arm masses reveals it is the dominant load-dependent friction mechanism.

3. **`load_friction_motor`: 0.429 → 0.177.** Old data over-estimated this because saturated frames (high motor torque, zero velocity) falsely correlated motor torque with friction. Clean data gives a more moderate estimate; much of what was attributed to motor torque load was actually from external load.

### 2.6 MuJoCo Export

MuJoCo's standard actuator model supports: `kp` (position gain), `damping` (velocity gain), `frictionloss` (Coulomb), `armature` (rotor inertia), and `forcerange` (torque clamp). This maps cleanly to M1. M6's load-dependent terms have **no direct MuJoCo equivalent**.

Export formula (given firmware `kp_fw=200`, `vin=7.4V`):

```
kp_mj        = error_gain × kp_fw × vin × kt / R
damping_mj   = friction_viscous + kt² / R
frictionloss = friction_base
forcerange   = ±(vin × kt / R)
```

where `error_gain = (4096 / 2π) / (256 × 885) = 0.002877` (XL330 encoder/PWM scaling).

| Export param | Old M6 | New M6 | New M1 |
|---|---|---|---|
| `kp` | 0.522 | **0.432** | 0.386 |
| `damping` | 0.0480 | 0.0418 | 0.0408 |
| `frictionloss` | 0.0060 | 0.0078 | **0.0317** |
| `armature` | 0.00196 | 0.00223 | 0.00207 |
| `forcerange` | ±0.908 Nm | ±0.750 Nm | ±0.670 Nm |

**The frictionloss discrepancy between M6_new (0.008) and M1_new (0.032)** is expected: M6 separates base Coulomb from Stribeck and load terms. M1 lumps everything into `friction_base`. The "effective" Coulomb near zero velocity for M6 is 0.008 + 0.013 (Stribeck) ≈ 0.021 Nm — closer to M1's 0.032.

The remaining gap is the unmodelled load-dependent friction. During walking, loaded joints carry significant reaction forces, making `load_friction_motor` and `load_friction_external` active. These terms can add 0.03–0.15 Nm of friction to loaded joints that simply doesn't exist in the MuJoCo sim.

---

## 3. Domain Randomization

Domain randomization (DR) is the primary strategy for making the policy robust to the sim2real gap. The following parameters are randomised at episode reset.

### 3.1 Physical Parameters

| Parameter | Range | Motivation |
|---|---|---|
| Centre of mass offset | ±3 mm (per axis) | Manufacturing asymmetry, CoM estimation error |
| Body mass | ×(0.95, 1.05) | Payload, wear, battery state |
| Body inertia | ×(0.95, 1.05) | Coupled with mass randomization |
| Motor Kp gain | ×(0.85, 1.15) | Firmware gain uncertainty, temperature drift |
| Motor Kd gain | ×(0.90, 1.10) | Damping estimation error |
| IMU mounting angle | ±1° | Sensor misalignment, mechanical flex |

Disabled (too destabilising at current training stage):
- Joint friction randomisation
- Joint damping randomisation
- Base orientation initialisation

### 3.2 Perturbations

**Velocity pushes:** Every 3–6 seconds, an instantaneous velocity impulse in the range ±0.3 m/s is applied to the base. This trains the policy to recover from pushes and prevents it from relying on a perfectly stable base assumption.

**Neck offset:** The neck/head target position is randomised every 2–5 s up to ±0.3 rad (introduced gradually via curriculum starting at iteration 12,000). This prevents the policy from learning a fixed head pose and trains robustness to head-induced inertial perturbations.

### 3.3 Observation Noise and Delays

Sensor noise is added to all observations before policy inference:

| Observation | Noise type | Range |
|---|---|---|
| Base angular velocity | Uniform | ±0.024 rad/s |
| Projected gravity | Uniform | ±0.007 |
| Joint position | Uniform | ±0.0006 rad |
| Joint velocity | Uniform | ±0.024 rad/s |

**Observation delays** simulate sensor latency and communication jitter. Each sensor modality is given an independent delay drawn uniformly from `[0, 3]` control steps (0–60 ms at 50 Hz), resampled every 64 steps:

```python
delay_min_lag=0, delay_max_lag=3, delay_update_period=64
```

This is particularly important for the IMU: the XL330 USB2Dynamixel bus scan and the IMU I²C read happen asynchronously, so the policy must be robust to misaligned IMU/joint data.

**Actuator delay:** Motor commands are also delayed by 0–3 steps, modelling the firmware processing latency and USB communication round-trip.

### 3.4 What DR Does NOT Cover (Known Gaps)

- **Load-dependent friction:** The MuJoCo friction model is velocity-dependent (Coulomb + viscous) but not load-dependent. Real XL330s show significant load-friction (M6 `load_friction_external = 0.333`). DR on `Kd` partially compensates but does not correctly model the mechanism.
- **Motor nonlinearity:** Back-EMF, PWM clipping, and the Stribeck effect are absent from the sim actuator.
- **Ground contact model:** Flat-floor sim vs real floor surface texture, compliance, and micro-geometry.
- **Leg asymmetry:** Left/right manufacturing differences, cable routing tension asymmetry.

---

## 4. RL Training Setup

### 4.1 Algorithm: PPO

Proximal Policy Optimisation with Generalised Advantage Estimation (GAE):

```
γ = 0.99     (discount factor)
λ = 0.95     (GAE smoothing)
ε = 0.2      (clip ratio)
K = 5        (epochs per update)
M = 4        (mini-batches)
α = 1e-3     (learning rate, adaptive schedule)
desired_KL = 0.01
```

Training runs 4096 parallel environments for 50,000 iterations, each collecting 24 steps before updating. That is ~5 billion environment steps total.

### 4.2 Network Architecture

Both actor and critic use MLP networks with ELU activations:

```
Actor:   [51] → [512] → [256] → [128] → [14]   (14 joint position targets)
Critic:  [54+] → [512] → [256] → [128] → [1]   (value function)
```

The critic receives additional **privileged observations** not available on the real robot: base linear velocity (3D) and foot heights. This is the standard asymmetric actor-critic setup for locomotion.

Policy output is a Gaussian distribution over joint position offsets from the default pose. At inference, the mean is taken (no sampling).

### 4.3 Observation Space (51-dimensional actor)

```
Dims  0:3   — base angular velocity (body frame)       3D
Dims  3:6   — projected gravity vector (body frame)    3D  [IMU-equivalent]
Dims  6:20  — joint positions relative to default     14D
Dims 20:34  — joint velocities                        14D
Dims 34:48  — last action                             14D
Dims 48:51  — velocity command [vx, vy, ω_z]          3D
```

**Note:** The projected gravity vector (`obs[3:6]`) is used instead of raw accelerometer readings. This is the gravity vector expressed in the body frame: `R_body_world.T @ [0, 0, -1]`. It encodes roll and pitch but not yaw, making it IMU-equivalent without needing a magnetometer.

Lateral lean can be read directly as:
```python
lateral_lean_deg = degrees(arcsin(clip(obs[4], -1, 1)))
```

### 4.4 Action Space

14-dimensional continuous joint position targets. The policy outputs **offsets from the default standing pose**:

```python
joint_target = DEFAULT_POSE + action * action_scale
```

`action_scale` is the critical sim2real tuning knob — it scales how aggressively the policy moves joints. Ideally 1.0; in practice it was ~0.65 with the old (contaminated) motor params.

### 4.5 Reward Function

The reward function balances task performance against stability and smoothness:

**Tracking rewards** (encourage velocity following):
```
track_linear_velocity:  w=3.0,  exp(−|v_cmd − v_base|² / 0.15)
track_angular_velocity: w=3.0,  exp(−|ω_cmd − ω_base|² / 0.40)
```

**Posture rewards** (encourage upright walking):
```
upright:       w=1.0   — penalise trunk tilt
pose:          w=2.0   — soft joint-angle targets (wide std during walking)
com_height:    w=1.2   — CoM height in [0.08, 0.11] m
```

**Gait rewards** (encourage good footfall patterns):
```
air_time:      w=5.0   — reward swing phases of 0.10–0.25 s
foot_clearance:        — reward lifting feet at least 2 cm
foot_slip:     w=−0.1  — penalise foot sliding at contact
soft_landing:  w=−1e-5 — penalise hard impacts
stillness_at_zero: w=3.0 — penalise motion when v_cmd=0
```

**Regularisation** (encourage smooth, efficient motion):
```
action_rate_l2:    w=−0.6 to −1.0  (curriculum-ramped)
joint_torques_l2:  w=−1e-3
body_ang_vel:      w=−0.05
angular_momentum:  w=−0.02
```

**`action_rate_l2`** is the most important regulariser for sim2real. It penalises `‖a_t - a_{t-1}‖²`, directly discouraging high-frequency joint oscillations. Its weight is ramped via curriculum so early training can explore freely before smoothness is enforced.

### 4.6 Curriculum Learning

Training progressively increases task difficulty across five axes simultaneously:

**Action rate penalty (iteration → weight):**
```
     0 → −0.4
  6000 → −0.8
 12000 → −1.0
```

**Standing environment fraction** (fraction of envs initialised standing still):
```
     0 →  2%
 12000 →  5%
 18000 → 10%
 24000 → 15%
 36000 → 20%
 48000 → 25%
```

**Velocity command range** (max commanded speed):
```
     0 → ±0.30 m/s lin,  ±1.5 rad/s ang
 12000 → ±0.35 m/s lin,  ±1.6 rad/s ang
 24000 → ±0.40 m/s lin,  ±1.7 rad/s ang
 36000 → ±0.50 m/s lin,  ±2.0 rad/s ang
```

**Neck perturbation amplitude:**
```
     0 → 0.0 rad
 12000 → 0.1 rad
 18000 → 0.2 rad
 24000 → 0.3 rad  (max)
```

### 4.7 Symmetry Augmentation

The robot is bilaterally symmetric. This is exploited with a **mirror loss**:

```
L_mirror = 0.5 × MSE(π(o), flip(π(flip(o))))
```

where `flip(o)` swaps left/right joint indices and negates the appropriate signs (yaw/roll axes reverse under left-right reflection). This enforces that the policy produces symmetric actions for mirrored observations, halving the effective sample complexity for symmetric gaits.

Data augmentation (storing mirrored trajectories in the replay buffer) is available but currently disabled in favour of the mirror loss.

---

## 5. Terrain and Environment Configuration

### 5.1 Flat Terrain

The primary training environment. A flat ground plane with foot contact friction coefficient 0.6. Most policies are trained flat-only to get clean gaits before introducing terrain.

### 5.2 Rough Terrain

A procedurally generated terrain with four zone types:

```
30% — Flat
20% — Pyramid stairs (step height 0–1.5 cm, step width 15 cm)
20% — Inverted pyramid stairs
30% — Random grid (grid pitch 12 cm, height 0–1.0 cm)
```

Maximum obstacle height is kept small (1–1.5 cm) relative to leg length, matching MicroDuck's capability. Full terrain curriculum (starting flat, adding rough) is available but not yet the primary training strategy.

---

## 6. Lateral Lean Investigation

**Observation:** The real robot consistently leans ~7–8° to the left when walking. Is this the policy's fault (a sim artefact) or a hardware asymmetry?

**Test:** Run the policy headless in simulation for 500 control steps and log the lateral component of the projected gravity vector:

```python
lateral_lean_deg = degrees(arcsin(clip(obs[4], -1, 1)))
# obs[4] = projected_gravity[1] = lateral lean indicator
```

**Result:**

| Environment | Mean lateral lean | Std |
|---|---|---|
| Simulation (headless, 500 steps) | ~0.0° | small |
| Real robot (walking) | **~7–8°** | moderate |

Simulation shows essentially zero lateral lean. The policy is not causing the lean — it is a **physical hardware asymmetry**: likely a combination of CoM offset (unbalanced leg/cable mass), left/right motor characteristic differences, and possibly IMU mounting tilt.

This directly motivates the `COM_RANDOMIZATION_RANGE` DR parameter. The current ±3 mm range may need to be widened, or an asymmetric CoM bias could be added during training to explicitly train robustness to the known offset direction.

---

## 7. MuJoCo System Identification (Walking Data)

### 7.1 Motivation

BAM identifies motor parameters from controlled testbench data. Can we identify sim parameters directly from walking robot data? MuJoCo's `mjx.sysid` framework (Gauss-Newton least-squares with FD Jacobians) was explored for this.

### 7.2 Full-Trajectory Rollout: Failure Mode

The standard sysid approach: given a sequence of states and controls, roll out the model, compute residuals at each timestep, differentiate with respect to parameters.

**Problem:** Biped walking trajectories are dynamically unstable. A small parameter mismatch (wrong `kp`, wrong `damping`) causes the trunk to fall differently, desynchronising ground contacts from the recorded data. By step 5–10, the simulated trajectory has diverged completely from the measured one, and the gradient is meaningless.

This is a fundamental obstacle: **sysid designed for stable, periodic systems (robot arms, pendulums) does not directly apply to underactuated biped walking.**

### 7.3 One-Step Sysid

**Approach:** Instead of rolling out multi-step trajectories, reset the simulation state to the measured state at every timestep `t`, step forward once, and compare the simulated `t+1` to the measured `t+1`.

```python
for t in range(T-1):
    set_state(data, measured_q[t], measured_dq[t], measured_quat[t])
    set_ctrl(data, ctrl[t])
    mj_step(model, data)
    residual[t] = data.qpos - measured_q[t+1]  # joint residual
```

The trunk quaternion is reconstructed from the projected gravity observation:

```python
def quat_from_proj_grav(pg):
    # pg = R.T @ [0,0,-1]: encodes roll and pitch, yaw=0
    gz = np.array([0, 0, -1])
    axis = np.cross(gz, -pg); axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(gz, -pg), -1, 1))
    return axis_angle_to_quat(axis, angle)
```

**Parameters identified:** log-space `[log_kp, log_damping, log_armature]`

**Results:**

| Parameter | Identified | Notes |
|---|---|---|
| `kp` | 0.200 (hit lower bound) | Identifiability issue |
| `damping` | 0.027 | Plausible but low |
| `armature` | 0.00005 | Implausibly low |

**The identifiability problem:** One-step sysid systematically biases toward **lower gains**. At one step, lower kp means smaller position correction — but the residual `q[t+1] - q_measured[t+1]` is also smaller because the joint didn't move much from its initial state. Lower gains are not penalised within a single step; only multi-step divergence would reveal that the gain is too low.

This approach is useful for identifying contact and friction parameters (where the one-step signal is more direct) but not for actuator gain identification.

**Conclusion:** MuJoCo sysid on walking data is fundamentally limited by trajectory instability. BAM on testbench data remains the right approach for motor parameters.

---

## 8. The action_scale Mystery

### 8.1 The Problem

When deploying a policy trained with `action_scale=1.0`, the real robot shakes violently. Reducing to `action_scale=0.65` gives stable walking. Why?

### 8.2 Root Cause Analysis

**Step 1: Motor strength mismatch (identified via BAM).**

Old M1 (contaminated data) identified a motor with stall torque 0.518 Nm. Real motor stall torque (new M1, clean data): 0.670 Nm. The motor is **29% stronger than the sim assumed**.

Policy trained with weak sim motor: learns **large actions** because a large position error is needed to generate enough torque. Real strong motor: same large position error → **1.29× more torque** → overshoots → oscillates.

Reducing `action_scale` to 0.65 compensates: it reduces the effective position error, bringing actual torque closer to what the policy expected.

**Step 2: Load-dependent friction (M6).**

The robot's joints, unlike the testbench arm, carry the body weight during walking. M6's `load_friction_motor=0.177` and `load_friction_external=0.333` contribute friction proportional to joint load — a mechanism absent from the MuJoCo sim. This additional damping in the real robot means the policy needs to command larger positions to achieve the same motion, partially offsetting the motor strength effect.

**Step 3: What remained after new M1 training.**

After retraining with new M1 (stronger, more friction), `action_scale` only improved slightly — still needed ~0.60–0.70. This points to the residual unmodelled friction (M6 terms) as the remaining cause.

**Step 4: The wrong fix (randomising action_scale).**

One proposed mitigation was to randomise `action_scale` during training (e.g., U(0.6, 1.2)). This was correctly rejected: it treats the symptom (policy over-actuation) rather than the cause (wrong motor model). A policy trained with randomised scale learns to be conservative everywhere, sacrificing performance at `scale=1.0`.

The correct fix is to improve the motor model in the sim (M6 params, or at minimum correct M1 params) so the policy never needs a gain reduction.

---

## 9. Summary: The Sim2Real Gap Stack

The following table enumerates every identified gap, its mechanism, its effect on the robot, and its current mitigation status:

| Gap | Mechanism | Effect on real robot | Mitigation | Status |
|---|---|---|---|---|
| Motor strength (kt, R) | BAM data contamination | Policy over-actuates → shaking | New M1/M6 (clean data) | Identified, retrain needed |
| Motor base friction | Same contamination | Less damping in sim | New M1 `frictionloss=0.032` | Identified, retrain needed |
| Load-dependent friction | M6 terms, no MuJoCo equivalent | Extra damping under load | None (unmodellable in standard MuJoCo) | Ongoing gap |
| Lateral CoM asymmetry | Hardware manufacturing | 7–8° lateral lean | DR on CoM ±3 mm (may need wider) | Partially mitigated |
| Actuator delay | Firmware + bus latency | Policy acts on stale state | Delay randomisation 0–3 steps | Mitigated |
| Sensor noise | IMU, encoder quantisation | Policy sees noisy observations | Noise injection (all modalities) | Mitigated |
| IMU mounting error | Mechanical misalignment | Gravity vector biased | IMU angle DR ±1° | Mitigated |
| Motor gain variability | Temperature, firmware tolerance | Wrong effective kp/kd | Kp/Kd DR ±15% / ±10% | Mitigated |
| Stribeck friction | Near-zero velocity physics | Different stick-slip behaviour | Not modelled in MuJoCo | Known gap |
| Ground contact model | Flat sim vs real surface | Different slip/bounce | Foot friction 0.6, slip penalty | Partial |

---

## 10. Current Actuator Parameters in Training XML

The training XML (`joints_properties.xml`) currently uses:

```xml
<default class="chosen_actuator">
  <joint damping="0.041" frictionloss="0.032" armature="0.002"/>
  <position kp="0.386" kv="0.0" forcerange="-0.67 0.67" ctrlrange="-10.0 10.0"/>
</default>
```

These correspond approximately to **New M1** values (kp=0.386, forcerange=0.670 Nm). Compared to what New M6 suggests (kp=0.432, forcerange=0.750 Nm), the current params are ~10% conservative on both gains and torque.

The next planned step is to update with New M6 values and retrain, then measure required `action_scale` empirically.

---

## 11. Open Questions and Next Steps

1. **Does retraining with New M6 close the action_scale gap?** Expected: yes partially. M6 gives kp=0.432 vs current 0.386 (+12%), and forcerange=0.750 vs 0.670 (+12%). But unmodelled load-friction still exists.

2. **Can load-friction be approximated in MuJoCo?** Options:
   - Increase `frictionloss` to a value that "averages" the load-dependent friction over the gait cycle (lossy but simple)
   - Domain-randomise `frictionloss` over a wide range (e.g., 0.008–0.030) to force robustness to this uncertainty
   - Use MuJoCo's `<tendon>` or custom force callbacks to implement load-proportional friction (complex)

3. **What causes the 7–8° lateral lean?** CoM offset measurement on the real robot (suspended pendulum method or IMU-based estimation during standing) would quantify this. The lean direction and magnitude could then inform an asymmetric bias in training.

4. **Is the observation delay model accurate?** The 0–3 step delay model is a rough approximation. If the actual delay distribution is different (e.g., bimodal due to USB scheduling), the policy may not be properly robust to it.

5. **Terrain transfer:** Current policies are trained on flat terrain. Rough terrain training has been set up but not yet fully evaluated on the real robot. The 1.5 cm maximum step height may be too conservative.

---

## Appendix A: Key File Locations

| File | Purpose |
|---|---|
| `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` | All DR, noise, curriculum, reward parameters |
| `src/mjlab_microduck/robot/microduck/joints_properties.xml` | Actuator parameters used in training |
| `src/mjlab_microduck/tasks/mdp.py` | Reward function implementations |
| `src/mjlab_microduck/tasks/symmetry.py` | Mirror loss and observation flip |
| `scripts/infer_policy.py` | Real-robot inference script |
| `scripts/export.py` | ONNX export from wandb checkpoints |
| `~/Rhoban/bam/params/xl330/m1_new.json` | Clean M1 identification result |
| `~/Rhoban/bam/params/xl330/m6_new.json` | Clean M6 identification result |

## Appendix B: Parameter Evolution Timeline

| Stage | kp | frictionloss | forcerange | action_scale needed |
|---|---|---|---|---|
| Old M1 (contaminated) | 0.522 | 0.016 | ±0.518 Nm | ~0.65 |
| Old M6 (contaminated) | 0.522 | 0.006 | ±0.908 Nm | unknown |
| New M1 (clean) | 0.386 | 0.032 | ±0.670 Nm | ~0.60–0.70 |
| New M6 (clean) — **planned** | **0.432** | 0.008 | **±0.750 Nm** | **TBD** |

The gap between Old M1 kp=0.522 and the real robot's effective kp is what forced action_scale downward. New M6 moves in the right direction. Whether it fully closes the gap depends on how much of the remaining mismatch is load-friction (unmodellable) vs electromechanical error (now corrected).
