# Action Space Bounding and Observation Normalization in PPO for Robotics

A literature and code review prompted by recurring training-instability issues on the sprung-leg MicroDuck. Written 2026-05-31 after multiple training runs collapsed via the action_rate_l2 → value-loss → NaN feedback loop, and after the workaround (`clip_actions=π`) was found to actively shape the policy toward sub-optimal hopping gaits.

---

## 1. Problem Statement

Two interlocking architectural questions surfaced during the sprung-leg training campaign:

1. **Action-space bounding** — should the policy network be free to emit any real value (current mjlab/rsl_rl default), should outputs be clipped at the env wrapper, or should the network architecture itself bound outputs (tanh, Beta)? Each affects training dynamics differently.
2. **Observation normalization** — should the network receive raw observations (current mjlab default) or normalized inputs (running mean/std)? With observation channels spanning ~10× different natural scales (gravity components ±1 vs. joint velocities ±10 rad/s), unnormalized inputs ask the first-layer weights to compensate for scale.

Both choices are framework defaults in mjlab/rsl_rl. The defaults work for many setups, but they are not consensus best practice in the broader RL literature, and they are not robust against pathologies that arose for us when spring dynamics created subtle observation-distribution shifts during convergence.

This report surveys the recent literature, examines reference implementations, and gives concrete recommendations for the MicroDuck project.

---

## 2. Methodology

Searched and reviewed:

- **Foundational papers** — SAC ([Haarnoja et al. 2018](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)) for tanh squashing; Beta policy ([Chou et al. 2017](https://proceedings.mlr.press/v70/chou17a/chou17a.pdf)) for bounded-support stochastic policy.
- **Empirical PPO study** — [Andrychowicz et al. ICLR 2021, "What Matters in On-Policy Reinforcement Learning"](https://openreview.net/pdf?id=nIAxjsniDzg) — a large-scale (~250k trained agents) ablation of every PPO design choice.
- **PPO implementation guides** — [Huang et al. "The 37 Implementation Details of PPO" (ICLR 2022 Blog Track)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/), the de facto reference for PPO engineering choices.
- **Recent (2024-2025) work** — [Corrected Soft Actor Critic](https://arxiv.org/html/2410.16739v1) (Chen et al. 2024); PPO+Beta paper ([Petrazzini & Gomes 2021](https://arxiv.org/pdf/2111.02202)); recent sim-to-real legged-robot work ([Systematic Sim-to-Real Transfer 2025](https://arxiv.org/html/2509.06342v1), [GPO 2025](https://arxiv.org/pdf/2601.20668)).
- **Reference code** — Stable Baselines3 (industrial-grade), CleanRL (research reference), Isaac Lab (NVIDIA's robotics framework), legged_gym (the original Isaac Gym legged-robot environment), rsl_rl (the PPO implementation mjlab uses).

---

## 3. Action Space Bounding — Approaches and Trade-offs

### 3.1 Unbounded policy + wrapper clip (mjlab/rsl_rl default)

**How it works**: Actor network ends in `Linear(hidden, n_actions)` with no output activation. Mean μ is unbounded. Sampling produces `a ~ Normal(μ, σ)`, also unbounded. The env wrapper applies `torch.clamp(a, -K, K)` before stepping the env (if `clip_actions=K` is set; if `None`, no clipping).

**Pros**:
- Simplest implementation — no log-probability correction needed.
- Universal across action types (position, velocity, torque).
- The PPO math is clean for actions deep inside the bound.

**Cons**:
- **The clipping introduces a log-probability bias near the bound.** PPO computes `log_prob` for the *sampled* (unclipped) action, but the env saw the *clipped* action. The advantage signal is correctly credited to the clipped action, but the gradient flows through the unclipped sample's log-density. This is small for samples deep within bounds but real near boundaries.
- **The network has no structural reason to keep outputs small.** Without architectural pressure, weights can drift such that occasional outputs are pathologically large (10⁸–10¹⁰), which feeds the action_rate_l2 → value-loss death spiral we observed.
- **Not sim-to-real friendly** — the policy may rely on the trained simulation environment's specific action-clipping semantics.

**Empirical evidence**:
> "tanh performs slightly better overall (in particular it improves the performance on HalfCheetah by 30%) compared to clipping when transforming unbounded sampled actions into the bounded [−1, 1] domain."
> — Andrychowicz et al. 2021, ICLR

### 3.2 Tanh squashing with Jacobian correction (SAC standard)

**How it works**: Actor outputs μ unbounded. Sample `u ~ Normal(μ, σ)`. The action is `a = tanh(u) ∈ (-1, 1)`. To preserve a valid probability distribution, the log-probability is corrected:

```
log π(a | s) = log p_Normal(u | s) − Σᵢ log(1 − tanh²(uᵢ))
            = log p_Normal(u | s) − Σᵢ log(1 − aᵢ²)
```

The subtracted term is the log Jacobian determinant of the tanh transform. Without it the density is not normalized over [-1, +1] and the entropy and KL terms in the loss are biased.

**Pros**:
- **Structural bound** — outputs cannot exceed (-1, +1) by construction.
- **Smooth, differentiable, infinitely-bounded** — no clipping boundary discontinuity.
- **Mathematically rigorous** when the Jacobian is included.
- Used in SAC and adopted by many SOTA continuous-control algorithms.

**Cons**:
- Requires more careful implementation. Adds a few lines to log_prob, sample, and entropy computations.
- The Jacobian correction is numerically sensitive near a = ±1 (the term blows up as `log(0)` is approached). Standard fix: `log(1 − a² + ε)` with `ε ≈ 1e-6`.
- **Recent critique** (Chen et al. 2024, "Corrected Soft Actor Critic"): even with the Jacobian, tanh-squashed Gaussian has its mode shifted away from `tanh(μ)`, and this is more severe in high-dimensional action spaces. They argue for further corrections.
- **PPO + tanh is theoretically less well-established than SAC + tanh.** SAC needs squashing for its entropy term to be well-defined on bounded actions; PPO doesn't strictly need it, so PPO+tanh is a hybrid practice rather than a textbook recipe.

**Empirical evidence**:
- Andrychowicz et al. (above) — tanh > clipping on continuous control benchmarks.
- The "37 Implementation Details" blog notes that tanh was *not* used by OpenAI's stable-baselines reference PPO, despite being shown to work better empirically. The convention persists for historical reasons.

### 3.3 Tanh squashing *without* Jacobian correction

A pragmatic shortcut sometimes seen in Isaac-Lab-style codebases: apply tanh on the actor's output to bound it, but don't bother with the Jacobian term.

**Pros**: simple — just add `nn.Tanh()` as the last layer.

**Cons**: the log_prob PPO sees is *wrong* (it's the unsquashed Normal's log_prob, but actions came from a squashed distribution). Empirically this can still train, but the policy gradient and value targets are technically biased. We don't recommend this — if you do tanh, do it properly.

### 3.4 Beta distribution policy

**How it works**: Actor outputs two parameters per action dimension: α, β > 0 (via softplus or exp). Action sampled from `Beta(α, β)` which has natural support [0, 1]; linearly rescaled to [-1, +1]. Beta has no Jacobian issue — it's naturally bounded.

**Pros**:
- **Naturally bounded support** — no clipping, no Jacobian.
- Bias-free near the boundaries (unlike Gaussian which always assigns mass outside the env action space).
- **[Chou et al. 2017](https://proceedings.mlr.press/v70/chou17a/chou17a.pdf)**: faster convergence and higher scores than Gaussian on multiple continuous-control benchmarks.
- **[Petrazzini & Gomes 2021](https://arxiv.org/pdf/2111.02202)**: PPO + Beta — 63% improvement in success rate on CarRacing, faster convergence, more stable training.

**Cons**:
- More implementation work — must add Beta head, positivity constraints on α, β.
- Beta distribution behavior is harder to reason about than Gaussian (e.g. multi-modal at α, β < 1).
- Less widely tested in robotics-specific PPO literature.
- Not currently in rsl_rl, stable-baselines3, or CleanRL.

**Verdict**: Promising research direction, but adoption is limited. Worth knowing about; not yet a default choice.

### 3.5 Per-joint scaling (the legged_gym convention)

**How it works**: Actor still outputs unbounded values (or clipped via wrapper). The env applies a *per-joint* multiplicative scale before sending to the actuator: `target = HOME + a × scale_per_joint`. Combined with a moderate `clip_actions`, this turns the action space into a normalized per-joint range with physically meaningful semantics.

This is what we set up most recently for the MicroDuck:

```python
joint_pos_action.scale = {
    r".*hip_yaw.*":   0.5,
    r".*hip_pitch.*": 0.7,
    r".*knee.*":      0.8,
    # ... etc
}
clip_actions = 1.0
```

The original [legged_gym](https://github.com/leggedrobotics/legged_gym) (the canonical Isaac Gym quadruped repo, which informs rsl_rl) uses **`action_scale = 0.5`** uniformly and **`clip_actions = 100.0`** (i.e. very loose clip).

Stock [Isaac Lab velocity locomotion](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py) uses **`scale = 0.5`** uniformly.

**Pros**:
- Physically meaningful action space — `a ∈ [-1, +1]` maps to "fraction of useful joint range".
- Per-joint customization possible (head joints can have different scale from leg joints).
- Inherits everything good from the bounded-action world.

**Cons**:
- Still has the wrapper-clip log-prob bias near ±1 (unlike a tanh head).
- Per-joint configuration adds setup complexity.
- The clip-vs-network-output gap remains: the network can still emit huge raw values, which only get truncated downstream.

---

## 4. Observation Normalization — Approaches and Trade-offs

### 4.1 Running mean/std normalization (Welford, EmpiricalNormalization)

**How it works**: A wrapper or module maintains running estimates of per-feature mean and standard deviation. Each observation is transformed: `obs_norm = (obs - running_mean) / (running_std + ε)`. The running statistics are updated continuously during training (typically frozen after a certain step count or at inference time).

This is the **Welford algorithm** when implemented online without re-accumulating; the rsl_rl version is `EmpiricalNormalization`.

**Pros**:
- **Inputs to the network are O(1) regardless of physical units.** First-layer weights don't have to learn the scale of each component.
- **Reduces sensitivity to obs distribution shifts** during training — the network's input distribution stays roughly N(0, 1).
- **Strong empirical evidence** for benefit (next subsection).

**Cons**:
- **Sim-to-real complication**: at deployment, the normalization statistics computed during training must travel with the policy (or the real-world obs must be normalized using the same stats). Easy to mishandle.
- Mid-training distribution shifts (e.g. curriculum-driven command range expansion) may make old statistics partially obsolete.
- Adds a tiny computational overhead.

**Empirical evidence**:
> "Input normalization is crucial for good performance on all environments apart from Hopper. [Walker2d, HalfCheetah, Ant, and Humanoid] showed substantial performance degradation without it."
> — Andrychowicz et al. 2021, ICLR

> "The 37 Implementation Details of PPO" notes observation normalization as one of two preprocessing steps (with reward scaling) that have "significantly larger impact" on PPO performance, citing Engstrom et al. 2020.

### 4.2 Manual per-component scaling (legged_gym convention)

**How it works**: Each observation component is multiplied by a hand-tuned constant before going to the network.

From [legged_gym base config](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py):

```python
class obs_scales:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05
    height_measurements = 5.0
```

**Pros**:
- Deterministic — no running statistics to manage.
- Sim-to-real friendly — fixed transformation, easy to replicate.
- Per-component scales reflect physical units explicitly (e.g. dof_vel × 0.05 acknowledges that joint velocities range ±20 rad/s).

**Cons**:
- Hand-tuned. The "right" constants depend on the robot and task.
- Doesn't adapt to distribution shifts.
- Some components vary by orders of magnitude across robots, so the constants are not transferable.

This is what the original Isaac Gym work used; mjlab dropped it without replacing it with running normalization.

### 4.3 No normalization (current mjlab default)

The MicroDuck currently feeds raw observations to the network — no Welford normalization, no manual scaling beyond what's defined per-observation-term (we have UniformNoise on some terms but not scaling). Observation magnitudes span ~10× across components.

This is unusual in the wider RL community. It works for the rigid microduck because the system is well-behaved enough that the first-layer weights can compensate, but it leaves performance on the table by most analyses.

### 4.4 Where to put normalization in the architecture

- **Env wrapper** (CleanRL, Gym `NormalizeObservation`): clean separation, normalization is part of the environment. Statistics travel with the env, not the policy. Easy to mess up at deployment.
- **First module of the network** (rsl_rl's `EmpiricalNormalization`): statistics are part of the policy. The exported policy is self-contained. Easier for sim-to-real.

mjlab uses the latter convention via rsl_rl, so flipping `actor_obs_normalization=True` is a one-line change.

---

## 5. Reference Implementation Survey

| Library / Project | Action squashing | Action clip default | Action scale | Obs normalization | Reward normalization | Advantage normalization |
|---|---|---|---|---|---|---|
| **Stable Baselines3** | Both `DiagGaussian` and `SquashedDiagGaussian` with full Jacobian. Choose via `use_sde` or distribution arg. | Env action space spec | n/a | `VecNormalize` wrapper (Welford + clip to ±10) | `VecNormalize` (running discounted std) | Per minibatch |
| **CleanRL** (`ppo_continuous_action.py`) | No tanh. Gaussian sample + env wrapper `ClipAction`. Recommends `init_std=0.01` on final layer. | Yes (`ClipAction`) | n/a | Yes (`NormalizeObservation` + clip to ±10) | Yes (`NormalizeReward` + clip to ±10) | Per minibatch |
| **rsl_rl** (what mjlab uses) | None — raw Gaussian. | Configurable via `clip_actions` (default `None`) | Per-joint dict supported in `JointPositionActionCfg` | `EmpiricalNormalization` available (Welford). Default off in mjlab. | `EmpiricalDiscountedVariationNormalization` available. Default off. | Yes (built into PPO update) |
| **Isaac Lab stock** (velocity locomotion) | None. | None default | **0.5** uniform on JointPositionActionCfg | Not configured in stock velocity_env_cfg | Not configured | Yes |
| **legged_gym** (Isaac Gym, original) | None. | **100.0** (very loose) | **0.5** uniform | Manual per-component (`obs_scales`) | None | Yes |
| **mjlab MicroDuck rigid** (working) | None. | None | 1.0 | False | False | Yes |
| **mjlab MicroDuck sprung** (current) | None. | **1.0** (just changed) | **per-joint dict 0.5–0.8** | False | False | Yes |

**Observations from this survey**:

1. **No major framework defaults to no-clipping AND no-scaling at the same time.** Either there's a moderate clip (`legged_gym`: 100), a scale (`Isaac Lab`: 0.5), or both. Our previous setup (no clip, scale=1.0) was an outlier.
2. **Observation normalization is the default in all research-grade implementations except rsl_rl/mjlab.** Even legged_gym uses manual obs_scales.
3. **Tanh squashing with Jacobian is implemented in SB3 but not in research-robotics PPO codebases.** This is a historical convention difference, not a principled one.
4. **Per-joint scaling exists in mjlab** (JointPositionActionCfg accepts `dict[str, float]`) but is not standard practice in mjlab velocity-locomotion tasks (rigid microduck uses uniform 1.0). The fact that Isaac Lab's stock velocity locomotion uses 0.5 suggests mjlab's 1.0 is again an outlier choice.

---

## 6. Recent Research (2022-2026)

### Findings most relevant to robotics PPO:

**On observation normalization**:
- [Andrychowicz et al. ICLR 2021](https://openreview.net/pdf?id=nIAxjsniDzg) (~250k trained agents): "Always use observation normalization and check if value function normalization improves performance." Effect size is 20–50% performance change on standard benchmarks.

**On tanh vs. clipping**:
- Andrychowicz et al. (same): tanh > clipping by ~30% on HalfCheetah; "use tanh both as the activation function (if the networks are not too deep) and to transform the samples from the normal distribution to the bounded action space."
- [Chen et al. 2024, "Corrected Soft Actor Critic"](https://arxiv.org/html/2410.16739v1): even with the Jacobian, tanh-squashed actions have a mode-shift problem in high dimensions. Proposes more elaborate corrections. *Probably applies to PPO+tanh too*, but unstudied for PPO.

**On Beta policies**:
- [Petrazzini & Gomes 2021](https://arxiv.org/pdf/2111.02202): PPO + Beta — 63% improvement in CarRacing success rate, faster convergence. Suggests the underlying Gaussian-with-clipping framework leaves real performance on the table for bounded action problems.

**On sim-to-real in legged robotics specifically**:
- [Systematic Sim-to-Real Transfer (2025)](https://arxiv.org/html/2509.06342v1): policy outputs joint position offsets via PD control; uses *joint-limit saturation* as a separate safety layer rather than relying on architecture-level bounds.
- [GPO (2025)](https://arxiv.org/pdf/2601.20668): time-varying action transformation that grows the effective action space during training — a curriculum on the action space itself. Empirically beats fixed-action-space PPO on legged locomotion.

**Notable trend**: the most recent legged-locomotion papers don't fundamentally challenge the unbounded-Gaussian-with-clip convention. They add safety layers on top (joint-limit saturation, action curricula, soft-bounded outputs) rather than replacing the underlying architecture. This suggests **the architectural choice is empirically less important than the surrounding engineering** — but it doesn't mean it's optimal.

---

## 7. Diagnosis of the MicroDuck Sprung-Leg Situation

Mapping the literature back to our specific failures:

**The death-spiral mechanism**: A converged policy produced occasional outlier actions of magnitude 10⁸–10¹⁰, which made `action_rate_l2` reward go to ~−10²⁵, which corrupted the value function via the GAE return. This is a textbook consequence of "no architectural bound on policy outputs" — and it's exactly the failure mode that tanh squashing or Beta policy would have prevented at the network level.

**The bouncing/hopping local optimum**: After patching with `clip_actions=π`, the policy converged on a 46%-flight bouncing gait. Our diagnosis was that the clip changed the training landscape — actions that *would* have been useful (large asymmetric leg extension for running strides) were truncated, so the policy found a different attractor. The literature agrees with this in spirit: Andrychowicz showed that tanh-squashed actions outperform clipping on locomotion *for a reason*, and that reason is exactly this kind of training-time landscape distortion.

**The unnormalized observations**: With `actor_obs_normalization=False` and observation channels spanning ~10× scales, the first-layer weights are doing scale-compensation work that could be free if normalization were enabled. Andrychowicz's "crucial for good performance" probably understates the effect for our specific task — sprung dynamics make obs distributions broader and shift over training.

---

## 8. Recommendations

### 8.1 Recommended sequence of single-variable tests

Each step should be its own training run, and we should compare against the baseline ("very good running" config from before all the recent changes).

1. **Step 0 (already done)**: per-joint scale + clip_actions=1.0 + reverted rewards. Verify clean running gait returns.

2. **Step 1**: enable observation normalization. Single config change: `actor_obs_normalization=True, critic_obs_normalization=True`. Expected effect: clearer training curves, possibly slight performance bump, smoother convergence. Sim-to-real consequence: the normalization statistics must be exported with the policy (rsl_rl handles this if it's part of the network module).

3. **Step 2 (longer-horizon)**: implement tanh-squashed Gaussian with Jacobian correction as a custom `ActorCritic` subclass for `rsl_rl`. This is the architectural fix that matches the SB3-quality reference and the Andrychowicz recommendation. Estimated work: 40–80 lines of subclass + integration tests. Once tested, this would replace the wrapper-level `clip_actions` entirely.

4. **Step 3 (research)**: prototype a Beta policy as a `ActorCritic` subclass. This is more speculative but the Petrazzini & Gomes results suggest it might give a real bump on bounded action spaces, especially for the sprung-leg task where actions naturally live within a fixed physical range.

### 8.2 What NOT to do

- **Don't add tanh squashing without the Jacobian correction.** The pragmatic shortcut introduces a real bias in PPO and is the worst of both worlds (extra complexity without the theoretical guarantee). Either do it properly or stick with the clip.
- **Don't enable obs normalization for actor without critic** (or vice versa). They need to be consistent or the value function and policy disagree about state representations.
- **Don't change multiple things at once after Step 0.** The previous attempts to "fix the gait" by stacking changes (clip + flight_phase cap + L2 penalty) produced compound effects we couldn't disentangle. Single-variable methodology going forward.

### 8.3 Open questions worth tracking but not yet acting on

- **Does the Chen et al. 2024 "Corrected SAC" critique apply to our setup?** The mode-shift problem in high-dim tanh-squashed Gaussians is real. For 14-dimensional actions, the effect should be small but nonzero. Not the highest-priority fix.
- **Should we add reward normalization (`EmpiricalDiscountedVariationNormalization`)?** Less consensus in the literature than obs normalization. mjlab's `nan_to_num` patches and our advantage normalization may be sufficient. Worth revisiting if value-loss issues recur.
- **Should we explore the GPO curriculum approach** (growing action space over training) for our sprung-leg case? Speculative but possibly relevant for the running-vs-walking spectrum.

---

## 9. Conclusion

The framework defaults in mjlab/rsl_rl (unbounded Gaussian + no obs normalization + no action clip) are at the conservative end of what the broader RL community considers best practice. They work for well-tuned, well-behaved tasks (rigid microduck), and they fail in edge cases (sprung microduck) by mechanisms that are well-documented in the literature.

Our recent intervention (per-joint scale + clip_actions=1.0) brings us closer to the legged_gym/Isaac Lab convention and should resolve the immediate symptoms. The principled next step is observation normalization, which is the single most agreed-upon improvement in the empirical PPO literature. Beyond that, a proper tanh-squashed Gaussian implementation is the theoretical north star and worth investing in if we plan to keep iterating on this stack.

---

## References

### Foundational
- Haarnoja et al. 2018. *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.* ICML. [PDF](https://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf)
- Chou et al. 2017. *Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution.* ICML. [PDF](https://proceedings.mlr.press/v70/chou17a/chou17a.pdf)

### Empirical
- Andrychowicz et al. 2021. *What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study.* ICLR. [OpenReview](https://openreview.net/pdf?id=nIAxjsniDzg) · [arXiv](https://ar5iv.labs.arxiv.org/html/2006.05990)
- Huang et al. 2022. *The 37 Implementation Details of Proximal Policy Optimization.* ICLR Blog Track. [URL](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

### Recent (2023+)
- Chen et al. 2024. *Corrected Soft Actor Critic for Continuous Control.* [arXiv](https://arxiv.org/html/2410.16739v1)
- Petrazzini & Gomes 2021. *Proximal Policy Optimization with Continuous Bounded Action Space via the Beta Distribution.* [arXiv](https://arxiv.org/pdf/2111.02202)
- (2025) *Systematic Sim-to-Real Transfer for Diverse Legged Robots.* [arXiv](https://arxiv.org/html/2509.06342v1)
- (2025) *Growing Policy Optimization (GPO) for Legged Robot Locomotion.* [arXiv](https://arxiv.org/pdf/2601.20668)

### Reference implementations
- [Stable Baselines3 distributions](https://stable-baselines3.readthedocs.io/en/v2.5.0/_modules/stable_baselines3/common/distributions.html)
- [CleanRL `ppo_continuous_action.py`](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)
- [legged_gym](https://github.com/leggedrobotics/legged_gym) (Isaac Gym legged robot envs)
- [Isaac Lab velocity_env_cfg](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)
- [rsl_rl actor_critic.py](https://github.com/leggedrobotics/rsl_rl) (the PPO impl mjlab uses)
