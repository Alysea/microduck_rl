# Phase B Step B1 — Touchpoint Map for `SquashedActorCritic`

This document is the output of Step B1: a code-reading-only analysis of how `rsl_rl`'s `ActorCritic` is consumed by `PPO`, identifying the minimum override surface needed to implement a tanh-squashed Gaussian policy with Jacobian-corrected log-probability.

No code is written yet. This is the plan that gates Step B2.

## Files read

- `rsl_rl/modules/actor_critic.py` — base class.
- `rsl_rl/algorithms/ppo.py` — calls actor-critic via the `self.policy.*` interface.
- `rsl_rl/runners/on_policy_runner.py` — instantiates actor-critic via `resolve_callable(class_name)`.
- `rsl_rl/utils/utils.py::resolve_callable` — supports fully-qualified strings like `"package.module:ClassName"`.

## How to plug in a custom actor-critic

`OnPolicyRunner._construct_algorithm` does this at line 270:

```python
actor_critic_class = resolve_callable(self.policy_cfg.pop("class_name"))
actor_critic = actor_critic_class(obs, obs_groups, num_actions, **policy_cfg).to(device)
```

`class_name` is a field on `RslRlPpoActorCriticCfg` (default `"ActorCritic"`). We set it to `"mjlab_microduck.policy.squashed_actor_critic:SquashedActorCritic"` and rsl_rl uses our class. **No monkey-patching needed, no fork required.**

## The `policy.*` interface PPO depends on

Read from `rsl_rl/algorithms/ppo.py`:

| Call | Caller location | What it returns | What we override |
|---|---|---|---|
| `policy.act(obs)` | line 129 (rollout), line 248 (re-eval in update) | The action sent to the env / used in log_prob. **Side-effect**: sets `self.distribution`. | YES — sample `u` from Normal, return `tanh(u)`. |
| `policy.evaluate(obs)` | line 130, 172 (compute_returns), 250 (update) | Critic value. | NO — critic untouched. |
| `policy.get_actions_log_prob(a)` | line 131 (rollout), 249 (update) | `log π(a | obs)` using `self.distribution`. | YES — recover `u = atanh(a)`, return `log_prob_Normal(u) - log(1-a²+ε)`. |
| `policy.action_mean` | line 132 (rollout), 252 (update for KL) | Mean of the underlying Normal. | NO — `self.distribution` still holds the Normal. The KL computation works on the underlying Normal (this is the SAC convention; the squashed-distribution KL has no closed form). |
| `policy.action_std` | line 133, 253 | Std of the underlying Normal. | NO. |
| `policy.entropy` | line 254 (update, for entropy bonus) | `Normal.entropy().sum(dim=-1)`. | NO — we use the Normal entropy as a proxy for the squashed entropy. The true squashed entropy has no closed form. **This is what SAC does** and is the conventional choice. Note: this means we're keeping a small approximation here; if it becomes a problem we can swap in a Monte Carlo entropy estimate later. |
| `policy.act_inference(obs)` | line 324 (symmetry loss, deterministic action) | Mean action for deterministic inference / symmetry. | YES — return `tanh(actor(obs))`. |
| `policy.update_normalization(obs)` | line 142 (process_env_step) | Updates running stats if obs normalization is on. | NO — inherited unchanged. |
| `policy.reset(dones)` | line 167 | No-op for stateless networks. | NO — inherited. |

## Subtle interactions worth flagging

### 1. `act()` is called *twice* per training step

- Once at rollout time (line 129): stores the action that goes to the env.
- Once during the PPO update (line 248): re-runs the forward pass to get the *current* distribution, which is then queried for `action_mean`, `action_std`, `entropy`, and used by `get_actions_log_prob`.

This means **`get_actions_log_prob(a)` always operates on `self.distribution` set by the most recent `act()` call**. Our override must keep this invariant: when `act()` is called (anywhere), `self.distribution` is the latest underlying `Normal(μ, σ)`.

The base class does this via `_update_distribution()` inside `act()`. We can keep that — our `act()` just additionally squashes the sample.

### 2. The action stored in the rollout buffer is the **squashed** action

`transition.actions = self.policy.act(obs).detach()` — line 129. So the rollout buffer stores `a = tanh(u)`. At update time, when we compute `log_prob`, we receive `a` and must invert it back to `u` for the underlying Normal's `log_prob`.

This is the Option 1 implementation we discussed. **Tradeoff for the file's header**: float32 round-trip precision of `atanh(tanh(u))` is ~1e-5 in `u`; downstream impact on log_prob gradient is similarly ~1e-5. Acceptable. If unit tests reveal worse than that, we pivot to Option 2 (extending the rollout buffer to store `u` directly).

### 3. KL divergence is computed on the underlying Normal (line 259–265)

```python
kl = log(σ_new/σ_old) + (σ_old² + (μ_old - μ_new)²) / (2 σ_new²) - 0.5
```

This is the closed-form KL between two Normals. For our squashed distribution, the *true* KL between the squashed policies has no closed form. **We accept the Normal-KL as a proxy** — same choice SAC makes. This drives the adaptive learning-rate schedule; the bias is small in the bulk of the distribution and only matters when both Normals have very different `μ` (which is what we want to penalize anyway).

### 4. Entropy bonus also uses the underlying Normal's entropy

`entropy = self.distribution.entropy().sum(dim=-1)` returns `Σᵢ (0.5 log(2πe σᵢ²))`. The *true* entropy of the tanh-squashed Normal is `entropy_Normal + E[Σᵢ log(1 - aᵢ²)]` (this comes from the change-of-variable formula for differential entropy).

The correction term is *positive* when `|a| < 1` (because `log(1-a²)` is negative — wait, sign: `log(1-a²)` is ≤ 0 for `|a| ≤ 1`. So `entropy_squashed = entropy_Normal - E[Σᵢ -log(1-a²+ε)] ≤ entropy_Normal`). Equivalently: the squashed entropy is *less* than the underlying Normal's entropy (tanh compresses the distribution).

We could compute the correction term via Monte Carlo when needed (sample u from the distribution, average `log(1-tanh(u)²)`), but it adds compute and the entropy bonus is multiplied by a small coefficient (0.01) anyway. **We keep `Normal.entropy()` for now**, same as SB3 and SAC. Document this in the class as "approximation — use Monte Carlo if entropy bonus appears to misbehave".

### 5. Symmetry loss uses `act_inference` (mean action)

Line 324 calls `policy.act_inference(obs)` to get the deterministic mean action for the symmetry-augmentation comparison. For us, `act_inference` returns `tanh(actor(obs))`, which is correct (tanh of mean is the deterministic policy output, even if it's not the mode of the squashed distribution per Chen et al. 2024 — we accept this approximation; symmetry is currently disabled anyway).

## Minimum override surface

A 4-method subclass:

```python
class SquashedActorCritic(ActorCritic):
    EPS = 1e-6
    SCALE_FINAL_LAYER_INIT = 0.01   # Andrychowicz et al. recommendation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Small-init on the final policy layer:
        #   For state_dependent_std=False (our case), the last layer is self.actor[-1].
        with torch.no_grad():
            self.actor[-1].weight.mul_(self.SCALE_FINAL_LAYER_INIT)
            self.actor[-1].bias.zero_()

    def act(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)   # sets self.distribution = Normal(μ, σ)
        u = self.distribution.sample()
        return torch.tanh(u)

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.state_dependent_std:
            mean = self.actor(obs)[..., 0, :]
        else:
            mean = self.actor(obs)
        return torch.tanh(mean)

    def get_actions_log_prob(self, actions):
        # actions are the squashed values stored in the rollout buffer
        a = actions.clamp(-1 + self.EPS, 1 - self.EPS)
        u = torch.atanh(a)
        log_prob_normal = self.distribution.log_prob(u).sum(dim=-1)
        # Jacobian correction: −Σᵢ log(1 − aᵢ²)
        correction = torch.log(1 - a.pow(2) + self.EPS).sum(dim=-1)
        return log_prob_normal - correction
```

Everything else (the critic, the obs normalizers, KL, entropy, distribution state) is inherited unchanged.

**Lines of code**: ~40 including docstrings. The actual logic fits in ~10 lines.

## Risks & open questions before Step B2

1. **The `_update_distribution` re-call during PPO update (line 248)**: this calls `self.policy.act(obs_batch, masks=..., hidden_state=...)`. Our `act` signature uses `(self, obs)` and ignores kwargs. The base class's signature is `act(self, obs, **kwargs)`. **Need to confirm our override preserves the `**kwargs` flexibility** so the recurrent variant's masks/hidden_state still get accepted (even if ignored for the non-recurrent case). Easy fix in the subclass — note for implementation.

2. **`init_noise_std` interaction with small-init**: with the default `init_noise_std=1.0` and a final layer initialized to ~0 (small-init), initial samples are `Normal(~0, 1)`. After tanh: roughly `tanh(N(0, 1))`, which is a unimodal distribution concentrated around 0 with support (-1, +1). That's actually a *nice* exploration distribution at init. Should work well.

3. **`std` parameter (separate `nn.Parameter`)**: The base class has `self.std = nn.Parameter(...)`. This stays unchanged in our subclass — std learns through PPO's normal updates. Good.

4. **`load_state_dict` compatibility**: our subclass adds no new parameters (the small-init touches existing weights). Loading a checkpoint trained with this class into the base class would work; the other direction would lose the small-init but otherwise work. Resume isn't a goal for Phase B (it's a from-scratch run), so this is academic.

5. **Numerical floor at saturation**: the unit test `test_numerical_stability_near_boundary` (Step B3) needs to verify that even when the policy outputs `μ` very large (saturated tanh), the log_prob doesn't explode. The clamp at `±(1-ε)` with `ε=1e-6` should be sufficient, but worth confirming.

## Configuration changes needed

A single line in `microduck_velocity_sprung_env_cfg.py`:

```python
policy=RslRlPpoActorCriticCfg(
    class_name="mjlab_microduck.policy.squashed_actor_critic:SquashedActorCritic",
    # ... all other params unchanged
),
```

No env-cfg changes, no algorithm-cfg changes.

## Where this code lives

`src/mjlab_microduck/policy/squashed_actor_critic.py` — single file, ~80 lines including docstrings and tests imports.

Tests in `tests/test_squashed_actor_critic.py` per Step B3 plan.

## Summary

The override surface is minimal, well-scoped, and matches Stable Baselines3's `SquashedDiagGaussianDistribution` pattern. Two approximations are taken (entropy uses Normal entropy; KL uses Normal KL) — both match SAC convention and Stable Baselines3 practice. The atanh-reconstruction (Option 1) is the only numerical sensitivity to watch in unit tests.

**Ready to proceed to Step B2 (implementation) once you've reviewed and approved this plan.**
