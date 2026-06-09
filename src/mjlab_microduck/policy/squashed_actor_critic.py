"""Tanh-squashed Gaussian actor-critic for PPO.

Drop-in replacement for ``rsl_rl.modules.ActorCritic`` that bounds the
policy's action output structurally via ``tanh`` instead of via the env
wrapper's ``clip_actions``.  This avoids the clip-bias pathology where
PPO's entropy bonus drives the action-noise σ unbounded because the
clip flattens the σ → behavior gradient (observed in run fktnuht3:
noise_std grew from 1.0 → 229 over 40k iters with clip_actions=1.0).

The literature support and full background is in
``docs/action_obs_normalization_report.md``.  This file is the
implementation; the rationale is there.

Two design decisions baked in:

(1) **Atanh reconstruction (Option 1, with two-epsilon refinement
    Fix A)** — at PPO update time we recover the pre-squashed Normal
    sample ``u`` from the stored squashed action ``a`` via
    ``u = atanh(a.clamp(-1+ATANH_CLAMP_EPS, 1-ATANH_CLAMP_EPS))``.
    Matches Stable Baselines3's ``SquashedDiagGaussianDistribution``
    structurally, but with a *larger* atanh clamp (1e-3 vs SB3's 1e-6)
    to bound |u| ≤ 3.8 and keep the PPO importance ratio from
    exploding when σ shifts between rollout and update.

    History of this design:
    - First version: single EPS = 1e-6 everywhere.  Surrogate loss
      spikes to 1e3–1e5 magnitude observed at iter 700+ of run
      ab_squashed; network NaN'd at iter 740.  Diagnosed as
      saturated-|u| ratio explosion.
    - Fix A (current): two epsilons — atanh clamp at 1e-3 (bounds u),
      Jacobian eps stays at 1e-6 (preserves precision).
    - If Fix A still produces ratio spikes, the next escalation is
      hard ratio clipping in the loss (changes rsl_rl PPO, not
      lightweight) or Option 2 (store u directly, also heavier).

(2) **Two SAC-convention approximations are kept**:
    - ``entropy`` returns the underlying Normal's entropy, not the
      true squashed entropy (which has no closed form).  Off by
      ``E[Σᵢ log(1-aᵢ²+ε)]`` (negative, so squashed entropy is
      smaller).  Entropy bonus is multiplied by entropy_coef=0.01, so
      the bias on the gradient is small.  Monte-Carlo correction is
      possible but adds compute.  Standard practice (SB3, SAC).
    - The PPO adaptive-LR KL is computed on the underlying Normal
      (closed form), not on the squashed distribution.  Same SAC
      convention.

Anything else (critic, obs normalization, ``_update_distribution``,
KL adaptation, value loss, advantage normalization) is inherited
unchanged from ``ActorCritic``.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from rsl_rl.modules.actor_critic import ActorCritic


class SquashedActorCritic(ActorCritic):
    """ActorCritic with tanh-squashed action output.

    The actor's MLP outputs the mean of an *underlying* Gaussian; the
    action delivered to the env is ``tanh(sample)`` (so always in
    ``(-1, +1)``).  The Jacobian correction is applied in
    :meth:`get_actions_log_prob`.

    Compared to :class:`ActorCritic` this overrides exactly four
    methods.  All other behaviour — critic, normalization, distribution
    storage, std parameter, KL adaptation — is inherited unchanged.

    Args:
        small_init_scale: Multiplicative factor applied to the final
            actor layer's weights at init time.  Andrychowicz et al.
            (ICLR 2021, "What Matters in On-Policy RL") recommend
            ~100× smaller weights on the final policy layer.  With
            tanh saturation this matters even more (large init →
            saturated outputs → no gradient).  Default 0.01.
    """

    # Two epsilons, used in two different places — see "Two-epsilon
    # design" below for the empirical reasoning.
    #
    #   ATANH_CLAMP_EPS:  clamp `a` to (-1+ε, 1-ε) before atanh.  This
    #                     caps |u| at atanh(1 − ATANH_CLAMP_EPS).
    #                     Larger ε ⇒ smaller |u|_max ⇒ more stable PPO
    #                     ratio, at the cost of biasing actions away
    #                     from the strict boundary.
    #   EPS:              add ε inside log(1 − a² + ε) for the Jacobian.
    #                     This needs to be small for precision.
    #
    # === Two-epsilon design ===
    # Before Fix A, both used 1e-6.  The 1e-6 atanh clamp allows
    # |u| up to ≈ 7, which is fine for *static* log-probability
    # evaluation but disastrous for the PPO importance ratio when σ
    # changes between rollout and update: a sample at u=7 with σ_old=1
    # vs σ_new=1.5 gives ratio ≈ exp(13) ≈ 545k, and PPO's max(unclipped,
    # clipped) lets the unclipped through when advantage is negative.
    # Result: surrogate-loss spikes of 1e3–1e5 corrupt the network
    # within ~50 iterations (observed in run ab_squashed at iter 700+).
    #
    # FIX A++: ATANH_CLAMP_EPS = 0.1 caps |u| at atanh(0.9) ≈ 1.47.
    # The earlier Fix A value (1e-3) bounded per-dim ratio to ~37, but
    # for our 14-D actions that multiplies to ~1.8M joint ratio, still
    # disastrous.  At ε=0.1, per-dim ratio under σ 1→1.5 is ~1.22 and
    # the joint ratio is ~16 — fully survivable by PPO.
    #
    # The bias is small *and only on log_prob* — the env still sees the
    # full ``tanh(u) ∈ (-1, +1)`` action range because the env-side
    # action is computed before any of this log_prob machinery.  This
    # clamp only affects how the *gradient* attributes credit to
    # saturated actions: actions at |a| > 0.9 are treated by the
    # gradient as if they were at |a| = 0.9.  The policy will learn to
    # avoid producing |a| > 0.9 magnitude because the gradient signal
    # is bounded there — which is exactly what we want, since saturated
    # actions are anyway capped at ±1 by the actuator's physical limits.
    #
    # The Jacobian's EPS stays at 1e-6 because that term is a static
    # density correction — no ratios are computed across it.  Smaller
    # ε gives more accurate Jacobian values.
    ATANH_CLAMP_EPS: float = 0.1
    EPS: float = 1e-6

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Pop our own kwarg before forwarding to ActorCritic, so the
        # base-class `kwargs` filter doesn't print a warning about it.
        small_init_scale: float = float(kwargs.pop("small_init_scale", 0.01))
        super().__init__(*args, **kwargs)

        # Small-init on the final actor layer.  Identify it by walking
        # backwards through `self.actor` (rsl_rl's `MLP`) — the last
        # `nn.Linear` is the output layer.
        import torch.nn as nn

        last_linear: nn.Linear | None = None
        for module in reversed(list(self.actor.modules())):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        assert last_linear is not None, (
            "SquashedActorCritic: could not find the final nn.Linear in self.actor"
        )
        with torch.no_grad():
            last_linear.weight.mul_(small_init_scale)
            if last_linear.bias is not None:
                last_linear.bias.zero_()

    # ── Overrides ─────────────────────────────────────────────────────

    def act(self, obs: TensorDict, **kwargs: Any) -> torch.Tensor:
        """Sample from the underlying Normal and squash with tanh.

        The ``**kwargs`` are preserved (rsl_rl's recurrent variants
        pass ``masks`` and ``hidden_state``) but ignored here, same as
        the non-recurrent base class.

        Side-effects:
        - ``_update_distribution`` sets ``self.distribution`` to the
          underlying ``Normal(μ, σ)``.
        - ``self._last_raw_u`` caches the pre-tanh sample for
          downstream use by :class:`SquashedPPO`, which passes it back
          to :meth:`get_actions_log_prob_with_raw_u` at update time.
          This sidesteps the atanh-reconstruction information loss
          (Option 2).  See the module docstring's "atanh
          reconstruction" entry for the design rationale.
        """
        del kwargs  # masks/hidden_state — irrelevant for our non-recurrent case
        obs_t = self.get_actor_obs(obs)
        obs_t = self.actor_obs_normalizer(obs_t)
        self._update_distribution(obs_t)
        u = self.distribution.sample()
        self._last_raw_u = u
        return torch.tanh(u)

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        """Deterministic policy output: ``tanh(μ)``.

        Used at evaluation time and by the (currently disabled)
        symmetry loss.  Note this is *not* the mode of the squashed
        distribution — the mode of ``tanh(N(μ, σ))`` deviates from
        ``tanh(μ)`` for σ > 0 (see Chen et al. 2024 "Corrected SAC").
        We use ``tanh(μ)`` because (a) it's the SAC convention,
        (b) the symmetry loss expects a deterministic function of
        obs, and (c) we don't have closed form for the squashed mode.
        """
        obs_t = self.get_actor_obs(obs)
        obs_t = self.actor_obs_normalizer(obs_t)
        if self.state_dependent_std:
            mean = self.actor(obs_t)[..., 0, :]
        else:
            mean = self.actor(obs_t)
        return torch.tanh(mean)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Log-probability of squashed actions under the current policy.

        ``actions`` are squashed (tanh output) — what got sent to the
        env and stored in the rollout buffer.  We invert via atanh to
        recover ``u``, score it under the underlying Normal, then
        subtract the Jacobian log-determinant.

        Mathematical form:

            log π(a | s) = log p_Normal(u | μ, σ) − Σᵢ log(1 − aᵢ² + ε)

        where ``u = atanh(a)`` (clamped for numerical safety) and the
        ``ε`` floor prevents ``log(0)`` when ``|a|`` is at the boundary.

        The ``self.distribution`` we read here is set by the most
        recent :meth:`act` call (see touchpoint #1 in the design doc:
        PPO calls ``act`` before ``get_actions_log_prob`` in both
        rollout and update paths).
        """
        # Aggressive clamp before atanh — bounds |u| to atanh(0.999) ≈
        # 3.8 and keeps the PPO importance ratio survivable under σ
        # changes.  See "Two-epsilon design" in the class header.
        a_for_atanh = actions.clamp(
            -1.0 + self.ATANH_CLAMP_EPS, 1.0 - self.ATANH_CLAMP_EPS
        )
        u = torch.atanh(a_for_atanh)
        log_prob_normal = self.distribution.log_prob(u).sum(dim=-1)
        # Jacobian correction: −Σ log(1 − a²+EPS).  We use the original
        # ``actions`` (not ``a_for_atanh``) and the smaller EPS here
        # because this term is a static density correction with no
        # ratio computed across it — accuracy beats stability margin.
        jacobian_log_det = torch.log(
            1.0 - actions.pow(2) + self.EPS
        ).sum(dim=-1)
        return log_prob_normal - jacobian_log_det

    def get_actions_log_prob_with_raw_u(
        self, actions: torch.Tensor, raw_u: torch.Tensor
    ) -> torch.Tensor:
        """Log-probability of squashed actions using the supplied raw u.

        This is the **Option 2** path — the caller (SquashedPPO) passes
        in the exact ``u`` that was sampled during rollout (stored by
        :class:`SquashedRolloutStorage`).  Bypasses atanh
        reconstruction entirely, avoiding the saturation-induced info
        loss that destabilizes Option 1 in high-dim PPO (see module
        docstring for the failure analysis).

        Form:

            log π(a | s) = log p_Normal(u | μ, σ) − Σᵢ log(1 − aᵢ² + ε)

        where ``u`` is the supplied raw sample (NOT recovered from
        ``a``) and the Jacobian uses the small EPS for precision.

        Note ``actions`` and ``raw_u`` must correspond: ``actions``
        should equal ``tanh(raw_u)`` modulo float32 precision, but we
        don't assert this — the rollout buffer is responsible for
        keeping them aligned.
        """
        log_prob_normal = self.distribution.log_prob(raw_u).sum(dim=-1)
        jacobian_log_det = torch.log(
            1.0 - actions.pow(2) + self.EPS
        ).sum(dim=-1)
        return log_prob_normal - jacobian_log_det

    # NOTE: not overridden, inherited from ActorCritic:
    #   - _update_distribution  (creates self.distribution = Normal(μ, σ))
    #   - action_mean / action_std  (queried for KL computation in PPO)
    #   - entropy  (used for entropy bonus — SAC-convention approximation)
    #   - evaluate  (critic, no actor-side change needed)
    #   - get_actor_obs, get_critic_obs, update_normalization
    #   - reset, forward, load_state_dict
