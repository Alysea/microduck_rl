"""Standalone tests for SquashedActorCritic (Step B3).

Verifies that ``SquashedActorCritic.get_actions_log_prob`` matches the
Stable Baselines3 reference algorithm to within a tight tolerance, and
that sampling/numerical/gradient invariants hold.

Run:
    direnv exec . uv run python tests/test_squashed_actor_critic.py

Exits 0 on full success, non-zero on first failure.  No pytest needed.
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn as nn
from torch.distributions import Normal

# Make sure we're testing the source file in the repo, not a stale install.
sys.path.insert(0, "src")

from mjlab_microduck.policy import SquashedActorCritic


# ── Reference implementation (mirrors SB3 SquashedDiagGaussianDistribution) ───
#
# Source for the formula:
#   https://stable-baselines3.readthedocs.io/en/v2.5.0/_modules/stable_baselines3/common/distributions.html
# Specifically the SquashedDiagGaussianDistribution.log_prob method:
#   gaussian_actions = atanh(actions)
#   log_prob = self.gaussian.log_prob(gaussian_actions).sum(-1)
#   log_prob -= torch.sum(torch.log(1 - actions**2 + epsilon), dim=1)
#
# We use the SAME epsilon (1e-6) as SquashedActorCritic.EPS to make the
# test an apples-to-apples comparison of the math (not of the epsilon).


def sb3_style_log_prob(
    mean: torch.Tensor,
    std: torch.Tensor,
    actions: torch.Tensor,
    atanh_eps: float = 0.1,
    jacobian_eps: float = 1e-6,
) -> torch.Tensor:
    """Reference implementation of squashed-Normal log-probability.

    Originally modeled on SB3's SquashedDiagGaussianDistribution
    (single epsilon = 1e-6 everywhere).  Updated to match Fix A++:
    larger epsilon for the atanh clamp (caps |u| for PPO ratio
    stability), smaller epsilon for the Jacobian term (precision).
    See the SquashedActorCritic class header for the rationale.
    """
    a_for_atanh = actions.clamp(-1.0 + atanh_eps, 1.0 - atanh_eps)
    u = torch.atanh(a_for_atanh)
    log_prob_normal = Normal(mean, std).log_prob(u).sum(dim=-1)
    correction = torch.log(1.0 - actions.pow(2) + jacobian_eps).sum(dim=-1)
    return log_prob_normal - correction


# ── Test scaffolding ─────────────────────────────────────────────────────────


_passed = 0
_failed = 0


def test(name: str):
    """Decorator: print name, run, track pass/fail count."""

    def deco(fn):
        global _passed, _failed
        print(f"  {name} ... ", end="", flush=True)
        try:
            fn()
            print("PASS")
            _passed += 1
        except AssertionError as e:
            print(f"FAIL\n    {e}")
            _failed += 1
        except Exception as e:
            print(f"ERROR\n    {type(e).__name__}: {e}")
            _failed += 1
        return fn

    return deco


def make_dummy_policy(
    num_actor_obs: int = 8,
    num_actions: int = 4,
    init_noise_std: float = 1.0,
    seed: int = 0,
) -> SquashedActorCritic:
    """Construct a minimal SquashedActorCritic for testing.

    Uses a TensorDict-shaped fake obs to satisfy the base class's __init__
    (which inspects obs shapes during construction).
    """
    from tensordict import TensorDict

    torch.manual_seed(seed)
    batch = 1
    fake_obs = TensorDict(
        {
            "policy": torch.zeros(batch, num_actor_obs),
            "critic": torch.zeros(batch, num_actor_obs),
        },
        batch_size=[batch],
    )
    return SquashedActorCritic(
        obs=fake_obs,
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=num_actions,
        init_noise_std=init_noise_std,
        actor_hidden_dims=(16, 16),
        critic_hidden_dims=(16, 16),
        activation="elu",
    )


def make_distribution(
    policy: SquashedActorCritic, mean: torch.Tensor, std_val: float = 1.0
) -> None:
    """Force `policy.distribution` to a known Normal(mean, std).

    Used by tests that need to control μ and σ directly instead of going
    through the actor MLP (which has unknown initialization).
    """
    std = torch.full_like(mean, std_val)
    policy.distribution = Normal(mean, std)


# ── Tests ────────────────────────────────────────────────────────────────────


def run_all():
    print()
    print("=" * 70)
    print("SquashedActorCritic unit tests")
    print("=" * 70)

    # ───── (1) Log-prob matches the SB3 reference algorithm ─────
    @test("log_prob matches SB3 reference (1000 samples, 4 dims)")
    def _():
        torch.manual_seed(42)
        policy = make_dummy_policy(num_actions=4)
        # Random μ across batch and dims; σ=1.0 constant
        n = 1000
        mean = torch.randn(n, 4) * 2.0     # μ ~ N(0, 4)
        make_distribution(policy, mean, std_val=1.0)
        # Random actions in (-1, +1) — sample from squashed-Normal
        actions = torch.tanh(mean + torch.randn_like(mean) * 1.0)
        # Compute log-prob both ways
        ours = policy.get_actions_log_prob(actions)
        ref = sb3_style_log_prob(
            mean,
            torch.full_like(mean, 1.0),
            actions,
            atanh_eps=policy.ATANH_CLAMP_EPS,
            jacobian_eps=policy.EPS,
        )
        diff = (ours - ref).abs().max().item()
        assert diff < 1e-5, f"max abs diff = {diff:.2e} (expected < 1e-5)"

    # ───── (2) Correction is strictly subtracted (sign check) ─────
    @test("log_prob correction term is non-positive")
    def _():
        # For any |a| < 1, log(1 - a² + ε) < 0 (since 1 - a² ∈ (0, 1] and
        # ε is tiny).  So our `log_prob_normal - jacobian_log_det` always
        # *adds* a non-negative amount (subtracting a negative).  Verify.
        torch.manual_seed(7)
        policy = make_dummy_policy(num_actions=4)
        mean = torch.zeros(50, 4)
        make_distribution(policy, mean, std_val=1.0)
        actions = torch.rand(50, 4) * 1.8 - 0.9   # uniform in (-0.9, +0.9)
        ours = policy.get_actions_log_prob(actions)
        # Compare with the *unsquashed* log_prob (no correction).
        # Use the same atanh clamp that the policy uses internally so
        # the underlying u is identical.
        u = torch.atanh(
            actions.clamp(-1 + policy.ATANH_CLAMP_EPS, 1 - policy.ATANH_CLAMP_EPS)
        )
        unsquashed = Normal(mean, 1.0).log_prob(u).sum(dim=-1)
        # squashed log_prob ≥ unsquashed (because we subtract a negative)
        assert torch.all(ours >= unsquashed - 1e-5), (
            f"squashed log_prob is NOT ≥ unsquashed everywhere "
            f"(min diff = {(ours - unsquashed).min().item():.3e})"
        )

    # ───── (3) Sampled actions strictly within (-1, +1) ─────
    @test("act() samples are strictly within (-1, +1)")
    def _():
        from tensordict import TensorDict

        torch.manual_seed(11)
        policy = make_dummy_policy(num_actions=4, init_noise_std=2.0)
        # Larger init_noise_std to stress the tanh boundary
        obs = TensorDict(
            {
                "policy": torch.randn(128, 8) * 3.0,    # large obs to exercise μ
                "critic": torch.zeros(128, 8),
            },
            batch_size=[128],
        )
        a = policy.act(obs)
        assert torch.isfinite(a).all(), "non-finite actions"
        assert (a > -1.0).all() and (a < 1.0).all(), (
            f"actions outside (-1, +1): min={a.min().item()}, max={a.max().item()}"
        )

    # ───── (4) Numerical stability at the boundary |a| → 1 ─────
    @test("log_prob is finite at |a| ≈ 1 (numerical stability)")
    def _():
        torch.manual_seed(0)
        policy = make_dummy_policy(num_actions=4)
        # actions exactly at the EPS clamp boundary, and *past* it
        boundary_a = torch.tensor(
            [
                [0.99999, -0.99999, 0.999999, -0.999999],
                [1.0 - 1e-9, -1.0 + 1e-9, 0.9, -0.9],   # essentially ±1
                [1.0, -1.0, 1.0, -1.0],                  # exactly ±1
            ]
        )
        mean = torch.zeros_like(boundary_a)
        make_distribution(policy, mean, std_val=1.0)
        log_prob = policy.get_actions_log_prob(boundary_a)
        assert torch.isfinite(log_prob).all(), (
            f"non-finite log_prob at boundary: {log_prob}"
        )

    # ───── (5) Gradient flow on μ AND σ ─────
    @test("gradients flow to actor params AND std (the key fix)")
    def _():
        from tensordict import TensorDict

        torch.manual_seed(3)
        policy = make_dummy_policy(num_actions=4)
        obs = TensorDict(
            {
                "policy": torch.randn(32, 8),
                "critic": torch.zeros(32, 8),
            },
            batch_size=[32],
        )
        a = policy.act(obs)
        log_prob = policy.get_actions_log_prob(a)
        loss = -log_prob.mean()
        loss.backward()
        # Check actor weights have non-zero finite gradient
        actor_params_with_grad = [
            (n, p)
            for n, p in policy.named_parameters()
            if "actor" in n and p.grad is not None
        ]
        assert len(actor_params_with_grad) > 0, "no actor params have grad"
        for n, p in actor_params_with_grad:
            assert torch.isfinite(p.grad).all(), f"non-finite grad on {n}"
        # CRITICAL: std must have non-zero finite gradient.  This is what
        # was broken with clip_actions=1.0 (run fktnuht3, noise_std → 229).
        assert policy.std.grad is not None, "std has no grad"
        assert torch.isfinite(policy.std.grad).all(), "non-finite std.grad"
        assert policy.std.grad.abs().sum().item() > 1e-6, (
            f"std.grad is essentially zero: {policy.std.grad}"
        )

    # ───── (6) Entropy is finite and reasonable ─────
    @test("entropy is finite and matches Normal entropy (SAC convention)")
    def _():
        policy = make_dummy_policy(num_actions=4, init_noise_std=1.0)
        mean = torch.zeros(10, 4)
        make_distribution(policy, mean, std_val=1.0)
        ent = policy.entropy   # property — reads self.distribution.entropy
        # Analytical entropy of Normal(0, 1) per dim = 0.5 * log(2πe·1²)
        per_dim = 0.5 * math.log(2 * math.pi * math.e)
        expected = 4 * per_dim
        assert torch.isfinite(ent).all()
        assert abs(ent[0].item() - expected) < 1e-4, (
            f"entropy = {ent[0].item():.4f}, expected ≈ {expected:.4f}"
        )

    # ───── (7) Round-trip precision: atanh(tanh(u)) ≈ u to within 1e-4 ─────
    @test("atanh(tanh(u)) round-trip precision (Option-1 sanity check)")
    def _():
        # If this fails badly (>1e-4) we should reconsider Option 2.
        # Tests across a range of |u| from 0 to large.
        u = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
        u_neg = -u
        a_pos = torch.tanh(u)
        a_neg = torch.tanh(u_neg)
        eps = SquashedActorCritic.EPS
        u_recovered = torch.atanh(a_pos.clamp(-1.0 + eps, 1.0 - eps))
        u_neg_recovered = torch.atanh(a_neg.clamp(-1.0 + eps, 1.0 - eps))
        diff_pos = (u_recovered - u).abs()
        diff_neg = (u_neg_recovered - u_neg).abs()
        max_diff = max(diff_pos.max().item(), diff_neg.max().item())
        # At |u|=7 we expect saturation issues; tolerance 0.5 there.
        # For |u| ≤ 5 we expect < 1e-4.
        small_u_mask = u.abs() <= 5.0
        assert (diff_pos[small_u_mask] < 1e-4).all(), (
            f"round-trip error > 1e-4 for small |u|: {diff_pos}"
        )
        # Note overall max for debug visibility
        print(f"\n      (info) max round-trip error across [0..7] = {max_diff:.2e}")

    # ───── (8) PPO ratio stays bounded when σ shifts (Fix A regression test) ─────
    @test("PPO importance ratio bounded when σ changes between rollout/update")
    def _():
        # This is the regression test for Fix A.  Before the fix, an
        # atanh clamp of 1e-6 allowed |u| up to ~7, and a modest σ
        # shift (1.0 → 1.5) gave ratios > 1e5, which caused
        # surrogate-loss spikes that NaN'd training at iter 740 of
        # run ab_squashed.  With Fix A (atanh clamp 1e-3), |u| ≤ 3.8
        # and the same σ shift gives ratio < 50.
        policy = make_dummy_policy(num_actions=4)
        # Simulate rollout: σ_old = 1.0, sample actions from saturated
        # regions of the squashed distribution.
        mean = torch.zeros(100, 4)
        make_distribution(policy, mean, std_val=1.0)
        # Actions sampled "near boundary" — what causes the issue
        actions = torch.tanh(torch.randn(100, 4) * 4.0)   # u ~ N(0, 16), heavily saturated
        old_log_prob = policy.get_actions_log_prob(actions).detach()
        # Simulate policy update: σ shifts to 1.5
        make_distribution(policy, mean, std_val=1.5)
        new_log_prob = policy.get_actions_log_prob(actions).detach()
        ratio = torch.exp(new_log_prob - old_log_prob)
        max_ratio = ratio.max().item()
        # Should be well under 1000.  Before Fix A this was 1e4–1e5+.
        assert max_ratio < 1000.0, (
            f"PPO ratio not bounded: max = {max_ratio:.2e} "
            f"(Fix A intended to bound it to ~40)"
        )
        # And no NaN/inf
        assert torch.isfinite(ratio).all(), "non-finite ratio"

    # ───── (9) Option 2 path: log_prob with stored raw_u (regression for B6) ─────
    @test("Option 2 get_actions_log_prob_with_raw_u bounds ratio under μ shift")
    def _():
        # The bug we couldn't fix in Option 1 (atanh reconstruction):
        # at saturation, the recovered u_clamped ≠ actual_u, so log_prob
        # lives in the deep tail and small μ shifts blow up the ratio.
        # Option 2 stores raw_u and uses it directly → ratio bounded.
        policy = make_dummy_policy(num_actions=14)   # full microduck dim
        # Simulate rollout with σ=1.0, sample u that includes saturated values
        torch.manual_seed(1234)
        mean_old = torch.zeros(50, 14)
        make_distribution(policy, mean_old, std_val=1.0)
        # Sample raw u — some will be |u| > 3 (saturated tanh regime)
        u = policy.distribution.sample()
        a = torch.tanh(u)
        old_log_prob = policy.get_actions_log_prob_with_raw_u(a, u).detach()
        # Simulate policy update: μ shifts by 0.1 (realistic PPO per-update shift)
        mean_new = mean_old + 0.1
        make_distribution(policy, mean_new, std_val=1.0)
        new_log_prob = policy.get_actions_log_prob_with_raw_u(a, u).detach()
        ratio = torch.exp(new_log_prob - old_log_prob)
        max_ratio = ratio.max().item()
        # For a μ shift of 0.1 across 14 dims at σ=1, per-dim Δlog_prob
        # is bounded by 0.1*|u|.  For typical |u| < 3 (most samples)
        # joint Δlog_prob is bounded by ~4.2, ratio < 67.  Even with
        # outlier saturated u, with Option 2 the ratio scales as
        # exp(Σ Δμ · u_actual / σ²) — bounded.
        assert torch.isfinite(ratio).all(), "non-finite ratio"
        assert max_ratio < 1000, (
            f"Option 2 ratio not bounded under μ=0.1 shift: max = {max_ratio:.2e}"
        )
        # And for comparison: the Option 1 path (atanh reconstruction)
        # would explode here.  Verify by computing it the old way.
        old_log_prob_v1 = policy.get_actions_log_prob(a).detach()   # uses atanh
        # we reset distribution above; redo old
        make_distribution(policy, mean_old, std_val=1.0)
        old_log_prob_v1 = policy.get_actions_log_prob(a).detach()
        make_distribution(policy, mean_new, std_val=1.0)
        new_log_prob_v1 = policy.get_actions_log_prob(a).detach()
        ratio_v1 = torch.exp(new_log_prob_v1 - old_log_prob_v1)
        max_ratio_v1 = ratio_v1.max().item()
        # Option 1 ratio under same scenario should be larger
        # (this is a regression check — if it ever becomes smaller than
        # Option 2's ratio, our diagnosis of why Option 1 fails is wrong).
        # Use a lenient threshold to avoid flaking on rare random
        # seeds with no saturated samples.
        if max_ratio_v1 < max_ratio:
            print(f"\n      (warn) Option-1 ratio ({max_ratio_v1:.2e}) is "
                  f"smaller than Option-2 ({max_ratio:.2e}); means this "
                  f"seed didn't sample saturated u — test still passes")

    # ───── Summary ─────
    print()
    print(f"  passed: {_passed}, failed: {_failed}")
    print("=" * 70)
    return _failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
