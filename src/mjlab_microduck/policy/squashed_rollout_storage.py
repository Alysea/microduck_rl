"""Rollout storage that carries the pre-squashed Normal sample ``u``.

Extends ``rsl_rl.storage.RolloutStorage`` to store the raw, un-squashed
action ``u`` (Normal sample) alongside the squashed action ``a = tanh(u)``
that goes to the environment.  This is *Option 2* in the
``squashed_actor_critic.py`` header.

Why this exists: at PPO update time, the importance ratio
``exp(log_prob_new − log_prob_old)`` must be computed at the *exact*
``u`` that was sampled during rollout.  The naïve approach (Option 1)
reconstructs ``u`` via ``atanh(a)``, but when ``|u|`` is large the
``tanh`` is saturated and the round-trip loses information — the
recovered ``u`` is clamped, while the actor's actual decision was at
a much larger value.  PPO then mis-attributes log_prob and the ratio
explodes when ``μ`` shifts between rollout and update (observed in
runs ab_squashed and ab_squashed_v2: ratios of 1e6–1e12, NaN'd
training within ~50–700 iterations).

By storing ``u`` directly, both old and new log_probs are evaluated at
the same ``u`` value, and the ratio is bounded by the ``μ`` shift —
which PPO's KL constraint keeps small per update.

This file only adds the storage layer; the consumer side (PPO update
loop) is in ``squashed_ppo.py``.
"""

from __future__ import annotations

from collections.abc import Generator

import torch
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage


class SquashedRolloutStorage(RolloutStorage):
    """RolloutStorage that also stores ``raw_actions`` = pre-tanh ``u``.

    Parallel buffer to ``self.actions``: same shape, same indexing.  When
    the mini-batch generator shuffles by an index permutation, both
    ``actions`` and ``raw_actions`` shuffle by the same indices, so
    alignment is automatic.
    """

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        super().__init__(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, device
        )
        # Parallel buffer to self.actions, holds the un-squashed u.
        # Only meaningful for RL training (PPO).  We allocate
        # unconditionally; cost is one extra (T, N, A) tensor.
        self.raw_actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Copy ``raw_actions`` from the Transition before delegating.

        The Transition object is duck-typed — base class expects
        ``actions``, ``rewards``, etc.  We additionally require
        ``raw_actions`` (set by SquashedPPO.act() from
        ``policy._last_raw_u``).  Raises informative error if missing.
        """
        if not hasattr(transition, "raw_actions") or transition.raw_actions is None:
            raise AttributeError(
                "SquashedRolloutStorage.add_transition: transition is missing "
                "`raw_actions`.  This storage requires SquashedPPO (which sets "
                "transition.raw_actions in act()).  If you see this from stock "
                "PPO, the algorithm class name in the cfg is wrong."
            )
        # IMPORTANT: copy raw_actions to slot self.step BEFORE super()
        # increments the counter.
        self.raw_actions[self.step].copy_(transition.raw_actions)
        super().add_transition(transition)

    def squashed_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator:
        """Generator that yields ``raw_actions_batch`` alongside the usual tuple.

        Yields the same 10-tuple as the parent ``mini_batch_generator``
        but with ``raw_actions_batch`` *appended at the end* (position 10).
        SquashedPPO.update() consumes the 11-tuple; stock PPO would only
        unpack the first 10 and would silently ignore the extra (or
        actually it would raise an unpack-size error — we don't use
        stock PPO with this storage, so we never test that).
        """
        if self.training_type != "rl":
            raise ValueError(
                "squashed_mini_batch_generator is only available for RL training."
            )
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        # Same flattens as parent:
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        raw_actions = self.raw_actions.flatten(0, 1)   # NEW: parallel to actions
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield (
                    observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),         # hidden_state_a/c (recurrent unused here)
                    None,                 # masks_batch
                    raw_actions[batch_idx],   # NEW: extra element at the end
                )
