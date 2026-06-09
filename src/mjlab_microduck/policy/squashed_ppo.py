"""PPO subclass for the tanh-squashed actor.

Threads the pre-tanh sample ``u`` through rollout → storage → update,
so that the importance ratio is computed at the *actual* ``u`` that was
sampled, not at an atanh-reconstructed approximation.

Three overrides:
- ``__init__``: replaces the inherited stock ``RolloutStorage`` with
  :class:`SquashedRolloutStorage` (which has a ``raw_actions`` buffer).
- ``act``: captures ``policy._last_raw_u`` into ``transition.raw_actions``.
- ``update``: iterates the squashed mini-batch generator (which yields
  ``raw_actions_batch`` as an extra element), and routes through
  :meth:`SquashedActorCritic.get_actions_log_prob_with_raw_u` instead
  of the atanh-reconstructing default.

Everything else (advantage normalization, value-loss clipping, KL
adaptation, entropy bonus, symmetry, RND) is inherited from PPO.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO

from mjlab_microduck.policy.squashed_rollout_storage import SquashedRolloutStorage


class SquashedPPO(PPO):
    """PPO that uses raw ``u`` (not atanh-reconstructed) for log-prob.

    Designed to pair with :class:`SquashedActorCritic` from
    ``squashed_actor_critic.py``.  Will fail at runtime if the policy
    doesn't expose ``_last_raw_u`` after :meth:`act`.
    """

    def __init__(self, policy, storage, *args: Any, **kwargs: Any) -> None:
        # Stock OnPolicyRunner constructs a `RolloutStorage` and passes
        # it in.  Replace it with our subclass that carries raw_actions.
        # We reconstruct it from the stock storage's metadata.
        if not isinstance(storage, SquashedRolloutStorage):
            squashed_storage = SquashedRolloutStorage(
                training_type=storage.training_type,
                num_envs=storage.num_envs,
                num_transitions_per_env=storage.num_transitions_per_env,
                obs=storage.observations[0],   # any single slice has the right shape
                actions_shape=storage.actions_shape,
                device=storage.device,
            )
            storage = squashed_storage
        super().__init__(policy, storage, *args, **kwargs)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Same as parent, plus stashing ``raw_actions`` into the transition.

        ``self.policy.act(obs)`` is a side-effect call that sets
        ``self.distribution = Normal(μ, σ)`` and returns ``tanh(u)``.
        Our :class:`SquashedActorCritic` additionally caches the raw
        sample ``u`` in ``policy._last_raw_u`` (same shape as the
        returned action) — we copy that into the transition so
        :class:`SquashedRolloutStorage` can persist it.
        """
        # Parent's act() does this whole flow.  We re-implement instead
        # of calling super() because we need to slot raw_actions into
        # self.transition before storage takes a snapshot.  Otherwise we
        # would have a race: super().act() sets actions/values/log_prob
        # but not raw_actions, and any storage.add_transition call
        # between act() and process_env_step would fail.
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = self.policy.act(obs).detach()
        # Capture the raw u that policy.act() just stashed.  We .detach()
        # for consistency with `actions` — these are training targets,
        # not graph nodes.
        assert hasattr(self.policy, "_last_raw_u"), (
            "SquashedPPO requires a policy that caches `_last_raw_u` in "
            "act() — i.e. SquashedActorCritic.  Got: "
            f"{type(self.policy).__name__}"
        )
        self.transition.raw_actions = self.policy._last_raw_u.detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        # log_prob at rollout time uses the actual sampled u directly
        # (no atanh reconstruction needed since we still have u).
        self.transition.actions_log_prob = (
            self.policy.get_actions_log_prob_with_raw_u(
                self.transition.actions, self.transition.raw_actions
            ).detach()
        )
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def update(self) -> dict[str, float]:
        """Same as parent's update(), but threading raw_actions_batch through.

        Implementation note: we copy the parent's body verbatim except for:
        (a) iterating ``storage.squashed_mini_batch_generator`` instead
            of ``storage.mini_batch_generator``;
        (b) unpacking the extra ``raw_actions_batch`` from the yield;
        (c) computing ``actions_log_prob_batch`` via
            ``get_actions_log_prob_with_raw_u(actions, raw_actions)``
            instead of the atanh-reconstructing default.

        Everything else — value loss, surrogate clipping, entropy
        bonus, KL-adaptive LR, symmetry, RND, multi-GPU sync, gradient
        clipping — is identical.
        """
        # Recurrent path not supported (no use case for us, and the
        # recurrent mini-batch generator would also need raw_actions).
        # Fail loudly if someone tries.
        if self.policy.is_recurrent:
            raise NotImplementedError(
                "SquashedPPO doesn't support recurrent policies yet.  "
                "Override recurrent_mini_batch_generator on the storage "
                "and add raw_actions to its yield, then remove this check."
            )

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        # MODIFIED: use the squashed generator (yields raw_actions_batch
        # appended after masks_batch).
        generator = self.storage.squashed_mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
            raw_actions_batch,        # MODIFIED: extra element from squashed generator
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std() + 1e-8
                    )

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Symmetry augmentation: replicate raw_actions_batch alongside
                # actions_batch so the indices stay aligned.  Currently
                # symmetry is disabled for us (ENABLE_SYMMETRY=False), but
                # implement defensively.
                raise NotImplementedError(
                    "SquashedPPO + symmetry data augmentation not implemented. "
                    "Disable symmetry or extend SquashedPPO.update() to "
                    "augment raw_actions_batch in lockstep with actions_batch."
                )

            # Recompute actions log prob and entropy under the CURRENT policy
            self.policy.act(
                obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0]
            )
            # MODIFIED: use raw_actions_batch directly instead of atanh-recovering u
            actions_log_prob_batch = self.policy.get_actions_log_prob_with_raw_u(
                actions_batch, raw_actions_batch
            )
            value_batch = self.policy.evaluate(
                obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1]
            )
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL adaptation — unchanged
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (
                            torch.square(old_sigma_batch)
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(
                            kl_mean, op=torch.distributed.ReduceOp.SUM
                        )
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss — unchanged
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value loss — unchanged
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Symmetry loss (skipped when symmetry is disabled — we
            # already raised NotImplementedError above if it's on with
            # data augmentation, but the non-augmentation path is still
            # legal in stock PPO.  Replicate it minimally.)
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch_sym, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"]
                    )
                    num_aug = int(obs_batch_sym.batch_size[0] / original_batch_size)
                    mean_actions_batch = self.policy.act_inference(
                        obs_batch_sym.detach().clone()
                    )
                    action_mean_orig = mean_actions_batch[:original_batch_size]
                    _, actions_mean_symm_batch = data_augmentation_func(
                        obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                    )
                    mse_loss = torch.nn.MSELoss()
                    symmetry_loss = mse_loss(
                        mean_actions_batch[original_batch_size:],
                        actions_mean_symm_batch.detach()[original_batch_size:],
                    )
                    if self.symmetry["use_mirror_loss"]:
                        loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                    else:
                        symmetry_loss = symmetry_loss.detach()

            # RND loss — unchanged
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(
                        obs_batch[:original_batch_size]
                    )
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Optimization step — unchanged
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        return loss_dict
