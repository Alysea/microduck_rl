"""Custom policy classes for mjlab_microduck training.

The full Option 2 stack for stable PPO + tanh squashing on high-dim
actions consists of three classes that must be used together:

- :class:`SquashedActorCritic` — tanh-squashed Gaussian actor with
  Jacobian-corrected log-probability.  Drop-in replacement for
  rsl_rl's stock :class:`ActorCritic`.
- :class:`SquashedRolloutStorage` — extends ``RolloutStorage`` with a
  ``raw_actions`` buffer that stores the pre-tanh sample ``u`` so the
  PPO update can re-evaluate log_prob at the exact sample (avoiding
  the atanh-reconstruction info loss at saturation).
- :class:`SquashedPPO` — wires the above together: replaces stock
  storage with the squashed variant, threads ``raw_actions`` through
  the rollout/storage/update pipeline.

Wire them into a task via the RslRl config:

    policy.class_name    = "mjlab_microduck.policy:SquashedActorCritic"
    algorithm.class_name = "mjlab_microduck.policy:SquashedPPO"

The trio fixes BOTH the clip-bias pathology (Phase B Step B1) AND the
saturated-tanh ratio explosion (Phase B Step B5 fix).
"""

from mjlab_microduck.policy.squashed_actor_critic import (
    SquashedActorCritic,
)
from mjlab_microduck.policy.squashed_rollout_storage import (
    SquashedRolloutStorage,
)
from mjlab_microduck.policy.squashed_ppo import (
    SquashedPPO,
)

__all__ = [
    "SquashedActorCritic",
    "SquashedRolloutStorage",
    "SquashedPPO",
]
