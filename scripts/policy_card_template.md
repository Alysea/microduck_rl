---
library_name: onnx
pipeline_tag: reinforcement-learning
tags: [{{TAGS}}]
license: apache-2.0
---

# {{NAME}} — a Microduck policy

{{DESCRIPTION}}

`policy.onnx`: `obs[1,61] f32 → actions[1,14] f32`, observation normalizer baked in, 50 Hz, joint-position
targets around HOME (action scale {{ACTION_SCALE}}). Kind: **{{KIND}}**. Entry pose: **{{ENTRY_POSE}}**.
Shared in the [Microduck policy format](https://github.com/pollen-robotics/microduck_rl/blob/main/docs/sharing-policies.md)
(`manifest.json` is the machine-readable version of this card).

## Command

The last 13 observation values are the command block `[twist(3), head(4), body(6)]`. This policy reads:

| slot | meaning |
|---|---|
{{TWIST_ROWS}}
| head (4) | {{HEAD}} |
| body (6) | {{BODY}} |

Idle / rest command: twist = `{{IDLE}}`.

{{EXTRA}}

## On the robot

Point a `[policy]` role in `deploy/robotd.toml` at the downloaded file (absolute path) and restart `robotd`;
a file that fails to load is reported as `policy unavailable: …` in `robot.health` while the loop keeps holding
its pose. The daemon must write the command slots above for this policy to do anything — the runtime needs a
role that knows this command scheme (see the daemon's policy roles). Start it from the entry pose.

## Provenance

- task: `{{TASK_ID}}` in [{{TRAIN_REPO}}](https://github.com/{{TRAIN_REPO}}) @ `{{COMMIT}}`
- training run: `{{RUN}}` (checkpoints + logs stay in the private training repo)
- exported with `scripts/export.py` (bakes the obs normalizer); validated with `scripts/publish_policy.py validate`

## Files

- `policy.onnx` — the policy
- `manifest.json` — schema v2: contract, command, provenance, eval
{{MEDIA_LINE}}
