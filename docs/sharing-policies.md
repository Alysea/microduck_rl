# Sharing Microduck policies on the Hugging Face Hub

A policy is shared as **one public model repo per policy** holding the exported ONNX, a
machine-readable manifest and a model card. Anyone — the Pollen team or the community — can
publish under their own namespace; the Hub tag `microduck-policy` is the registry.

```
<namespace>/microduck-<name>          # model repo, public by default
├── policy.onnx                       # obs[1,61] f32 → actions[1,14] f32, obs normalizer baked in
├── manifest.json                     # schema below — what the policy is, what it reads, where it came from
├── README.md                         # model card (front-matter tags + the sections below)
└── media/preview.mp4                 # optional, one short sim clip
```

Publish with `scripts/publish_policy.py` — it validates the ONNX against the daemon contract,
writes the manifest and the card from `scripts/policy_card_template.md`, and uploads. Never
publish checkpoints (`model_*.pt`), hand-converted ONNX, or source tarballs into a policy repo;
those stay in the private training-run repos.

## Find and fetch

```bash
# every shared policy, any namespace
hf models ls --filter microduck-policy          # or: HfApi().list_models(filter="microduck-policy")
# get one
uv run scripts/publish_policy.py fetch RemiFabre/microduck-flamingo-cycle --to /tmp/flamingo
# → /tmp/flamingo/policy.onnx (re-validated) + manifest.json; then:
uv run scripts/infer_policy.py --walking /tmp/flamingo/policy.onnx
```

## The contract (what the robot daemon accepts)

Enforced by `robotd` at load (`duck-control/src/policy.rs`) and by `publish_policy.py validate`:

- input tensor named `obs`, f32, trailing dim **61**; first output f32, trailing dim **14**
- observation normalizer baked in (`scripts/export.py` does this; in-sim play hides its absence)
- obs layout: 48 proprioception + 13-D command `[twist(3), head_pose(4), body_pose(6)]`
- the daemon only feeds command slots twist 48–50, head 51–54, body z/roll/pitch 57–59;
  body x/y (55, 56) and body yaw (60) are always 0 on the robot
- actions are joint-position targets around HOME, unfiltered, 50 Hz

## `manifest.json` (schema_version 2)

Superset of the studio's schema v1 (`microduck_rl_studio/rl_space/contract.py`), adding the
command contract and provenance. Extend it **with the daemon team**, not unilaterally.

| field | meaning |
|---|---|
| `schema_version` | 2 |
| `model_api` | 1 (the 61/14 contract) |
| `name` | short slug, same as the repo suffix |
| `kind` | `perpetual` (runs until told otherwise: walking, standing, flamingo) or `episodic` (runs `duration_s` then returns: kick, roulade) |
| `obs_len`, `action_len` | 61, 14 |
| `action_scale` | multiplier the daemon applies to the network output (1.0 for the current envs) |
| `entry_pose` | state the robot must be in when the policy takes over (`standing`, `sitting`, `lying_face_down`, …) |
| `duration_s` | episodic only |
| `command.twist` | 3 strings: meaning of each twist slot (`"unused"` if ignored) |
| `command.head`, `command.body` | `"unused (zeros)"` or the meaning |
| `command.idle` | the 3 twist values that mean "do nothing" — the daemon's rest state |
| `robot` | `{model, hw_rev, servos, control_hz}` |
| `training` | `{task_id, repo, commit, run}` — enough to retrain or continue |
| `eval` | free-form: what was measured in sim and the known limits |
| `description` | one sentence |

## The card (`README.md`)

Front-matter (this is what makes the policy findable):

```yaml
---
library_name: onnx
pipeline_tag: reinforcement-learning
tags: [microduck, microduck-policy, mjlab, robotics]
license: apache-2.0
---
```

Sections, in this order: **What it does** · **Command** (a table of the twist/head/body slots) ·
**Known limits** (honest, measured: "falls on backward pushes ≥ 0.18 m/s") · **Try it in sim** ·
**On the robot** (`deploy/robotd.toml` `[policy]` path, `entry_pose`) · **Provenance** (task id,
training repo + commit, run) · **Files**.

## Publishing

```bash
cd microduck_rl
uv run scripts/publish_policy.py validate /path/to/policy.onnx
uv run scripts/publish_policy.py publish /path/to/policy.onnx \
    --name flamingo-cycle --namespace RemiFabre \
    --manifest /path/to/manifest.json \            # the fields above; the script fills obs/action lens, checks the rest
    --card-extra /path/to/card_body.md \           # your "What it does / Command / Limits / …" sections
    --media /path/to/preview.mp4 \
    --public                                       # default is private; --public is the intended setting for sharing
```

The script refuses to publish an ONNX that fails the contract, and it never uploads anything
but `policy.onnx`, `manifest.json`, `README.md` and `media/*`.

## On the robot

The daemon does not read the manifest yet: point a `[policy]` role in `deploy/robotd.toml` at the
downloaded `policy.onnx` (absolute path) and restart `robotd`. The `[policy]` roles are what the
daemon knows how to drive (walk, stand, sitstand, ground_pick, kicks, roulade …); a policy with a
new command scheme needs a matching role in the daemon. The updater's planned `model` component
(`microduck/policies/README.md`) is meant to install exactly these repos.
