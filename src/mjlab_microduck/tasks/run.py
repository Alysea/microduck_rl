"""Run task variant — push the velocity task toward an ALTERNATING running gait.

``make_run_variant(cfg)`` turns any microduck velocity-family env cfg into its
running counterpart, in the same shape as ``tasks/backlash.py``. Kept as a
transform rather than a new env cfg so it composes: the sprung phase becomes
``make_sprung_variant(make_run_variant(cfg))`` instead of a fourth copy of the
velocity env — the duplication that stranded the previous campaign.

Six changes:

1. Activate the posture running regime. ``variable_posture`` gates on
   ``|lin| + |ang|`` with ``running_threshold`` defaulting to 1.5, which the
   stock command ranges only reach with both maxed — so the regime is dead code
   today, and ``std_running`` is aliased to ``std_walking`` anyway.
2. Swap ``air_time`` to ``feet_air_time_capped`` and shorten its window. The
   stock reward pays double for simultaneous two-foot flight, which rewards the
   bouncing gait; the stock window (0.10-0.25 s) was tuned to slow the gait.
3. Add ``alternating_flight``, which rewards flight only when the feet are
   genuinely alternating.
4. Add ``action_magnitude_monitor`` (zero contribution, non-zero weight).
5. Add ``forward_speed_monitor`` (zero contribution, non-zero weight) — the
   spec's success criterion is a *measured forward-speed plateau*, and the
   pre-existing ``error_vel_xy`` is isotropic so it cannot isolate forward
   tracking.
6. Retarget the speed curriculum: ramp forward speed only. ``forward_only=True``
   makes ``lin_vel_x`` grow as ``(0, range)`` instead of ``(-range, range)``,
   and ``update_lin_vel_y=False`` leaves the lateral range pinned at the
   velocity env's base value. ``ang_vel_range`` is held constant across stages,
   so forward speed is the single moving variable and the plateau measurement is
   a forward-speed number rather than an isotropic xy error.
"""

from copy import deepcopy
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg

SENSOR_NAME = "feet_ground_contact"

# Posture tolerances for the running regime. Looser than walking on the joints
# that must swing, but hip_roll is deliberately UNCHANGED — loosening roll is
# what produced leg splay (see the tuning notes in microduck_velocity_env_cfg.py
# lines 168 and 177).
STD_RUNNING = {
    r".*hip_yaw.*": 0.5,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.8,
    r".*knee.*": 0.8,
    r".*ankle.*": 0.5,
}

# Commanded-speed threshold above which the running posture regime engages.
#
# IMPORTANT: mjlab's `variable_posture` gates on the MIXED total
# `|lin| + |ang|`, not on linear speed alone. `ang_vel_range` is held at 1.0
# through every curriculum stage, so a threshold at or below 1.0 would be
# reachable by YAW ALONE — a spin-in-place command at zero linear velocity would
# have been granted the loose hip_pitch/knee tolerance meant for running.
# 1.2 sits above the max |ang| of 1.0, so yaw can no longer reach it by itself.
#
# Consequence: with |ang| up to 1.0 available, the regime engages once
# `lin_vel_range` reaches 0.2 in the worst case, but for a purely forward
# command it needs lin >= 1.2 — i.e. only the last two curriculum stages
# (1.2 and 1.5) can trigger it on forward speed alone. Provisional; revisit once
# the plateau is measured.
RUNNING_THRESHOLD = 1.2

# Swing-time window. Stock is (0.10, 0.25), explicitly raised to slow the gait
# down; running needs faster strides.
AIR_TIME_WINDOW = (0.05, 0.15)

# Steps are env steps (iteration * num_steps_per_env=24).
VELOCITY_STAGES = [
    {"step": 0,         "lin_vel_range": 0.5, "ang_vel_range": 1.0},
    {"step": 1000 * 24, "lin_vel_range": 0.7, "ang_vel_range": 1.0},
    {"step": 2000 * 24, "lin_vel_range": 0.9, "ang_vel_range": 1.0},
    {"step": 3000 * 24, "lin_vel_range": 1.2, "ang_vel_range": 1.0},
    {"step": 4000 * 24, "lin_vel_range": 1.5, "ang_vel_range": 1.0},
]

ALTERNATING_FLIGHT_WEIGHT = 3.0


def make_run_variant(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
    """Convert a microduck velocity-family env cfg into the Run task."""
    # 1. Posture: activate the running regime with its own tolerances.
    pose = cfg.rewards["pose"]
    pose.params["std_running"] = dict(STD_RUNNING)
    pose.params["running_threshold"] = RUNNING_THRESHOLD

    # 2. Air time: stop paying double for simultaneous two-foot flight, and
    #    shorten the swing window. Params are unchanged — the capped function is
    #    deliberately signature-compatible with the stock one, so `command_name`
    #    and `command_threshold` survive the `.func` swap and the speed gate
    #    keeps working.
    air = cfg.rewards["air_time"]
    air.func = microduck_mdp.feet_air_time_capped
    air.params["sensor_name"] = SENSOR_NAME
    air.params["threshold_min"] = AIR_TIME_WINDOW[0]
    air.params["threshold_max"] = AIR_TIME_WINDOW[1]

    # 3. Reward genuinely alternating flight.
    cfg.rewards["alternating_flight"] = RewardTermCfg(
        func=microduck_mdp.alternating_flight,
        weight=ALTERNATING_FLIGHT_WEIGHT,
        params={
            "sensor_name": SENSOR_NAME,
            "command_name": "twist",
            "command_threshold": 0.01,
        },
    )

    # 4. Action-magnitude watchdog. Returns zeros, so the weight only has to be
    #    non-zero for RewardManager.compute to call it at all.
    cfg.rewards["action_magnitude_monitor"] = RewardTermCfg(
        func=microduck_mdp.action_magnitude_monitor,
        weight=1.0,
        params={},
    )

    # 5. Forward-speed metric. Also zero-contribution, same non-zero-weight
    #    requirement. This is the number the sprung phase is compared against.
    cfg.rewards["forward_speed_monitor"] = RewardTermCfg(
        func=microduck_mdp.forward_speed_monitor,
        weight=1.0,
        params={},
    )

    # 6. Speed curriculum — forward speed is the ONLY moving variable.
    #    `velocity_command_ranges_curriculum` defaults to `forward_only=False`
    #    and `update_lin_vel_y=True`, which would ramp backward and lateral
    #    speed alongside forward speed (the last stage would set BOTH
    #    lin_vel_x=(-1.5, 1.5) and lin_vel_y=(-1.5, 1.5)). `update_ang_vel_z` is
    #    left at its default because `ang_vel_range` is constant across stages
    #    anyway.
    curriculum_params = cfg.curriculum["velocity_command_ranges"].params
    curriculum_params["velocity_stages"] = [dict(stage) for stage in VELOCITY_STAGES]
    curriculum_params["forward_only"] = True
    curriculum_params["update_lin_vel_y"] = False

    return cfg


# Same hyperparameters as the velocity task — Phase 1 changes the task, not the
# learner. Only the logging identity differs, so the baseline and the later
# sprung runs land in separate wandb groups.
#
# The nested `actor` / `critic` / `algorithm` cfgs are DEEP-COPIED. `replace` is
# shallow, so without this they would be the *same objects* as
# `MicroduckRlCfg`'s, and the Phase 3 escape hatch — swapping the policy
# distribution via `actor.distribution_cfg["class_name"]` — would silently
# mutate the Velocity task too, destroying the experimental control this
# baseline exists to provide. Values are identical; only identity differs.
MicroduckRunRlCfg = replace(
    MicroduckRlCfg,
    actor=deepcopy(MicroduckRlCfg.actor),
    critic=deepcopy(MicroduckRlCfg.critic),
    algorithm=deepcopy(MicroduckRlCfg.algorithm),
    experiment_name="run",
    run_name="run",
)
