"""Run task variant — push the velocity task toward an ALTERNATING running gait.

``make_run_variant(cfg)`` turns any microduck velocity-family env cfg into its
running counterpart, in the same shape as ``tasks/backlash.py``. Kept as a
transform rather than a new env cfg so it composes: the sprung phase becomes
``make_sprung_variant(make_run_variant(cfg))`` instead of a fourth copy of the
velocity env — the duplication that stranded the previous campaign.

Four changes:

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

The speed curriculum ramps ``lin_vel_range`` only; ``ang_vel_range`` is held so
forward speed is the single moving variable.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp

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

# Total commanded speed (|lin| + |ang|) above which the running posture regime
# engages. Provisional — revisit once the plateau is measured.
RUNNING_THRESHOLD = 0.6

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
    #    deliberately signature-compatible with the stock one.
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

    # 5. Speed curriculum.
    cfg.curriculum["velocity_command_ranges"].params["velocity_stages"] = [
        dict(stage) for stage in VELOCITY_STAGES
    ]

    return cfg
