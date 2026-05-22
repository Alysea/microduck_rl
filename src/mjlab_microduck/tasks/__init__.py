import os
import sys
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}

from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .testbench_env_cfg import (
    make_testbench_env_cfg,
    MicroduckTestbenchRlCfg,
)
from .slip_biped_balance_env_cfg import (
    make_slip_biped_balance_env_cfg,
    SlipBipedBalanceRlCfg,
)
from .slip_biped_balance_3d_env_cfg import (
    make_slip_biped_balance_3d_env_cfg,
    SlipBipedBalance3DRlCfg,
)
from .slip_biped_hop_3d_env_cfg import (
    make_slip_biped_hop_3d_env_cfg,
    SlipBipedHop3DRlCfg,
)
from .slip_biped_velocity_3d_env_cfg import (
    make_slip_biped_velocity_3d_env_cfg,
    SlipBipedVelocity3DRlCfg,
)
from .slip_biped_run_3d_env_cfg import (
    make_slip_biped_run_3d_env_cfg,
    SlipBipedRun3DRlCfg,
)
from .microduck_hop_sprung_env_cfg import (
    make_microduck_hop_sprung_env_cfg,
    MicroduckHopSprungRlCfg,
)
from .microduck_hop_forward_sprung_env_cfg import (
    make_microduck_hop_forward_sprung_env_cfg,
    MicroduckHopForwardSprungRlCfg,
)
from .microduck_velocity_sprung_env_cfg import (
    make_microduck_velocity_sprung_env_cfg,
    MicroduckVelocitySprungRlCfg,
)

def _make_roller_get_base_metadata():
    """Return a get_base_metadata replacement that skips joints with no actuator.

    Needed for the roller skate robot where passive wheel joints exist as DOFs
    but have no position actuators, causing a KeyError in the stock implementation.
    """
    import torch
    from mjlab.envs.mdp.actions import JointPositionAction

    def roller_get_base_metadata(env, run_path):
        robot = env.scene["robot"]
        joint_action = env.action_manager.get_term("joint_pos")
        assert isinstance(joint_action, JointPositionAction)

        joint_name_to_ctrl_id = {
            act.target.split("/")[-1]: act.id
            for act in robot.spec.actuators
        }

        # Filter to joints that have an actuator, preserving natural order
        all_names = list(robot.joint_names)
        actuated_idx = [i for i, n in enumerate(all_names) if n in joint_name_to_ctrl_id]
        joint_names = [all_names[i] for i in actuated_idx]
        ctrl_ids = [joint_name_to_ctrl_id[n] for n in joint_names]

        joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids, 0]
        joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]

        return {
            "run_path": run_path,
            "joint_names": joint_names,
            "joint_stiffness": joint_stiffness.tolist(),
            "joint_damping": joint_damping.tolist(),
            "default_joint_pos": robot.data.default_joint_pos[0][actuated_idx].cpu().tolist(),
            "command_names": list(env.command_manager.active_terms),
            "observation_names": env.observation_manager.active_terms["policy"],
            "action_scale": joint_action._scale[0].cpu().tolist()
            if isinstance(joint_action._scale, torch.Tensor)
            else joint_action._scale,
        }

    return roller_get_base_metadata


class MicroduckRollersOnPolicyRunner(MicroduckOnPolicyRunner):
    """Runner for the roller skate task.

    Overrides save() to patch get_base_metadata in the velocity exporter module
    for the duration of the call, filtering out passive wheel joints that have no
    actuators and would cause a KeyError in the stock implementation.
    """

    def save(self, path, *args, **kwargs):
        import mjlab.tasks.velocity.rl.runner as _vel_runner
        orig = _vel_runner.get_base_metadata
        _vel_runner.get_base_metadata = _make_roller_get_base_metadata()
        try:
            super().save(path, *args, **kwargs)
        finally:
            _vel_runner.get_base_metadata = orig

# Roller skate velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckRollersOnPolicyRunner,
)
print("✓ Rollers task registered: Mjlab-Velocity-Flat-MicroDuck-Rollers")

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ StandUp task registered: Mjlab-StandUp-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ StandUp task registered: Mjlab-StandUp-Rough-MicroDuck")

# Ground pick task — episodic policy: crouch, touch ground with mouth, return to standing
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Ground pick task registered: Mjlab-GroundPick-Flat-MicroDuck")

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Ground pick task registered: Mjlab-GroundPick-Rough-MicroDuck")

# XL330 test-bench task — single-DOF sim2real validation rig
register_mjlab_task(
    task_id="Mjlab-Testbench-XL330",
    env_cfg=make_testbench_env_cfg(),
    play_env_cfg=make_testbench_env_cfg(play=True),
    rl_cfg=MicroduckTestbenchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
print("✓ Testbench task registered: Mjlab-Testbench-XL330")

# SLIP biped 2D toy — first RL experiment on the sprung-leg model
register_mjlab_task(
    task_id="Mjlab-Balance-SlipBiped-2D",
    env_cfg=make_slip_biped_balance_env_cfg(),
    play_env_cfg=make_slip_biped_balance_env_cfg(play=True),
    rl_cfg=SlipBipedBalanceRlCfg,
    runner_cls=None,  # default OnPolicyRunner is fine — no symmetry
)
print("✓ SLIP biped balance task registered: Mjlab-Balance-SlipBiped-2D")

# SLIP biped 3D — same don't-fall task, free joint + hip_roll added
register_mjlab_task(
    task_id="Mjlab-Balance-SlipBiped-3D",
    env_cfg=make_slip_biped_balance_3d_env_cfg(),
    play_env_cfg=make_slip_biped_balance_3d_env_cfg(play=True),
    rl_cfg=SlipBipedBalance3DRlCfg,
    runner_cls=None,
)
print("✓ SLIP biped balance task registered: Mjlab-Balance-SlipBiped-3D")

# SLIP biped 3D — hop in place (bounce vertically without horizontal drift)
register_mjlab_task(
    task_id="Mjlab-Hop-SlipBiped-3D",
    env_cfg=make_slip_biped_hop_3d_env_cfg(),
    play_env_cfg=make_slip_biped_hop_3d_env_cfg(play=True),
    rl_cfg=SlipBipedHop3DRlCfg,
    runner_cls=None,
)
print("✓ SLIP biped hop task registered: Mjlab-Hop-SlipBiped-3D")

# SLIP biped 3D — forward velocity tracking (first locomotion task)
register_mjlab_task(
    task_id="Mjlab-Velocity-SlipBiped-3D",
    env_cfg=make_slip_biped_velocity_3d_env_cfg(),
    play_env_cfg=make_slip_biped_velocity_3d_env_cfg(play=True),
    rl_cfg=SlipBipedVelocity3DRlCfg,
    runner_cls=None,
)
print("✓ SLIP biped velocity task registered: Mjlab-Velocity-SlipBiped-3D")

# SLIP biped 3D — running (fast forward velocity, airborne phases encouraged)
register_mjlab_task(
    task_id="Mjlab-Run-SlipBiped-3D",
    env_cfg=make_slip_biped_run_3d_env_cfg(),
    play_env_cfg=make_slip_biped_run_3d_env_cfg(play=True),
    rl_cfg=SlipBipedRun3DRlCfg,
    runner_cls=None,
)
print("✓ SLIP biped run task registered: Mjlab-Run-SlipBiped-3D")

# MicroDuck sprung-shank prototype: hop in place
register_mjlab_task(
    task_id="Mjlab-Hop-MicroDuck-Sprung",
    env_cfg=make_microduck_hop_sprung_env_cfg(),
    play_env_cfg=make_microduck_hop_sprung_env_cfg(play=True),
    rl_cfg=MicroduckHopSprungRlCfg,
    runner_cls=None,
)
print("✓ MicroDuck sprung hop task registered: Mjlab-Hop-MicroDuck-Sprung")

# MicroDuck sprung-shank prototype: hop forward at commanded velocity
register_mjlab_task(
    task_id="Mjlab-HopForward-MicroDuck-Sprung",
    env_cfg=make_microduck_hop_forward_sprung_env_cfg(),
    play_env_cfg=make_microduck_hop_forward_sprung_env_cfg(play=True),
    rl_cfg=MicroduckHopForwardSprungRlCfg,
    runner_cls=None,
)
print("✓ MicroDuck sprung hop-forward task registered: Mjlab-HopForward-MicroDuck-Sprung")

# MicroDuck sprung-shank velocity task — built on the stock velocity env
# template with feet_air_time + foot-quality rewards (same recipe as the
# rigid microduck task). Encourages alternating walking/running gait.
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Sprung",
    env_cfg=make_microduck_velocity_sprung_env_cfg(),
    play_env_cfg=make_microduck_velocity_sprung_env_cfg(play=True),
    rl_cfg=MicroduckVelocitySprungRlCfg,
    runner_cls=None,
)
print("✓ MicroDuck sprung velocity task registered: Mjlab-Velocity-Flat-MicroDuck-Sprung")
