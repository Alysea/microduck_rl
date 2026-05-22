"""Hand-coded Raibert-style controller for slip_biped_2d.xml.

NOTE: This file expects MOTOR (torque) hip actuators.  The model now uses
POSITION actuators for RL training, so running this script will not work
out of the box -- swap the <position> actuators back to <motor> in
slip_biped_2d.xml (and re-add ctrlrange="-1.0 1.0") to revive it.  Kept
here as a reference for the SLIP dynamics + body-attitude / foot-placement
analysis we ran through in conversation.

Implements the classic decoupling from Marc Raibert's hopping work
(MIT Leg Lab, late 1980s), adapted for a 2-DoF biped (hip pitch only,
no active leg extension):

  1. Body attitude     — during stance, the planted-leg hip applies
                         tau = -Kp_att·pitch - Kd_att·pitch_rate
                         to right the torso.

  2. Foot placement    — during flight, the leg swings to a target
                         angle so it lands at the Raibert "neutral point":
                            x_fp = (T_stance/2)·v_x + k_fp·(v_x - v_target)
                         (the first term is symmetric stance, the second
                         is a velocity-correction gain.)

  3. Hopping height    — no explicit thrust (no active leg actuator).
                         Sustained by low spring damping (0.2 N·s/m → ~5%
                         loss/cycle) plus the work done by stance hip
                         torque.  The controller will MAINTAIN hopping
                         from an initial drop; it can't pump from rest.

Phase detection (per leg) is by spring compression: leg is in stance
if its slide-joint qpos exceeds STANCE_THRESHOLD (~1 mm).  When both
legs are in stance, the attitude torque is split between them so the
effective gain doesn't double.

Usage
-----
    uv run python slip_biped/raibert_controller.py            # hop in place
    uv run python slip_biped/raibert_controller.py --v 0.3    # target 0.3 m/s
    uv run python slip_biped/raibert_controller.py --drop 0.08  # drop from 8 cm
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# -------------------------------------------------------------------------
# Controller
# -------------------------------------------------------------------------

class RaibertController:
    """3-part Raibert controller (attitude + foot placement, no leg thrust)."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        v_x_target: float = 0.0,
        T_stance: float = 0.13,      # stance time estimate (s) — tunable
        k_fp: float = 0.04,          # velocity-correction gain in foot-placement law
        L_leg: float = 0.10,         # nominal hip-to-foot length (m)
        kp_att: float = 6.0,         # body attitude Kp (Nm/rad)
        kd_att: float = 0.4,         # body attitude Kd (Nm·s/rad)
        kp_swing: float = 2.5,       # in-flight hip Kp
        kd_swing: float = 0.08,      # in-flight hip Kd
        stance_threshold: float = 0.0010,   # spring compression to detect contact (m)
        tau_max: float = 1.0,        # hip torque clip (matches XML ctrlrange)
    ):
        self.model = model
        self.data = data

        self.v_x_target = v_x_target
        self.T_stance = T_stance
        self.k_fp = k_fp
        self.L_leg = L_leg
        self.kp_att = kp_att
        self.kd_att = kd_att
        self.kp_swing = kp_swing
        self.kd_swing = kd_swing
        self.stance_threshold = stance_threshold
        self.tau_max = tau_max

        # Resolve qpos / qvel / actuator indices by name (robust to MJCF order)
        def qadr(name: str) -> int:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            assert jid >= 0, f"joint {name!r} not found"
            return model.jnt_qposadr[jid]

        def vadr(name: str) -> int:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return model.jnt_dofadr[jid]

        def aidx(name: str) -> int:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            assert aid >= 0, f"actuator {name!r} not found"
            return aid

        # State indices
        self.q_slider_x = qadr("slider_x")
        self.q_pitch    = qadr("hinge_pitch")
        self.q_hip_L    = qadr("left_hip_pitch")
        self.q_spring_L = qadr("left_spring")
        self.q_hip_R    = qadr("right_hip_pitch")
        self.q_spring_R = qadr("right_spring")

        self.v_slider_x = vadr("slider_x")
        self.v_pitch    = vadr("hinge_pitch")
        self.v_hip_L    = vadr("left_hip_pitch")
        self.v_hip_R    = vadr("right_hip_pitch")

        self.a_hip_L = aidx("left_hip_pitch_act")
        self.a_hip_R = aidx("right_hip_pitch_act")

        # Logging — emit one line per touchdown / liftoff
        self._prev_stance_L = False
        self._prev_stance_R = False

    # ------------------------------------------------------------------ #
    # Sub-controllers
    # ------------------------------------------------------------------ #

    def _attitude_torque(self, pitch: float, pitch_rate: float) -> float:
        """Body-attitude PD — applied to whichever hip is in stance.

        Sign: hip motor torque +T on the leg applies -T to the torso about
        the pitch axis (Newton's 3rd, hinge convention).  So to push the
        body back toward upright when pitch>0, we need +T on the hip.
        """
        return +self.kp_att * pitch + self.kd_att * pitch_rate

    def _foot_placement_target(self, v_x: float) -> float:
        """Raibert foot placement → target hip angle in BODY frame.

        Body-frame target, not world-frame: keeping the leg vertical in
        world during flight would require an external torque the robot
        doesn't have, so the swing controller just positions the leg in
        body frame and trusts the stance attitude controller to keep
        the body upright.

        Sign: with hinge axis +Y, positive hip_pitch swings the leg toward
        -X (backward).  For forward foot placement (x_fp>0), hip_target<0.
        """
        x_fp = (self.T_stance / 2.0) * v_x + self.k_fp * (v_x - self.v_x_target)
        ratio = float(np.clip(x_fp / self.L_leg, -0.6, 0.6))
        return -math.asin(ratio)

    def _swing_torque(self, hip_q: float, hip_v: float, hip_target: float) -> float:
        """In-flight hip PD around the foot-placement target."""
        return -self.kp_swing * (hip_q - hip_target) - self.kd_swing * hip_v

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self) -> None:
        d = self.data

        pitch       = d.qpos[self.q_pitch]
        pitch_rate  = d.qvel[self.v_pitch]
        v_x         = d.qvel[self.v_slider_x]

        spring_L = d.qpos[self.q_spring_L]
        spring_R = d.qpos[self.q_spring_R]
        stance_L = spring_L > self.stance_threshold
        stance_R = spring_R > self.stance_threshold

        # Body attitude torque is split between stance legs so the effective
        # gain stays the same regardless of single/double stance.
        n_stance = int(stance_L) + int(stance_R)
        if n_stance > 0:
            tau_att_total = self._attitude_torque(pitch, pitch_rate)
            tau_att_per_leg = tau_att_total / n_stance
        else:
            tau_att_per_leg = 0.0

        # Foot-placement target (same for both legs — symmetry takes care
        # of which one is currently swinging vs. planted)
        hip_target = self._foot_placement_target(v_x)

        # Per-leg torque
        hip_L_q, hip_L_v = d.qpos[self.q_hip_L], d.qvel[self.v_hip_L]
        hip_R_q, hip_R_v = d.qpos[self.q_hip_R], d.qvel[self.v_hip_R]

        tau_L = (tau_att_per_leg if stance_L
                 else self._swing_torque(hip_L_q, hip_L_v, hip_target))
        tau_R = (tau_att_per_leg if stance_R
                 else self._swing_torque(hip_R_q, hip_R_v, hip_target))

        d.ctrl[self.a_hip_L] = float(np.clip(tau_L, -self.tau_max, self.tau_max))
        d.ctrl[self.a_hip_R] = float(np.clip(tau_R, -self.tau_max, self.tau_max))

        # Phase-change logging
        if stance_L and not self._prev_stance_L:
            print(f"t={d.time:6.3f}  L touchdown  v_x={v_x:+.3f}  pitch={math.degrees(pitch):+5.1f}°")
        if not stance_L and self._prev_stance_L:
            print(f"t={d.time:6.3f}  L liftoff    v_x={v_x:+.3f}")
        if stance_R and not self._prev_stance_R:
            print(f"t={d.time:6.3f}  R touchdown  v_x={v_x:+.3f}  pitch={math.degrees(pitch):+5.1f}°")
        if not stance_R and self._prev_stance_R:
            print(f"t={d.time:6.3f}  R liftoff    v_x={v_x:+.3f}")
        self._prev_stance_L = stance_L
        self._prev_stance_R = stance_R


# -------------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    _DEFAULT_XML = (Path(__file__).resolve().parents[1] / "src" /
                    "mjlab_microduck" / "robot" / "slip_biped" / "slip_biped_2d.xml")
    ap.add_argument("--xml", default=str(_DEFAULT_XML))
    ap.add_argument("--v", "--v-target", dest="v_target", type=float, default=0.0,
                    help="target forward velocity (m/s)")
    ap.add_argument("--drop", type=float, default=0.05,
                    help="initial drop height above nominal (m)")
    ap.add_argument("--stagger", action="store_true",
                    help="give the right hip a small initial offset to break "
                         "left-right symmetry (encourage alternating gait)")
    ap.add_argument("--headless", action="store_true",
                    help="run without viewer, print state every 0.5 s")
    ap.add_argument("--duration", type=float, default=5.0,
                    help="headless run duration (s)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    ctrl = RaibertController(model, data, v_x_target=args.v_target)

    # Initial conditions: drop from height + (optional) hip offset to
    # break left-right symmetry, encouraging an alternating gait rather
    # than perfectly synchronized hopping.
    sz_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slider_z")
    data.qpos[model.jnt_qposadr[sz_id]] = args.drop
    if args.stagger:
        data.qpos[ctrl.q_hip_R] = -0.15
    mujoco.mj_forward(model, data)

    print(f"\nRaibert controller — slip_biped_2d")
    print(f"  v_target = {args.v_target} m/s")
    print(f"  drop     = {args.drop*100:.1f} cm above nominal")
    if args.stagger:
        print(f"  stagger  = right hip offset -0.15 rad")
    print()

    if args.headless:
        n_steps = int(args.duration / model.opt.timestep)
        log_every = int(0.5 / model.opt.timestep)
        for i in range(n_steps):
            ctrl.step()
            mujoco.mj_step(model, data)
            if i % log_every == 0:
                z = data.qpos[model.jnt_qposadr[sz_id]]
                pitch_deg = math.degrees(data.qpos[ctrl.q_pitch])
                v_x = data.qvel[ctrl.v_slider_x]
                print(f"  t={data.time:5.2f}  z={z*1000:+6.1f}mm  "
                      f"pitch={pitch_deg:+5.1f}°  v_x={v_x:+.3f}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.time()
        while viewer.is_running():
            ctrl.step()
            mujoco.mj_step(model, data)
            viewer.sync()
            target = wall_start + data.time
            now = time.time()
            if target > now:
                time.sleep(target - now)


if __name__ == "__main__":
    main()
