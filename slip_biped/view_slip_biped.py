"""Sanity-check viewer for slip_biped_2d.xml.

Loads the model and (optionally) drives the hips with an anti-symmetric sinusoid
so you can watch the springs compress and rebound on each footfall.

    uv run python slip_biped/view_slip_biped.py            # passive (hips held at 0)
    uv run python slip_biped/view_slip_biped.py --swing    # slow walking-like swing
"""

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def main():
    ap = argparse.ArgumentParser()
    # XML now lives inside the package so mjlab can discover it.
    _DEFAULT_XML = (Path(__file__).resolve().parents[1] / "src" /
                    "mjlab_microduck" / "robot" / "slip_biped" / "slip_biped_2d.xml")
    ap.add_argument("--xml", default=str(_DEFAULT_XML))
    ap.add_argument("--swing", action="store_true",
                    help="Drive hip_pitch with an anti-symmetric sinusoid")
    ap.add_argument("--swing-amp", type=float, default=0.4, help="amplitude (rad)")
    ap.add_argument("--swing-freq", type=float, default=1.5, help="frequency (Hz)")
    ap.add_argument("--drop", type=float, default=0.0,
                    help="extra height (m) added to initial torso z, for a drop test")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # Joint / actuator inventory
    print(f"\nLoaded {args.xml}")
    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}")
    print(f"  total mass: {sum(model.body_mass):.4f} kg\n")
    print("  Joints:")
    jnt_type_name = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"    [{i}] {name:25s} type={jnt_type_name[model.jnt_type[i]]}")
    print("\n  Actuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"    [{i}] {name}")
    print()

    # Optional drop test: bump torso z up by --drop metres
    if args.drop > 0:
        slider_z_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "slider_z")
        data.qpos[model.jnt_qposadr[slider_z_id]] = args.drop
        print(f"Drop test: torso lifted {args.drop*1000:.0f} mm above nominal\n")

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.time()
        while viewer.is_running():
            if args.swing:
                phase = 2 * math.pi * args.swing_freq * data.time
                data.ctrl[0] =  args.swing_amp * math.sin(phase)   # left hip target (rad)
                data.ctrl[1] = -args.swing_amp * math.sin(phase)   # right hip target (rad), anti-phase

            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time pacing
            target = wall_start + data.time
            now = time.time()
            if target > now:
                time.sleep(target - now)


if __name__ == "__main__":
    main()
