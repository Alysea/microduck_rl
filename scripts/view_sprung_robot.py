"""Visual inspection of the sprung-shank microduck.

Bypasses mjlab entirely — we just load the raw MJCF, add a floor, crank up
the XL330 actuator's position gain to a non-physical value so it can
statically hold the home pose, and hand off to MuJoCo's blocking viewer.

This is purely a visualisation aid: the boosted gains do NOT match what
training sees.  Use it to confirm:
  • The kinematic chain is right (no weird mesh intersections).
  • The spring joint exists and compresses on landing impact.
  • Both legs settle symmetrically.

Controls (blocking MuJoCo viewer):
  Space          pause / play
  Right-arrow    single step
  Left-drag a body → apply force
  Right-click → Joints → slider panel.  Scroll to *_shank_spring to read
                                         the compression in metres in real time.
"""

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer


# Option A "half-bent" home pose: knee bend halved (-1.2 → -0.6) plus
# proportional hip+ankle reduction.  Shank world angle becomes ~17° instead
# of ~34°, so spring compression has roughly half the horizontal foot-drift
# component, much less tipping moment.  Leg is ~11 mm longer extended, so
# trunk starts ~11 mm higher.
HOME_RIGID = {
    "left_hip_pitch": 0.6,   "right_hip_pitch": -0.6,
    "left_knee":     -1.2,   "right_knee":      1.2,
    "left_ankle":     0.6,   "right_ankle":    -0.6,
}
HOME_SPRUNG = {
    "left_hip_pitch": 0.3,   "right_hip_pitch": -0.3,
    "left_knee":     -0.6,   "right_knee":      0.6,
    "left_ankle":     0.3,   "right_ankle":    -0.3,
}
HOME_COMMON = {
    "left_hip_yaw":   0.0,   "right_hip_yaw":   0.0,
    "left_hip_roll":  0.0,   "right_hip_roll":  0.0,
    "neck_pitch":    -0.3491, "head_pitch":     0.3491,
    "head_yaw":       0.0,   "head_roll":       0.0,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigid", action="store_true")
    ap.add_argument("--drop", type=float, default=0.03)
    args = ap.parse_args()

    xml_dir = (Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck"
               / "robot" / "microduck")
    fname = "robot_walk.xml" if args.rigid else "robot_walk_sprung.xml"
    xml = (xml_dir / fname).read_text()
    # absolute meshdir so MuJoCo can find the STLs from any CWD
    xml = xml.replace('meshdir="assets"', f'meshdir="{(xml_dir / "assets").resolve()}"')
    # add a ground plane (real microduck XML relies on mjlab to add one)
    xml = xml.replace(
        '<worldbody>',
        '<worldbody>\n'
        '    <geom name="_view_floor" type="plane" size="2 2 0.05" pos="0 0 0" '
        'rgba="0.85 0.9 0.85 1"/>\n'
        '    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>',
    )

    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    # Crank up the position-actuator gain hard.  XL330's real kp≈0.43; we
    # use ≈200 Nm/rad here so the robot can hold the home pose under
    # gravity without sagging visually.  Force range also boosted.
    # (Position actuator: gainprm[0]=+kp, biasprm[1]=-kp, biasprm[2]=-kv)
    KP_BOOST = 500.0
    m.actuator_gainprm[:, 0] *= KP_BOOST
    m.actuator_biasprm[:, 1] *= KP_BOOST
    m.actuator_forcerange[:, 0] *= KP_BOOST
    m.actuator_forcerange[:, 1] *= KP_BOOST

    # Pick HOME pose + nominal standing height per model.  The "half-bent"
    # sprung pose has a longer effective leg (less knee bend) so the trunk
    # starts ~11 mm higher to keep the foot on the floor.
    HOME = {**HOME_COMMON, **(HOME_RIGID if args.rigid else HOME_SPRUNG)}
    nominal_trunk_z = 0.120 if args.rigid else 0.131

    for name, val in HOME.items():
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            d.qpos[m.jnt_qposadr[jid]] = val
    for i in range(m.nu):
        aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if aname in HOME:
            d.ctrl[i] = HOME[aname]

    # Trunk z = nominal + drop
    fj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    if fj >= 0:
        d.qpos[m.jnt_qposadr[fj] + 2] = nominal_trunk_z + args.drop

    mujoco.mj_forward(m, d)

    # Report what we see
    label = "rigid" if args.rigid else "sprung (half-bent HOME)"
    print(f"\n{label} microduck   nq={m.nq}  nv={m.nv}  nu={m.nu}")
    print(f"Initial trunk z = {d.qpos[m.jnt_qposadr[fj] + 2]*1000:.1f} mm")
    print(f"  HOME knee = {HOME.get('left_knee', 0):+.2f} rad   "
          f"hip_pitch = {HOME.get('left_hip_pitch', 0):+.2f} rad   "
          f"ankle = {HOME.get('left_ankle', 0):+.2f} rad")
    for jn in ("left_shank_spring", "right_shank_spring"):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            print(f"  {jn}: range {m.jnt_range[jid][0]*1000:.0f}-"
                  f"{m.jnt_range[jid][1]*1000:.0f} mm, "
                  f"k={m.jnt_stiffness[jid]:.0f} N/m, "
                  f"c={m.dof_damping[m.jnt_dofadr[jid]]:.2f} N·s/m")
        elif not args.rigid:
            print(f"  {jn}: NOT FOUND")
    print()
    print("Blocking viewer.  Space=pause, →=step, drag bodies to perturb.\n")

    mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    main()
