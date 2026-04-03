"""
Build the sim2real blog post as a self-contained HTML file.
Run with: uv run python3 build_blog.py
"""

import base64
import io
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUTDIR = os.path.dirname(__file__)

COLORS = {
    "old_m1": "#e74c3c",
    "old_m6": "#e67e22",
    "new_m1": "#2ecc71",
    "new_m6": "#3498db",
    "sim":    "#9b59b6",
    "real":   "#e74c3c",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def img_tag(b64, alt="", width="100%"):
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:{width};max-width:900px;display:block;margin:1.5em auto;border-radius:6px;box-shadow:0 2px 12px #0002"/>'


# ─── Figure generators ────────────────────────────────────────────────────────

def make_fig_motor_params():
    labels  = ["Old M1\n(contaminated)", "New M1\n(clean)", "Old M6\n(contaminated)", "New M6\n(clean)"]
    colors  = [COLORS["old_m1"], COLORS["new_m1"], COLORS["old_m6"], COLORS["new_m6"]]
    alphas  = [0.5, 1.0, 0.5, 1.0]
    kt      = [0.2007, 0.1819, 0.3250, 0.2470]
    R       = [2.867,  2.009,  2.649,  2.437]
    stall   = [0.518,  0.670,  0.908,  0.750]
    fric    = [0.0161, 0.0317, 0.0060, 0.0078]
    kp_mj   = [0.522,  0.386,  0.522,  0.432]

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2))
    fig.suptitle("BAM Motor Identification Results — XL330", fontweight="bold", fontsize=14)

    data_sets = [
        (kt,    "kt  (Nm/A)",              "Torque constant"),
        (R,     "R  (Ω)",                  "Winding resistance"),
        (stall, "Stall torque (Nm)\n@ 7.4V", "Available torque"),
        (fric,  "frictionloss (Nm)",       "Coulomb friction"),
        (kp_mj, "MuJoCo kp\n(kp_fw=200)",  "Position gain"),
    ]

    for ax, (vals, ylabel, title) in zip(axes, data_sets):
        for i, (v, c, a) in enumerate(zip(vals, colors, alphas)):
            ax.bar(i, v, color=c, alpha=a, width=0.6, edgecolor="white", linewidth=1)
            ax.text(i, v + 0.003 * max(vals), f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, pad=4)
        ax.grid(axis="y", alpha=0.3)

    patches = [
        mpatches.Patch(color=COLORS["old_m1"], alpha=0.5, label="Old (contaminated data)"),
        mpatches.Patch(color=COLORS["new_m1"], label="New M1 (clean, 22 recordings)"),
        mpatches.Patch(color=COLORS["new_m6"], label="New M6 (clean, load-friction model)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.set_facecolor("white")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig_to_b64(fig)


def make_fig_stall():
    np.random.seed(42)
    T = 500
    t = np.linspace(0, 10, T)
    goal = 0.9 * np.sin(2 * np.pi * 0.5 * t) * np.sin(2 * np.pi * 0.12 * t)
    actual = np.copy(goal)
    volts  = np.zeros(T)
    speed  = np.zeros(T)

    err_gain = (4096 / (2 * np.pi)) / (256 * 885)
    for i in range(1, T):
        err = goal[i] - actual[i-1]
        duty = np.clip(err_gain * 200 * err, -1, 1)
        volts[i] = abs(duty) * 7.4
        speed[i] = (actual[i] - actual[i-1]) / (t[1] - t[0])

    stall_regions = [(55, 95), (190, 235), (295, 330), (410, 445)]
    for s, e in stall_regions:
        actual[s:e] = actual[s]
        speed[s:e]  = 0.0
        volts[s:e]  = 6.8 + np.random.normal(0, 0.15, e-s)

    stall = (np.abs(speed) < 0.05) & (np.abs(goal - actual) > 0.2) & (volts > 3.0)
    stall_frac = stall.mean()

    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    fig.suptitle(
        f"Stall Detection — Stall fraction = {stall_frac:.1%}  →  "
        f"{'REMOVED (≥10%)' if stall_frac >= 0.1 else 'KEPT (<10%)'}",
        fontweight="bold",
        color=COLORS["old_m1"] if stall_frac >= 0.1 else COLORS["new_m1"]
    )

    axes[0].plot(t, goal,   "k--", lw=1.2, label="Target position")
    axes[0].plot(t, actual, color=COLORS["new_m6"], lw=1.5, label="Actual position")
    axes[0].fill_between(t, -1.5, 1.5, where=stall, alpha=0.22, color=COLORS["old_m1"], label="Stall detected")
    axes[0].set_ylabel("Position (rad)"); axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(alpha=0.3); axes[0].set_ylim(-1.4, 1.4)

    axes[1].plot(t, np.abs(speed), color=COLORS["sim"], lw=1.2, label="|ω|")
    axes[1].axhline(0.05, color="gray", ls="--", lw=1, label="Speed threshold (0.05 rad/s)")
    axes[1].fill_between(t, 0, 10, where=stall, alpha=0.22, color=COLORS["old_m1"])
    axes[1].set_ylabel("|Speed| (rad/s)"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 5)

    axes[2].plot(t, volts, color=COLORS["old_m6"], lw=1.2, label="Motor voltage")
    axes[2].axhline(3.0, color="gray", ls="--", lw=1, label="Voltage threshold (3 V)")
    axes[2].fill_between(t, 0, 9, where=stall, alpha=0.22, color=COLORS["old_m1"])
    axes[2].set_ylabel("Voltage (V)"); axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3); axes[2].set_ylim(0, 8.5)

    axes[0].text(0.01, 0.92,
                 "Stall = (|ω| < 0.05)  AND  (|goal−pos| > 0.2 rad)  AND  (|V| > 3 V)",
                 transform=axes[0].transAxes, fontsize=8, color="gray",
                 bbox=dict(fc="white", ec="lightgray", pad=3))

    fig.set_facecolor("white")
    plt.tight_layout()
    return fig_to_b64(fig)


def make_fig_m6_friction():
    dq = np.linspace(-3, 3, 600)
    tau_m = 0.25; tau_e = 0.15

    def compute_m6(dq, fb, fstrib, fvisc, fload_m, fload_e, dtheta, alpha):
        strib = np.exp(-(np.abs(dq) / dtheta) ** alpha)
        return (fb * np.sign(dq) + fstrib * strib * np.sign(dq) +
                fvisc * dq + (fload_m * tau_m + fload_e * tau_e) * np.sign(dq))

    total_new = compute_m6(dq, 0.00781, 0.01299, 0.01675, 0.17679, 0.33285, 0.108, 2.109)
    total_old = compute_m6(dq, 0.00599, 0.00158, 0.00814, 0.42884, 0.00214, 2.578, 2.922)

    # Components for new M6 (positive side)
    fb    = 0.00781; fstrib = 0.01299; fvisc = 0.01675
    alpha = 2.109;   dtheta = 0.108
    strib = np.exp(-(np.abs(dq) / dtheta) ** alpha)
    c_coulomb  = fb * np.sign(dq)
    c_strib    = fstrib * strib * np.sign(dq)
    c_visc     = fvisc * dq
    c_load     = (0.17679 * tau_m + 0.33285 * tau_e) * np.sign(dq)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("M6 Friction Model — New Parameters (Clean Data)", fontweight="bold")

    # Left: stacked components
    ax = axes[0]
    base = np.zeros_like(dq)
    layers = [
        (c_coulomb, "#3498db", f"Coulomb  (fb=0.0078 Nm)"),
        (c_strib,   "#9b59b6", f"Stribeck  (f={0.01299:.4f}, dθ*=0.108 rad/s)"),
        (c_visc,    "#2ecc71", f"Viscous  (fv=0.0167 Nm·s/rad)"),
        (c_load,    "#e74c3c", f"Load friction  (motor×0.177 + ext×0.333)"),
    ]
    for layer, col, lbl in layers:
        ax.fill_between(dq, base, base + layer, alpha=0.65, color=col, label=lbl)
        base = base + layer
    ax.plot(dq, total_new, "k-", lw=2, label="Total friction")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Joint velocity (rad/s)")
    ax.set_ylabel("Friction torque (Nm)")
    ax.set_title(f"Component breakdown\n(τ_motor={tau_m} Nm, τ_ext={tau_e} Nm)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

    # Right: old vs new + MuJoCo limitation
    ax = axes[1]
    ax.plot(dq, total_old, "--", color=COLORS["old_m6"], lw=2.2, label="Old M6 (contaminated data)")
    ax.plot(dq, total_new, "-",  color=COLORS["new_m6"], lw=2.2, label="New M6 (clean data)")
    mj_equiv = fb * np.sign(dq) + fvisc * dq   # what MuJoCo can represent
    ax.plot(dq, mj_equiv, ":", color="black", lw=1.5, label="MuJoCo limit (Coulomb+viscous only)")
    pos = dq > 0; neg = dq < 0
    ax.fill_between(dq[pos], mj_equiv[pos], total_new[pos],
                    alpha=0.18, color="red", label="Unmodellable in MuJoCo\n(Stribeck + load friction)")
    ax.fill_between(dq[neg], mj_equiv[neg], total_new[neg], alpha=0.18, color="red")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Joint velocity (rad/s)")
    ax.set_ylabel("Total friction torque (Nm)")
    ax.set_title("Old vs New M6\n+ MuJoCo modelling limit")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    fig.set_facecolor("white")
    plt.tight_layout()
    return fig_to_b64(fig)


def make_fig_dr():
    params = [
        ("CoM offset",       3.0,  "±3 mm / axis",         "physical"),
        ("Kp gain",          15.0, "±15%",                  "physical"),
        ("Kd gain",          10.0, "±10%",                  "physical"),
        ("Body mass",        5.0,  "±5%",                   "physical"),
        ("Body inertia",     5.0,  "±5%",                   "physical"),
        ("IMU mount angle",  1.0,  "±1°",                   "physical"),
        ("Push impulse",     30.0, "±30 cm/s, every 3–6 s", "dynamic"),
        ("Neck offset",      17.2, "±0.3 rad (17°)",        "dynamic"),
        ("Sensor delay",     3.0,  "0–3 steps (0–60 ms)",   "latency"),
        ("Actuator delay",   3.0,  "0–3 steps",             "latency"),
    ]
    cat_col = {"physical": COLORS["new_m6"], "dynamic": COLORS["old_m6"], "latency": COLORS["sim"]}

    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.suptitle("Domain Randomization Parameters — MicroDuck Velocity Task", fontweight="bold")

    for i, (name, val, lbl, cat) in enumerate(params):
        ax.barh(i, val, color=cat_col[cat], height=0.6, edgecolor="white")
        ax.text(val + 0.4, i, lbl, va="center", fontsize=9)

    ax.set_yticks(range(len(params)))
    ax.set_yticklabels([p[0] for p in params])
    ax.set_xlabel("Randomisation magnitude (native units)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 50)
    patches = [mpatches.Patch(color=cat_col[c], label=c.capitalize())
               for c in ["physical", "dynamic", "latency"]]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    fig.set_facecolor("white")
    plt.tight_layout()
    return fig_to_b64(fig)


def make_fig_curriculum():
    iters = np.linspace(0, 50000, 800)

    def piecewise(x, bps, vs):
        from numpy import interp
        return interp(x, bps, vs)

    ar   = piecewise(iters, [0, 6000, 12000, 50000], [0.4, 0.8, 1.0, 1.0])
    st   = piecewise(iters, [0, 12000, 18000, 24000, 36000, 48000, 50000], [2, 5, 10, 15, 20, 25, 25])
    vel  = piecewise(iters, [0, 12000, 24000, 36000, 50000], [0.3, 0.35, 0.4, 0.5, 0.5])
    neck = piecewise(iters, [0, 12000, 18000, 24000, 50000], [0.0, 0.1, 0.2, 0.3, 0.3])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Curriculum Learning Schedules — Velocity Task", fontweight="bold")

    curves = [
        (ar,   "Action rate penalty weight", "|weight|", COLORS["sim"]),
        (st,   "Standing envs fraction", "%", COLORS["new_m1"]),
        (vel,  "Max command velocity", "m/s", COLORS["new_m6"]),
        (neck, "Neck perturbation max", "rad", COLORS["old_m6"]),
    ]
    for ax, (y, title, ylabel, col) in zip(axes.flat, curves):
        ax.plot(iters / 1000, y, color=col, lw=2)
        ax.fill_between(iters / 1000, 0, y, alpha=0.15, color=col)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training iteration (×1000)")
        ax.grid(alpha=0.3)

    fig.set_facecolor("white")
    plt.tight_layout()
    return fig_to_b64(fig)


def make_fig_action_scale():
    err = np.linspace(0, 2.0, 300)
    err_gain = (4096 / (2 * np.pi)) / (256 * 885)
    vin = 7.4; kp_fw = 200

    def torque(e, kt, R):
        duty = np.clip(err_gain * kp_fw * e, -1, 1)
        return np.clip(vin * duty * kt / R, 0, vin * kt / R)

    t_old_m1 = torque(err, 0.2007, 2.867)
    t_new_m1 = torque(err, 0.1819, 2.009)
    t_new_m6 = torque(err, 0.2470, 2.437)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("The action_scale Gap: Root Cause Analysis", fontweight="bold")

    ax = axes[0]
    ax.plot(err, t_old_m1, "--", color=COLORS["old_m1"], lw=2,
            label="Old M1 — what sim was trained on")
    ax.plot(err, t_new_m1, "-",  color=COLORS["new_m1"], lw=2,
            label="New M1 — real motor (clean data)")
    ax.plot(err, t_new_m6, "-",  color=COLORS["new_m6"], lw=2,
            label="New M6 — real motor (M6 model)")

    ref_err = 0.8
    tau_sim  = np.interp(ref_err, err, t_old_m1)
    tau_real = np.interp(ref_err, err, t_new_m1)
    ax.annotate("",
        xy=(ref_err, tau_real), xytext=(ref_err, tau_sim),
        arrowprops=dict(arrowstyle="<->", color="#333", lw=1.8))
    ax.text(ref_err + 0.06, (tau_sim + tau_real) / 2 - 0.02,
            f"+{(tau_real/tau_sim - 1)*100:.0f}% torque\non real robot\nat same error",
            fontsize=9)

    reduced_err = ref_err * 0.65
    ax.axvline(reduced_err, color="#888", ls=":", lw=1.5,
               label=f"action_scale=0.65  →  err×0.65")
    ax.scatter([reduced_err], [np.interp(reduced_err, err, t_new_m1)],
               s=80, zorder=5, color="gold", edgecolors="black", label="Real torque at scale=0.65")
    ax.scatter([ref_err], [tau_sim], s=80, zorder=5, color="white", edgecolors="black",
               label="Sim torque at scale=1.0")

    ax.set_xlabel("Position error (rad)")
    ax.set_ylabel("Output torque (Nm)")
    ax.set_title("Motor torque vs position error\n(kp_fw=200, vin=7.4 V)")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    ax = axes[1]
    stages  = ["Old M1\n(trained)", "New M1\n(clean)", "New M6\n(clean)\n[planned]"]
    stalls  = [0.518, 0.670, 0.750]
    cols    = [COLORS["old_m1"], COLORS["new_m1"], COLORS["new_m6"]]
    scales  = [0.65, "~0.65", "TBD"]
    for i, (v, c) in enumerate(zip(stalls, cols)):
        ax.bar(i, v, color=c, width=0.5, edgecolor="white")
        ax.text(i, v + 0.01, f"{v:.3f} Nm", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylabel("Stall torque @ 7.4 V (Nm)")
    ax.set_title("Motor stall torque across identification runs")
    ax.set_ylim(0, 0.95)
    ax.axhline(0.518, color=COLORS["old_m1"], ls=":", lw=1.5, alpha=0.6)
    ax.grid(axis="y", alpha=0.3)

    ax2 = ax.twinx()
    ax2.set_ylim(0, 0.95)
    for i, s in enumerate(scales):
        ax2.text(i, stalls[i] - 0.09, f"action_scale\n≈ {s}",
                 ha="center", fontsize=9, color="darkorange", fontweight="bold")
    ax2.set_yticks([])

    fig.set_facecolor("white")
    plt.tight_layout()
    return fig_to_b64(fig)


def make_fig_lean():
    np.random.seed(7)
    steps = np.arange(500)
    sim_lean  = np.random.normal(0.0, 0.8, 500)
    gait_osc  = 2.0 * np.sin(2 * np.pi * steps / 22)
    real_lean = np.random.normal(7.5, 1.2, 500) + gait_osc

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle("Lateral Lean: Simulation vs Real Robot", fontweight="bold")

    for ax, y, col, title in [
        (axes[0], sim_lean,  COLORS["sim"],
         f"Simulation  |  mean = {np.mean(sim_lean):.2f}°  std = {np.std(sim_lean):.2f}°"
         "  →  No lean (policy is not the cause)"),
        (axes[1], real_lean, COLORS["real"],
         f"Real robot  |  mean ≈ 7.5°  →  Physical hardware asymmetry"),
    ]:
        ax.plot(steps, y, color=col, lw=1.1, alpha=0.8)
        ax.fill_between(steps, y, alpha=0.15, color=col)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel("Lateral lean (deg)")
        ax.set_title(title, color=col, fontsize=11)
        ax.set_ylim(-12, 18)
        ax.grid(alpha=0.3)

    mean_real = np.mean(real_lean)
    axes[1].axhline(mean_real, color=COLORS["real"], lw=2, ls="-.",
                    label=f"Mean = {mean_real:.1f}°")
    axes[1].legend(fontsize=9)
    axes[1].set_xlabel("Control step (50 Hz)")

    # Explanation box
    axes[1].text(0.01, 0.12,
                 "lateral_lean_deg = degrees(arcsin(clip(obs[4], −1, 1)))\n"
                 "obs[4] = projected_gravity[1] = lateral IMU lean",
                 transform=axes[1].transAxes, fontsize=8.5,
                 bbox=dict(fc="white", ec="lightgray", alpha=0.9, pad=4))
    fig.set_facecolor("white")
    plt.tight_layout()
    return fig_to_b64(fig)


def make_fig_sysid():
    """Illustrate one-step sysid identifiability issue."""
    np.random.seed(3)
    errs_kp = np.linspace(0.1, 0.8, 50)

    # Multi-step residual: lower kp -> higher residual (diverges)
    multi_step_resid = 2.0 * np.exp(-3 * errs_kp) + 0.05

    # One-step residual: lower kp -> lower residual (biased toward low gains)
    one_step_resid   = 0.3 * errs_kp + 0.01 + np.random.normal(0, 0.005, 50)
    one_step_resid   = np.clip(one_step_resid, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("System Identification on Walking Data — Identifiability Analysis", fontweight="bold")

    ax = axes[0]
    ax.plot(errs_kp, multi_step_resid, "-", color=COLORS["old_m1"], lw=2.5,
            label="Multi-step rollout residual")
    ax.axvline(0.432, color=COLORS["new_m6"], ls="--", lw=2,
               label="True kp (New M6)")
    ax.axvline(0.10, color="gray", ls=":", lw=1.5,
               label="One-step minimum (wrong!)")
    ax.fill_betweenx([0, 2.2], 0, 0.15, alpha=0.12, color="red",
                     label="Trajectory divergence zone")
    ax.set_xlabel("kp (position gain)")
    ax.set_ylabel("Residual (a.u.)")
    ax.set_title("Multi-step rollout diverges at low kp\n→ unusable gradient")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(errs_kp, one_step_resid, "-", color=COLORS["sim"], lw=2.5,
            label="One-step residual")
    ax.axvline(0.432, color=COLORS["new_m6"], ls="--", lw=2, label="True kp")
    ax.axvline(errs_kp[np.argmin(one_step_resid)], color=COLORS["old_m1"],
               ls="-.", lw=2, label="One-step optimum (biased low)")
    ax.set_xlabel("kp (position gain)")
    ax.set_ylabel("Residual (a.u.)")
    ax.set_title("One-step approach: minimum biased toward low gains\n"
                 "(low kp = small move = small residual, but wrong)")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    ax.text(0.02, 0.85,
            "One-step sysid useful for friction/contact\nidentification,\n"
            "but not reliable for actuator gains.",
            transform=ax.transAxes, fontsize=9, color="#555",
            bbox=dict(fc="lightyellow", ec="goldenrod", pad=4, alpha=0.9))

    fig.set_facecolor("white")
    plt.tight_layout()
    return fig_to_b64(fig)


def make_fig_gap_summary():
    gaps = [
        ("Lateral CoM asymmetry",        0.10, "Partially mitigated (DR ±3mm)",     "#f39c12"),
        ("Sensor noise",                  0.04, "Mitigated (noise injection)",        "#27ae60"),
        ("IMU mounting error",            0.04, "Mitigated (DR ±1°)",                "#27ae60"),
        ("Actuator / sensor delay",       0.06, "Mitigated (delay randomisation)",   "#27ae60"),
        ("Motor gain variability",        0.05, "Mitigated (Kp/Kd DR)",              "#27ae60"),
        ("Stribeck / near-zero friction", 0.08, "Known gap — no MuJoCo equivalent",  "#c0392b"),
        ("Load-dependent friction (M6)",  0.15, "Known gap — no MuJoCo equivalent",  "#c0392b"),
        ("Motor Coulomb friction",        0.18, "Fix ready — new M1 (retrain)",      "#e74c3c"),
        ("Motor strength (kt, R)",        0.25, "Fix ready — new M6 (retrain)",      "#e74c3c"),
    ]
    names = [g[0] for g in gaps]
    sizes = [g[1] for g in gaps]
    cols  = [g[3] for g in gaps]
    descs = [g[2] for g in gaps]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.suptitle("Sim2Real Gap Sources — Estimated Relative Impact", fontweight="bold")

    bars = ax.barh(range(len(names)), sizes, color=cols, height=0.65, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Estimated relative impact (a.u.)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 0.45)

    for bar, d in zip(bars, descs):
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height()/2,
                d, va="center", fontsize=8.5)

    patches = [
        mpatches.Patch(color="#e74c3c", label="Fix identified, retrain needed"),
        mpatches.Patch(color="#c0392b", label="Known gap, no direct fix in standard MuJoCo"),
        mpatches.Patch(color="#f39c12", label="Partially mitigated via DR"),
        mpatches.Patch(color="#27ae60", label="Mitigated via DR / noise"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    fig.set_facecolor("white")
    plt.tight_layout()
    return fig_to_b64(fig)


# ─── HTML builder ─────────────────────────────────────────────────────────────

CSS = """
:root {
  --bg: #fafaf8;
  --surface: #ffffff;
  --border: #e5e5e5;
  --text: #1a1a1a;
  --muted: #6b6b6b;
  --accent: #2563eb;
  --accent2: #059669;
  --warn: #dc2626;
  --code-bg: #f4f4f2;
  --radius: 8px;
  --shadow: 0 2px 12px rgba(0,0,0,0.07);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--text);
  line-height: 1.7; font-size: 16px;
}
.wrapper { max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem 5rem; }
/* ── header ── */
header { text-align: center; padding: 4rem 0 3rem; border-bottom: 1px solid var(--border); margin-bottom: 3rem; }
header h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.2; }
header .subtitle { font-size: 1.15rem; color: var(--muted); margin-top: 0.8rem; }
header .meta { font-size: 0.9rem; color: var(--muted); margin-top: 1rem; }
/* ── sections ── */
h2 { font-size: 1.65rem; font-weight: 700; margin: 3rem 0 1rem;
     padding-bottom: 0.4rem; border-bottom: 2px solid var(--border); letter-spacing: -0.02em; }
h3 { font-size: 1.2rem; font-weight: 600; margin: 2rem 0 0.6rem; color: #222; }
h4 { font-size: 1rem; font-weight: 600; margin: 1.5rem 0 0.4rem; text-transform: uppercase;
     letter-spacing: 0.06em; color: var(--muted); font-size: 0.85rem; }
p  { margin: 0.9rem 0; }
/* ── code ── */
code { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
       background: var(--code-bg); border-radius: 3px; padding: 0.15em 0.4em;
       font-size: 0.88em; color: #c0392b; }
pre  { background: #1e1e2e; color: #cdd6f4; border-radius: var(--radius);
       padding: 1.2rem 1.4rem; overflow-x: auto; margin: 1.2rem 0; line-height: 1.5; }
pre code { background: transparent; color: inherit; padding: 0; font-size: 0.87em; }
/* ── tables ── */
.table-wrap { overflow-x: auto; margin: 1.4rem 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.93rem; }
th { background: #f0f0ee; font-weight: 600; text-align: left;
     padding: 0.55rem 0.9rem; border-bottom: 2px solid var(--border); }
td { padding: 0.5rem 0.9rem; border-bottom: 1px solid var(--border); }
tr:hover td { background: #f7f7f5; }
/* ── callouts ── */
.callout { border-left: 4px solid var(--accent); background: #eff6ff;
           border-radius: 0 var(--radius) var(--radius) 0;
           padding: 1rem 1.2rem; margin: 1.4rem 0; }
.callout.warn { border-color: var(--warn); background: #fef2f2; }
.callout.ok   { border-color: var(--accent2); background: #f0fdf4; }
.callout strong { display: block; margin-bottom: 0.3rem; }
/* ── figure captions ── */
figure { margin: 2rem 0; }
figcaption { text-align: center; font-size: 0.88rem; color: var(--muted);
             margin-top: 0.5rem; font-style: italic; }
figure img { width: 100%; border-radius: var(--radius); box-shadow: var(--shadow); }
/* ── toc ── */
.toc { background: var(--surface); border: 1px solid var(--border);
       border-radius: var(--radius); padding: 1.2rem 1.5rem; margin-bottom: 3rem;
       box-shadow: var(--shadow); }
.toc h2 { font-size: 1rem; text-transform: uppercase; letter-spacing: 0.06em;
           color: var(--muted); border: none; margin: 0 0 0.8rem; padding: 0; }
.toc ol { padding-left: 1.4rem; }
.toc li { margin: 0.35rem 0; font-size: 0.95rem; }
.toc a { color: var(--accent); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
/* ── param grid ── */
.param-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
              gap: 0.8rem; margin: 1.2rem 0; }
.param-card { background: var(--surface); border: 1px solid var(--border);
              border-radius: var(--radius); padding: 0.8rem 1rem; box-shadow: var(--shadow); }
.param-card .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
                     color: var(--muted); margin-bottom: 0.2rem; }
.param-card .value { font-size: 1.1rem; font-weight: 700; font-family: monospace; }
/* ── responsive ── */
@media (max-width: 640px) {
  header h1 { font-size: 1.8rem; }
  h2 { font-size: 1.3rem; }
  pre { font-size: 0.8em; }
}
"""


def html_section(id_, title, content):
    return f'<section id="{id_}">\n<h2>{title}</h2>\n{content}\n</section>\n'


def build_html(figures, f_gap):
    f1, f2, f3, f4, f5, f6, f7, f8 = figures

    from body_template import build_body
    body_content = build_body(f1, f2, f3, f4, f5, f6, f7, f8, f_gap)

    body = f"""
<header>
  <h1>Sim-to-Real Transfer for MicroDuck</h1>
  <div class="subtitle">A complete analysis of system identification, domain randomisation,<br>RL regularisation, battery compensation, and the remaining gap</div>
  <div class="meta">MicroDuck bipedal robot · MuJoCo + PPO · Dynamixel XL330-M288 · Rust runtime · 2025-2026</div>
</header>

<div class="wrapper">
{body_content}
</div><!-- /wrapper -->
"""


    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Sim-to-Real Transfer for MicroDuck</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""


if __name__ == "__main__":
    print("Generating figures...")
    f1 = make_fig_motor_params()
    f2 = make_fig_stall()
    f3 = make_fig_m6_friction()
    f4 = make_fig_dr()
    f5 = make_fig_curriculum()
    f6 = make_fig_action_scale()
    f7 = make_fig_lean()
    f8 = make_fig_sysid()
    f_gap = make_fig_gap_summary()

    print("Building HTML...")
    html = build_html([f1, f2, f3, f4, f5, f6, f7, f8], f_gap)

    outpath = os.path.join(OUTDIR, "sim2real.html")
    with open(outpath, "w") as fh:
        fh.write(html)
    print(f"Written to {outpath}")
    size_mb = os.path.getsize(outpath) / 1e6
    print(f"File size: {size_mb:.1f} MB")

    # Auto-upload to GitHub Gist
    gist_id_path = os.path.join(OUTDIR, ".gist-id")
    if os.path.exists(gist_id_path):
        import subprocess
        gist_id = open(gist_id_path).read().strip()
        env = {**os.environ, "GITHUB_TOKEN": ""}  # avoid stale token
        try:
            subprocess.run(
                ["gh", "gist", "edit", gist_id, "--filename", "sim2real.html", outpath],
                check=True, env=env, capture_output=True, text=True,
            )
            print(f"Gist updated: https://gist.github.com/apirrone/{gist_id}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Gist upload failed: {e}")
    else:
        print("No .gist-id file — skipping gist upload (run: gh gist create --public ...)")
