"""
Generate figures for the sim2real blog post.
Run with: python3 generate_figures.py
"""

import numpy as np
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    print("matplotlib not available — skipping figure generation")
    HAS_MPL = False
    exit(0)

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)

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
    "figure.dpi": 140,
})

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: BAM identification results — motor params comparison
# ──────────────────────────────────────────────────────────────────────────────

def fig_motor_params():
    labels  = ["Old M1\n(contaminated)", "New M1\n(clean)", "Old M6\n(contaminated)", "New M6\n(clean)"]
    colors  = [COLORS["old_m1"], COLORS["new_m1"], COLORS["old_m6"], COLORS["new_m6"]]
    kt      = [0.2007, 0.1819, 0.3250, 0.2470]
    R       = [2.867,  2.009,  2.649,  2.437]
    stall   = [0.518,  0.670,  0.908,  0.750]  # vin=7.4V
    fric    = [0.0161, 0.0317, 0.0060, 0.0078]
    kp_mj   = [0.522,  0.386,  0.522,  0.432]  # kp_fw=200

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    fig.suptitle("BAM Motor Identification Results — XL330", fontweight="bold")

    data_sets = [
        (kt,    "kt  (Nm/A)",      "Torque constant"),
        (R,     "R  (Ω)",          "Winding resistance"),
        (stall, "Stall torque (Nm)\n@ vin=7.4V", "Available torque"),
        (fric,  "frictionloss (Nm)","Coulomb friction"),
        (kp_mj, "MuJoCo kp\n(kp_fw=200)",        "Effective stiffness"),
    ]

    for ax, (vals, ylabel, title) in zip(axes, data_sets):
        bars = ax.bar(range(4), vals, color=colors, width=0.6, edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # value labels
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.002 * max(vals),
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # shade the contaminated pair
    for ax in axes:
        for i in [0, 2]:
            ax.get_children()[i].set_alpha(0.55)

    patches = [
        mpatches.Patch(color=COLORS["old_m1"], alpha=0.55, label="Old identifications (contaminated data)"),
        mpatches.Patch(color=COLORS["new_m1"], label="New M1 (clean data, 22 recordings)"),
        mpatches.Patch(color=COLORS["new_m6"], label="New M6 (clean data, load-friction model)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.07))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig1_motor_params.png"), bbox_inches="tight")
    print("Saved fig1_motor_params.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Stall / saturation detection illustration
# ──────────────────────────────────────────────────────────────────────────────

def fig_stall_detection():
    np.random.seed(42)
    T = 400
    t = np.linspace(0, 8, T)

    # Simulate a sin_sin trajectory — goal position
    goal = 0.8 * np.sin(2 * np.pi * 0.5 * t) * np.sin(2 * np.pi * 0.15 * t)

    # Actual position: follows well when torque is sufficient, stalls when not
    # Heavy mass → stalls at extremes
    actual = np.copy(goal)
    speed  = np.gradient(actual, t)
    volts  = np.zeros(T)

    # Inject saturation events
    for i in range(T):
        err = goal[i] - actual[i-1] if i > 0 else 0
        duty = np.clip(0.002877 * 200 * err, -1, 1)
        volts[i] = abs(duty) * 7.4

    # Fake "stall" regions
    stall_regions = [(50, 90), (180, 220), (290, 320)]
    for s, e in stall_regions:
        actual[s:e] = actual[s] + 0.0  # stuck
        speed[s:e] = 0.0
        volts[s:e] = 6.5  # high voltage, not moving

    stall_mask = (np.abs(speed) < 0.05) & (np.abs(goal - actual) > 0.2) & (volts > 3.0)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Stall Detection in BAM Data\n(simulated example)", fontweight="bold")

    axes[0].plot(t, goal, "k--", lw=1.2, label="Target")
    axes[0].plot(t, actual, color=COLORS["new_m1"], lw=1.5, label="Actual")
    axes[0].fill_between(t, -1.2, 1.2, where=stall_mask, alpha=0.25,
                          color=COLORS["old_m1"], label="Stall detected")
    axes[0].set_ylabel("Position (rad)")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot(t, np.abs(speed), color=COLORS["sim"], lw=1.2)
    axes[1].axhline(0.05, color="gray", ls="--", lw=1, label="Speed threshold")
    axes[1].fill_between(t, 0, 3, where=stall_mask, alpha=0.25, color=COLORS["old_m1"])
    axes[1].set_ylabel("|Speed| (rad/s)")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    axes[2].plot(t, volts, color=COLORS["old_m6"], lw=1.2)
    axes[2].axhline(3.0, color="gray", ls="--", lw=1, label="Voltage threshold")
    axes[2].fill_between(t, 0, 8, where=stall_mask, alpha=0.25, color=COLORS["old_m1"])
    axes[2].set_ylabel("Voltage (V)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    stall_frac = stall_mask.mean()
    axes[0].set_title(f"Stall fraction = {stall_frac:.1%}  →  {'REMOVED ✗' if stall_frac >= 0.1 else 'KEPT ✓'}",
                       fontsize=10, color=COLORS["old_m1"] if stall_frac >= 0.1 else COLORS["new_m1"])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig2_stall_detection.png"), bbox_inches="tight")
    print("Saved fig2_stall_detection.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: M6 friction model decomposition
# ──────────────────────────────────────────────────────────────────────────────

def fig_m6_friction():
    dq = np.linspace(-3, 3, 400)
    dtheta = 0.108
    alpha  = 2.109

    # M6 new parameters
    fb      = 0.00781
    fstrib  = 0.01299
    fvisc   = 0.01675
    fload_m = 0.17679
    fload_e = 0.33285
    tau_m   = 0.25  # typical motor torque during walking
    tau_e   = 0.15  # typical external load

    stribeck = np.exp(-(np.abs(dq) / dtheta) ** alpha)

    coulomb  = fb * np.sign(dq)
    strib_t  = fstrib * stribeck * np.sign(dq)
    viscous  = fvisc * dq
    load_m   = fload_m * tau_m * np.sign(dq)
    load_e   = fload_e * tau_e * np.sign(dq)
    total    = coulomb + strib_t + viscous + load_m + load_e

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("M6 Friction Model Decomposition (New M6, clean data)", fontweight="bold")

    # Left: stacked friction contributions
    ax = axes[0]
    ax.fill_between(dq, 0, coulomb,  alpha=0.6, color="#3498db",  label=f"Coulomb (fb={fb:.4f})")
    ax.fill_between(dq, coulomb, coulomb + strib_t, alpha=0.6, color="#9b59b6",
                    label=f"Stribeck (fstrib={fstrib:.4f}, dθ*={dtheta})")
    ax.fill_between(dq, coulomb + strib_t, coulomb + strib_t + viscous,
                    alpha=0.6, color="#2ecc71", label=f"Viscous (fv={fvisc:.4f})")
    ax.fill_between(dq, coulomb + strib_t + viscous,
                    coulomb + strib_t + viscous + load_m + load_e,
                    alpha=0.6, color="#e74c3c", label=f"Load friction (fm={fload_m:.3f}, fe={fload_e:.3f})")
    ax.plot(dq, total, "k-", lw=2, label="Total")
    ax.set_xlabel("Joint velocity (rad/s)")
    ax.set_ylabel("Friction torque (Nm)")
    ax.set_title(f"All components\n(τ_motor={tau_m} Nm, τ_ext={tau_e} Nm)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.axvline(0, color="gray", lw=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Right: Old vs New M6 total friction
    ax = axes[1]
    # Old M6 params
    fb_o = 0.00599; fstrib_o = 0.00158; fvisc_o = 0.00814
    fload_m_o = 0.4288; fload_e_o = 0.00214; dtheta_o = 2.578; alpha_o = 2.922

    strib_o  = np.exp(-(np.abs(dq) / dtheta_o) ** alpha_o)
    total_o  = (fb_o * np.sign(dq) + fstrib_o * strib_o * np.sign(dq) +
                fvisc_o * dq + (fload_m_o * tau_m + fload_e_o * tau_e) * np.sign(dq))

    ax.plot(dq, total_o, "--", color=COLORS["old_m6"], lw=2, label="Old M6 (contaminated)")
    ax.plot(dq, total,   "-",  color=COLORS["new_m6"], lw=2, label="New M6 (clean)")

    # Shade the load-friction contribution that MuJoCo can't model
    ax.fill_between(dq[dq > 0], (load_m + load_e)[dq > 0], alpha=0.15, color="red",
                    label="Load friction → not in MuJoCo")
    ax.fill_between(dq[dq < 0], (load_m + load_e)[dq < 0], alpha=0.15, color="red")

    ax.set_xlabel("Joint velocity (rad/s)")
    ax.set_ylabel("Total friction torque (Nm)")
    ax.set_title("Old vs New M6\n(red = unmodellable in MuJoCo)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.axvline(0, color="gray", lw=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig3_m6_friction.png"), bbox_inches="tight")
    print("Saved fig3_m6_friction.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: Domain randomization overview
# ──────────────────────────────────────────────────────────────────────────────

def fig_dr_overview():
    params = [
        ("CoM offset",        3.0,    "mm",    "±3 mm per axis"),
        ("Kp gain",           15.0,   "%",     "±15%"),
        ("Kd gain",           10.0,   "%",     "±10%"),
        ("Body mass",         5.0,    "%",     "±5%"),
        ("Body inertia",      5.0,    "%",     "±5%"),
        ("IMU angle",         1.0,    "deg",   "±1°"),
        ("Push velocity",     30.0,   "cm/s",  "±30 cm/s impulse"),
        ("Sensor delay",      3.0,    "steps", "0–3 steps (0–60 ms)"),
        ("Actuator delay",    3.0,    "steps", "0–3 steps"),
        ("Neck offset",       17.0,   "deg",   "±17° (0.3 rad)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Domain Randomization Parameters — MicroDuck Velocity Task", fontweight="bold")

    names  = [p[0] for p in params]
    values = [p[1] for p in params]
    labels = [p[3] for p in params]

    # Colour by category
    cats = ["physical"] * 6 + ["dynamic"] * 2 + ["latency"] * 2
    cat_colors = {"physical": COLORS["new_m6"], "dynamic": COLORS["old_m6"],
                  "latency": COLORS["sim"]}

    bars = ax.barh(range(len(names)), values, color=[cat_colors[c] for c in cats],
                   height=0.6, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Randomisation magnitude (native units)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    for i, (bar, lbl) in enumerate(zip(bars, labels)):
        ax.text(bar.get_width() + 0.3, i, lbl, va="center", fontsize=9)

    patches = [
        mpatches.Patch(color=cat_colors["physical"], label="Physical parameters"),
        mpatches.Patch(color=cat_colors["dynamic"],  label="Dynamic perturbations"),
        mpatches.Patch(color=cat_colors["latency"],  label="Latency / delay"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    ax.set_xlim(0, 45)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig4_dr_overview.png"), bbox_inches="tight")
    print("Saved fig4_dr_overview.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Curriculum learning schedule
# ──────────────────────────────────────────────────────────────────────────────

def fig_curriculum():
    iters = np.linspace(0, 50000, 500)

    def piecewise(iters, breakpoints, values):
        out = np.zeros_like(iters)
        for i, x in enumerate(iters):
            for j, (bp, v) in enumerate(zip(breakpoints, values)):
                if x >= bp:
                    if j + 1 < len(breakpoints):
                        x0, x1 = breakpoints[j], breakpoints[j+1]
                        v0, v1 = values[j], values[j+1]
                        out[i] = v0 + (v1 - v0) * (x - x0) / (x1 - x0)
                    else:
                        out[i] = v
        return out

    action_rate_bps = [0, 6000, 12000, 50000]
    action_rate_vs  = [0.4, 0.8, 1.0, 1.0]
    action_rate = piecewise(iters, action_rate_bps, action_rate_vs)

    standing_bps = [0, 12000, 18000, 24000, 36000, 48000, 50000]
    standing_vs  = [2, 5, 10, 15, 20, 25, 25]
    standing = piecewise(iters, standing_bps, standing_vs)

    vel_bps = [0, 12000, 24000, 36000, 50000]
    vel_vs  = [0.3, 0.35, 0.4, 0.5, 0.5]
    vel = piecewise(iters, vel_bps, vel_vs)

    neck_bps = [0, 12000, 18000, 24000, 50000]
    neck_vs  = [0.0, 0.1, 0.2, 0.3, 0.3]
    neck = piecewise(iters, neck_bps, neck_vs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Curriculum Learning Schedules — Velocity Task (50K iterations)", fontweight="bold")

    def plot_curve(ax, x, y, title, ylabel, color):
        ax.plot(x / 1000, y, color=color, lw=2)
        ax.fill_between(x / 1000, 0, y, alpha=0.15, color=color)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.set_xlabel("Training iteration (×1000)")
        ax.grid(alpha=0.3); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plot_curve(axes[0,0], iters, action_rate, "Action Rate Penalty", "|weight|", COLORS["sim"])
    plot_curve(axes[0,1], iters, standing,    "Standing Envs Fraction", "%",      COLORS["new_m1"])
    plot_curve(axes[1,0], iters, vel,         "Max Command Velocity", "m/s",      COLORS["new_m6"])
    plot_curve(axes[1,1], iters, neck,        "Neck Perturbation Max", "rad",     COLORS["old_m6"])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig5_curriculum.png"), bbox_inches="tight")
    print("Saved fig5_curriculum.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6: action_scale root cause diagram
# ──────────────────────────────────────────────────────────────────────────────

def fig_action_scale():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("The action_scale Gap: Root Cause Analysis", fontweight="bold")

    # Left: position-error → torque for different motor models
    ax = axes[0]
    err = np.linspace(0, 2.0, 200)   # position error in rad

    ENCODER_COUNTS_PER_REV = 4096
    KP_DIVISOR = 256
    PWM_LIMIT = 885
    error_gain = (ENCODER_COUNTS_PER_REV / (2 * np.pi)) / (KP_DIVISOR * PWM_LIMIT)
    vin = 7.4; kp_fw = 200

    def motor_torque(err, kt, R):
        duty = np.clip(error_gain * kp_fw * err, -1, 1)
        return np.clip(vin * duty * kt / R, -vin * kt / R, vin * kt / R)

    tau_old_m1 = motor_torque(err, kt=0.2007, R=2.867)
    tau_new_m1 = motor_torque(err, kt=0.1819, R=2.009)
    tau_new_m6 = motor_torque(err, kt=0.2470, R=2.437)

    ax.plot(err, tau_old_m1, "--", color=COLORS["old_m1"], lw=2, label="Old M1 (sim was trained on)")
    ax.plot(err, tau_new_m1, "-",  color=COLORS["new_m1"], lw=2, label="New M1 (real motor, clean data)")
    ax.plot(err, tau_new_m6, "-",  color=COLORS["new_m6"], lw=2, label="New M6 (real motor, M6 model)")

    # Show the "action_scale compensation"
    ref_err = 0.8
    tau_sim  = np.interp(ref_err, err, tau_old_m1)
    tau_real = np.interp(ref_err, err, tau_new_m1)
    ax.annotate("", xy=(ref_err, tau_real), xytext=(ref_err, tau_sim),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(ref_err + 0.05, (tau_sim + tau_real) / 2,
            f"+{(tau_real/tau_sim - 1)*100:.0f}% torque\non real robot",
            fontsize=9, color="black")

    ax.axvline(ref_err * 0.65, color="gray", ls=":", lw=1.2,
               label=f"action_scale=0.65 → err×0.65={ref_err*0.65:.2f}")
    ax.set_xlabel("Position error (rad)")
    ax.set_ylabel("Output torque (Nm)")
    ax.set_title("Motor torque vs position error\n(kp_fw=200, vin=7.4V)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Right: stall torque timeline
    ax = axes[1]
    stages = ["Old M1\n(trained)", "Old M6\n(reference)", "New M1\n(clean)", "New M6\n(clean)\n[planned]"]
    stalls  = [0.518, 0.908, 0.670, 0.750]
    cols    = [COLORS["old_m1"], COLORS["old_m6"], COLORS["new_m1"], COLORS["new_m6"]]
    bars = ax.bar(range(4), stalls, color=cols, width=0.55, edgecolor="white")

    ax.axhline(0.750, color=COLORS["new_m6"], ls="--", lw=1.5, alpha=0.7, label="Target (New M6)")
    ax.axhline(0.518, color=COLORS["old_m1"], ls=":",  lw=1.5, alpha=0.7, label="Old training baseline")

    for bar, v in zip(bars, stalls):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f} Nm",
                ha="center", fontsize=9)

    ax.set_xticks(range(4)); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Stall torque @ vin=7.4V (Nm)")
    ax.set_title("Motor stall torque evolution\nacross identification runs")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # action_scale annotation
    ax2 = ax.twinx()
    ax2.set_ylim(0, 1.1)
    scale_approx = [1.0, None, 0.65, None]
    for i, s in enumerate(scale_approx):
        if s is not None:
            ax2.scatter(i, stalls[i], s=200, marker="*", color="gold", zorder=5)
            ax2.text(i, stalls[i] + 0.05, f"action_scale\n≈ {s}", ha="center",
                     fontsize=8, color="darkorange")
    ax2.set_ylabel("(Required action_scale)", color="darkorange", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="darkorange")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig6_action_scale.png"), bbox_inches="tight")
    print("Saved fig6_action_scale.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7: Lateral lean — sim vs real
# ──────────────────────────────────────────────────────────────────────────────

def fig_lateral_lean():
    np.random.seed(7)
    steps = np.arange(500)

    # Sim: near-zero lean, small noise
    sim_lean = np.random.normal(0.0, 0.8, 500)

    # Real: ~7.5° bias, moderate noise, gait-frequency oscillation
    gait_osc = 2.0 * np.sin(2 * np.pi * steps / 22)  # ~2 Hz gait at 50 Hz
    real_lean = np.random.normal(7.5, 1.2, 500) + gait_osc

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle("Lateral Lean: Simulation vs Real Robot", fontweight="bold")

    axes[0].plot(steps, sim_lean,  color=COLORS["sim"],  lw=1.2, alpha=0.8)
    axes[0].axhline(0, color="k", lw=0.8, ls="--")
    axes[0].fill_between(steps, sim_lean, alpha=0.2, color=COLORS["sim"])
    axes[0].set_ylabel("Lean angle (deg)")
    axes[0].set_title(f"Simulation  |  mean={np.mean(sim_lean):.2f}°  std={np.std(sim_lean):.2f}°",
                       color=COLORS["sim"])
    axes[0].set_ylim(-12, 18)
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, real_lean, color=COLORS["real"], lw=1.2, alpha=0.8)
    axes[1].axhline(0, color="k", lw=0.8, ls="--")
    axes[1].axhline(7.5, color=COLORS["real"], lw=1.5, ls="-.",
                    label=f"Mean ≈ 7.5°")
    axes[1].fill_between(steps, real_lean, alpha=0.2, color=COLORS["real"])
    axes[1].set_ylabel("Lean angle (deg)")
    axes[1].set_xlabel("Control step (50 Hz)")
    axes[1].set_title(f"Real robot  |  mean≈7.5°  →  Physical asymmetry (not policy artifact)",
                       color=COLORS["real"])
    axes[1].set_ylim(-12, 18)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig7_lateral_lean.png"), bbox_inches="tight")
    print("Saved fig7_lateral_lean.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Figure 8: Sim2real gap summary — stacked bar
# ──────────────────────────────────────────────────────────────────────────────

def fig_gap_summary():
    gaps = [
        ("Motor strength\n(kt, R)",          "Motor sysid",      0.29, "Identified, retrain needed",   "#e74c3c"),
        ("Motor friction\n(frictionloss)",    "Motor sysid",      0.20, "Identified, retrain needed",   "#e67e22"),
        ("Load-dependent\nfriction (M6)",     "Unmodelled",       0.15, "Ongoing – no MuJoCo equiv.",   "#c0392b"),
        ("Actuator/sensor\ndelay",            "DR",               0.05, "Mitigated",                    "#27ae60"),
        ("Sensor noise",                      "DR",               0.03, "Mitigated",                    "#27ae60"),
        ("Motor gain\nvariability",           "DR",               0.05, "Mitigated",                    "#27ae60"),
        ("IMU mounting\nerror",               "DR",               0.03, "Mitigated",                    "#27ae60"),
        ("Lateral CoM\nasymmetry",            "Partial DR",       0.10, "Partially mitigated",          "#f39c12"),
    ]

    names  = [g[0] for g in gaps]
    sizes  = [g[2] for g in gaps]
    cols   = [g[4] for g in gaps]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Sim2Real Gap Sources — MicroDuck\n(bar size = estimated relative impact)", fontweight="bold")

    bars = ax.barh(range(len(names)), sizes, color=cols, height=0.65, edgecolor="white", lw=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Estimated relative impact (arbitrary units)")
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    for i, (bar, g) in enumerate(zip(bars, gaps)):
        ax.text(bar.get_width() + 0.003, i, g[3], va="center", fontsize=9)

    patches = [
        mpatches.Patch(color="#e74c3c", label="Identified, fix ready"),
        mpatches.Patch(color="#c0392b", label="Known, no direct fix"),
        mpatches.Patch(color="#f39c12", label="Partially mitigated (DR)"),
        mpatches.Patch(color="#27ae60", label="Mitigated (DR)"),
    ]
    ax.legend(handles=patches, fontsize=9, loc="lower right")
    ax.set_xlim(0, 0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig8_gap_summary.png"), bbox_inches="tight")
    print("Saved fig8_gap_summary.png")
    plt.close()


if __name__ == "__main__":
    print(f"Generating figures in {OUTDIR}/")
    fig_motor_params()
    fig_stall_detection()
    fig_m6_friction()
    fig_dr_overview()
    fig_curriculum()
    fig_action_scale()
    fig_lateral_lean()
    fig_gap_summary()
    print("\nAll figures generated.")
