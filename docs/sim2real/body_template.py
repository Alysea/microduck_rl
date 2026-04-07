"""HTML body template for the sim2real blog post.
Called from build_blog.py with figure base64 strings.
"""


def html_section(id_, title, content):
    return f'<section id="{id_}">\n<h2>{title}</h2>\n{content}\n</section>\n'


def img_tag(b64, alt="", width="100%"):
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:{width};max-width:900px;display:block;margin:1.5em auto;border-radius:6px;box-shadow:0 2px 12px #0002"/>'


def build_body(f1, f2, f3, f4, f5, f6, f7, f8, f_gap):
    return f"""

<nav class="toc">
  <h2>Contents</h2>
  <ol>
    <li><a href="#robot">The Robot and the Challenge</a></li>
    <li><a href="#motor">The XL330-M288 Motor</a></li>
    <li><a href="#bam">System Identification with BAM</a></li>
    <li><a href="#dr">Domain Randomisation</a></li>
    <li><a href="#rl">RL Training Setup</a></li>
    <li><a href="#runtime">Runtime and Battery Compensation</a></li>
    <li><a href="#lean">Lateral Lean Investigation</a></li>
    <li><a href="#sysid">MuJoCo System Identification on Walking Data</a></li>
    <li><a href="#action_scale">The action_scale Mystery</a></li>
    <li><a href="#summary">Summary: The Sim2Real Gap Stack</a></li>
    <li><a href="#actuator_params">Current Actuator Parameters</a></li>
    <li><a href="#future">Beyond Standard MuJoCo: Actuator Nets &amp; Custom Models</a></li>
    <li><a href="#next">Open Questions, Opinions, and Next Steps</a></li>
    <li><a href="#sim2real_recipes">Sim2Real Tips &amp; Recipes</a></li>
    <li><a href="#update-apr2">Update (2 Apr 2026): BAM M6 Actuator in MuJoCo Warp</a></li>
    <li><a href="#update-apr3">Update (3 Apr 2026): Debugging the Training Regression</a></li>
    <li><a href="#update-apr4">Update (4 Apr 2026): Next Experiments</a></li>
    <li><a href="#update-apr5">Update (5 Apr 2026): Tuesday Test Checklist</a></li>
    <li><a href="#update-apr7">Update (7 Apr 2026): Two Key Findings</a></li>
  </ol>
</nav>

<!-- ── 1. Robot ── -->
{html_section("robot", "1. The Robot and the Challenge", '''
<p>MicroDuck is a miniature bipedal robot weighing <strong>755 g</strong> and standing roughly <strong>25 cm</strong> tall in its crouched home pose.
It has <strong>15 actuated joints</strong>: 5 per leg (hip yaw, hip roll, hip pitch, knee, ankle), 4 for the neck/head (neck pitch, head pitch, head yaw, head roll),
and 1 for the mouth (opening/closing) — though the mouth DOF is not part of the learned policy and is controlled independently.
All joints are driven by <strong>Dynamixel XL330-M288</strong> servo motors running firmware position control.</p>

<p>The runtime is written in <strong>Rust</strong>, running on a Raspberry Pi Zero 2W. ONNX policies are loaded via <code>ort</code> (ONNX Runtime).
A BNO055 or BNO08X IMU provides orientation and angular velocity via I&sup2;C.
Motor communication goes through a Dynamixel TTL bus at 1 Mbps. The control loop runs at <strong>50 Hz</strong>.
An Xbox controller provides velocity commands wirelessly.</p>

<p>The walking policy is trained entirely in simulation using PPO via <strong>mjlab</strong>, a lightweight RL framework built on <strong>MuJoCo Warp</strong> (NVIDIA Warp-based GPU acceleration of MuJoCo, distinct from the JAX-based MJX). mjlab borrows Isaac Lab&rsquo;s manager-based API design but uses MuJoCo as the physics engine. Policies are exported to ONNX and deployed on-robot.
The central goal is that a policy trained at <code>action_scale=1.0</code> in simulation should transfer directly to the real robot without any gain reduction.</p>

<h3>Why small bipeds are harder</h3>
<ul>
  <li><strong>Actuator nonlinearity dominates.</strong> The XL330 has a plastic gear train with a 288:1 ratio. Motor friction and armature inertia are a large fraction of the available torque budget. A 10% error in torque constant has proportionally more effect than on a humanoid.</li>
  <li><strong>Battery voltage fluctuates.</strong> The XL330 is rated for 3.7-6.0 V. MicroDuck runs on 2S LiPo (7.4 V nominal, 8.4 V full, 6.0 V discharged). Since motor kp is proportional to supply voltage, the effective gains <em>drift during a session</em>.</li>
  <li><strong>Body asymmetry matters more.</strong> A lateral CoM offset of 2 mm on a 100 g leg segment is a meaningful bias torque. The same offset on a large robot is negligible.</li>
  <li><strong>Control latency is a larger fraction of the gait cycle.</strong> At 50 Hz, 3 steps of delay = 60 ms, a significant chunk of a ~300 ms step period.</li>
</ul>
''')}

<!-- ── 2. Motor ── -->
{html_section("motor", "2. The XL330-M288 Motor", '''
<p>All joints use the <strong>Dynamixel XL330-M288-T</strong>, a micro servo from Robotis weighing just 18 g.</p>

<div class="table-wrap"><table>
  <tr><th>Spec</th><th>Value</th></tr>
  <tr><td>Weight</td><td>18 g</td></tr>
  <tr><td>Gear ratio</td><td>288.4 : 1 (plastic spur gears)</td></tr>
  <tr><td>Motor type</td><td>Cored DC motor</td></tr>
  <tr><td>Encoder</td><td>Contactless absolute (ams AS5601), 12-bit, 4096 pulses/rev</td></tr>
  <tr><td>Input voltage</td><td>3.7 &ndash; 6.0 V (recommended 5.0 V)</td></tr>
  <tr><td>Stall torque (5.0 V)</td><td>0.52 Nm @ 1.47 A</td></tr>
  <tr><td>Stall torque (6.0 V)</td><td>0.60 Nm @ 1.74 A</td></tr>
  <tr><td>No-load speed (5.0 V)</td><td>103 RPM (10.8 rad/s)</td></tr>
  <tr><td>PWM limit</td><td>885 counts (100% duty)</td></tr>
  <tr><td>Current limit</td><td>1750 mA</td></tr>
  <tr><td>Communication</td><td>TTL half-duplex, Protocol 2.0, up to 4 Mbps</td></tr>
  <tr><td>Default position P gain</td><td>400 (register 84, scaled by K_PP = value/128)</td></tr>
  <tr><td>Operating modes</td><td>Position, Extended Position, Velocity, Current, Current-based Position, PWM</td></tr>
</table></div>

<div class="callout">
  <strong>Over-voltage operation.</strong> MicroDuck runs the XL330 at <strong>7.4 V</strong> (2S LiPo), above the rated 6.0 V max. This provides more torque but the firmware voltage limiter must be raised. The motor identification (BAM) is done at vin=7.4 V, matching the actual operating condition.
  When running at 7.4 V, the firmware kp_fw=200 gives an effective stiffness much higher than at rated voltage. BAM captures this scaling: <code>kp_mj = error_gain &times; kp_fw &times; vin &times; kt / R</code>.
</div>

<h3>Motor firmware control law</h3>
<p>In position control mode, the XL330 firmware applies:</p>
<pre><code>duty_cycle = error_gain &times; kp_fw &times; (goal_position - present_position)
duty_cycle = clip(duty_cycle, -PWM_LIMIT, PWM_LIMIT)
voltage = vin &times; duty_cycle
torque = (voltage &times; kt / R) - (kt&sup2; / R) &times; dq   # back-EMF</code></pre>
<p>where <code>error_gain = (4096 / 2&pi;) / (KP_DIVISOR &times; PWM_LIMIT) = 0.002877</code> converts radians to the firmware&rsquo;s internal pulse/gain units. KP_DIVISOR=256 was empirically determined (the manual says 128, but measurements show 256).</p>
''')}

<!-- ── 3. BAM ── -->
{html_section("bam", "3. System Identification with BAM", f'''
<h3>3.1 What is BAM?</h3>
<p><strong>BAM (Better Actuator Model)</strong> is an open-source motor identification framework
(<a href="https://arxiv.org/abs/2410.08650" style="color:var(--accent)">Duclusaud, Passault, Padois &amp; Ly, 2024</a>).
A pendulum arm is attached to the motor output; position, velocity, and PWM are logged while the motor tracks reference trajectories.
A physics model is then fitted using <strong>CMA-ES</strong> (via Optuna), minimising the <strong>Mean Absolute Error</strong> between simulated and measured position traces.
Data is split 75/25 for identification/validation.</p>

<h3>3.2 The Six Motor Models</h3>
<p>BAM defines six friction models of increasing complexity. Each defines the maximum friction budget &tau;<sub>f</sub><sup>m</sup> available to oppose motion:</p>

<div class="table-wrap"><table>
  <tr><th>Model</th><th>#Params</th><th>Friction budget &tau;<sub>f</sub><sup>m</sup></th><th>What it adds</th></tr>
  <tr><td><strong>M1</strong> Coulomb-Viscous</td><td>2</td>
      <td><code>K_v|dq| + K_c</code></td>
      <td>Baseline. Standard in MuJoCo/Isaac.</td></tr>
  <tr><td><strong>M2</strong> Stribeck</td><td>5</td>
      <td><code>M1 + K_c<sup>s</sup> exp(-(|dq|/dq*)^&alpha;)</code></td>
      <td>Extra static friction near zero velocity (Stribeck effect).</td></tr>
  <tr><td><strong>M3</strong> Load-dependent</td><td>3</td>
      <td><code>M1 + K_l |&tau;<sub>m</sub> - &tau;<sub>e</sub>|</code></td>
      <td>Friction proportional to net gearbox load.</td></tr>
  <tr><td><strong>M4</strong> Stribeck + Load</td><td>7</td>
      <td><code>M2 + K_l|load| + K_l<sup>s</sup>|load| exp(...)</code></td>
      <td>Combined Stribeck and load-dependent, including Stribeck-load interaction.</td></tr>
  <tr><td><strong>M5</strong> Directional</td><td>9</td>
      <td><code>M4 but K_m&tau;<sub>m</sub> and K_e&tau;<sub>e</sub> split</code></td>
      <td>Separates motor-side vs external-side friction (directional gearbox efficiency).</td></tr>
  <tr><td><strong>M6</strong> Quadratic</td><td>11</td>
      <td><code>M5 + K_m<sup>q</sup>&tau;<sub>m</sub>&sup2; + K_e<sup>q</sup>&tau;<sub>e</sub>&sup2;</code></td>
      <td>Quadratic load-dependent terms (observed in harmonic drives).</td></tr>
</table></div>

<p>The BAM paper reports that for Dynamixel spur-gear servos (MX-64, MX-106), <strong>M4-M5 give the best results</strong> (2-3&times; MAE reduction over M1).
For harmonic drives (eRob80:100), M6 is best. For the XL330 specifically, we explored M1 and M6. M5 (directional) is likely the sweet spot for the XL330
and should be explored in future work.</p>

<h3>3.3 Testbench Setup</h3>
<p>A rigid arm of length 0.1 m is attached to the motor output. Different arm masses (0.112, 0.534, 1.012 kg) and springs are used
to vary the loading conditions. Four trajectory types are recorded at 200 Hz (dt=0.005 s):</p>
<ul>
  <li><strong>sin_time_square</strong> &mdash; sinusoidal position with frequency increasing as t&sup2; (sweeps phase/amplitude response)</li>
  <li><strong>sin_sin</strong> &mdash; double sinusoid with rich frequency content</li>
  <li><strong>up_and_down</strong> &mdash; slow lift-and-lower, emphasises static friction and Stribeck transitions</li>
  <li><strong>lift_and_drop</strong> &mdash; lifts mass then releases (torque_enable=false), isolates viscous friction from back-EMF</li>
</ul>

<h3>3.4 Data Contamination and Filtering</h3>
<p>The original dataset had <strong>53 recordings</strong>. Analysis revealed that recordings with heavy arm mass (0.223+ kg) and stiff springs (k&ge;800)
hit <strong>motor torque saturation</strong>: the motor was at maximum PWM with near-zero velocity.</p>

<div class="callout warn">
  <strong>Contamination mechanism:</strong>
  Saturated frames (high voltage, zero velocity) inflate the estimated resistance R &mdash;
  the model explains &ldquo;why didn&rsquo;t it move? must be high R&rdquo;. Then kt compensates downward to preserve the duty-cycle-torque ratio.
  Net effect: old M1 identified a motor that was <strong>29% weaker</strong> than reality.
</div>

{img_tag(f2, "Stall detection in BAM data", "100%")}
<figure>
  <figcaption><strong>Figure 1.</strong> Stall detection: frames where the motor strains against its voltage limit (high V, zero speed, large position error) are flagged. Files &ge;10% stall fraction are discarded.</figcaption>
</figure>

<p>Stall criterion: <code>|&omega;| &lt; 0.05 AND |goal-pos| &gt; 0.2 rad AND |V| &gt; 3 V</code>.
Filter reduced the dataset: <strong>53 &rarr; 22 clean recordings</strong> (17 sin_time_square, 3 sin_sin, 2 up_and_down; all lift_and_drop removed).
Original data preserved in <code>processed.bak/</code>.</p>

<h3>3.5 M1 Identification Results</h3>
{img_tag(f1, "Motor identification results comparison")}
<figure>
  <figcaption><strong>Figure 2.</strong> Key motor parameters across identification runs. Faded bars = contaminated data.</figcaption>
</figure>

<div class="table-wrap"><table>
  <tr><th>Parameter</th><th>Old M1</th><th>New M1</th><th>Change</th></tr>
  <tr><td><code>kt</code> (Nm/A)</td><td>0.2007</td><td>0.1819</td><td>&minus;9%</td></tr>
  <tr><td><code>R</code> (&Omega;)</td><td>2.867</td><td><strong>2.009</strong></td><td><strong>&minus;30%</strong></td></tr>
  <tr><td><code>armature</code> (kg&middot;m&sup2;)</td><td>0.00153</td><td>0.00207</td><td>+35%</td></tr>
  <tr><td><code>friction_base</code> (Nm)</td><td>0.0161</td><td><strong>0.0317</strong></td><td><strong>+97%</strong></td></tr>
  <tr><td><code>friction_viscous</code></td><td>0.0182</td><td>0.0243</td><td>+34%</td></tr>
  <tr><td><strong>Stall torque</strong> @ 7.4V</td><td>0.518 Nm</td><td><strong>0.670 Nm</strong></td><td><strong>+29%</strong></td></tr>
</table></div>

<h3>3.6 M6 Identification Results</h3>
<div class="table-wrap"><table>
  <tr><th>Parameter</th><th>Old M6</th><th>New M6</th><th>Change</th></tr>
  <tr><td><code>kt</code></td><td>0.325</td><td>0.247</td><td>&minus;24%</td></tr>
  <tr><td><code>R</code></td><td>2.649</td><td>2.437</td><td>&minus;8%</td></tr>
  <tr><td><code>load_friction_motor</code></td><td>0.429</td><td><strong>0.177</strong></td><td><strong>&minus;59%</strong></td></tr>
  <tr><td><code>load_friction_external</code></td><td>0.002</td><td><strong>0.333</strong></td><td><strong>+15483%</strong></td></tr>
  <tr><td><code>dtheta_stribeck</code></td><td>2.579</td><td><strong>0.108</strong></td><td><strong>&minus;96%</strong></td></tr>
  <tr><td><strong>Stall torque</strong> @ 7.4V</td><td>0.908 Nm</td><td>0.750 Nm</td><td>&minus;17%</td></tr>
</table></div>

<p>Three major changes: (1) <strong>dtheta_stribeck</strong> dropped from 2.58 to 0.108 rad/s &mdash; now physically correct (near-zero velocity);
(2) <strong>load_friction_external</strong> went from near-zero to 0.333 &mdash; the dominant load-dependent friction mechanism, invisible in contaminated data;
(3) <strong>load_friction_motor</strong> halved &mdash; old data falsely correlated motor torque with friction due to saturation.</p>

{img_tag(f3, "M6 friction model decomposition")}
<figure>
  <figcaption><strong>Figure 3.</strong> Left: M6 friction decomposed into components at typical walking loads. Right: Old vs new M6, with MuJoCo&rsquo;s modelling limit shown. The red-shaded region is friction the sim cannot represent.</figcaption>
</figure>

<h3>3.7 MuJoCo Export and Its Limits</h3>
<p>MuJoCo supports: <code>kp</code>, <code>damping</code>, <code>frictionloss</code>, <code>armature</code>, <code>forcerange</code>. This maps to M1. <strong>M2&ndash;M6 terms have no direct MuJoCo equivalent.</strong>
The BAM export script warns: &ldquo;Model other than m1 can&rsquo;t be exported exactly to MuJoCo.&rdquo;</p>

<div class="table-wrap"><table>
  <tr><th>MuJoCo param</th><th>Old M6 (trained on)</th><th>New M6 (planned)</th><th>New M1</th></tr>
  <tr><td><code>kp</code></td><td>0.522</td><td><strong>0.432</strong></td><td>0.386</td></tr>
  <tr><td><code>damping</code></td><td>0.0480</td><td>0.0418</td><td>0.0408</td></tr>
  <tr><td><code>frictionloss</code></td><td>0.0060</td><td>0.0078</td><td><strong>0.0317</strong></td></tr>
  <tr><td><code>armature</code></td><td>0.00196</td><td>0.00223</td><td>0.00207</td></tr>
  <tr><td><code>forcerange</code></td><td>&plusmn;0.908 Nm</td><td>&plusmn;0.750 Nm</td><td>&plusmn;0.670 Nm</td></tr>
</table></div>

<div class="callout">
  <strong>frictionloss discrepancy M6 (0.008) vs M1 (0.032):</strong>
  M6 decomposes friction into Coulomb + Stribeck + load terms; M1 lumps everything into friction_base.
  Near zero velocity, M6&rsquo;s total is ~0.021 Nm. The remaining gap (vs M1&rsquo;s 0.032) reflects unmodelled load friction implicitly captured by M1&rsquo;s holistic fit.
</div>
''')}

<!-- ── 4. DR ── -->
{html_section("dr", "4. Domain Randomisation", f'''
<p>Domain randomisation (DR) is the primary strategy for robustifying the policy against sim-to-real mismatches.
Parameters are re-sampled at each episode reset.</p>

{img_tag(f4, "Domain randomisation parameters overview")}
<figure>
  <figcaption><strong>Figure 4.</strong> All DR parameters and their magnitudes.</figcaption>
</figure>

<h3>4.1 Physical Parameters</h3>
<div class="table-wrap"><table>
  <tr><th>Parameter</th><th>Range</th><th>Motivation</th></tr>
  <tr><td>Centre of mass offset</td><td>&plusmn;3 mm / axis</td><td>Manufacturing asymmetry, cable routing, CoM estimation error</td></tr>
  <tr><td>Motor Kp gain</td><td>&times;(0.85, 1.15)</td><td>Firmware gain uncertainty, temperature drift, per-motor variability</td></tr>
  <tr><td>Motor Kd gain</td><td>&times;(0.90, 1.10)</td><td>Damping estimation error</td></tr>
  <tr><td>Body mass</td><td>&times;(0.95, 1.05)</td><td>Payload, wear, battery state</td></tr>
  <tr><td>Body inertia</td><td>&times;(0.95, 1.05)</td><td>Coupled with mass randomisation (physically consistent)</td></tr>
  <tr><td>IMU mounting angle</td><td>&plusmn;1&deg;</td><td>Sensor misalignment, mechanical flex</td></tr>
</table></div>

<p>Currently <strong>disabled</strong> (too destabilising): joint friction randomisation, joint damping randomisation, base orientation initialisation.</p>

<h3>4.2 Dynamic Perturbations</h3>
<p><strong>Velocity pushes:</strong> Every 3-6 s, a velocity impulse of &plusmn;0.3 m/s is applied to the base.
Trains recovery from external perturbations.</p>
<p><strong>Neck offset:</strong> Head target randomised every 2-5 s up to &plusmn;0.3 rad (curriculum-gated, starts at iteration 12K).
Trains robustness to head-induced inertial perturbations.</p>

<h3>4.3 Observation Noise and Delays</h3>
<div class="table-wrap"><table>
  <tr><th>Observation</th><th>Noise</th><th>Delay</th></tr>
  <tr><td>Base angular velocity</td><td>Uniform &plusmn;0.024 rad/s</td><td>0-3 steps (0-60 ms)</td></tr>
  <tr><td>Projected gravity</td><td>Uniform &plusmn;0.007</td><td>0-3 steps</td></tr>
  <tr><td>Joint position</td><td>Uniform &plusmn;0.0006 rad</td><td>None (same bus read)</td></tr>
  <tr><td>Joint velocity</td><td>Uniform &plusmn;0.024 rad/s</td><td>None</td></tr>
</table></div>
<p>Delays are resampled every 64 steps. Actuator commands are also delayed by 0-3 steps, modelling firmware processing + USB round-trip.</p>

<h3>4.4 Known DR Gaps</h3>
<div class="callout warn">
  <strong>What DR does NOT cover:</strong>
  <ul style="margin-top:0.5rem">
    <li><strong>Load-dependent friction</strong> &mdash; MuJoCo only has velocity-dependent friction. M6&rsquo;s <code>load_friction_external=0.333</code> is significant and unmodelled. Kp/Kd DR partially compensates but doesn&rsquo;t model the mechanism correctly.</li>
    <li><strong>Battery voltage drift</strong> &mdash; supply voltage drops ~25% over a discharge cycle (8.4V &rarr; 6.3V), changing motor kp proportionally. Not randomised in sim (but see runtime voltage compensation below).</li>
    <li><strong>Motor nonlinearity</strong> &mdash; back-EMF clipping, PWM saturation, Stribeck effect absent from sim actuator.</li>
    <li><strong>Hardware asymmetry</strong> &mdash; left/right manufacturing differences, per-motor friction variation.</li>
  </ul>
</div>

<div class="callout ok">
  <strong>Opinion: what should be improved.</strong>
  <ul style="margin-top:0.5rem">
    <li>Battery voltage DR seems like low-hanging fruit: randomise <code>kp</code> multiplicatively by &times;(0.75, 1.15) to cover the voltage range, rather than the current &times;(0.85, 1.15).</li>
    <li>Per-motor gain randomisation (different kp per joint) would better capture the per-unit variability vs the current global scale factor.</li>
    <li>The CoM randomisation range (&plusmn;3 mm) may be too narrow given the observed 7-8&deg; lean. Either widen to &plusmn;5-8 mm or add a directional bias.</li>
    <li>Joint friction randomisation was disabled because it was &ldquo;too destabilising&rdquo; &mdash; but that may indicate the range was too wide. A narrower range (e.g., frictionloss &times;(0.8, 1.5)) could work and would help close the load-friction gap.</li>
  </ul>
</div>
''')}

<!-- ── 5. RL ── -->
{html_section("rl", "5. RL Training Setup", f'''
<h3>5.1 Algorithm: PPO with GAE</h3>
<div class="param-grid">
  <div class="param-card"><div class="label">Discount &gamma;</div><div class="value">0.99</div></div>
  <div class="param-card"><div class="label">GAE &lambda;</div><div class="value">0.95</div></div>
  <div class="param-card"><div class="label">PPO clip &epsilon;</div><div class="value">0.20</div></div>
  <div class="param-card"><div class="label">PPO epochs</div><div class="value">5</div></div>
  <div class="param-card"><div class="label">Mini-batches</div><div class="value">4</div></div>
  <div class="param-card"><div class="label">Learning rate</div><div class="value">1e-3 (adaptive)</div></div>
  <div class="param-card"><div class="label">Desired KL</div><div class="value">0.01</div></div>
  <div class="param-card"><div class="label">Entropy coeff</div><div class="value">0.01</div></div>
  <div class="param-card"><div class="label">Envs &times; steps</div><div class="value">4096 &times; 24</div></div>
  <div class="param-card"><div class="label">Max iterations</div><div class="value">50,000</div></div>
</div>

<h3>5.2 Network Architecture</h3>
<p>Asymmetric actor-critic with privileged critic observations (base linear velocity, foot heights, terrain scan):</p>
<pre><code>Actor:  [51] &rarr; ELU[512] &rarr; ELU[256] &rarr; ELU[128] &rarr; [14]   (joint position offsets)
Critic: [54+] &rarr; ELU[512] &rarr; ELU[256] &rarr; ELU[128] &rarr; [1]   (value function)</code></pre>

<h3>5.3 Observation Space (51D actor)</h3>
<pre><code>dims  0: 3  &mdash; base angular velocity (body frame)          3D
dims  3: 6  &mdash; projected gravity vector (body frame)       3D
dims  6:20  &mdash; joint positions relative to default pose   14D
dims 20:34  &mdash; joint velocities                           14D
dims 34:48  &mdash; last action                                14D
dims 48:51  &mdash; velocity command [vx, vy, &omega;z]              3D</code></pre>

<h3>5.4 Reward Function</h3>
<div class="table-wrap"><table>
  <tr><th>Term</th><th>Weight</th><th>Purpose</th></tr>
  <tr><td>Track linear velocity</td><td>3.0</td><td>Gaussian, &sigma;&sup2;=0.15, track vx/vy</td></tr>
  <tr><td>Track angular velocity</td><td>3.0</td><td>Gaussian, &sigma;&sup2;=0.40, track &omega;z</td></tr>
  <tr><td>Upright</td><td>1.0</td><td>Penalise trunk tilt</td></tr>
  <tr><td>Pose</td><td>2.0</td><td>Soft joint-angle targets (wide &sigma; during walking)</td></tr>
  <tr><td>CoM height</td><td>1.2</td><td>Keep CoM in 0.08-0.11 m range</td></tr>
  <tr><td>Air time</td><td>5.0</td><td>Reward swing phases 0.10-0.25 s</td></tr>
  <tr><td>Foot clearance</td><td>&mdash;</td><td>Lift feet &ge; 2 cm during swing</td></tr>
  <tr><td>Foot slip</td><td>&minus;0.1</td><td>Penalise sliding at contact</td></tr>
  <tr><td>Stillness at zero cmd</td><td>3.0</td><td>Penalise motion when v_cmd=0</td></tr>
  <tr><td><strong>Action rate L2</strong></td><td>&minus;0.6&rarr;&minus;1.0</td><td><strong>Penalise &Vert;a<sub>t</sub>&minus;a<sub>t-1</sub>&Vert;&sup2; &mdash; key for sim2real</strong></td></tr>
  <tr><td>Joint torques L2</td><td>&minus;1e-3</td><td>Energy efficiency</td></tr>
  <tr><td>Body angular velocity</td><td>&minus;0.05</td><td>Reduce trunk wobble</td></tr>
  <tr><td>Angular momentum</td><td>&minus;0.02</td><td>Reduce spinning tendency</td></tr>
</table></div>

<div class="callout ok">
  <strong>action_rate_l2 is the most important sim2real regulariser.</strong>
  It directly penalises high-frequency oscillations. Its weight is curriculum-ramped so early training explores freely.
</div>

{img_tag(f5, "Curriculum learning schedules")}
<figure>
  <figcaption><strong>Figure 5.</strong> Curriculum axes: action rate penalty, standing environment fraction, max command velocity, neck perturbation amplitude.</figcaption>
</figure>

<h3>5.5 Symmetry</h3>
<p>Mirror loss (coeff=0.5) enforces bilateral symmetry: <code>L_mirror = 0.5 &times; MSE(&pi;(o), flip(&pi;(flip(o))))</code>.
Left/right joint indices are swapped and signs negated for yaw/roll axes.</p>
''')}

<!-- ── 6. Runtime ── -->
{html_section("runtime", "6. Runtime and Battery Compensation", '''
<p>The on-robot runtime is written in Rust (<code>microduck_runtime</code>), running on a <strong>Raspberry Pi Zero 2W</strong>.
Policy inference uses ONNX Runtime (ort) with 2 threads, matching the Pi&rsquo;s dual-core ARM.</p>

<h3>6.1 Control Loop</h3>
<pre><code>loop at 50 Hz:
    motor_state = motors.read_state()           # bulk sync_read: current, velocity, position (10 bytes &times; 14 motors)
    imu_data    = imu.read()                    # BNO055/BNO08X quaternion &rarr; projected gravity + gyro
    obs         = build_observation(motor_state, imu_data, last_action, command)
    action      = policy.infer(obs, command)     # ONNX Runtime
    targets     = DEFAULT_POSE + action * effective_action_scale
    motors.write_goal_positions(targets)
    sleep_until(next_tick)                       # maintain 50 Hz</code></pre>

<p>Motor communication is via TTL at 1 Mbps. Bulk sync_read fetches 10 bytes per motor (2 current + 4 velocity + 4 position) in a single bus transaction.
The IMU runs on a separate I&sup2;C bus.</p>

<h3>6.2 Battery Voltage Compensation</h3>
<p>The XL330 motor kp is proportional to supply voltage: <code>kp &prop; vin</code>. As the battery discharges (8.4V &rarr; 6.3V over a session),
the effective motor stiffness drops ~25%. A policy trained at a fixed vin=7.4V will experience different dynamics depending on battery state.</p>

<p>The runtime implements <strong>voltage-adaptive action scaling</strong> (<code>--voltage-adapt</code> flag):</p>
<pre><code>effective_action_scale = action_scale &times; (nominal_voltage / measured_voltage)
                       = action_scale &times; (7.4 / voltage_ema)</code></pre>

<p>where <code>voltage_ema</code> is an exponential moving average of the motor bus voltage, updated every second.
When the battery is full (8.4V), the effective scale is reduced (7.4/8.4 = 0.88&times;); when depleted (6.5V), it&rsquo;s increased (7.4/6.5 = 1.14&times;).
This compensates for the voltage-proportional gain change without retraining.</p>

<div class="callout">
  <strong>Recording metadata.</strong> When recording walking data (for sysid or analysis), the runtime saves the measured voltage alongside the action_scale.
  This allows post-hoc correction: <code>effective_scale = base_scale &times; (7.4 / recorded_voltage)</code>.
</div>

<h3>6.3 Other Runtime Details</h3>
<ul>
  <li><strong>PID gains:</strong> kp_fw=200, ki=0, kd=0 (set at startup via <code>--kp 200</code>)</li>
  <li><strong>PWM slope:</strong> set to 255 (fastest PWM ramp, checked and corrected at startup)</li>
  <li><strong>Fall detection:</strong> runtime detects falls via projected gravity and can auto-recover using the standing policy</li>
  <li><strong>Mouth motor:</strong> independent control (ID 34), not part of the policy observation/action space</li>
  <li><strong>Gamepad:</strong> Xbox controller via gilrs. Left stick = linear velocity, right stick = angular velocity. Y button = head control mode, B = body pose mode</li>
  <li><strong>Battery benchmark mode:</strong> walks until battery dies, logging time and voltage</li>
</ul>

<div class="callout warn">
  <strong>Opinion: voltage compensation is treating a symptom.</strong>
  The correct fix would be to include battery voltage as a training observation or to domain-randomise vin during training.
  The runtime compensation works in practice but breaks the assumption that the policy was trained for a fixed voltage.
  A voltage-aware policy could make better decisions (e.g., slower gait when voltage is low).
</div>
''')}

<!-- ── 7. Lean ── -->
{html_section("lean", "7. Lateral Lean Investigation", f'''
<p><strong>Observation:</strong> The real robot consistently leans ~7-8&deg; to the left when walking.</p>

{img_tag(f7, "Lateral lean sim vs real")}
<figure>
  <figcaption><strong>Figure 6.</strong> Lateral lean: simulation shows zero bias, real robot shows ~7.5&deg; persistent lean. Source is hardware asymmetry, not policy.</figcaption>
</figure>

<div class="callout ok">
  <strong>Confirmed:</strong> The policy does not cause the lean. Source is physical &mdash; likely CoM offset from cable routing, left/right motor friction differences, or IMU mounting tilt.
</div>
''')}

<!-- ── 8. Sysid ── -->
{html_section("sysid", "8. MuJoCo System Identification on Walking Data", f'''
<h3>8.1 Full-Trajectory Rollout: Failure</h3>
<p>Standard sysid (multi-step rollout with parameter gradients) fails for biped walking: trajectory diverges within 5-10 steps
due to contact desynchronisation. The gradient carries no useful signal.</p>

<h3>8.2 One-Step Sysid</h3>
<p>Alternative: reset state from measurement each frame, step once, compare. Avoids divergence but biases toward lower gains
(lower kp = smaller move = smaller one-step residual, regardless of correctness).</p>

{img_tag(f8, "Sysid identifiability analysis")}
<figure>
  <figcaption><strong>Figure 7.</strong> One-step sysid biases toward low gains. Useful for friction/contact identification, not for actuator gains.</figcaption>
</figure>

<p><strong>Conclusion:</strong> MuJoCo sysid on walking data is fundamentally limited by trajectory instability.
BAM on testbench data remains the right approach for motor parameters.</p>
''')}

<!-- ── 9. action_scale ── -->
{html_section("action_scale", "9. The action_scale Mystery", f'''
<p>Policy trained at <code>action_scale=1.0</code>: real robot shakes violently. Needs &asymp;0.65 to walk.</p>

{img_tag(f6, "action_scale root cause analysis")}
<figure>
  <figcaption><strong>Figure 8.</strong> Root cause: the real motor delivers more torque than the sim assumed. Reducing action_scale compensates by shrinking the position error.</figcaption>
</figure>

<h3>Root Cause Chain</h3>
<ol>
  <li><strong>Motor strength mismatch:</strong> Old M1 (contaminated): stall=0.518 Nm. Real (new M1): 0.670 Nm. Motor 29% stronger than sim assumed. Policy learned large actions; real motor overshoots.</li>
  <li><strong>Load-dependent friction:</strong> Real joints carry body weight &rarr; M6 load_friction adds 0.03-0.15 Nm friction absent from sim. Partially offsets motor strength but not enough.</li>
  <li><strong>After new M1 retraining:</strong> Still needed action_scale ~0.6-0.7. Residual gap = unmodelled load friction.</li>
  <li><strong>Rejected fix: randomising action_scale.</strong> Treats symptom, not cause. Policy becomes conservative everywhere. Correct fix: improve motor model.</li>
</ol>
''')}

<!-- ── 10. Summary ── -->
{html_section("summary", "10. Summary: The Sim2Real Gap Stack", f'''
{img_tag(f_gap, "Sim2real gap summary")}
<figure>
  <figcaption><strong>Figure 9.</strong> All identified sim2real gap sources, estimated impact, and mitigation status.</figcaption>
</figure>

<div class="table-wrap"><table>
  <tr><th>Gap</th><th>Mechanism</th><th>Mitigation</th><th>Status</th></tr>
  <tr><td>Motor strength (kt, R)</td><td>BAM data contamination</td><td>New M1/M6 clean data</td><td style="color:#e74c3c">Fix ready, retrain needed</td></tr>
  <tr><td>Motor Coulomb friction</td><td>Same contamination</td><td>New M1 frictionloss=0.032</td><td style="color:#e74c3c">Fix ready, retrain needed</td></tr>
  <tr><td>Load-dependent friction</td><td>M6 terms, no MuJoCo equiv</td><td>None standard; see &sect;11</td><td style="color:#c0392b">Ongoing gap</td></tr>
  <tr><td>Battery voltage drift</td><td>vin drops 8.4&rarr;6.3V</td><td>Runtime --voltage-adapt</td><td style="color:#f39c12">Runtime fix (not in training)</td></tr>
  <tr><td>Lateral CoM asymmetry</td><td>Hardware manufacturing</td><td>DR &plusmn;3 mm</td><td style="color:#f39c12">Partially mitigated</td></tr>
  <tr><td>Actuator/sensor delay</td><td>Firmware + bus latency</td><td>Delay DR 0-3 steps</td><td style="color:#27ae60">Mitigated</td></tr>
  <tr><td>Sensor noise</td><td>IMU, encoder quantisation</td><td>Noise injection</td><td style="color:#27ae60">Mitigated</td></tr>
  <tr><td>IMU mounting error</td><td>Mechanical misalignment</td><td>DR &plusmn;1&deg;</td><td style="color:#27ae60">Mitigated</td></tr>
  <tr><td>Motor gain variability</td><td>Temperature, per-unit</td><td>Kp DR &plusmn;15%</td><td style="color:#27ae60">Mitigated</td></tr>
  <tr><td>Stribeck friction</td><td>Near-zero velocity physics</td><td>Not modelled</td><td style="color:#c0392b">Known gap</td></tr>
</table></div>

<h3>Parameter Evolution</h3>
<div class="table-wrap"><table>
  <tr><th>Stage</th><th>kp</th><th>frictionloss</th><th>forcerange</th><th>action_scale</th></tr>
  <tr><td>Old M1 (contaminated)</td><td>0.522</td><td>0.016</td><td>&plusmn;0.518 Nm</td><td>~0.65</td></tr>
  <tr><td>New M1 (clean)</td><td>0.386</td><td>0.032</td><td>&plusmn;0.670 Nm</td><td>~0.60-0.70</td></tr>
  <tr><td><strong>New M6 (clean) &mdash; planned</strong></td><td><strong>0.432</strong></td><td>0.008</td><td><strong>&plusmn;0.750 Nm</strong></td><td><strong>TBD</strong></td></tr>
</table></div>
''')}

<!-- ── 10b. Current actuator params ── -->
{html_section("actuator_params", "10b. Current Actuator Parameters in Training XML", '''
<p>The training XML (<code>joints_properties.xml</code>) currently uses:</p>
<pre><code>&lt;default class="chosen_actuator"&gt;
  &lt;joint damping="0.041" frictionloss="0.032" armature="0.002"/&gt;
  &lt;position kp="0.386" kv="0.0" forcerange="-0.67 0.67" ctrlrange="-10.0 10.0"/&gt;
&lt;/default&gt;</code></pre>
<p>These correspond approximately to <strong>New M1</strong> values (kp=0.386, forcerange=0.670 Nm).
Compared to what New M6 suggests (kp=0.432, forcerange=0.750 Nm), the current params are ~10% conservative on both gains and torque.</p>
<p>The next planned step is to update with New M6 values and retrain, then measure the required <code>action_scale</code> empirically.</p>
''')}

<!-- ── 11. Future ── -->
{html_section("future", "11. Beyond Standard MuJoCo: Actuator Nets &amp; Custom Models", '''
<p>The fundamental problem: MuJoCo&rsquo;s built-in actuator model (kp + damping + frictionloss) cannot represent load-dependent friction, Stribeck effects, or the full BAM M6 model.
mjlab uses <strong>MuJoCo Warp</strong> (NVIDIA Warp-based GPU acceleration) rather than MJX (JAX-based). This means custom actuator models must be implemented as Warp kernels rather than JAX functions. MuJoCo Warp does not yet support automatic differentiation through the sim, but Warp kernels are JIT-compiled to CUDA and run at full GPU speed.
Several approaches could close this gap:</p>

<h3>11.1 Actuator Networks</h3>
<p>Pioneered by <strong>Hwangbo et al. (Science Robotics, 2019)</strong> for ANYmal: train a small MLP on real motor data
(input: desired position, current position, velocity &rarr; output: torque), then use this net as the actuator model during RL training.
The network implicitly captures friction nonlinearity, backlash, Stribeck, load-dependence &mdash; everything BAM models explicitly, plus phenomena the analytical model misses.</p>

<div class="callout ok">
  <strong>This is the most promising approach for MicroDuck.</strong>
  BAM already has excellent testbench data at multiple loads and velocities. An actuator net could be trained directly on this data.
  The BAM M6 identification effectively validates what the net should learn.
</div>

<p><strong>Implementation path in MuJoCo Warp (mjlab&rsquo;s backend):</strong> MuJoCo Warp runs the physics pipeline on GPU via NVIDIA Warp.
Custom actuator dynamics can be implemented as <strong>Warp kernels</strong> that modify <code>data.qfrc_applied</code> before stepping.
Note: MuJoCo Warp does <em>not</em> support PLUGIN-type actuators, so custom models must be applied as external force overrides.
Warp kernels are JIT-compiled to CUDA &mdash; no speed penalty vs the built-in model. However, unlike MJX (JAX-based), MuJoCo Warp does not yet support automatic differentiation, so the actuator net would need to be trained separately (not end-to-end through the sim).</p>

<pre><code># Pseudocode for actuator net in MuJoCo Warp
import warp as wp
import torch  # or warp.torch interop

def custom_actuator_step(model, data, actuator_net):
    # Compute desired position from ctrl
    q_desired = data.ctrl[joint_indices]
    q_current = data.qpos[joint_indices]
    dq = data.qvel[joint_indices]

    # Actuator net: trained on BAM testbench data
    net_input = wp.concat([q_desired, q_current, dq])
    torque = actuator_net(net_input)  # small MLP, runs on GPU

    # Apply as external force (replaces built-in actuator)
    data.qfrc_applied[joint_indices] = torque
    return data</code></pre>

<h3>11.2 Analytical BAM Model in MuJoCo Warp</h3>
<p>Instead of a neural net, implement the full BAM M5 or M6 friction model analytically as a Warp kernel.
This is simpler (no training data needed beyond BAM identification) and directly uses the identified parameters:</p>
<pre><code>@wp.kernel
def m6_friction_kernel(dq: wp.array, tau_motor: wp.array, tau_external: wp.array,
                       params: M6Params, friction_out: wp.array):
    i = wp.tid()
    strib = wp.exp(-wp.pow(wp.abs(dq[i]) / params.dtheta_stribeck, params.alpha))
    f = (params.friction_base * wp.sign(dq[i])
       + params.friction_stribeck * strib * wp.sign(dq[i])
       + params.load_friction_motor * wp.abs(tau_motor[i]) * wp.sign(dq[i])
       + params.load_friction_external * wp.abs(tau_external[i]) * wp.sign(dq[i])
       + params.friction_viscous * dq[i])
    friction_out[i] = f</code></pre>

<p>The challenge is computing <code>tau_motor</code> and <code>tau_external</code> inside the step. Motor torque requires the voltage control law;
external torque requires reading constraint/gravity forces from <code>data.qfrc_bias</code>. Both are available in MuJoCo Warp&rsquo;s data structure.</p>

<div class="callout">
  <strong>Opinion: start with the analytical model.</strong> An actuator net requires careful data collection and training.
  The BAM M5/M6 analytical model is already identified, tested, and validated. Implementing it as a Warp kernel is a weekend project.
  If it doesn&rsquo;t close the gap, <em>then</em> train an actuator net.
</div>

<h3>11.3 Feasibility Assessment</h3>
<div class="table-wrap"><table>
  <tr><th>Approach</th><th>Effort</th><th>Expected impact</th><th>Risk</th></tr>
  <tr><td>Update XML with new M6 export</td><td>Low (1 hour)</td><td>Moderate (+12% kp accuracy)</td><td>Low</td></tr>
  <tr><td>Analytical M6 in MuJoCo Warp</td><td>Medium (days)</td><td>High (full load-friction)</td><td>Medium (Warp kernel integration)</td></tr>
  <tr><td>Actuator net in MuJoCo Warp</td><td>High (weeks)</td><td>Highest (captures everything)</td><td>Medium (training stability, no auto-diff through sim)</td></tr>
  <tr><td>DR-only compensation</td><td>Low (hours)</td><td>Moderate (hides gap)</td><td>Low (but doesn&rsquo;t fix root cause)</td></tr>
</table></div>
''')}

<!-- ── 12. Next steps ── -->
{html_section("next", "12. Open Questions, Opinions, and Next Steps", '''
<h3>Immediate actions</h3>
<ol>
  <li><strong>Retrain with New M6 export params</strong> (kp=0.432, forcerange=0.750, damping=0.042). Low effort, should close ~50% of action_scale gap.</li>
  <li><strong>Widen Kp DR range</strong> to cover battery voltage variation: &times;(0.75, 1.15) instead of &times;(0.85, 1.15).</li>
  <li><strong>Try M5 identification</strong> on clean data. M5 (directional) is the best model for Dynamixel spur gears per the BAM paper. We only tried M1 and M6.</li>
</ol>

<h3>Medium-term improvements</h3>
<ol>
  <li><strong>Implement M5/M6 analytical friction in MuJoCo Warp.</strong> The identified parameters are ready. Write a Warp kernel that applies BAM friction as <code>qfrc_applied</code>. This eliminates the biggest known gap (load-dependent friction).</li>
  <li><strong>Add battery voltage to observation space.</strong> The runtime already measures it. A voltage-aware policy could adapt its aggressiveness to battery state instead of relying on runtime compensation.</li>
  <li><strong>Measure real CoM offset.</strong> Suspend the robot and measure the lean angle to quantify the CoM offset. Use this to set an asymmetric bias in training.</li>
  <li><strong>Per-motor friction randomisation.</strong> Each of the 14 XL330s has slightly different friction. Randomising frictionloss per-joint would help.</li>
</ol>

<h3>Research directions</h3>
<ol>
  <li><strong>Actuator net from BAM data.</strong> Train a small MLP on the 22 clean testbench recordings. Integrate into MuJoCo Warp training loop as a Warp/PyTorch module. Could capture phenomena that even M6 misses.</li>
  <li><strong>Re-record BAM data with chirp trajectory.</strong> The current dataset lacks armature identification coverage. A chirp (frequency sweep) at low mass would better excite the inertial dynamics.</li>
  <li><strong>Short-window sysid.</strong> One-step sysid biased actuator gains, but multi-step sysid fails due to divergence. A middle ground: use short (5-10 step) windows with periodic resets and optimise via CMA-ES or finite differences. Could recover better parameters than pure one-step. (Note: MuJoCo Warp does not yet support auto-diff, so gradient-based sysid would require MJX or finite-difference approximations.)</li>
</ol>

<h3>What works well</h3>
<ul>
  <li>The <strong>action_rate_l2 curriculum</strong> is highly effective. Without it, policies are unusable on real hardware.</li>
  <li><strong>Observation delays</strong> (0-3 steps on both sensors and actuators) capture the real latency well.</li>
  <li><strong>Mirror loss</strong> is elegant and effective for bilateral gaits.</li>
  <li>The <strong>Rust runtime</strong> is fast and reliable. Bulk sync_read for 14 motors in one bus transaction is important for timing.</li>
  <li><strong>BAM&rsquo;s stall detection</strong> (our contribution) dramatically improved identification quality. The old contaminated params were the single biggest source of sim2real gap.</li>
</ul>

<h3>What needs work</h3>
<ul>
  <li>The <strong>MuJoCo actuator model</strong> is the bottleneck. Standard kp+damping+frictionloss misses too much physics for small servo motors with plastic gears.</li>
  <li><strong>Battery voltage</strong> is a first-order effect that&rsquo;s not in the training loop at all.</li>
  <li>The <strong>lateral lean</strong> suggests the CoM model is wrong and/or left-right motor characteristics differ significantly. Neither is addressed in training.</li>
  <li><strong>Terrain transfer</strong> is entirely untested on the real robot.</li>
</ul>
''')}

<!-- ── Sim2Real Recipes ── -->
{html_section("sim2real_recipes", "What Actually Worked: MicroDuck Sim2Real Lessons", '''
<p>Concrete findings from this project, in roughly the order we discovered them.</p>

<h3>Motor Identification</h3>
<ul>
  <li><strong>BAM testbench data can be contaminated without obvious signs.</strong> Our initial M1/M6 fits used data from servos with dirty commutators. The identified params looked plausible but inflated friction and distorted kt/R. Re-running identification on cleaned servos produced noticeably different &mdash; and better-transferring &mdash; params.</li>
  <li><strong>The XL330 firmware kp register is not MuJoCo&rsquo;s kp.</strong> Direct translation fails. We used the BAM voltage-control law to derive the equivalent MuJoCo kp from the firmware register value, supply voltage, and encoder resolution. Getting this right was necessary to match sim stiffness to the real robot.</li>
</ul>

<h3>Domain Randomisation</h3>
<ul>
  <li><strong>Kp DR covers battery discharge.</strong> As the LiPo drains, effective motor gain drops. Randomising Kp &times;(0.85, 1.15) made the policy robust to this without needing explicit voltage compensation in training.</li>
  <li><strong>Floor friction DR range matters.</strong> We trained with friction (0.3, 1.2) to cover slippery to grippy surfaces. After switching to a grippier footpad, tightening the range to (0.7, 1.3) produced a more planted, confident gait &mdash; the policy stopped wasting capacity on low-friction strategies it no longer needed.</li>
  <li><strong>Observation and action delay randomisation is necessary on USB-connected hardware.</strong> USB scheduling jitter means the actual delay varies per step. Without 0&ndash;3 step delay DR, the policy was brittle to this.</li>
</ul>

<h3>Training</h3>
<ul>
  <li><strong>Export early (~2000 iterations), not at the reward peak.</strong> At 2000 iterations the policy walks robustly and transfers cleanly (action_scale ~0.65). Continuing to train pushes the reward higher but the policy starts exploiting simulator-specific dynamics &mdash; exact contact timing, warp kernel artefacts &mdash; that don&rsquo;t exist on hardware. The result is high-frequency oscillation even at the same action_scale that worked earlier.</li>
  <li><strong>Symmetry loss hurt our sim2real transfer.</strong> With symmetry enabled, the policy couldn&rsquo;t develop asymmetric compensation for the real robot&rsquo;s left-right motor variation and CoM offset. Disabling it improved transfer significantly.</li>
  <li><strong>Lock your simulation dependencies and treat them like production code.</strong> A single commit that bumped mjlab/MuJoCo/warp introduced a months-long regression that was invisible in training metrics. Rolling back to the exact working versions (committed <code>uv.lock</code>) immediately fixed it. Before updating any sim dependency, train a short walk policy and test on hardware.</li>
</ul>

<h3>Physical Model</h3>
<ul>
  <li><strong>The 80g of unmodelled mass (wiring, PCB, screws) matters.</strong> Lumping it all on the trunk raised the CoM and increased pitch inertia, causing backward-falling instability on the real robot. A more realistic distribution is needed.</li>
  <li><strong>Small joint offsets shift the CoM significantly on a small robot.</strong> Setting neck_pitch=&minus;20&deg; and head_pitch=+20&deg; moved enough head mass rearward to visibly improve stability. The absolute mass moved is small (&lt;100g) but the moment arm is large relative to foot width.</li>
</ul>

<h3>Deployment</h3>
<ul>
  <li><strong>Battery voltage compensation in the runtime runtime is necessary.</strong> We measure voltage each step and scale actions by Vin/Vnom. Without it, the robot walks fine on a full battery and poorly on a depleted one.</li>
  <li><strong>Head/neck joints need a low-pass filter when unloaded.</strong> These joints have very little inertia and the firmware gain is tuned for loaded leg joints. At action_scale=0.8 they oscillate when standing still. An exponential filter (&alpha;&asymp;0.3) on the runtime command eliminates this with no visible effect on intentional head movement.</li>
  <li><strong>action_scale needs to be swept empirically on the real robot.</strong> Despite careful sysid, the remaining sim2real gap means you can&rsquo;t predict the right scale from training alone. We start at 0.5 and increment until the gait becomes unstable.</li>
</ul>
''')}

<!-- ── 13. Update ── -->
{html_section("update-apr2", "Update &mdash; 2 April 2026: BAM M6 Actuator in MuJoCo Warp", '''
<p>Following the analysis above, we implemented the full BAM M6 friction model as a custom mjlab actuator and validated it against real testbench data. This section documents the implementation, the bug we found, and the results so far.</p>

<h3>13.1 Implementation</h3>
<p>We created a <code>BamM6Actuator</code> class that replaces MuJoCo&rsquo;s built-in <code>&lt;position&gt;</code> actuator with direct torque control implementing the full BAM chain:</p>
<ol>
  <li><strong>XL330 firmware control law:</strong> position error &rarr; duty cycle (clipped to &plusmn;1.0) &rarr; voltage</li>
  <li><strong>DC motor equation:</strong> voltage &rarr; motor torque, with back-EMF subtraction (<code>kt&sup2;/R &times; vel</code>)</li>
  <li><strong>M6 friction:</strong> Coulomb + Stribeck + directional motor-load + external-load + quadratic + viscous</li>
  <li><strong>Static friction clipping:</strong> BAM&rsquo;s Algorithm 1 &mdash; friction cannot exceed the torque needed to stop the joint in one timestep</li>
</ol>

<p>The existing XML <code>&lt;position&gt;</code> actuators are converted to <code>&lt;motor&gt;</code> actuators (direct torque mode) via <code>set_to_motor()</code> in the MuJoCo spec.
Joint damping and frictionloss are zeroed out &mdash; all friction is handled by the M6 model.
The actuator wraps cleanly in mjlab&rsquo;s <code>DelayedActuator</code> for delay randomisation.</p>

<h3>13.2 The Sign Bug</h3>
<p>Initial training with the M6 actuator made sim2real <strong>worse</strong> (action_scale dropped from 0.65 to 0.5).
We validated the kernel against BAM&rsquo;s own Python simulator by replaying real testbench recordings in MuJoCo and comparing position traces.</p>

<p>The validation revealed a <strong>sign convention mismatch</strong> between BAM and MuJoCo:</p>

<div class="callout warn">
  <strong>The bug:</strong> BAM computes gravity bias as <code>bias = m &times; g &times; l &times; sin(q)</code> with <code>g = &minus;9.81</code>,
  so <code>bias_torque</code> is <em>negative</em> when gravity pulls in the positive-q direction.
  MuJoCo&rsquo;s <code>qfrc_bias</code> uses the <em>opposite</em> convention &mdash; it&rsquo;s positive in the same scenario.
  <br><br>
  This caused the directional friction formula <code>|K_e &times; ext &minus; K_m &times; mot|</code> to compute the wrong gearbox load:
  friction was <em>underestimated</em> when motor and gravity opposed each other (the common case during walking),
  and <em>overestimated</em> when they were aligned.
</div>

<p><strong>Fix:</strong> negate <code>qfrc_bias</code> before using it as BAM&rsquo;s <code>external_torque</code> in the friction model.
The <code>tau_stop</code> computation (for static friction clipping) correctly uses the un-negated MuJoCo convention since it predicts the actual MuJoCo dynamics.</p>

<h3>13.3 Testbench Validation Results</h3>
<p>After the fix, we replayed 10 real testbench recordings through both BAM&rsquo;s Python simulator and MuJoCo with our M6 kernel:</p>

<div class="table-wrap"><table>
  <tr><th>Comparison</th><th>MAE (rad)</th><th>Notes</th></tr>
  <tr><td>BAM Python vs Real</td><td>0.030</td><td>BAM&rsquo;s own simulator &mdash; the reference</td></tr>
  <tr><td><strong>MuJoCo M6 vs Real</strong></td><td><strong>0.029</strong></td><td>Our kernel matches reality as well as BAM does</td></tr>
  <tr><td>BAM vs MuJoCo M6</td><td>0.024</td><td>Residual gap from CAD mass model vs BAM&rsquo;s point-mass simplification</td></tr>
</table></div>

<p>The MuJoCo M6 kernel now matches real testbench data with the same accuracy as BAM&rsquo;s own simulator.
The remaining BAM-vs-MuJoCo gap (0.024 rad) is expected: MuJoCo uses the full CAD geometry for the testbench arm (distributed mass, mesh inertia), while BAM uses a simplified point-mass pendulum (<code>I = m &times; l&sup2;</code>).</p>

<h3>13.4 Why M1 Export Was Wrong for MuJoCo</h3>
<p>An important conceptual insight emerged during this work. We initially planned to use M6&rsquo;s exported kt/R values in MuJoCo&rsquo;s standard actuator model (which only supports M1-level physics). This was wrong:</p>
<ul>
  <li>M6 parameters are <strong>co-optimised</strong>. Its kt, R, and friction_base only make sense <em>together with</em> the load-dependent friction terms.</li>
  <li>Stripping the load terms and using M6&rsquo;s kt/R in an M1 model gives a motor that&rsquo;s too strong with too little friction &mdash; worse than using M1 parameters directly.</li>
  <li><strong>New M1</strong> (identified on clean data) is the correct export for MuJoCo&rsquo;s standard model. M1&rsquo;s higher <code>friction_base=0.032</code> implicitly absorbs load-dependent friction as a best-fit constant.</li>
  <li>To actually benefit from M6 physics, <strong>you must implement the full model</strong> &mdash; which is what the custom actuator does.</li>
</ul>

<h3>13.5 Current Status and Next Steps</h3>
<ul>
  <li><strong>Sign bug fixed, kernel validated.</strong> MuJoCo M6 matches real testbench data (MAE 0.029 rad).</li>
  <li><strong>Retraining in progress</strong> with the corrected kernel. The previous training (with the sign bug) converged normally but gave worse sim2real &mdash; the corrected version should improve.</li>
  <li><strong>Remaining question:</strong> does the M6 model (identified on a single-joint pendulum testbench) capture the friction physics of multi-joint walking?
    The testbench arm loads are small (0.1 Nm max); robot joint loads during walking can be larger.
    If retraining doesn&rsquo;t close the action_scale gap, re-recording BAM data at more representative loads or using an actuator net trained on walking data would be the next step.</li>
</ul>
''')}

<!-- ── 14. Update 3 Apr ── -->
{html_section("update-apr3", "Update &mdash; 3 April 2026: Debugging the Training Regression", '''
<p>After fixing the sign bug and retraining with the corrected M6 kernel, sim2real transfer was <em>still</em> worse than the old best policy (from February, commit <code>41b4a41</code>).
The old policy &mdash; trained with contaminated BAM values on an older version of mjlab &mdash; consistently outperformed all new policies on both the old and new robots.
This pointed to a <strong>training pipeline regression</strong>, not an actuator model problem.</p>

<h3>14.1 The Clue: Old Policy Robustness</h3>
<p>Key observations:</p>
<ul>
  <li>The old policy (run <code>hhfzw7an</code>, Feb 26) transfers well even to a <em>new robot</em> with different mass distribution. It&rsquo;s remarkably robust.</li>
  <li>All policies trained after the mjlab upgrade (late March onwards) transfer poorly &mdash; regardless of actuator model (XML or M6).</li>
  <li>The regression correlates with the mjlab version upgrade, not with any specific actuator or reward change.</li>
  <li>A hardware debugging red herring: the new robot had hip pitch and hip roll <strong>inverted left-right</strong>. The old policy still walked (barely) even with this mounting error, further demonstrating its robustness.</li>
</ul>

<h3>14.2 Systematic Comparison: Old vs New</h3>
<p>We compared wandb configs between the old good run (<code>hhfzw7an</code>) and a recent run (<code>0894egj0</code>):</p>

<div class="table-wrap"><table>
  <tr><th>Change</th><th>Old (good)</th><th>New (bad)</th><th>Suspected impact</th></tr>
  <tr><td><strong>Symmetry mirror loss</strong></td><td>None (disabled)</td><td>coeff=0.5</td><td style="color:#e74c3c"><strong>High</strong> &mdash; constrains policy to symmetric gaits, reduces ability to compensate for real-world asymmetries</td></tr>
  <tr><td><strong>Motor gain DR with BamM6</strong></td><td>Works (XmlPosition)</td><td>Silent no-op!</td><td style="color:#e74c3c"><strong>High</strong> &mdash; policy trains with zero gain variation, brittle to real motor differences</td></tr>
  <tr><td>air_time weight</td><td>5.0</td><td>6.0</td><td>Low</td></tr>
  <tr><td>Contact sensor mode</td><td>subtree</td><td>geom</td><td>Low (same result, different API)</td></tr>
  <tr><td>obs_groups naming</td><td>&ldquo;policy&rdquo;</td><td>&ldquo;actor&rdquo;</td><td>None (API rename)</td></tr>
</table></div>

<h3>14.3 The Motor Gain DR Bug</h3>
<p>The <code>randomize_delayed_actuator_gains</code> function checked <code>isinstance(base_actuator, XmlPositionActuator)</code> before applying gain randomisation.
With the BamM6Actuator, this check fails silently &mdash; <strong>no gain randomisation happens at all</strong>.
The policy trains with perfectly consistent motors across all environments, making it brittle to the real robot&rsquo;s per-motor variability, temperature drift, and voltage changes.</p>

<div class="callout warn">
  <strong>Fix applied:</strong> The BamM6Actuator now has <code>kp_scale</code> and <code>kd_scale</code> per-environment tensors.
  The DR function detects BamM6Actuator and calls <code>set_gains()</code> to apply per-env scaling to the firmware kp and back-EMF damping.
  Verified: after reset, <code>kp_scale</code> varies across environments (e.g., 0.97&ndash;1.01 with the &plusmn;15% range).
</div>

<h3>14.4 The Symmetry Hypothesis</h3>
<p>The mirror loss (coeff=0.5) was introduced after the old good policy. It forces the policy to produce identical actions for left-right mirrored observations.
While this is theoretically sound for a symmetric robot, MicroDuck has significant real-world asymmetries:</p>
<ul>
  <li>7&ndash;8&deg; lateral lean (CoM offset or IMU mounting)</li>
  <li>Per-motor friction differences (14 individual XL330s)</li>
  <li>Cable routing asymmetry</li>
</ul>
<p>A symmetric policy <em>cannot</em> compensate for these asymmetries. The old policy (no mirror loss) was free to learn asymmetric compensatory strategies, which may explain its superior real-world robustness.</p>

<h3>14.5 Results: Symmetry Off + M6 Kernel + DR Fix</h3>
<p>Retraining with symmetry disabled, M6 kernel (sign-fixed), and motor gain DR fixed produced <strong>significantly better sim2real transfer</strong> than previous new-mjlab attempts.
However, the walk quality still didn&rsquo;t match the old best policy (run <code>hhfzw7an</code>). The policy required <code>action_scale &asymp; 0.65</code> and walked reasonably but with less robustness than the old policy.</p>

<p>This confirmed that symmetry was a <em>major</em> factor, but not the only one. Something else changed between the old and new training setups.</p>

<h3>14.6 Isolation Experiment: Reproducing the Old Setup on New mjlab</h3>
<p>To isolate whether the remaining gap comes from the M6 kernel or from mjlab internals, we set up a controlled experiment reproducing the old good training setup as closely as possible on the current mjlab version:</p>

<div class="table-wrap"><table>
  <tr><th>Setting</th><th>Value (matching old run <code>hhfzw7an</code>)</th></tr>
  <tr><td>Actuator</td><td>XML <code>&lt;position&gt;</code> (MuJoCo built-in PD), not M6 kernel</td></tr>
  <tr><td>Motor params</td><td>Old contaminated M6 export: kp=0.52, damping=0.048, frictionloss=0.006</td></tr>
  <tr><td>Symmetry</td><td>Disabled</td></tr>
  <tr><td>air_time weight</td><td>5.0</td></tr>
  <tr><td>air_time thresholds</td><td>min=0.10, max=0.25</td></tr>
  <tr><td>All DR params</td><td>Identical (CoM &plusmn;3mm, Kp &plusmn;15%, Kd &plusmn;10%, mass &plusmn;5%, IMU &plusmn;1&deg;)</td></tr>
</table></div>

<p>The only difference from the old good run is the mjlab framework version itself (0.1.0 &rarr; 1.2.0).</p>

<h3>14.7 Result: Still Worse</h3>
<p>The isolation experiment produced policies <strong>comparable to the M6 kernel training but still clearly worse than the old best policy</strong>.
Since the physics, DR, noise, delays, rewards, and actuator model are all identical to the old run, the regression must come from the framework upgrade itself.</p>

<h3>14.8 Root Cause: rsl_rl Version Jump</h3>
<p>Deeper investigation revealed:</p>
<ul>
  <li><strong>rsl_rl went from 3.3.0 to 5.0.1</strong> &mdash; a major version jump</li>
  <li>The PPO implementation was significantly refactored: the single <code>ActorCritic</code> module was split into separate <code>actor</code>/<code>critic</code> <code>MLPModel</code>s</li>
  <li>Batch storage, gradient computation, and the optimizer chain were all restructured</li>
  <li>The core loss formulas (surrogate, value clipping, entropy) appear equivalent, but subtle differences in gradient flow or normalization could change what policies are learned</li>
</ul>

<p>We also verified that all base environment defaults in the new mjlab match the old version:</p>
<ul>
  <li>Physics: <code>timestep=0.005</code>, <code>integrator=implicitfast</code>, <code>solver=newton</code>, <code>iterations=10</code>, <code>decimation=4</code> &mdash; identical</li>
  <li>Push robot: we fully override with our params (x/y only, &plusmn;0.3 m/s, interval 3&ndash;6 s) &mdash; identical</li>
  <li>Reset: <code>z=(0.12, 0.13)</code>, no velocity randomization &mdash; identical</li>
  <li>New default events (<code>encoder_bias</code>, <code>base_com</code>): explicitly deleted in our config</li>
</ul>

<p>Pinning rsl_rl back to 3.3.0 is the obvious test, but the new mjlab API expects rsl_rl 5.0.1 interfaces (separate actor/critic models, new config dataclasses), making a simple version pin impractical without significant adaptation work.</p>

<h3>14.9 Current Status</h3>
<p>The situation as of 3 April 2026:</p>
<div class="table-wrap"><table>
  <tr><th>Component</th><th>Status</th></tr>
  <tr><td>BAM M6 actuator kernel</td><td style="color:#27ae60">Validated against testbench data (MAE 0.029 rad)</td></tr>
  <tr><td>qfrc_bias sign bug</td><td style="color:#27ae60">Fixed</td></tr>
  <tr><td>Motor gain DR for BamM6</td><td style="color:#27ae60">Fixed</td></tr>
  <tr><td>Symmetry mirror loss</td><td style="color:#27ae60">Identified as harmful, disabled</td></tr>
  <tr><td>rsl_rl 3.3.0 &rarr; 5.0.1 regression</td><td style="color:#e74c3c">Identified but not yet resolved</td></tr>
</table></div>

<h3>14.10 New Strategy: Perfect the Sim</h3>
<p>Rather than chasing the rsl_rl regression, we&rsquo;re shifting focus to making the simulation as accurate as possible.
The reasoning: if the sim perfectly matches reality, <em>any</em> well-trained policy should transfer &mdash; no &ldquo;luck&rdquo; required.
The old good policy may have just been lucky (robust to the specific sim2real gaps by chance), while a more accurate sim would make all policies transfer reliably.</p>

<h4>Fresh BAM Data Collection Plan</h4>
<p>Re-recording testbench data on a fresh XL330 motor with controlled conditions:</p>

<div class="table-wrap"><table>
  <tr><th>Parameter</th><th>Plan</th><th>Rationale</th></tr>
  <tr><td>Power supply</td><td><strong>Lab supply at 7.4V</strong> (matching runtime nominal_voltage). Additional runs at 6.5V and 8.4V for voltage validation.</td><td>Eliminates battery voltage drift. Extra voltages validate the voltage-proportional kp assumption.</td></tr>
  <tr><td>Arm mass</td><td>0.112 kg (light) + 0.3&ndash;0.5 kg (medium). Avoid &gt;0.8 kg.</td><td>Light for base friction/armature. Medium for load-dependent terms. Heavy caused saturation before.</td></tr>
  <tr><td>Firmware kp</td><td>kp_fw=200 (primary) + 100, 300, 400 (validation)</td><td>200 matches runtime. Others for cross-validation.</td></tr>
  <tr><td>Trajectories</td><td>sin_time_square, sin_sin, up_and_down, <strong>+ chirp</strong> (frequency sweep 0.1&ndash;5 Hz)</td><td>Chirp excites inertial dynamics better for armature identification.</td></tr>
  <tr><td>Multiple motors</td><td>2&ndash;3 different XL330 units</td><td>Captures per-motor variability, informs DR ranges.</td></tr>
  <tr><td>Motor condition</td><td>Fresh motor (not worn)</td><td>Avoids gear wear artefacts from the old motor.</td></tr>
</table></div>

<h4>XML Model Audit</h4>
<p>Full audit of the MuJoCo robot model against the real robot. Key checks:</p>

<div class="table-wrap"><table>
  <tr><th>Check</th><th>XML value</th><th>Action</th></tr>
  <tr><td>Total mass</td><td>0.770 kg</td><td>Weigh real robot with battery. Real robot is ~755g &mdash; 2% off, worth correcting.</td></tr>
  <tr><td>CoM position (home pose)</td><td>[10mm forward, 0mm lateral, 150mm up]</td><td>Real robot leans 7&ndash;8&deg; left, suggesting lateral CoM offset not captured in XML. Measure by suspending the robot or from IMU standing data.</td></tr>
  <tr><td>Joint limits</td><td>14 hinge joints, asymmetric left/right</td><td>Verify each joint&rsquo;s physical range matches XML. A wrong limit means the policy explores unreachable poses.</td></tr>
  <tr><td>Body masses</td><td>Individual link masses from CAD</td><td>Check head assembly (98g), trunk (350g), feet (15g each). Cable mass, battery mass, PCB mass may differ from CAD.</td></tr>
  <tr><td>Collision geometry</td><td>Foot collision geoms</td><td>Verify foot contact shape/position matches real TPU foot pads. Contact point location directly affects balance.</td></tr>
  <tr><td>Joint axes</td><td>All z-axis hinge joints</td><td>Already found hip pitch/roll inversion on new robot. Triple-check all 14 joints.</td></tr>
  <tr><td>Foot friction</td><td>DR range (0.3, 1.2)</td><td>Measure actual surface friction. If walking on a specific surface, narrow the range.</td></tr>
</table></div>

<p>With fresh BAM data, a validated M6 kernel, and a corrected XML model, the simulation accuracy should improve enough that the rsl_rl version becomes less critical &mdash; a robust sim2real gap of near-zero means the policy doesn&rsquo;t need to be &ldquo;lucky&rdquo; to transfer.</p>

<h3>14.11 Fresh BAM Data and Re-identification</h3>
<p>New testbench data was recorded on a fresh XL330 motor with a lab power supply at 7.4V (current limit set correctly this time).
The dataset covers 4 masses (0.08, 0.15, 0.202, 0.26 kg), 4 trajectories (sin_time_square, sin_sin, up_and_down, lift_and_drop), and kp=100&ndash;500. Total: 80 recordings.</p>

<div class="callout warn">
  <strong>Casualties of data collection:</strong> Recording at 0.5 kg with kp=400 stripped a gear tooth in the XL330, destroying the motor.
  The XL330&rsquo;s plastic spur gears cannot handle sustained high-torque tracking at heavy loads.
  Final dataset uses max 0.26 kg (0.255 Nm max external torque).
</div>

<p>Winding resistance was measured directly with a multimeter: <strong>R = 2.5&Omega;</strong>.
This was used to constrain the identification (R bounded to 2.4&ndash;2.6&Omega;) after discovering that with R as a free parameter,
the optimizer consistently pushed R to its upper bound (3.0) and inflated kt to compensate &mdash;
an identifiability issue caused by the entanglement of kt, R, and back-EMF damping with light-load data.</p>

<h4>Identification Results (R constrained to 2.4&ndash;2.6&Omega;)</h4>

<p>Initial identification used only masses 0.08&ndash;0.26 kg. The dataset was then extended with <strong>0.52 kg</strong> recordings
(lift_and_drop, up_and_down, sin_time_square at kp=100&ndash;200) to improve load-friction identifiability.</p>

<h4>Identification Results (R constrained to 2.4&ndash;2.6&Omega;, with 0.52 kg data)</h4>

<div class="table-wrap"><table>
  <tr><th></th><th>kt</th><th>R</th><th>Stall</th><th>MJ kp</th><th>f_base</th><th>f_visc</th><th>armature</th></tr>
  <tr><td><strong>M1 (fresh, R=2.5)</strong></td><td>0.263</td><td>2.60</td><td>0.749 Nm</td><td>0.431</td><td>0.0132</td><td>0.0171</td><td>0.00176</td></tr>
  <tr><td>M5 (fresh, R=2.5)</td><td>0.411</td><td>2.55</td><td>1.194 Nm</td><td>0.687</td><td>0.0056</td><td>0.0049</td><td>0.00171</td></tr>
  <tr><td>M6 (fresh, R=2.5)</td><td>0.389</td><td>2.43</td><td>1.185 Nm</td><td>0.682</td><td>0.0012</td><td>0.0050</td><td>0.00167</td></tr>
  <tr><td><em>M1 (old motor, ref)</em></td><td><em>0.182</em></td><td><em>2.01</em></td><td><em>0.670 Nm</em></td><td><em>0.386</em></td><td><em>0.0317</em></td><td><em>0.0243</em></td><td><em>0.00207</em></td></tr>
  <tr><td><em>M6 (old motor, ref)</em></td><td><em>0.247</em></td><td><em>2.44</em></td><td><em>0.750 Nm</em></td><td><em>0.432</em></td><td><em>0.0078</em></td><td><em>0.0168</em></td><td><em>0.00223</em></td></tr>
</table></div>

<h4>M5/M6 load-friction terms</h4>
<div class="table-wrap"><table>
  <tr><th></th><th>load_motor</th><th>load_ext</th><th>d&theta;_strib</th><th>&alpha;</th></tr>
  <tr><td>M5 (fresh, with 0.52kg)</td><td>0.085</td><td>0.098</td><td>0.807</td><td>0.502</td></tr>
  <tr><td>M6 (fresh, with 0.52kg)</td><td>0.224</td><td>0.085</td><td>2.689</td><td>0.604</td></tr>
  <tr><td><em>M5 (fresh, without 0.52kg)</em></td><td><em>0.140</em></td><td><em>0.282</em></td><td><em>0.523</em></td><td><em>1.923</em></td></tr>
  <tr><td><em>M6 (fresh, without 0.52kg)</em></td><td><em>0.251</em></td><td><em>0.241</em></td><td><em>0.444</em></td><td><em>2.465</em></td></tr>
  <tr><td><em>M6 (old motor, ref)</em></td><td><em>0.177</em></td><td><em>0.333</em></td><td><em>0.108</em></td><td><em>2.109</em></td></tr>
</table></div>

<h4>Key Finding: Structural Identifiability Problem in M5/M6</h4>
<p><strong>Even with 0.52 kg data (0.51 Nm max external torque), the kt inflation problem persists.</strong>
M5 and M6 still show kt~0.39&ndash;0.41, stall torque ~1.19 Nm (60% above datasheet), and friction_viscous ~0.005 (vs M1&rsquo;s 0.017).</p>

<p>The 0.52 kg data helped marginally: friction_viscous improved from 0.001 to 0.005 (4&times;), and
<code>load_friction_external</code> dropped from 0.24&ndash;0.28 to 0.085&ndash;0.098 (confirming the previous values were inflated by data scarcity).
But the fundamental problem is <strong>structural, not data-limited</strong>: on a single-DOF pendulum, back-EMF damping (<code>kt&sup2;/R &times; vel</code>)
and viscous friction (<code>friction_viscous &times; vel</code>) are perfectly collinear. No amount of pendulum data can separate them
when the M5/M6 load-dependent terms absorb the static torque constraints that would otherwise pin kt/R.</p>

<p><strong>M1 is unaffected</strong> because it has no load-dependent terms &mdash; kt/R must simultaneously explain static torque and dynamic damping.
M1 results are identical with and without the 0.52 kg data (kt=0.263 vs 0.265), confirming it was already well-constrained.</p>

<p>Velocity damping breakdown (back-EMF + viscous):</p>
<div class="table-wrap"><table>
  <tr><th></th><th>back-EMF (kt&sup2;/R)</th><th>friction_viscous</th><th>Total</th></tr>
  <tr><td>M1 (fresh)</td><td>0.027</td><td>0.017</td><td>0.044</td></tr>
  <tr><td>M5 (fresh)</td><td>0.066</td><td>0.005</td><td>0.071</td></tr>
  <tr><td>M6 (fresh)</td><td>0.062</td><td>0.005</td><td>0.067</td></tr>
</table></div>

<div class="callout ok">
  <strong>Decision: train with M1 newest (R=2.5).</strong>
  The M1 parameters (kt=0.263, R=2.60, stall=0.75 Nm) are physically consistent, stable across dataset variations, and trustworthy.
  The fresh motor is genuinely different from the old motor: higher kt/R ratio (0.101 vs 0.091), lower friction_base (0.013 vs 0.032), lower friction_viscous (0.017 vs 0.024).
  For future M5/M6 use with the BAM M6 kernel, pinning friction_viscous to M1&rsquo;s value (~0.017) during M5/M6 identification would break the kt/viscous entanglement.
</div>

<h3>14.12 Results: Fresh M1 Params &mdash; Major Improvement</h3>
<p>Training with the fresh motor M1 params (kt=0.263, R=2.60, symmetry off) produced <strong>the best sim2real transfer since the mjlab upgrade</strong>:</p>
<ul>
  <li><strong>action_scale improved to ~0.8</strong> (up from 0.65 with old params). This is the biggest action_scale improvement we&rsquo;ve achieved.</li>
  <li>Forward walking works well &mdash; approaching the quality of the old best policy.</li>
  <li>Turning and backward walking still cause occasional backward falls, suggesting a <strong>forward CoM bias</strong> in the sim model vs reality.</li>
  <li>Head shaking while standing at action_scale=0.8 &mdash; the head/neck joints are unloaded and may have different friction characteristics than leg joints (all 14 joints share the same BAM params).</li>
</ul>

<h3>14.13 CoM Randomisation Curriculum</h3>
<p>To address the backward falling without destabilising early training, we added a <strong>curriculum on CoM randomisation range</strong>:</p>

<div class="table-wrap"><table>
  <tr><th>Training step</th><th>CoM range</th></tr>
  <tr><td>0 &ndash; 1000 iterations</td><td>&plusmn;3 mm (original value)</td></tr>
  <tr><td>1000 &ndash; 2000 iterations</td><td>&plusmn;5 mm</td></tr>
  <tr><td>2000+ iterations</td><td>&plusmn;8 mm</td></tr>
</table></div>

<p>Direct jump to &plusmn;8 mm caused the policy to learn a weird gait with excessive head motion &mdash;
the larger CoM uncertainty was too disruptive for early training.
The curriculum lets the policy first learn a stable gait with small CoM variation, then gradually become robust to larger offsets.</p>

<p>Result: the CoM curriculum did <strong>not</strong> fix the backward falling. The policy still tips over when turning or walking backward,
despite being robust to &plusmn;8 mm CoM offsets in sim. This suggests the issue is not (only) CoM offset but possibly:</p>
<ul>
  <li><strong>Inaccurate body inertias</strong> &mdash; the CAD model may not reflect the real mass distribution (battery placement, cables, PCB weight)</li>
  <li><strong>Contact model mismatch</strong> &mdash; foot geometry, friction coefficient, or ground compliance differ between sim and real</li>
  <li><strong>Actuator asymmetry</strong> &mdash; real motors have per-unit friction differences not captured by uniform DR</li>
  <li><strong>The rsl_rl 5.0.1 PPO</strong> may produce policies that are less robust to backward motion specifically (the old best policy handled this fine)</li>
</ul>

<h3>14.14 Summary of Progress</h3>
<div class="table-wrap"><table>
  <tr><th>Configuration</th><th>action_scale on real robot</th><th>Walk quality</th></tr>
  <tr><td>Old best policy (Feb, old mjlab)</td><td>~0.65</td><td>Excellent</td></tr>
  <tr><td>New mjlab + old params + symmetry ON</td><td>~0.50</td><td>Poor</td></tr>
  <tr><td>New mjlab + M6 kernel + symmetry OFF</td><td>~0.65</td><td>OK</td></tr>
  <tr><td>New mjlab + old XML actuator (isolation)</td><td>~0.65</td><td>OK</td></tr>
  <tr><td><strong>New mjlab + fresh M1 params + symmetry OFF</strong></td><td><strong>~0.80</strong></td><td><strong>Good (forward), backward falls</strong></td></tr>
  <tr><td>+ CoM curriculum (&plusmn;3&rarr;&plusmn;8 mm)</td><td>~0.80</td><td>Same &mdash; still falls backward on turns/reverse</td></tr>
</table></div>

<p>The action_scale progression (0.50 &rarr; 0.65 &rarr; 0.80) shows that each fix contributed meaningfully:
disabling symmetry, fixing motor params with fresh BAM data on a measured R, and training with the correct motor for the robot being tested.</p>
''')}


{html_section("update-apr4", "Update (4 Apr 2026): Next Experiments", '''
<p>With the CoM curriculum not fully resolving the backward-falling issue, we analysed the remaining gap systematically and identified four concrete experiments to try next.</p>

<h3>Why Are We Still Behind the Old Best Policy?</h3>

<p>Several hypotheses, ranked by likelihood:</p>

<ol>
  <li><strong>80g mass approximation.</strong> The real robot is ~80g heavier than the CAD model (screws, wires, PCB, connectors).
  We lumped all 80g onto <code>trunk_base</code>, but the real mass is distributed &mdash; some of it at knee level, foot level, along the arms.
  Adding it all at the top makes the simulated robot artificially top-heavy, raising the CoM and increasing the pitch moment of inertia.
  This would directly cause backward-falling instability: the sim robot is harder to tip than the real one in the forward direction, and easier to tip backward.
  <strong>Fix:</strong> redistribute the 80g more realistically &mdash; e.g. 40g at trunk_base, remainder at knee or foot attachment points near where the PCB/battery actually sits.</li>

  <li><strong>rsl_rl 3.3.0 &rarr; 5.0.1 architecture change.</strong> The upgrade split the actor and critic into separate MLP modules with different weight initialisation.
  Even if the PPO math is identical, the inductive bias of the network may differ. The old policy&rsquo;s weights may have had properties (e.g. more conservative action magnitudes) that naturally transferred better.
  This is hard to isolate without retraining with the pinned old version.</li>

  <li><strong>Per-motor friction variation.</strong> All 14 joints use identical BAM params in sim, but real servos vary by unit.
  The neck/head joints in particular are unloaded and may have very different friction characteristics than the loaded leg joints.
  The head shaking at action_scale=0.8 is a symptom of this.</li>

  <li><strong>Floor friction mismatch.</strong> The old policy was trained with floor friction randomised over (0.3, 1.2) with a base of 0.6 &mdash;
  covering everything from slippery tile to carpet. With the new grippier footpad, the real robot now operates at friction ~1.0.
  If the policy was trained spending most time in low-friction envs, it may have learned a gait optimised for sliding feet rather than planting them.</li>
</ol>

<h3>Experiment 1: Redistribute the 80g Mass</h3>
<p>Instead of all 80g on <code>trunk_base</code>, split it more realistically.
The real robot&rsquo;s extra mass comes from: battery connector + wiring harness (near trunk, but lower), PCB board (mid-torso), screws at every joint (distributed).
A rough but better split: ~40g at trunk_base (upper torso electronics), ~20g at each hip yaw link (screws + wire strain relief).
This lowers the effective CoM and reduces pitch inertia, which should help backward stability.</p>

<h3>Experiment 2: Tighter Floor Friction for the Grippier Footpad</h3>
<p>The new footpad material is significantly grippier. We updated the sim accordingly:</p>
<ul>
  <li><strong>Base friction:</strong> 0.6 &rarr; 1.0 (in the robot XML collision geometry)</li>
  <li><strong>Randomisation range:</strong> (0.3, 1.2) &rarr; (0.7, 1.3) &mdash; removes the low-friction tail the policy was wasting capacity on</li>
</ul>
<p>This should produce a more planted gait and potentially fix the sliding/tipping seen on turns.</p>

<h3>Experiment 3: Later Checkpoint Export</h3>
<p>Current export strategy: wait for CoM curriculum to end (iter 2000), then +500 steps = export at ~2500.
The concern is that this may be too early &mdash; the policy has only had 500 iterations to consolidate under full randomisation.
We&rsquo;ll try exporting at 3500&ndash;4000 iterations (reward plateau after curriculum), watching for the common pattern where
the <em>best sim checkpoint is the worst real-robot checkpoint</em> because it over-exploits sim-specific dynamics.
The right timing is just after the reward plateaus, not the absolute peak.</p>

<h3>Experiment 4: Pinned friction_viscous for M6 Identification</h3>
<p>The structural identifiability problem in M5/M6 (back-EMF and viscous friction are collinear on a pendulum testbench)
means kt is inflated and friction_viscous is under-estimated in M6 fits.
The fix: pin friction_viscous to M1&rsquo;s value (~0.017) during M6 identification, which breaks the entanglement and lets kt converge to its true value.
If kt comes out close to the M1 value (~0.263), the M6 load-friction terms can then be trusted for training with the full BAM M6 kernel.</p>

<div class="callout warn">
  <strong>Current status:</strong> Experiments 1&ndash;3 are queued for the next training run. Experiment 4 requires re-running BAM identification with the viscous pin.
  We are close &mdash; action_scale 0.8 with good forward walking suggests the remaining gap is a few targeted fixes rather than a fundamental problem.
</div>
''')}


{html_section("update-apr5", "Update (5 Apr 2026): Tuesday Test Checklist", '''
<p>A new policy has been trained with two changes applied simultaneously:</p>
<ul>
  <li><strong>New default neck/head pose:</strong> neck_pitch=&minus;20&deg;, head_pitch=+20&deg; (head stays flat &mdash; DOFs are inverted &mdash; but the neck assembly tilts backward, shifting head mass rearward to align CoM over the feet)</li>
  <li><strong>Tighter floor friction:</strong> base friction 0.6&rarr;1.0, randomisation range (0.3,&thinsp;1.2)&rarr;(0.7,&thinsp;1.3) to match the new grippier footpad</li>
</ul>

<p>Robot access resumes Tuesday. Things to test, in order:</p>

<h3>1. Basic Transfer &mdash; Does the New Policy Walk?</h3>
<p>Two policies to test:</p>
<div class="table-wrap"><table>
  <tr><th>Run</th><th>Changes vs previous best</th></tr>
  <tr><td><code>pollen-robotics/mjlab_microduck/elque458</code></td><td>New friction only (base 1.0, range 0.7&ndash;1.3)</td></tr>
  <tr><td><code>pollen-robotics/mjlab_microduck/izc73yop</code></td><td>New friction <strong>+</strong> neck&minus;20&deg;/head+20&deg; default pose</td></tr>
</table></div>
<div class="callout">
  <code>uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path pollen-robotics/mjlab_microduck/elque458 --action-scale 0.8</code><br/>
  <code>uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path pollen-robotics/mjlab_microduck/izc73yop --action-scale 0.8</code>
</div>

<h3>2. Head Shaking &mdash; Test the Low-Pass Filter</h3>
<p>At action_scale=0.8 the head oscillates while standing still because the neck/head joints are lightly loaded and the BAM friction model was identified under load.
A low-pass filter has been added to the runtime (<code>--head-low-pass</code>, default &alpha;=0.3).
Test with and without it:</p>
<div class="callout">
  <code>microduck --action-scale 0.8 --head-low-pass</code><br/>
  <code>microduck --action-scale 0.8 --head-low-pass --head-low-pass-alpha 0.5</code>
</div>
<p>If &alpha;=0.3 is too sluggish for intentional head movements (head mode), try 0.5. If still shaky, try 0.2.</p>

<h3>3. Backward Falling &mdash; Is the CoM Fix Working?</h3>
<p>The &minus;20&deg;/+20&deg; neck/head default was specifically designed to align the CoM over the feet.
Test: walk forward, then turn, then walk backward. The key question is whether the robot still falls backward on turns and reverse &mdash; this is the clearest symptom of a forward-biased CoM in the model.</p>

<h3>4. Friction Feel &mdash; Is the Gait More Planted?</h3>
<p>With floor friction base raised to 1.0 and the low-friction tail removed from training, the policy should produce a more confident, planted gait.
Subjectively: does it feel less &ldquo;slidy&rdquo; than before? Does it handle turning better on the test surface?</p>

<h3>5. Checkpoint Comparison (if time)</h3>
<p>The current export is at ~2500 iterations (500 after CoM curriculum ends). Try also exporting at ~3500&ndash;4000 (reward plateau).
Later checkpoints sometimes transfer worse due to sim over-fitting, but sometimes better if the curriculum consolidation needed more time.</p>

<div class="callout warn">
  <strong>Priority order:</strong> 1 (does it work at all?) &rarr; 3 (backward falling fixed?) &rarr; 2 (head shaking) &rarr; 4 (gait feel) &rarr; 5 (checkpoint timing).
  If backward falling is not fixed, the most likely remaining causes are: (a) the 80g extra mass is all on trunk_base instead of distributed, (b) the rsl_rl 5.0.1 architecture change.
</div>
''')}

<!-- ── Update 7 Apr ── -->
{html_section("update-apr7", "Update (7 Apr 2026): Two Key Findings", '''
<h3>Finding 1: The mjlab Update Was the Root Cause All Along</h3>
<p>After months investigating sim2real failures (motor model, CoM placement, reward weights, observation noise),
the true culprit was a single dependency update commit <code>dbaec69</code> (&ldquo;updating mjlab to latest release&rdquo;) that bumped:</p>
<div class="table-wrap"><table>
  <thead><tr><th>Dependency</th><th>Before (working)</th><th>After (broken)</th></tr></thead>
  <tbody>
    <tr><td>mjlab rev</td><td><code>d1d32d8b&hellip;</code></td><td><code>5af32e37&hellip;</code></td></tr>
    <tr><td>rsl-rl-lib</td><td>3.3.0</td><td>5.0.1</td></tr>
    <tr><td>mujoco</td><td>3.4.0</td><td>3.6.0</td></tr>
    <tr><td>warp-lang</td><td>1.11.0</td><td>1.12.0</td></tr>
  </tbody>
</table></div>
<p>The update likely changed subtle simulation behaviour &mdash; contact dynamics, constraint solver parameters,
or warp kernel numerics &mdash; in ways invisible during training but catastrophic at deployment.
Rolling back to the old <code>uv.lock</code> and retraining immediately produced a policy matching the quality
of the original good policy.</p>
<pre><code># pyproject.toml
mjlab = { git = "https://github.com/mujocolab/mjlab.git",
          rev = "d1d32d8b86e68fe317356de2561f4efc63ffcc29" }
override-dependencies = ["mujoco&gt;=3.4.0"]</code></pre>
<div class="callout warn">
  <strong>Lesson:</strong> Framework updates are not free. Even a &ldquo;minor&rdquo; sim backend bump can silently break
  sim2real with no visible degradation in training metrics. Before updating any simulation dependency,
  train a short walk policy and test on hardware &mdash; if action_scale drops vs the known-good baseline, revert.
</div>

<h3>Finding 2: Over-Training Hurts Sim2Real</h3>
<p>With the rolled-back mjlab, a second finding emerged: <strong>training too long degrades transfer</strong>.</p>
<ul>
  <li>At the <strong>final checkpoint</strong> (normal training length): robot shakes at action_scale=0.65.</li>
  <li>At <strong>~2000 iterations</strong>: clean transfer, no shaking.</li>
</ul>
<p>Past the point where the policy walks robustly, the optimizer exploits simulation-specific dynamics &mdash;
precise contact timing, exact actuator curves, numerical artefacts in warp kernels. These strategies are
brittle: any sim2real gap causes them to break down as high-frequency oscillation.
Short-trained policies use a more conservative strategy that tolerates model mismatch naturally.</p>
<div class="callout">
  <strong>Protocol:</strong> export at <strong>~2000 iterations</strong> as the primary candidate. Test on hardware before
  training further. Run the action_scale sweep on the early checkpoint &mdash; a policy needing
  action_scale=0.5 at the end may work fine at 0.65&ndash;0.8 at iter 2000.
</div>
''')}

</div><!-- /wrapper -->
"""

    return body
