"""Interactive play for any trained microduck velocity / hop-forward task.

Replaces the auto-sampled twist command with keyboard input so you can
drive the trained policy in real time and verify its response to
arbitrary (vx, vy, ωz) commands.  Works for any task that uses mjlab's
stock `UniformVelocityCommand` — pass the task id with --task.

Examples:
    # Local checkpoint:
    uv run python scripts/play_hop_forward_interactive.py \\
        --task Mjlab-Velocity-Flat-MicroDuck-Sprung \\
        --checkpoint_file logs/rsl_rl/microduck_velocity_sprung/<ts>/model_9999.pt

    # Checkpoint downloaded from a wandb run (matches the stock `play`
    # CLI's --wandb-run-path).  Requires real wandb, not trackio:
    MJLAB_MICRODUCK_LOGGER=wandb uv run python scripts/play_hop_forward_interactive.py \\
        --task Mjlab-Velocity-Flat-MicroDuck-Sprung \\
        --wandb-run-path entity/mjlab_microduck/<run_id>

    # Previous (legacy) hop-forward task — works the same way:
    uv run python scripts/play_hop_forward_interactive.py \\
        --task Mjlab-HopForward-MicroDuck-Sprung \\
        --checkpoint_file logs/rsl_rl/microduck_hop_forward_sprung/<ts>/model_XXXX.pt

Controls (focus the viewer window first):
    ↑ / ↓       lin_vel_x  (forward / backward)
    ← / →       ang_vel_z  (turn left / right)
    Q / E       lin_vel_y  (strafe left / right)
    Space       zero all commands
    R           reset the robot (use when it falls and can't recover)
    K           toggle the env's random pushes (the periodic velocity
                perturbations the training events apply).  NOTE: don't
                use X for this — mujoco's native viewer binds X to
                `mjVIS_TEXTURE` (toggles ground texture).
    V           toggle camera follow (track the robot's trunk vs
                free-look mode)
    P           print current command
    + / -       widen / shrink the per-keypress step size

Terminations (fallen, bad orientation, timeout) are disabled so the env
runs continuously — useful for interactive driving, but means the robot
won't auto-reset if it falls.  Use `R` (mjlab's reset shortcut, in the
viewer's right-click menu) or restart the script if it crashes badly.

Plus everything `NativeMujocoViewer` already supports:
    Mouse-drag a body → apply force (test push recovery)
    Right-click → Joints   slider panel (drag *_shank_spring to compress)

Example:
    uv run python scripts/play_hop_forward_interactive.py \\
        --checkpoint_file logs/rsl_rl/microduck_hop_forward_sprung/<ts>/model_5999.pt
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import mujoco
import torch
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401  (task registration)
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.viewer import NativeMujocoViewer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-Velocity-Flat-MicroDuck-Sprung", type=str,
                    help="Registered task id (must use UniformVelocityCommand)")
    # Either a local checkpoint file OR a wandb run path — but not both.
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint_file", type=str,
                     help="Local path to a saved model_XXXX.pt checkpoint")
    src.add_argument("--wandb-run-path", type=str,
                     help="<entity>/<project>/<run_id> — downloads model_XXXX.pt "
                          "from this wandb run.  By default picks the latest "
                          "checkpoint; pair with --checkpoint N for a specific "
                          "iteration.  Requires MJLAB_MICRODUCK_LOGGER=wandb.")
    ap.add_argument("--checkpoint", type=int, default=None,
                    help="With --wandb-run-path: download model_<N>.pt instead "
                         "of the latest.  Useful for rolling back to a checkpoint "
                         "from before a training collapse.")
    ap.add_argument("--device", default=("cuda:0" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--vx-max", type=float, default=1.0,
                    help="cap |vx|. NOTE: training range was -0.15..+0.25 — "
                         "commanding beyond that is out-of-distribution for the policy")
    ap.add_argument("--vy-max", type=float, default=0.5,
                    help="cap |vy|. Training range was ±0.15 — beyond is OOD")
    ap.add_argument("--wz-max", type=float, default=2.0,
                    help="cap |ωz|. Training range was ±0.5 — beyond is OOD")
    args = ap.parse_args()

    # Build the env (play mode = num_envs=1, no command resampling needed
    # since we'll override it anyway).
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(args.task).clip_actions)

    # Resolve checkpoint — local file or download from wandb
    agent_cfg = load_rl_cfg(args.task)
    if args.checkpoint_file is not None:
        checkpoint_path = Path(args.checkpoint_file).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loading local checkpoint: {checkpoint_path}")
    else:
        # Verify we're not aliased to trackio — wandb.Api() in trackio doesn't work
        import wandb
        if not hasattr(wandb, "Api") or wandb.__name__ == "trackio":
            raise RuntimeError(
                "--wandb-run-path requires real wandb.  Prefix the command with "
                "MJLAB_MICRODUCK_LOGGER=wandb so the trackio→wandb alias is skipped."
            )
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        log_root_path.mkdir(parents=True, exist_ok=True)

        if args.checkpoint is not None:
            # Pick a specific iteration — mirrors mjlab/scripts/play.py logic.
            checkpoint_filename = f"model_{args.checkpoint}.pt"
            api = wandb.Api()
            wandb_run = api.run(str(args.wandb_run_path))
            run_id = args.wandb_run_path.split("/")[-1]
            download_dir = log_root_path / "wandb_checkpoints" / run_id
            checkpoint_path = download_dir / checkpoint_filename
            if checkpoint_path.exists():
                print(f"Using cached {checkpoint_filename} (run: {run_id})")
            else:
                available = [f.name for f in wandb_run.files() if "model" in f.name]
                if checkpoint_filename not in available:
                    raise FileNotFoundError(
                        f"Checkpoint '{checkpoint_filename}' not found in wandb run.  "
                        f"Available: {sorted(available)}"
                    )
                wandb_run.file(checkpoint_filename).download(
                    str(download_dir), replace=True,
                )
                print(f"Downloaded {checkpoint_filename} (run: {run_id})")
        else:
            print(f"Resolving latest checkpoint from wandb run {args.wandb_run_path} ...")
            checkpoint_path, was_cached = get_wandb_checkpoint_path(
                log_root_path, Path(args.wandb_run_path)
            )
            print(f"Loaded {'cached' if was_cached else 'downloaded'}: {checkpoint_path.name}")

    runner = OnPolicyRunner(wrapped_env, asdict(agent_cfg), device=args.device)
    runner.load(str(checkpoint_path), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)

    # --- Disable auto-reset on termination ------------------------------------
    # For interactive play we want to keep running through falls / timeouts.
    # Patch the termination manager's compute() so it always returns "no
    # termination", and clears its internal buffers each step.
    no_done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    def _no_terminate():
        env.termination_manager._truncated_buf[:] = False
        env.termination_manager._terminated_buf[:] = False
        return no_done
    env.termination_manager.compute = _no_terminate

    # --- Override the velocity command term -----------------------------------
    # `UniformVelocityCommand` stores the active body-frame command in
    # `vel_command_b` (shape: num_envs × 3) and resamples it periodically via
    # `_resample_command`.  We disable the auto-resample and write directly to
    # the tensor from the keyboard callback.
    # `_update_command` is a no-op for our task (heading_command=False, and
    # rel_standing_envs=0 so is_standing_env is all False).
    twist_term = env.command_manager.get_term("twist")
    twist_term._resample_command = lambda env_ids: None    # no auto-resample

    cmd = torch.zeros(3, device=args.device)               # vx, vy, ωz
    # Larger step sizes to match the wider default ranges (so you don't
    # need 20 key-presses to reach the new cap).  Adjust with `+`/`−`.
    step = {"vx": 0.1, "vy": 0.05, "wz": 0.2}
    bounds = {"vx": args.vx_max, "vy": args.vy_max, "wz": args.wz_max}

    def clamp(v, lim):
        return max(-lim, min(lim, v))

    def print_cmd():
        print(f"\rcmd: vx={cmd[0].item():+.3f}  vy={cmd[1].item():+.3f}  "
              f"ωz={cmd[2].item():+.3f}   "
              f"(step vx±{step['vx']:.02f} vy±{step['vy']:.02f} wz±{step['wz']:.02f})",
              flush=True)

    def write_cmd_to_term():
        twist_term.vel_command_b[:] = cmd.unsqueeze(0)

    write_cmd_to_term()
    print_cmd()

    # GLFW keycodes — match what mjlab's _safe_key_callback forwards.
    KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
    KEY_SPACE, KEY_P, KEY_Q, KEY_E = 32, 80, 81, 69
    KEY_R = 82
    # NOTE: avoid letter keys mujoco's native viewer binds to mjVIS_*
    # rendering flag toggles.  The reserved letters (mjVIS abbreviations)
    # are: A B C D E H I J L M N O P R S T U W X.  Safe letters: G K Q V
    # Y Z.  We use K for kicks (push toggle) and V for view (camera
    # follow).  Don't reuse X — it toggles mjVIS_TEXTURE.
    KEY_K, KEY_V = 75, 86
    KEY_PLUS_1, KEY_PLUS_2 = 61, 93     # '=' and ']'
    KEY_MINUS_1, KEY_MINUS_2 = 45, 47   # '-' and '/'

    # Reset flag — set by R key, consumed in the policy wrapper before the
    # next env.step (we can't call env.reset() directly from the keyboard
    # callback because we're not in the viewer's main loop).
    reset_pending = [False]

    # Wrap the env's `push_robot` event so we can toggle it from a keypress.
    # The stock velocity-env template adds a periodic-interval push event
    # that applies random velocity disturbances every 1-3 s for robustness
    # training.  Useful when training; sometimes distracting when watching
    # the gait.  Default ON (matches training behaviour).
    pushes_enabled = [True]
    try:
        _push_cfg = env.event_manager.get_term_cfg("push_robot")
        _push_orig_func = _push_cfg.func
        def _gated_push(*a, **k):
            if pushes_enabled[0]:
                _push_orig_func(*a, **k)
        _push_cfg.func = _gated_push
        _has_push_event = True
    except (KeyError, ValueError):
        _has_push_event = False
        print("  (No 'push_robot' event in this task — K key will be a no-op.)")

    # Camera-follow state.  The NativeMujocoViewer is created below; we
    # capture it via a list so the key_callback (defined first) can reach
    # through to the underlying mujoco passive viewer's `cam` once the
    # viewer is constructed.  The robot's root body id is resolved up-front.
    viewer_ref = [None]
    camera_follow = [False]
    robot_root_body_id = env.scene["robot"].indexing.root_body_id

    def key_callback(keycode: int):
        if keycode == KEY_UP:
            cmd[0] = clamp(cmd[0].item() + step["vx"], bounds["vx"])
        elif keycode == KEY_DOWN:
            cmd[0] = clamp(cmd[0].item() - step["vx"], bounds["vx"])
        elif keycode == KEY_LEFT:
            cmd[2] = clamp(cmd[2].item() + step["wz"], bounds["wz"])
        elif keycode == KEY_RIGHT:
            cmd[2] = clamp(cmd[2].item() - step["wz"], bounds["wz"])
        elif keycode == KEY_Q:
            cmd[1] = clamp(cmd[1].item() + step["vy"], bounds["vy"])
        elif keycode == KEY_E:
            cmd[1] = clamp(cmd[1].item() - step["vy"], bounds["vy"])
        elif keycode == KEY_SPACE:
            cmd[:] = 0.0
        elif keycode == KEY_R:
            reset_pending[0] = True
            cmd[:] = 0.0                                    # zero command on reset
            print("\n  RESET requested — env will reset on next step")
            return
        elif keycode == KEY_K:
            if _has_push_event:
                pushes_enabled[0] = not pushes_enabled[0]
                print(f"\n  random pushes: {'ON' if pushes_enabled[0] else 'OFF'}")
            else:
                print("\n  No 'push_robot' event in this task — K is a no-op")
            return
        elif keycode == KEY_V:
            # Toggle camera tracking on/off.  When ON, mujoco follows the
            # robot's trunk; when OFF, free-look camera (default).
            if viewer_ref[0] is None or viewer_ref[0].viewer is None:
                print("\n  Viewer not ready yet — try again in a moment")
                return
            mjcam = viewer_ref[0].viewer.cam
            camera_follow[0] = not camera_follow[0]
            if camera_follow[0]:
                mjcam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
                mjcam.trackbodyid = robot_root_body_id
                mjcam.fixedcamid = -1
                print("\n  camera follow: ON (tracking robot)")
            else:
                mjcam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
                mjcam.trackbodyid = -1
                mjcam.fixedcamid = -1
                print("\n  camera follow: OFF (free-look)")
            return
        elif keycode == KEY_P:
            print_cmd(); return
        elif keycode in (KEY_PLUS_1, KEY_PLUS_2):
            for k in step: step[k] *= 1.5
        elif keycode in (KEY_MINUS_1, KEY_MINUS_2):
            for k in step: step[k] /= 1.5
        else:
            return                                          # not our key
        write_cmd_to_term()
        print_cmd()

    print("\nControls: ↑/↓ vx, ←/→ ωz, Q/E vy, Space=zero, R=reset, K=pushes, V=cam-follow, P=print, +/- adjust step")
    print(f"vx limit ±{args.vx_max}, vy limit ±{args.vy_max}, ωz limit ±{args.wz_max}\n")

    # Wrap policy so we re-apply our command override before every step in
    # case mjlab's internal command_manager.compute() resets _command for any
    # reason (cheap insurance).  Also: print cmd vs actual velocity once per
    # second so it's easy to tell whether the policy is responding to commands.
    import time
    base_policy = policy
    last_print = [time.time()]
    asset = env.scene["robot"]

    n_actions = env.action_manager.action_term_dim[0]
    zero_action = torch.zeros((env.num_envs, n_actions), device=args.device)

    def policy_with_cmd_override(obs):
        # Handle pending reset (R key) — call env.reset() and return a zero
        # action so the next env.step doesn't immediately destabilise the
        # freshly-reset state with whatever the policy would have output.
        if reset_pending[0]:
            env.reset()
            write_cmd_to_term()
            reset_pending[0] = False
            return zero_action

        write_cmd_to_term()
        now = time.time()
        if now - last_print[0] > 1.0:
            with torch.no_grad():
                vxy = asset.data.root_link_lin_vel_b[0, :2]
                wz  = asset.data.root_link_ang_vel_b[0, 2]
                z   = asset.data.root_link_pos_w[0, 2]
            print(f"\r  cmd: vx={cmd[0].item():+.2f} vy={cmd[1].item():+.2f} ωz={cmd[2].item():+.2f}  "
                  f"|  actual: vx={vxy[0].item():+.2f} vy={vxy[1].item():+.2f} ωz={wz.item():+.2f}  "
                  f"|  trunk_z={z.item()*1000:.0f}mm",
                  flush=True)
            last_print[0] = now
        return base_policy(obs)

    viewer = NativeMujocoViewer(
        env=wrapped_env,
        policy=policy_with_cmd_override,
        key_callback=key_callback,
        enable_perturbations=True,
    )
    viewer_ref[0] = viewer    # let the V-key handler reach through to viewer.cam
    viewer.run()


if __name__ == "__main__":
    main()
