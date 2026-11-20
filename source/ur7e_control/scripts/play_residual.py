"""Roll out a trained residual policy (or baseline) and plot / view the trajectory.

Examples:
  python -m ur7e_control.scripts.play_residual --seed 0
  python -m ur7e_control.scripts.play_residual --model logs/residual/residual_ppo.zip --seed 0
  mjpython -m ur7e_control.scripts.play_residual --model logs/residual/residual_ppo.zip --viewer
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ur7e_control.backends.sim_ur7e import SimPlantParams, SimUR7e
from ur7e_control.control.limits import CONTROL_DT, HOME_Q
from ur7e_control.control.logging import TrajectoryLogger
from ur7e_control.control.tracker import JointTracker
from ur7e_control.envs.trajectory import INNER_STEPS, RESIDUAL_SCALE
from ur7e_control.scripts.trajectory_visualization import TrajectoryOverlay, plot_comparison
from ur7e_control.trajectory.min_jerk import MinJerkTrajectory, random_waypoints_near_home


def _load_policy(model_path: Path | None):
    if model_path is None:
        return None
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install RL extras: uv pip install -e '.[rl]'") from exc
    model = PPO.load(str(model_path))

    def policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float64).reshape(6)

    return policy


def _obs(q, qd, q_ref, qd_ref, qdd_ref, prev_action, t, duration) -> np.ndarray:
    e = q_ref - q
    phase = 0.0 if duration <= 0 else t / duration
    return np.concatenate(
        [e / 0.1, qd / 1.0, qd_ref / 1.0, qdd_ref / 5.0, prev_action, [phase]]
    ).astype(np.float32)


def _rollout(
    traj: MinJerkTrajectory,
    *,
    policy,
    plant: SimPlantParams,
    kp: float,
    visualize: bool,
    speed: float = 1.0,
    loop: bool = False,
) -> TrajectoryLogger:
    if speed <= 0 or not np.isfinite(speed):
        raise ValueError("speed must be finite and greater than zero")
    robot = SimUR7e(q0=traj.waypoints[0], params=plant, visualize=visualize)
    tracker = JointTracker(kp=kp)
    tracker.reset()
    logger = TrajectoryLogger()
    prev_a = np.zeros(6)
    t = 0.0
    n_steps = int(np.ceil(traj.duration / CONTROL_DT)) + 1
    viewer = None
    overlay = None
    if visualize:
        import mujoco
        import mujoco.viewer

        assert robot._model is not None and robot._data is not None
        viewer = mujoco.viewer.launch_passive(robot._model, robot._data)
        overlay = TrajectoryOverlay(
            robot._model, traj, actual_label="PPO + tracker" if policy is not None else "Scripted tracker",
        )

    try:
        while True:
            robot.reset(traj.waypoints[0])
            tracker.reset()
            logger = TrajectoryLogger()
            prev_a = np.zeros(6)
            t = 0.0
            residual = None
            if overlay is not None:
                overlay.reset()
            for step in range(n_steps):
                if viewer is not None and not viewer.is_running():
                    break
                step_start = time.perf_counter()
                q, qd = robot.read()
                q_ref, qd_ref, qdd_ref = traj.sample(t)
                # Hold each action for ten control ticks, as in training.
                if policy is not None and step % INNER_STEPS == 0:
                    obs = _obs(q, qd, q_ref, qd_ref, qdd_ref, prev_a, t, traj.duration)
                    action = np.clip(policy(obs), -1.0, 1.0)
                    residual = action * RESIDUAL_SCALE
                    prev_a = action
                qd_cmd = tracker.compute(q, q_ref, qd_ref, qdd_ref, residual_qd=residual)
                robot.command_velocity(qd_cmd)
                logger.record(t, q_ref, qd_ref, q, qd, qd_cmd)
                if viewer is not None:
                    next_t = min(t + CONTROL_DT, traj.duration)
                    next_ref, _, _ = traj.sample(next_t)
                    overlay.update(viewer, next_ref, robot.q, next_t)
                    viewer.sync()
                    leftover = CONTROL_DT / speed - (time.perf_counter() - step_start)
                    if leftover > 0:
                        time.sleep(leftover)
                t += CONTROL_DT
                if t > traj.duration:
                    break
            if not loop or viewer is None or not viewer.is_running():
                break
    finally:
        robot.stop()
        if viewer is not None:
            # Leave the final pose visible briefly, then close.
            if viewer.is_running():
                time.sleep(1.0)
            viewer.close()
    return logger


def _plot(log: TrajectoryLogger, waypoints: np.ndarray, out: Path, title: str) -> Path:
    return plot_comparison(log, waypoints, out, title)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Visualize residual / baseline trajectory tracking")
    p.add_argument("--model", type=Path, default=Path("logs/residual/residual_ppo.zip"))
    p.add_argument("--baseline", action="store_true", help="Ignore model; zero residual only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offset-scale", type=float, default=0.25)
    p.add_argument("--kp", type=float, default=8.0)
    p.add_argument("--outdir", type=Path, default=Path("logs/play"))
    p.add_argument("--viewer", action="store_true", help="MuJoCo passive viewer (use mjpython on macOS)")
    p.add_argument("--speed", type=float, default=1.0, help="Viewer playback speed (0.25 = quarter speed)")
    p.add_argument("--loop", action="store_true", help="Repeat the trajectory until the viewer is closed")
    args = p.parse_args(argv)
    if args.speed <= 0 or not np.isfinite(args.speed):
        p.error("--speed must be finite and greater than zero")
    if args.loop and not args.viewer:
        p.error("--loop requires --viewer")

    rng = np.random.default_rng(args.seed)
    waypoints = random_waypoints_near_home(
        HOME_Q, n_waypoints=4, offset_scale=args.offset_scale, rng=rng
    )
    traj = MinJerkTrajectory(waypoints)
    # Fixed mild plant so plots are comparable.
    plant = SimPlantParams(delay_s=0.006, tau_s=0.025, gain=np.ones(6), meas_noise_std=0.0)

    print("Trajectory: 4 joint-space waypoints (home -> 2 random offsets -> home)")
    print("  interpolated with rest-to-rest min-jerk quintics")
    print(f"  duration={traj.duration:.2f}s  offset_scale={args.offset_scale}  seed={args.seed}")
    print("Waypoints q (rad):")
    for i, w in enumerate(waypoints):
        print(f"  [{i}] {np.array2string(w, precision=3, suppress_small=True)}")

    policy = None if args.baseline else _load_policy(args.model if args.model.is_file() else None)
    if policy is None and not args.baseline:
        print(f"Model not found at {args.model}; running baseline (zero residual).")
    label = "baseline" if policy is None else "residual"
    controller_label = "scripted tracker" if policy is None else "PPO + tracker"
    print(f"Orange dashed: reference | Blue: {controller_label} | paths at Hand-E tip")
    if args.loop:
        print("Repeating trajectory; close the viewer window to quit.")
    log = _rollout(
        traj, policy=policy, plant=plant, kp=args.kp, visualize=args.viewer,
        speed=args.speed, loop=args.loop,
    )
    out = args.outdir / f"play_{label}_seed{args.seed}.png"
    path = _plot(log, waypoints, out, title=f"Reference vs {controller_label} (seed={args.seed})")
    print(f"RMS tracking error: {log.rms_tracking_error():.6f} rad")
    print(f"saved {path}")
    print(f"saved {path.with_stem(path.stem + '_joints')}")
    print(f"saved {path.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
