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
from ur7e_control.control.fk import fk_position
from ur7e_control.control.limits import CONTROL_DT, HOME_Q
from ur7e_control.control.logging import TrajectoryLogger
from ur7e_control.control.tracker import JointTracker
from ur7e_control.envs.trajectory import RESIDUAL_SCALE
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
) -> TrajectoryLogger:
    robot = SimUR7e(q0=traj.waypoints[0], params=plant, visualize=visualize)
    tracker = JointTracker(kp=kp)
    tracker.reset()
    logger = TrajectoryLogger()
    prev_a = np.zeros(6)
    t = 0.0
    n_steps = int(np.ceil(traj.duration / CONTROL_DT)) + 1
    viewer = None
    if visualize:
        import mujoco
        import mujoco.viewer

        assert robot._model is not None and robot._data is not None
        viewer = mujoco.viewer.launch_passive(robot._model, robot._data)

    try:
        for _ in range(n_steps):
            q, qd = robot.read()
            q_ref, qd_ref, qdd_ref = traj.sample(t)
            residual = None
            if policy is not None:
                obs = _obs(q, qd, q_ref, qd_ref, qdd_ref, prev_a, t, traj.duration)
                action = np.clip(policy(obs), -1.0, 1.0)
                residual = action * RESIDUAL_SCALE
                prev_a = action
            qd_cmd = tracker.compute(q, q_ref, qd_ref, qdd_ref, residual_qd=residual)
            step_start = time.time()
            robot.command_velocity(qd_cmd)
            logger.record(t, q_ref, qd_ref, q, qd, qd_cmd)
            if viewer is not None:
                viewer.sync()
                leftover = CONTROL_DT - (time.time() - step_start)
                if leftover > 0:
                    time.sleep(leftover)
            t += CONTROL_DT
            if t > traj.duration:
                break
    finally:
        robot.stop()
        if viewer is not None:
            # Leave the final pose visible briefly, then close.
            time.sleep(1.0)
            viewer.close()
    return logger


def _plot(log: TrajectoryLogger, waypoints: np.ndarray, out: Path, title: str) -> Path:
    # mjpython / MuJoCo viewer leaves us off the Cocoa main thread; use Agg.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = log.as_arrays()
    t = data["t"]
    q_ref, q = data["q_ref"], data["q"]
    err = q - q_ref

    # Cartesian path of the reference (DH FK) for intuition.
    xyz_ref = np.array([fk_position(qi) for qi in q_ref])
    xyz = np.array([fk_position(qi) for qi in q])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(title)

    ax = axes[0, 0]
    for j in range(6):
        ax.plot(t, q_ref[:, j], "--", alpha=0.7, label=f"ref j{j}" if j < 3 else None)
        ax.plot(t, q[:, j], alpha=0.9, label=f"q j{j}" if j < 3 else None)
    ax.set_ylabel("q (rad)")
    ax.set_title("Joint trajectory (ref dashed)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for j in range(6):
        ax.plot(t, err[:, j], label=f"j{j}")
    ax.set_ylabel("q error (rad)")
    ax.set_title(f"Tracking error  RMS={log.rms_tracking_error():.4f}")
    ax.legend(ncol=6, fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(xyz_ref[:, 0], xyz_ref[:, 1], "--", label="ref XY")
    ax.plot(xyz[:, 0], xyz[:, 1], label="actual XY")
    ax.scatter(xyz_ref[0, 0], xyz_ref[0, 1], c="g", s=40, zorder=5, label="start")
    ax.scatter(xyz_ref[-1, 0], xyz_ref[-1, 1], c="r", s=40, zorder=5, label="end")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Flange path (top view)")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, xyz_ref[:, 2], "--", label="ref z")
    ax.plot(t, xyz[:, 2], label="actual z")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("z (m)")
    ax.set_title("Flange height")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)

    # Also dump waypoints for clarity.
    np.savez(out.with_suffix(".npz"), **data, waypoints=waypoints)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Visualize residual / baseline trajectory tracking")
    p.add_argument("--model", type=Path, default=Path("logs/residual/residual_ppo.zip"))
    p.add_argument("--baseline", action="store_true", help="Ignore model; zero residual only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offset-scale", type=float, default=0.25)
    p.add_argument("--kp", type=float, default=8.0)
    p.add_argument("--outdir", type=Path, default=Path("logs/play"))
    p.add_argument("--viewer", action="store_true", help="MuJoCo passive viewer (use mjpython on macOS)")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    waypoints = random_waypoints_near_home(
        HOME_Q, n_waypoints=4, offset_scale=args.offset_scale, rng=rng
    )
    traj = MinJerkTrajectory(waypoints)
    # Fixed mild plant so plots are comparable.
    plant = SimPlantParams(delay_s=0.006, tau_s=0.025, gain=np.ones(6), meas_noise_std=0.0)

    print("Trajectory: 4 joint-space waypoints (home → 2 random offsets → home)")
    print("  interpolated with rest-to-rest min-jerk quintics")
    print(f"  duration={traj.duration:.2f}s  offset_scale={args.offset_scale}  seed={args.seed}")
    print("Waypoints q (rad):")
    for i, w in enumerate(waypoints):
        print(f"  [{i}] {np.array2string(w, precision=3, suppress_small=True)}")

    policy = None if args.baseline else _load_policy(args.model if args.model.is_file() else None)
    if policy is None and not args.baseline:
        print(f"Model not found at {args.model}; running baseline (zero residual).")
    label = "baseline" if policy is None else "residual"
    log = _rollout(traj, policy=policy, plant=plant, kp=args.kp, visualize=args.viewer)
    out = args.outdir / f"play_{label}_seed{args.seed}.png"
    path = _plot(log, waypoints, out, title=f"UR7e joint traj — {label} (seed={args.seed})")
    print(f"RMS tracking error: {log.rms_tracking_error():.6f} rad")
    print(f"saved {path}")
    print(f"saved {path.with_suffix('.npz')}")


if __name__ == "__main__":
    main()
