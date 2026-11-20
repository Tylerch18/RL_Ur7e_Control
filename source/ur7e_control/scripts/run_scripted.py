"""Run the scripted joint tracker in sim (default) or on a real UR via RTDE."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ur7e_control.backends.rtde_ur7e import RTDEUR7e
from ur7e_control.backends.sim_ur7e import SimPlantParams, SimUR7e
from ur7e_control.control.limits import HOME_Q, SAFE_A_MAX, SAFE_V_MAX
from ur7e_control.control.runner import run_scripted_tracking
from ur7e_control.control.tracker import JointTracker
from ur7e_control.trajectory.min_jerk import MinJerkTrajectory, random_waypoints_near_home


def _plot(log_path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return None

    data = np.load(log_path)
    t = data["t"]
    err = data["q"] - data["q_ref"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for j in range(6):
        axes[0].plot(t, err[:, j], label=f"j{j}")
    axes[0].set_ylabel("q error (rad)")
    axes[0].legend(ncol=6, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, np.linalg.norm(err, axis=1))
    axes[1].set_ylabel("||e||")
    axes[1].set_xlabel("t (s)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = log_path.with_suffix(".png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Scripted UR7e trajectory tracking")
    p.add_argument("--backend", choices=("sim", "real"), default="sim")
    p.add_argument("--ip", type=str, default=None, help="Robot IP (real only)")
    p.add_argument(
        "--i-am-at-the-robot",
        action="store_true",
        help="Required with --backend real to enable live motion",
    )
    p.add_argument("--live", action="store_true", help="Disable dry-run on real backend")
    p.add_argument("--logdir", type=Path, default=Path("logs/scripted"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offset-scale", type=float, default=0.2)
    p.add_argument("--kp", type=float, default=8.0)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--visualize", action="store_true", help="Mirror sim state into MuJoCo")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    waypoints = random_waypoints_near_home(
        HOME_Q, n_waypoints=4, offset_scale=args.offset_scale, rng=rng
    )

    if args.backend == "sim":
        # Mild plant lag so the tracker has something to correct.
        robot = SimUR7e(
            q0=waypoints[0],
            params=SimPlantParams(delay_s=0.006, tau_s=0.025, gain=np.ones(6)),
            visualize=args.visualize,
            rng=rng,
        )
        v_max, a_max = None, None
    else:
        live = bool(args.live and args.i_am_at_the_robot and args.ip)
        if args.live and not live:
            raise SystemExit("Refusing live motion: need --ip and --i-am-at-the-robot")
        robot = RTDEUR7e(
            ip=args.ip,
            live=live,
            i_am_at_the_robot=args.i_am_at_the_robot,
            v_max=SAFE_V_MAX,
            a_max=SAFE_A_MAX,
            q0_dry=waypoints[0],
        )
        v_max, a_max = SAFE_V_MAX, SAFE_A_MAX

    traj = MinJerkTrajectory(waypoints, v_max=v_max, a_max=a_max)
    tracker = JointTracker(kp=args.kp, v_max=v_max, a_max=a_max)

    print(f"backend={args.backend} live={getattr(robot, 'live', False)} duration={traj.duration:.2f}s")
    logger = run_scripted_tracking(robot, traj, tracker)
    args.logdir.mkdir(parents=True, exist_ok=True)
    log_path = args.logdir / f"scripted_{args.backend}_seed{args.seed}.npz"
    logger.save(log_path)
    rms = logger.rms_tracking_error()
    print(f"RMS tracking error: {rms:.6f} rad")
    print(f"saved {log_path}")
    if not args.no_plot:
        plot_path = _plot(log_path)
        if plot_path:
            print(f"plot {plot_path}")


if __name__ == "__main__":
    main()
