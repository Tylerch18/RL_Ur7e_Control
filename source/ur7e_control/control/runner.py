"""Shared scripted tracking loop for sim and real backends."""

from __future__ import annotations

import numpy as np

from ur7e_control.control.interface import RobotInterface
from ur7e_control.control.limits import CONTROL_DT
from ur7e_control.control.logging import TrajectoryLogger
from ur7e_control.control.tracker import JointTracker
from ur7e_control.trajectory.types import Trajectory
from ur7e_control.trajectory.validate import validate_trajectory


def run_scripted_tracking(
    robot: RobotInterface,
    traj: Trajectory,
    tracker: JointTracker | None = None,
    *,
    dt: float = CONTROL_DT,
    error_abort: float = 0.15,
    validate: bool = True,
    residual_fn=None,
) -> TrajectoryLogger:
    """Track ``traj`` until ``duration``; log full-rate signals.

    ``residual_fn(t, q, qd, q_ref, qd_ref, qdd_ref) -> residual_qd | None`` is
    optional (used by the residual Gym env).
    """
    if validate:
        validate_trajectory(traj, dt=dt)

    tracker = tracker or JointTracker(dt=dt)
    q0, qd0 = robot.read()
    tracker.reset(qd0)

    # Align to start.
    q_start, _, _ = traj.sample(0.0)
    if hasattr(robot, "move_to_start"):
        robot.move_to_start(q_start)  # type: ignore[attr-defined]

    logger = TrajectoryLogger()
    t = 0.0
    n_steps = int(np.ceil(traj.duration / dt)) + 1

    try:
        for _ in range(n_steps):
            q, qd = robot.read()
            q_ref, qd_ref, qdd_ref = traj.sample(t)
            err = np.max(np.abs(q - q_ref))
            if err > error_abort:
                robot.stop()
                raise RuntimeError(f"tracking error {err:.4f} rad exceeds abort limit {error_abort}")

            residual = None
            if residual_fn is not None:
                residual = residual_fn(t, q, qd, q_ref, qd_ref, qdd_ref)

            qd_cmd = tracker.compute(q, q_ref, qd_ref, qdd_ref, residual_qd=residual)
            robot.command_velocity(qd_cmd)
            logger.record(t, q_ref, qd_ref, q, qd, qd_cmd)
            robot.wait_period()
            t += dt
    except Exception:
        robot.stop()
        raise
    finally:
        robot.stop()

    return logger
