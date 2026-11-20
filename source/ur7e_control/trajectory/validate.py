"""Validate joint trajectories against position / velocity / acceleration limits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ur7e_control.control.limits import A_MAX, JOINT_HIGH, JOINT_LOW, V_MAX
from ur7e_control.trajectory.types import Trajectory

Array = NDArray[np.floating]


class TrajectoryValidationError(ValueError):
    """Raised when a trajectory violates limits."""


def validate_waypoints(
    waypoints: Array,
    *,
    q_low: Array = JOINT_LOW,
    q_high: Array = JOINT_HIGH,
) -> None:
    """Ensure every waypoint lies inside the joint box (no wrapping)."""
    w = np.asarray(waypoints, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 6:
        raise TrajectoryValidationError(f"waypoints must be (N, 6), got {w.shape}")
    if np.any(w < q_low) or np.any(w > q_high):
        raise TrajectoryValidationError("waypoint outside conservative joint box")


def validate_trajectory(
    traj: Trajectory,
    *,
    dt: float = 0.002,
    q_low: Array = JOINT_LOW,
    q_high: Array = JOINT_HIGH,
    v_max: Array = V_MAX,
    a_max: Array = A_MAX,
    atol_v: float = 1e-3,
    atol_a: float = 1e-2,
) -> None:
    """Dense-sample ``traj`` and check q / qd / qdd limits."""
    if traj.duration <= 0:
        raise TrajectoryValidationError("duration must be positive")
    times = np.arange(0.0, traj.duration + 0.5 * dt, dt)
    for t in times:
        q, qd, qdd = traj.sample(float(t))
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        qdd = np.asarray(qdd, dtype=np.float64)
        if q.shape != (6,) or qd.shape != (6,) or qdd.shape != (6,):
            raise TrajectoryValidationError("sample() must return vectors of length 6")
        if np.any(q < q_low - 1e-9) or np.any(q > q_high + 1e-9):
            raise TrajectoryValidationError(f"q out of box at t={t:.4f}")
        if np.any(np.abs(qd) > v_max + atol_v):
            raise TrajectoryValidationError(f"|qd| exceeds vmax at t={t:.4f}")
        if np.any(np.abs(qdd) > a_max + atol_a):
            raise TrajectoryValidationError(f"|qdd| exceeds amax at t={t:.4f}")
